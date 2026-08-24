"""Stage 5: deploy the locked ASL definition to AWS Step Functions.

Deploy is deliberately the LAST stage and the only one that mutates a
real AWS account -- every earlier stage (extraction, enrichment, ASL
generation/validation) is either an LLM call or pure local
computation, and asl_service.validate_asl() only ever READS from AWS
(ValidateStateMachineDefinition). Because deploy is the one
irreversible step, this module re-checks readiness immediately before
the API call rather than trusting whatever validation result the
caller hands it:

  1. The Stage 4 validation report must be `ok` -- structural,
     reachability, spec-consistency, AWS-constraint, and extras groups
     all passed.
  2. If Stage 4's live ValidateStateMachineDefinition call hard-failed
     (aws_validate_ok is False, not None), that blocks deploy too. A
     *skipped/soft-failed* attempt is allowed through -- validate-time
     credentials may differ from deploy-time credentials (e.g. CI vs a
     local dev profile), and CreateStateMachine/UpdateStateMachine
     re-validate the definition server-side regardless.
  3. Zero TODO_ placeholders anywhere in the ASL document. This is the
     one check that is NEW here, not inherited from Stage 4:
     enrichment_service.py's ARN whitelist check *intentionally* lets
     TODO_ placeholders through, because at Stage 3.5 a placeholder is
     a valid sentinel for "a human still needs to fill this in" -- and
     find_placeholders() in plan/render.py only ever looked at the
     thin Plan's service/description text, not the fully-enriched ASL,
     so a placeholder introduced later (e.g. enrichment writing
     TODO_UNKNOWN_INTEGRATION_ARN into a Parameters value) was never
     re-checked. A TODO_ token deployed as a literal resource ARN
     either fails CreateStateMachine outright or, worse, deploys a
     state machine that only fails once a real execution hits it.
  4. A resolvable IAM execution role ARN. Step Functions needs this to
     run the state machine at all -- it is not something this module
     can safely infer or create (that would mean minting IAM policies
     on the caller's behalf), so it must come from settings
     (AWS_STATE_MACHINE_ROLE_ARN) or the request explicitly.

Any failure raises PreDeployCheckError with the FULL list of reasons
(not just the first) so the caller can show the user everything that
needs fixing in one pass instead of a fix-one-see-the-next loop.
deploy_state_machine() never partially deploys: checks run first,
the AWS call happens only if every check passes (or `skip_checks`
was explicitly set, which the router should treat as a deliberate,
logged override, not a default).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from orchestrator.api.config.settings import get_api_settings
from orchestrator.api.services.asl_service import ValidationReport, find_asl_placeholders

# Step Functions state machine names: no whitespace, no
# < > { } [ ] ? * " # % \ ^ | ~ ` $ & , ; : / and max 80 chars.
_NAME_DISALLOWED_RE = re.compile(r"[\s<>{}\[\]?*\"#%\\^|~`$&,;:/]+")
MAX_NAME_LEN = 80


class PreDeployCheckError(Exception):
    """One or more pre-deploy checks failed. `.reasons` holds every
    failure found, not just the first.
    """

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__("; ".join(reasons))


class DeploymentError(Exception):
    """The AWS API call itself failed (bad role, permissions,
    throttling, malformed definition AWS rejected server-side, ...).
    Distinct from PreDeployCheckError so callers can tell "we caught
    it before AWS did" apart from "AWS rejected it".
    """


@dataclass
class DeployResult:
    state_machine_arn: str
    action: Literal["created", "updated"]
    state_machine_name: str
    region: str
    role_arn: str
    checks_passed: list[str] = field(default_factory=list)
    # AWS account id the deploy actually landed in -- parsed from
    # state_machine_arn (arn:aws:states:REGION:ACCOUNT_ID:...), not
    # trusted from caller input. Lets the caller sanity-check "did this
    # really go to MY account" instead of just hoping deploy_role_arn
    # was typed correctly.
    account_id: str = ""


CHECK_LABELS = [
    "validation_report_ok",
    "aws_validate_not_failed",
    "no_unresolved_placeholders",
    "execution_role_configured",
]



def _report_ok(validations: ValidationReport | dict | None) -> bool | None:
    if validations is None:
        return None
    if isinstance(validations, ValidationReport):
        return validations.ok
    return validations.get("ok")


def _aws_validate_ok(validations: ValidationReport | dict | None) -> bool | None:
    if validations is None:
        return None
    if isinstance(validations, ValidationReport):
        return validations.aws_validate_ok
    return validations.get("aws_validate_ok")


def run_pre_deploy_checks(
    asl: dict | None,
    validations: ValidationReport | dict | None,
    role_arn: str | None,
) -> list[str]:
    """Returns a list of human-readable failure reasons. Empty list
    means clear to deploy.
    """
    reasons: list[str] = []

    if not isinstance(asl, dict) or not asl.get("States"):
        return ["No ASL definition (or it has no States) to deploy."]

    report_ok = _report_ok(validations)
    if report_ok is not True:
        reasons.append(
            "ASL validation is not passing yet -- re-run "
            "/session/{id}/asl/validate and resolve the reported errors "
            "before deploying."
            if report_ok is None
            else "ASL validation is failing -- resolve the reported "
            "errors before deploying."
        )

    if _aws_validate_ok(validations) is False:
        reasons.append(
            "AWS's own ValidateStateMachineDefinition call reported this "
            "definition as invalid."
        )

    leftover = find_asl_placeholders(asl)
    if leftover:
        reasons.append(
            "Unresolved placeholders remain in the definition: "
            + ", ".join(leftover)
            + " -- fill these in (POST /session/{id}/approve with "
            "overrides covering every token) before deploying."
        )

    if not role_arn:
        reasons.append(
            "No IAM execution role ARN configured -- set "
            "AWS_STATE_MACHINE_ROLE_ARN, or pass role_arn explicitly "
            "in the deploy request."
        )

    return reasons


def slugify_state_machine_name(raw: str) -> str:
    """Best-effort conversion of a free-text plan name into a valid
    Step Functions state machine name (see _NAME_DISALLOWED_RE above).
    """
    slug = _NAME_DISALLOWED_RE.sub("-", (raw or "").strip()).strip("-")
    slug = slug or "workflow"
    return slug[:MAX_NAME_LEN]


def _base_boto3_session():
    """The API's own identity -- whatever credentials it normally runs
    as (AWS_PROFILE, env vars, or an instance role). This is the
    identity that calls sts:AssumeRole in cross-account mode; it is
    NEVER used to call CreateStateMachine/UpdateStateMachine directly
    once a deploy_role_arn is given (see _sfn_client below) -- the base
    identity typically has no Step Functions permissions of its own,
    only sts:AssumeRole into each tenant's deploy role.
    """
    import boto3

    settings = get_api_settings()
    session_kwargs: dict[str, Any] = {}
    if settings.aws_profile:
        session_kwargs["profile_name"] = settings.aws_profile
    return boto3.Session(**session_kwargs)


def _assume_deploy_role(deploy_role_arn: str, external_id: str | None) -> dict[str, str]:
    """STS AssumeRole into a tenant's own account. Returns temporary
    credentials (valid ~1h) scoped to whatever the deploy role's
    policy allows -- never the API's own long-lived credentials.

    external_id is optional but strongly recommended by AWS for
    third-party cross-account access (the "confused deputy" problem:
    without it, anyone who learns the deploy_role_arn and convinces
    ANY caller with sts:AssumeRole to assume it can deploy into that
    account). If the tenant's trust policy requires an ExternalId and
    none is supplied here, AWS rejects the AssumeRole call itself --
    fails closed, not open.
    """
    sts = _base_boto3_session().client("sts")
    kwargs: dict[str, Any] = {
        "RoleArn": deploy_role_arn,
        "RoleSessionName": "orchestrator-deploy",
    }
    if external_id:
        kwargs["ExternalId"] = external_id
    try:
        response = sts.assume_role(**kwargs)
    except Exception as exc:  # noqa: BLE001 — surfaced as DeploymentError below
        raise DeploymentError(
            f"Could not assume deploy_role_arn {deploy_role_arn}: {exc}. "
            "Check the role's trust policy names this API's own identity "
            "as a trusted principal, and that the external_id (if the "
            "trust policy requires one) matches."
        ) from exc
    creds = response["Credentials"]
    return {
        "aws_access_key_id": creds["AccessKeyId"],
        "aws_secret_access_key": creds["SecretAccessKey"],
        "aws_session_token": creds["SessionToken"],
    }


def _sfn_client(deploy_role_arn: str | None = None, external_id: str | None = None):
    try:
        import boto3  # noqa: F401 — import-checked here so the ImportError below is accurate
    except ImportError as exc:
        raise DeploymentError(
            "boto3 is not installed — pip install boto3 to enable deploy"
        ) from exc

    settings = get_api_settings()

    if deploy_role_arn:
        # Cross-account mode: assume the tenant's deploy role, build the
        # client from the resulting TEMPORARY credentials. The API's own
        # base credentials are never used to touch Step Functions
        # directly in this mode.
        temp_creds = _assume_deploy_role(deploy_role_arn, external_id)
        session = boto3.Session(**temp_creds)
        return session.client("stepfunctions", region_name=settings.aws_region)

    # Single-account mode (unchanged): the API's own base credentials
    # talk to Step Functions directly. This is what a solo/local setup
    # (no deploy_role_arn ever passed) continues to use.
    return _base_boto3_session().client("stepfunctions", region_name=settings.aws_region)


def _find_state_machine_arn_by_name(client, name: str) -> str | None:
    """Paginated lookup by name -- ListStateMachines has no server-side
    name filter, so this scans the account's state machines. Used both
    to decide create-vs-update up front, and (if a race loses to
    another deployer between our lookup and our create call) to
    recover the ARN after a StateMachineAlreadyExists error instead of
    parsing it out of the exception message text.
    """
    paginator = client.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page.get("stateMachines", []):
            if sm.get("name") == name:
                return sm.get("stateMachineArn")
    return None


def _account_id_from_arn(arn: str) -> str:
    """arn:aws:states:REGION:ACCOUNT_ID:stateMachine:NAME -- account_id
    is field index 4. Parsed from AWS's own response, not from caller
    input, so it's a reliable "which account did this actually land
    in" signal regardless of what deploy_role_arn the caller passed.
    """
    parts = arn.split(":")
    return parts[4] if len(parts) > 4 else ""


def _sync_tags(client, state_machine_arn: str, tags: dict[str, str]) -> None:
    if not tags:
        return
    client.tag_resource(
        resourceArn=state_machine_arn,
        tags=[{"key": k, "value": v} for k, v in tags.items()],
    )


def deploy_state_machine(
    name: str,
    asl: dict,
    *,
    role_arn: str | None = None,
    deploy_role_arn: str | None = None,
    external_id: str | None = None,
    validations: ValidationReport | dict | None = None,
    tags: dict[str, str] | None = None,
    skip_checks: bool = False,
) -> DeployResult:
    """Create the state machine if a state machine of this name doesn't
    exist yet, or update it in place if it does. Never raises
    partway through a partial deploy: pre-deploy checks (unless
    explicitly skipped) run to completion first, and the single AWS
    mutating call (create or update) either fully succeeds or raises
    DeploymentError.

    deploy_role_arn selects WHICH AWS ACCOUNT this deploys into (a
    role in the tenant's own account that this API's base identity
    assumes via STS -- see _assume_deploy_role). This is a completely
    different role from role_arn/AWS_STATE_MACHINE_ROLE_ARN, which is
    the execution role Step Functions itself assumes at RUNTIME inside
    whichever account was selected. Omitting deploy_role_arn preserves
    the original single-account behavior (this API's own base
    credentials talk to Step Functions directly) -- nothing breaks for
    callers that don't need multi-account.
    """
    settings = get_api_settings()
    resolved_role_arn = role_arn or settings.aws_state_machine_role_arn

    if not skip_checks:
        reasons = run_pre_deploy_checks(asl, validations, resolved_role_arn)
        if reasons:
            raise PreDeployCheckError(reasons)

    if not resolved_role_arn:
        # Even with skip_checks=True, there is no sane way to deploy
        # without an execution role -- AWS would just reject the call.
        raise PreDeployCheckError(["No IAM execution role ARN configured."])

    full_name = (
        f"{settings.aws_deploy_name_prefix}{name}"
        if settings.aws_deploy_name_prefix
        else name
    )
    sm_type = (settings.aws_state_machine_type or "STANDARD").upper()
    if sm_type not in ("STANDARD", "EXPRESS"):
        sm_type = "STANDARD"
    definition = json.dumps(asl)

    client = _sfn_client(deploy_role_arn, external_id)

    try:
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        raise DeploymentError(
            "boto3/botocore is not installed — pip install boto3 to enable deploy"
        ) from exc

    try:
        existing_arn = _find_state_machine_arn_by_name(client, full_name)

        if existing_arn is not None:
            client.update_state_machine(
                stateMachineArn=existing_arn,
                definition=definition,
                roleArn=resolved_role_arn,
            )
            _sync_tags(client, existing_arn, tags or {})
            return DeployResult(
                state_machine_arn=existing_arn,
                action="updated",
                state_machine_name=full_name,
                region=settings.aws_region,
                role_arn=resolved_role_arn,
                checks_passed=[] if skip_checks else list(CHECK_LABELS),
                account_id=_account_id_from_arn(existing_arn),
            )

        try:
            response = client.create_state_machine(
                name=full_name,
                definition=definition,
                roleArn=resolved_role_arn,
                type=sm_type,
                tags=[{"key": k, "value": v} for k, v in (tags or {}).items()],
            )
            return DeployResult(
                state_machine_arn=response["stateMachineArn"],
                action="created",
                state_machine_name=full_name,
                region=settings.aws_region,
                role_arn=resolved_role_arn,
                checks_passed=[] if skip_checks else list(CHECK_LABELS),
                account_id=_account_id_from_arn(response["stateMachineArn"]),
            )
        except client.exceptions.StateMachineAlreadyExists:
            # Lost a race: something else created `full_name` between
            # our lookup and our create call. Recover the ARN the same
            # reliable way as the up-front lookup, and fall back to
            # update rather than surfacing a spurious error.
            existing_arn = _find_state_machine_arn_by_name(client, full_name)
            if existing_arn is None:
                raise
            client.update_state_machine(
                stateMachineArn=existing_arn,
                definition=definition,
                roleArn=resolved_role_arn,
            )
            _sync_tags(client, existing_arn, tags or {})
            return DeployResult(
                state_machine_arn=existing_arn,
                action="updated",
                state_machine_name=full_name,
                region=settings.aws_region,
                role_arn=resolved_role_arn,
                checks_passed=[] if skip_checks else list(CHECK_LABELS),
                account_id=_account_id_from_arn(existing_arn),
            )
    except (BotoCoreError, ClientError) as exc:
        raise DeploymentError(str(exc)) from exc