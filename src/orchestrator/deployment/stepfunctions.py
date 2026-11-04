"""Deploy an approved, validated ASL definition as an AWS Step Function via boto3.

Phase 1 scope: boto3 only (no Terraform / CloudFormation / CDK).

This module is the public deployment surface for the synthesis package and CLI.
It wraps ``api.services.deploy.service`` so callers outside the API layer can
deploy without depending on FastAPI session models, while still enforcing:

  * human approval gate (``approved=True`` required)
  * pre-deploy validation + placeholder ban
  * idempotent create-or-update by stable state-machine name
  * execution-role resolution from settings when not passed explicitly
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class DeployOutcome:
    """Stable result shape for CLI / synthesis / API callers."""

    ok: bool
    state_machine_arn: str | None = None
    action: Literal["created", "updated"] | None = None
    state_machine_name: str | None = None
    region: str | None = None
    role_arn: str | None = None
    account_id: str | None = None
    checks_passed: list[str] = field(default_factory=list)
    blocked_reasons: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_role_arn(explicit: str | None = None) -> str | None:
    if explicit:
        return explicit
    try:
        from orchestrator.config import get_settings

        arn = get_settings().step_functions_execution_role_arn
        if arn:
            return arn
    except Exception:  # noqa: BLE001
        pass
    try:
        from orchestrator.api.config.settings import get_api_settings

        s = get_api_settings()
        return (
            getattr(s, "aws_state_machine_role_arn", None)
            or getattr(s, "step_functions_execution_role_arn", None)
        )
    except Exception:  # noqa: BLE001
        return None


def pre_deploy_gate(
    asl: dict[str, Any],
    *,
    validation_report: dict[str, Any] | None = None,
    approved: bool = False,
    role_arn: str | None = None,
) -> list[str]:
    """Return human-readable blockers. Empty list means clear to deploy."""
    reasons: list[str] = []

    if not approved:
        reasons.append(
            "Human approval required before deploy "
            "(pass approved=True only after the user reviews the ASL)."
        )

    if not isinstance(asl, dict) or not asl.get("States"):
        reasons.append("No ASL definition (or it has no States) to deploy.")
        return reasons

    from orchestrator.synthesis.validator import pre_deploy_checks

    reasons.extend(
        pre_deploy_checks(
            asl,
            validation_ok=(
                None if validation_report is None else bool(validation_report.get("ok"))
            ),
            execution_role_arn=role_arn or _resolve_role_arn(),
        )
    )

    # De-dupe while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for r in reasons:
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def deploy_state_machine(
    asl: dict[str, Any],
    *,
    name: str | None = None,
    role_arn: str | None = None,
    validation_report: dict[str, Any] | None = None,
    approved: bool = False,
    skip_checks: bool = False,
    deploy_role_arn: str | None = None,
    external_id: str | None = None,
    tags: dict[str, str] | None = None,
    state_machine_type: str = "STANDARD",
) -> DeployOutcome:
    """Create or update a Step Functions state machine from ASL.

    Parameters
    ----------
    asl:
        Amazon States Language document (must contain ``States`` and ``StartAt``).
    name:
        Desired state machine name. Slugified; if omitted, derived from ASL Comment
        or defaults to ``OAF-Workflow``.
    role_arn:
        Execution role the state machine assumes at runtime.
    validation_report:
        Output of ``validate_asl`` / Stage-4 report. Required unless ``skip_checks``.
    approved:
        Explicit human approval flag — mandatory enterprise gate.
    skip_checks:
        Dangerous escape hatch; only for controlled tests. Still requires ``approved``.
    """
    role = role_arn or _resolve_role_arn()

    if not skip_checks:
        blockers = pre_deploy_gate(
            asl,
            validation_report=validation_report,
            approved=approved,
            role_arn=role,
        )
        if blockers:
            return DeployOutcome(ok=False, blocked_reasons=blockers, role_arn=role)
    elif not approved:
        return DeployOutcome(
            ok=False,
            blocked_reasons=[
                "Even with skip_checks, approved=True is required "
                "(human review gate is never bypassed)."
            ],
            role_arn=role,
        )

    try:
        from orchestrator.api.services.deploy.service import (
            DeploymentError,
            PreDeployCheckError,
            deploy_state_machine as _deploy,
            slugify_state_machine_name,
        )
    except ImportError as exc:
        return DeployOutcome(
            ok=False,
            error=f"Deploy service unavailable: {exc}",
            role_arn=role,
        )

    sm_name = slugify_state_machine_name(
        name
        or (asl.get("Comment") if isinstance(asl.get("Comment"), str) else None)
        or "OAF-Workflow"
    )

    # Build a minimal validation object the deploy service understands
    validations = validation_report
    if validations is None and skip_checks:
        validations = {"ok": True, "aws_validate_ok": None}

    try:
        result = _deploy(
            asl=asl,
            name=sm_name,
            role_arn=role,
            validations=validations,
            skip_checks=skip_checks,
            deploy_role_arn=deploy_role_arn,
            external_id=external_id,
            tags=tags or {"ManagedBy": "OrchestrationAccelerator"},
            state_machine_type=state_machine_type,
        )
    except TypeError:
        # Older signature without some kwargs
        try:
            result = _deploy(
                asl=asl,
                name=sm_name,
                role_arn=role,
                validations=validations,
                skip_checks=skip_checks,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("deploy failed")
            return DeployOutcome(ok=False, error=f"{type(exc).__name__}: {exc}", role_arn=role)
    except PreDeployCheckError as exc:
        return DeployOutcome(
            ok=False,
            blocked_reasons=list(getattr(exc, "reasons", None) or [str(exc)]),
            role_arn=role,
        )
    except DeploymentError as exc:
        return DeployOutcome(ok=False, error=str(exc), role_arn=role)
    except Exception as exc:  # noqa: BLE001
        logger.exception("deploy failed")
        return DeployOutcome(ok=False, error=f"{type(exc).__name__}: {exc}", role_arn=role)

    return DeployOutcome(
        ok=True,
        state_machine_arn=getattr(result, "state_machine_arn", None)
        or (result.get("state_machine_arn") if isinstance(result, dict) else None),
        action=getattr(result, "action", None)
        or (result.get("action") if isinstance(result, dict) else None),
        state_machine_name=getattr(result, "state_machine_name", None)
        or sm_name,
        region=getattr(result, "region", None)
        or (result.get("region") if isinstance(result, dict) else None),
        role_arn=getattr(result, "role_arn", None) or role,
        account_id=getattr(result, "account_id", None)
        or (result.get("account_id") if isinstance(result, dict) else None),
        checks_passed=list(getattr(result, "checks_passed", None) or []),
    )
