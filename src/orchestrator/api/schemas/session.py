"""HTTP request/response schemas for session routes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from orchestrator.api.schemas.common import Plan


class StartSessionResponse(BaseModel):
    session_id: str


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1)


class MessageResponse(BaseModel):
    status: Literal["ready", "needs_clarification", "out_of_scope"]
    plan: Plan | None = None
    question: str | None = None
    # Set when status == "out_of_scope": refusal explaining domain limits.
    message: str | None = None
    # Populated only when status == "ready" -- see plan_render_service.
    # Sent automatically so the client never has to make a second call
    # just to get a human-friendly view of the plan.
    summary_markdown: str | None = None
    diagram_dot: str | None = None
    placeholders: list[str] | None = None
    # Full Stage 1 Pipeline Analysis artifact -- see
    # schemas/pipeline_analysis.py:PipelineAnalysis. Populated only when
    # status == "ready".
    pipeline_analysis: dict | None = None

    # --- Value-only patch after ASL (cosmetic rename / TODO fill) ---
    # When intent == "patch_params", the client should stay on the ASL
    # review surface and render asl + asl_graph_dot instead of re-showing
    # plan review. Structural revise leaves these unset.
    intent: str | None = None
    phase: str | None = None
    workflow_spec: dict | None = None
    remaining_placeholders: list[str] | None = None
    spec_placeholders: list[str] | None = None
    asl: dict | None = None
    asl_graph_dot: str | None = None
    asl_validations: dict | None = None
    asl_error: str | None = None


class ApproveRequest(BaseModel):
    # e.g. {"TODO_S3_BUCKET": "my-uploads-bucket", "TODO_GLUE_JOB_NAME": "nightly-etl-job"}
    # Any placeholder not included here is left unresolved in the
    # approved plan -- see apply_overrides() in plan_render_service for why
    # that's allowed rather than blocked.
    overrides: dict[str, str] = Field(default_factory=dict)


class ApproveResponse(BaseModel):
    status: Literal["approved"]
    resolved_plan: Plan
    remaining_placeholders: list[str]
    workflow_spec: dict | None = None
    enrichment_error: str | None = None
    # Set when enrichment paused to ask the user something -- answer
    # via POST /session/{id}/enrichment/answer.
    enrichment_question: str | None = None
    # TODO_ tokens still present in the locked workflow_spec after
    # enrichment (or after a partial POST /spec/placeholders call).
    # Empty means Stage 4 was allowed to run. Fill via
    # POST /session/{id}/spec/placeholders.
    spec_placeholders: list[str] = []
    # Short human-readable descriptions for each entry in
    # spec_placeholders (token → "what to fill"). Derived from
    # workflow_spec parameters / integrations / components so the UI
    # can label inputs beyond the bare TODO_ name.
    placeholder_details: dict[str, str] = Field(default_factory=dict)
    # Tokens from `overrides` that were REJECTED for having the wrong
    # shape (e.g. a bare bucket name where a full ARN is required, or
    # vice versa) and so were NOT applied -- they stay listed in
    # spec_placeholders as still-unresolved. token → user-facing reason.
    # See asl_service.validate_placeholder_value.
    placeholder_errors: dict[str, str] = Field(default_factory=dict)
    # Stage 4 — populated only when workflow_spec is complete and
    # spec_placeholders is empty.
    asl: dict | None = None
    asl_graph_dot: str | None = None
    asl_validations: dict | None = None
    asl_error: str | None = None


class SpecPlaceholdersRequest(BaseModel):
    # e.g. {"TODO_S3_BUCKET": "my-uploads-bucket", "TODO_GLUE_JOB_NAME": "nightly-etl-job"}
    # Applied across the entire locked workflow_spec -- see
    # session_service.fill_spec_placeholders. Any token not included here
    # is left unresolved and reported back in spec_placeholders.
    overrides: dict[str, str] = Field(default_factory=dict)


class EnrichmentAnswerRequest(BaseModel):
    answer: str = Field(..., min_length=1)


class RejectRequest(BaseModel):
    feedback: str = Field(..., min_length=1, description="What should change about the plan.")


class AslValidateRequest(BaseModel):
    """Optional body for re-validating a client-supplied or stored ASL."""

    asl: dict | None = None


class AslValidateResponse(BaseModel):
    asl: dict | None = None
    validations: dict
    graph_dot: str | None = None


class DeployRequest(BaseModel):
    """Body for POST /session/{id}/deploy.

    Nothing is required: state_machine_name defaults to a slugified
    version of the resolved plan's name, and role_arn defaults to
    AWS_STATE_MACHINE_ROLE_ARN from settings.

    deploy_role_arn and role_arn are two DIFFERENT roles, in the same
    target AWS account, used for two different things -- don't
    confuse them:
      - deploy_role_arn: WHICH ACCOUNT this deploys into. A role in
        the tenant's own account that this API's own identity assumes
        via STS to make the CreateStateMachine/UpdateStateMachine
        calls. Omit for single-account setups (this API's own base
        AWS credentials talk to Step Functions directly, as before).
      - role_arn: the execution role Step Functions itself assumes at
        RUNTIME inside that account to actually run the workflow.
    external_id is optional and only relevant alongside
    deploy_role_arn -- pass it if the tenant's trust policy requires
    one (recommended; see deploy/service.py's cross-account docstring
    for why).
    """

    state_machine_name: str | None = None
    role_arn: str | None = None
    deploy_role_arn: str | None = None
    external_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    # Skips the pre-deploy checks (validation report, AWS validate,
    # leftover placeholders, role ARN presence). Off by default on
    # purpose -- this should be a deliberate, visible choice by
    # whatever's calling the API, not something a client can do by
    # forgetting a flag.
    force: bool = False


class DeployResponse(BaseModel):
    status: Literal["deployed"]
    state_machine_arn: str
    action: Literal["created", "updated"]
    state_machine_name: str
    region: str
    role_arn: str
    # Parsed from AWS's own response ARN, not from the request -- lets
    # the caller confirm the deploy actually landed in the account
    # they intended, independent of what deploy_role_arn they typed.
    account_id: str = ""


class SessionStateResponse(BaseModel):
    """Full read-only view of a session -- everything a client needs
    to render its UI after a page load/refresh, in one call.
    """

    session_id: str
    history: list[dict]
    latest_plan: Plan | None = None
    locked_spec: dict | None = None
    # Recomputed from latest_plan (not cached) so a page refresh gets
    # the same complete, one-shot list of TODO_ tokens the original
    # /message response carried -- the client should never need to
    # reconstruct this itself just because the page reloaded between
    # the plan going "ready" and the user hitting Approve.
    placeholders: list[str] | None = None