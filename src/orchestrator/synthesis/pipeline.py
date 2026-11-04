"""End-to-end enterprise synthesis pipeline.

Orchestrates the full path required by the Orchestration Accelerator scope:

  user NL
    → canonicalize (paraphrase collapse)
    → repo RAG + AWS discovery tools
    → plan synthesis (LLM + template match + canonical step ids)
    → optional self-consistency gate
    → repo grounding (real job names / entry points)
    → enrichment → deterministic ASL
    → multi-group validation
    → human approval gate
    → idempotent boto3 deploy (only when approved)

This is the single entry point the CLI and higher-level services should call.
All intermediate stages remain independently testable via ``synthesis.planner``,
``synthesis.asl_generator``, ``synthesis.validator``, and ``deployment``.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)

PipelineStatus = Literal[
    "ready",
    "needs_clarification",
    "out_of_scope",
    "validation_failed",
    "blocked",
    "deployed",
    "error",
]


@dataclass
class PipelineResult:
    status: PipelineStatus
    plan: dict[str, Any] | None = None
    asl: dict[str, Any] | None = None
    validation: dict[str, Any] | None = None
    grounding: dict[str, Any] | None = None
    consistency: dict[str, Any] | None = None
    deploy: dict[str, Any] | None = None
    clarifying_question: str | None = None
    message: str | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    stages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_pipeline(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    current_plan: dict[str, Any] | None = None,
    # Consistency
    use_cache: bool = True,
    self_consistency: bool = False,
    # ASL / validation
    generate_asl: bool = True,
    include_aws_validate: bool = False,
    # Grounding
    ground_against_repo: bool = True,
    require_repo_grounding: bool = False,
    # Deploy (off by default — human approval required)
    deploy: bool = False,
    approved: bool = False,
    state_machine_name: str | None = None,
    role_arn: str | None = None,
) -> PipelineResult:
    """Run the full synthesis pipeline for one user request.

    Deploy never runs unless *both* ``deploy=True`` and ``approved=True``.
    """
    stages: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------ plan
    stages.append("plan")
    try:
        from orchestrator.synthesis.planner import generate_plan

        plan_result = generate_plan(
            user_message,
            history=history,
            session_id=session_id,
            current_plan=current_plan,
            use_cache=use_cache,
            self_consistency=self_consistency,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("pipeline plan stage failed")
        return PipelineResult(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            stages=stages,
        )

    if plan_result.status == "needs_clarification":
        return PipelineResult(
            status="needs_clarification",
            clarifying_question=plan_result.clarifying_question,
            consistency=plan_result.consistency,
            stages=stages,
        )
    if plan_result.status == "out_of_scope":
        return PipelineResult(
            status="out_of_scope",
            message=plan_result.message,
            stages=stages,
        )
    if plan_result.status != "ready" or not plan_result.plan:
        return PipelineResult(
            status="error",
            error=plan_result.error or "plan_not_ready",
            message=plan_result.message,
            consistency=plan_result.consistency,
            stages=stages,
        )

    plan = plan_result.plan
    consistency = plan_result.consistency
    if plan_result.from_cache:
        stages.append("cache_hit")
        warnings.append("Plan served from paraphrase-stable cache.")

    # ------------------------------------------------------------------ ground
    grounding_dict: dict[str, Any] | None = None
    if ground_against_repo:
        stages.append("ground")
        try:
            from orchestrator.synthesis.grounding import (
                apply_grounding_to_plan,
                ground_plan_against_repo,
            )

            greport = ground_plan_against_repo(
                plan, require_all=require_repo_grounding
            )
            grounding_dict = greport.to_dict()
            if greport.enabled and greport.hits:
                plan = apply_grounding_to_plan(plan, greport)
            if require_repo_grounding and not greport.ok:
                return PipelineResult(
                    status="blocked",
                    plan=plan,
                    grounding=grounding_dict,
                    consistency=consistency,
                    message=(
                        "Plan references resources not found in the repository index. "
                        "Name real jobs from the repo or disable require_repo_grounding."
                    ),
                    warnings=warnings
                    + list(greport.notes)
                    + [f"unresolved: {u}" for u in greport.unresolved[:12]],
                    stages=stages,
                )
            warnings.extend(greport.notes)
        except Exception as exc:  # noqa: BLE001
            logger.warning("grounding failed: %s", exc)
            warnings.append(f"grounding_skipped: {type(exc).__name__}: {exc}")

    if not generate_asl:
        return PipelineResult(
            status="ready",
            plan=plan,
            grounding=grounding_dict,
            consistency=consistency,
            warnings=warnings,
            stages=stages,
        )

    # ------------------------------------------------------------------ ASL
    stages.append("asl")
    try:
        from orchestrator.synthesis.asl_generator import generate_asl as _gen_asl

        asl_result = _gen_asl(
            plan,
            include_validation=True,
            include_aws_validate=include_aws_validate,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("ASL generation failed")
        return PipelineResult(
            status="error",
            plan=plan,
            grounding=grounding_dict,
            consistency=consistency,
            error=f"{type(exc).__name__}: {exc}",
            warnings=warnings,
            stages=stages,
        )

    if asl_result.error:
        return PipelineResult(
            status="error",
            plan=plan,
            grounding=grounding_dict,
            consistency=consistency,
            error=asl_result.error,
            warnings=warnings + list(asl_result.warnings or []),
            stages=stages,
        )

    warnings.extend(asl_result.warnings or [])
    validation = asl_result.validation
    stages.append("validate")

    if validation and not validation.get("ok", True):
        return PipelineResult(
            status="validation_failed",
            plan=plan,
            asl=asl_result.asl,
            validation=validation,
            grounding=grounding_dict,
            consistency=consistency,
            warnings=warnings,
            message="ASL failed validation — fix issues before deploy.",
            stages=stages,
        )

    # ------------------------------------------------------------------ deploy
    if not deploy:
        return PipelineResult(
            status="ready",
            plan=plan,
            asl=asl_result.asl,
            validation=validation,
            grounding=grounding_dict,
            consistency=consistency,
            warnings=warnings,
            message=(
                "Plan and ASL ready. Review the definition, then re-run with "
                "deploy=True and approved=True to publish."
            ),
            stages=stages,
        )

    stages.append("deploy")
    if not approved:
        return PipelineResult(
            status="blocked",
            plan=plan,
            asl=asl_result.asl,
            validation=validation,
            grounding=grounding_dict,
            consistency=consistency,
            warnings=warnings,
            message="Deploy requested but approved=False — human review gate blocked deploy.",
            stages=stages,
        )

    try:
        from orchestrator.deployment.stepfunctions import deploy_state_machine

        outcome = deploy_state_machine(
            asl_result.asl or {},
            name=state_machine_name or (plan.get("name") if plan else None),
            role_arn=role_arn,
            validation_report=validation,
            approved=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("deploy stage failed")
        return PipelineResult(
            status="error",
            plan=plan,
            asl=asl_result.asl,
            validation=validation,
            grounding=grounding_dict,
            consistency=consistency,
            error=f"{type(exc).__name__}: {exc}",
            warnings=warnings,
            stages=stages,
        )

    deploy_dict = outcome.to_dict()
    if not outcome.ok:
        return PipelineResult(
            status="blocked",
            plan=plan,
            asl=asl_result.asl,
            validation=validation,
            grounding=grounding_dict,
            consistency=consistency,
            deploy=deploy_dict,
            message="; ".join(outcome.blocked_reasons)
            or outcome.error
            or "Deploy blocked",
            warnings=warnings,
            stages=stages,
        )

    return PipelineResult(
        status="deployed",
        plan=plan,
        asl=asl_result.asl,
        validation=validation,
        grounding=grounding_dict,
        consistency=consistency,
        deploy=deploy_dict,
        warnings=warnings,
        message=f"Deployed {outcome.action}: {outcome.state_machine_arn}",
        stages=stages,
    )
