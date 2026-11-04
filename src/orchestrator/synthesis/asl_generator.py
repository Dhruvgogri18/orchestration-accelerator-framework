"""ASL generation from a locked plan / workflow spec.

Delegates to the existing deterministic translator in ``asl_service``
(and optional LLM path via ``build_asl_from_spec``).
"""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.synthesis.models import AslResult

logger = logging.getLogger(__name__)


def generate_asl(
    plan_or_spec: dict[str, Any],
    *,
    workflow_spec: dict[str, Any] | None = None,
    include_validation: bool = True,
    include_aws_validate: bool = False,
) -> AslResult:
    """Produce Amazon States Language from a plan or full workflow_spec."""
    if not plan_or_spec and not workflow_spec:
        return AslResult(error="empty_input")

    spec = workflow_spec if workflow_spec is not None else _ensure_spec(plan_or_spec)

    try:
        from orchestrator.api.services.asl_service import build_asl_from_spec

        asl, warnings = build_asl_from_spec(spec)
        result = AslResult(asl=asl, warnings=list(warnings or []))
    except Exception:
        try:
            from orchestrator.api.services.asl_service import generate_asl as _gen

            asl, warnings = _gen(spec)
            result = AslResult(asl=asl, warnings=list(warnings or []))
        except Exception as exc2:  # noqa: BLE001
            logger.exception("ASL generation failed")
            return AslResult(error=f"{type(exc2).__name__}: {exc2}")

    if include_validation and result.asl:
        from orchestrator.synthesis.validator import validate_asl

        v = validate_asl(
            result.asl,
            spec=spec,
            include_aws_validate=include_aws_validate,
        )
        result.validation = {
            "ok": v.ok,
            "issues": v.issues,
            "groups": v.groups,
            "report": v.report,
        }
        if not v.ok:
            result.warnings = list(result.warnings) + [
                f"validation: {i}" for i in v.issues[:12]
            ]

    return result


def _ensure_spec(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("steps") and plan.get("StartAt"):
        return plan
    if plan.get("workflow_spec"):
        return plan["workflow_spec"]

    steps = plan.get("steps") or []
    if not steps:
        return plan

    try:
        from orchestrator.api.schemas.common import Plan
        from orchestrator.api.services.enrichment.service import enrich_plan

        p = Plan.model_validate(plan)
        enriched = enrich_plan(p)
        if isinstance(enriched, dict) and enriched.get("workflow_spec"):
            return enriched["workflow_spec"]
        if hasattr(enriched, "workflow_spec"):
            return enriched.workflow_spec
        if isinstance(enriched, dict):
            return enriched
    except Exception as exc:  # noqa: BLE001
        logger.info("enrichment skipped, using minimal spec: %s", exc)

    states = []
    for i, s in enumerate(steps):
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or f"Step{i+1}")
        nxt = s.get("next")
        states.append(
            {
                "id": sid,
                "type": "Task",
                "service": s.get("service") or "Lambda",
                "description": s.get("description") or sid,
                "next": nxt,
                "end": nxt is None,
                "branches": s.get("branches") or [],
                "on_failure": s.get("on_failure") or [],
                "on_timeout": s.get("on_timeout") or [],
                "component_type": s.get("component_type"),
            }
        )
    start = states[0]["id"] if states else "Start"
    return {
        "name": plan.get("name") or "GeneratedWorkflow",
        "summary": plan.get("summary") or "",
        "StartAt": start,
        "steps": states,
        "plan": plan,
    }
