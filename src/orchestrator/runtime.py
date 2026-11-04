"""Public enterprise runtime API.

Preferred one-liners for product code, CLI, and notebooks:

    from orchestrator.runtime import nl_to_plan, nl_to_workflow

    plan_pkg = nl_to_plan("orchestrate invoice validation then archive to S3")
    full     = nl_to_workflow("...", include_asl=True)   # plan + ASL + validation
"""

from __future__ import annotations

from typing import Any


def nl_to_plan(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Natural language → Step Functions **plan** (RAG + tools + LangGraph).

    No prompt cache. No domain hardcoding. Returns a dict with ``status``,
    ``plan``, ``summary_markdown``, ``diagram_dot``, ``placeholders``, etc.
    """
    from orchestrator.graph.plan_graph import nl_to_plan as _nl

    return _nl(user_message, history=history, session_id=session_id)


def nl_to_workflow(
    user_message: str,
    *,
    history: list[dict] | None = None,
    session_id: str | None = None,
    include_asl: bool = True,
    include_aws_validate: bool = False,
    deploy: bool = False,
    approved: bool = False,
) -> dict[str, Any]:
    """NL → plan → (optional ASL + validation) → optional deploy after approval.

    Deploy never runs unless ``deploy=True`` and ``approved=True``.
    """
    from orchestrator.observability.otel import configure_otel, span

    configure_otel()
    with span("runtime.nl_to_workflow", include_asl=include_asl, deploy=deploy):
        plan_pkg = nl_to_plan(
            user_message, history=history, session_id=session_id
        )
        out: dict[str, Any] = {
            "plan_package": plan_pkg,
            "status": plan_pkg.get("status"),
            "plan": plan_pkg.get("plan"),
            "summary_markdown": plan_pkg.get("summary_markdown"),
            "diagram_dot": plan_pkg.get("diagram_dot"),
            "placeholders": plan_pkg.get("placeholders") or [],
            "clarifying_question": plan_pkg.get("clarifying_question"),
            "message": plan_pkg.get("message"),
            "error": plan_pkg.get("error"),
            "stages": list(plan_pkg.get("stages") or []),
            "warnings": list(plan_pkg.get("warnings") or []),
            "session_id": plan_pkg.get("session_id"),
            "asl": None,
            "validation": None,
            "deploy": None,
        }

        if plan_pkg.get("status") != "ready" or not plan_pkg.get("plan"):
            return out

        if not include_asl:
            return out

        from orchestrator.synthesis.asl_generator import generate_asl

        asl_result = generate_asl(
            plan_pkg["plan"],
            include_validation=True,
            include_aws_validate=include_aws_validate,
        )
        out["asl"] = asl_result.asl
        out["validation"] = asl_result.validation
        out["warnings"] = list(out["warnings"]) + list(asl_result.warnings or [])
        out["stages"] = list(out["stages"]) + ["asl", "validate"]

        if asl_result.error:
            out["status"] = "error"
            out["error"] = asl_result.error
            return out

        if asl_result.validation and not asl_result.validation.get("ok", True):
            out["status"] = "validation_failed"
            out["message"] = "ASL failed validation — fix issues before deploy."
            return out

        if not deploy:
            out["status"] = "ready"
            out["message"] = (
                "Plan and ASL ready. Review, then call again with "
                "deploy=True and approved=True to publish."
            )
            return out

        if not approved:
            out["status"] = "blocked"
            out["message"] = (
                "Deploy requested but approved=False — human review gate blocked deploy."
            )
            return out

        from orchestrator.deployment.stepfunctions import deploy_state_machine

        outcome = deploy_state_machine(
            asl_result.asl or {},
            name=(plan_pkg.get("plan") or {}).get("name"),
            validation_report=asl_result.validation,
            approved=True,
        )
        out["deploy"] = outcome.to_dict()
        out["stages"] = list(out["stages"]) + ["deploy"]
        if outcome.ok:
            out["status"] = "deployed"
            out["message"] = f"Deployed {outcome.action}: {outcome.state_machine_arn}"
        else:
            out["status"] = "blocked"
            out["message"] = (
                "; ".join(outcome.blocked_reasons) or outcome.error or "Deploy blocked"
            )
        return out
