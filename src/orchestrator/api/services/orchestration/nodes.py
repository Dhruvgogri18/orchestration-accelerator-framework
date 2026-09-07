"""Graph nodes — thin wrappers around existing domain services.

Each node:
  * reads what it needs from WorkflowState
  * calls existing orchestrator services (no prompt/logic duplication)
  * returns a partial state update dict

Domain imports are deferred inside node bodies so the graph module can
load (and be unit-tested for topology/routing) even when optional
monorepo packages (e.g. indexing gateway) are not on PYTHONPATH.

HITL nodes use langgraph.types.interrupt so the HTTP layer can pause
and later resume with Command(resume=...).
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from .state import WorkflowState


def _history_from_state(state: WorkflowState) -> list[dict]:
    """Normalize LangGraph messages into OpenAI-style role/content dicts."""
    out: list[dict] = []
    for m in state.get("messages") or []:
        if isinstance(m, dict):
            role = m.get("role") or m.get("type") or "user"
            content = m.get("content") or ""
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            out.append({"role": role, "content": content})
        else:
            role = getattr(m, "type", None) or getattr(m, "role", "user")
            content = getattr(m, "content", "") or ""
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            out.append({"role": role, "content": content})
    return out


def _plan_from_state(raw: Any):
    from orchestrator.api.schemas.common import Plan

    if raw is None:
        return None
    if isinstance(raw, Plan):
        return raw
    return Plan.model_validate(raw)


# ---------------------------------------------------------------------------
# Stage 1 — scope + pipeline analysis
# ---------------------------------------------------------------------------


def classify_scope_node(state: WorkflowState) -> dict:
    """Optional standalone scope node (analyze_pipeline also scopes internally)."""
    from orchestrator.api.services.pipeline_analysis.service import classify_scope

    history = _history_from_state(state)
    try:
        scope = classify_scope(history)
        status = getattr(scope, "status", None)
        if status in ("in_scope", "out_of_scope"):
            in_scope = status == "in_scope"
        else:
            in_scope = status != "out_of_scope"
        return {
            "scope_status": "in_scope" if in_scope else "out_of_scope",
            "out_of_scope_message": getattr(scope, "message", None),
            "last_stage": "classify_scope",
        }
    except Exception as exc:  # noqa: BLE001 — surface as graph error
        return {
            "scope_status": "out_of_scope",
            "out_of_scope_message": str(exc),
            "error": str(exc),
            "last_stage": "classify_scope",
        }


def pipeline_analysis_node(state: WorkflowState) -> dict:
    """Run the full Prompt2DAG-aligned five-sub-stage analysis chain."""
    from orchestrator.api.services.pipeline_analysis.service import (
        ExtractionError,
        analyze_pipeline,
    )

    history = _history_from_state(state)
    try:
        result = analyze_pipeline(history)
    except ExtractionError as exc:
        return {
            "extraction_status": "error",
            "error": str(exc),
            "last_stage": "pipeline_analysis",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "extraction_status": "error",
            "error": f"pipeline_analysis failed: {exc}",
            "last_stage": "pipeline_analysis",
        }

    update: dict[str, Any] = {
        "extraction_status": result.status,
        "last_stage": "pipeline_analysis",
    }
    if result.status == "ready" and result.plan is not None:
        update["plan"] = result.plan.model_dump()
        update["analysis"] = result.analysis
        update["scope_status"] = "in_scope"
    elif result.status == "needs_clarification":
        update["clarification_message"] = result.clarifying_question
        update["scope_status"] = "in_scope"
    elif result.status == "out_of_scope":
        update["out_of_scope_message"] = result.message
        update["scope_status"] = "out_of_scope"
    return update


# ---------------------------------------------------------------------------
# Deterministic presentation + diagnostics
# ---------------------------------------------------------------------------


def render_plan_node(state: WorkflowState) -> dict:
    from orchestrator.api.services.plan.render import (
        find_placeholders,
        render_plan_diagram,
        render_plan_summary,
    )

    plan = _plan_from_state(state.get("plan"))
    if plan is None:
        return {
            "error": "render_plan: no plan in state",
            "last_stage": "render_plan",
        }
    return {
        "summary_markdown": render_plan_summary(plan),
        "diagram_dot": render_plan_diagram(plan),
        "placeholders": find_placeholders(plan),
        "last_stage": "render_plan",
    }


def diagnose_match_node(state: WorkflowState) -> dict:
    """Matching / canonicalize diagnostics only — never overwrites plan."""
    from orchestrator.api.services.plan.matching import match as match_plan

    plan = _plan_from_state(state.get("plan"))
    if plan is None:
        return {"last_stage": "diagnose_match"}
    try:
        match_result = match_plan(
            plan,
            session_id=state.get("session_id") or "",
            user_message=state.get("user_message") or "",
        )
        return {
            "match_tier": match_result.tier,
            "match_archetype": getattr(match_result, "archetype", None)
            or getattr(match_result, "template_name", None),
            "match_score": getattr(match_result, "score", None),
            "signature_hash": getattr(match_result, "signature_hash", None),
            "last_stage": "diagnose_match",
        }
    except Exception:  # noqa: BLE001 — diagnostics must not fail the turn
        return {"last_stage": "diagnose_match"}


# ---------------------------------------------------------------------------
# Human plan review (HITL)
# ---------------------------------------------------------------------------


def plan_review_interrupt_node(state: WorkflowState) -> dict:
    """Pause until the client resumes with approve/reject.

    Resume payload expected shape:
      {"action": "approve"|"reject", "overrides": {...}, "feedback": "..."}
    """
    decision = interrupt(
        {
            "type": "plan_review",
            "session_id": state.get("session_id"),
            "plan": state.get("plan"),
            "summary_markdown": state.get("summary_markdown"),
            "diagram_dot": state.get("diagram_dot"),
            "placeholders": state.get("placeholders"),
            "pipeline_analysis": state.get("analysis"),
            "match_tier": state.get("match_tier"),
            "match_archetype": state.get("match_archetype"),
        }
    )
    if not isinstance(decision, dict):
        decision = {"action": "pending"}

    action = (decision.get("action") or "pending").lower()
    if action not in ("approve", "reject", "pending"):
        action = "pending"

    update: dict[str, Any] = {
        "plan_decision": action,
        "plan_overrides": decision.get("overrides") or {},
        "reject_feedback": decision.get("feedback"),
        "last_stage": "plan_review",
    }
    if action == "reject":
        feedback = decision.get("feedback") or "Please revise the plan."
        update["messages"] = [
            {
                "role": "user",
                "content": (
                    "REJECTED — revise the previous plan. Do not return the same "
                    f"plan unchanged.\n\nFeedback: {feedback}"
                ),
            }
        ]
        update["user_message"] = feedback
    return update


def apply_overrides_node(state: WorkflowState) -> dict:
    from orchestrator.api.services.plan.render import apply_overrides

    plan = _plan_from_state(state.get("plan"))
    if plan is None:
        return {
            "error": "apply_overrides: no plan in state",
            "enrichment_status": "error",
            "last_stage": "apply_overrides",
        }
    overrides = state.get("plan_overrides") or {}
    resolved, remaining = apply_overrides(plan, overrides)
    return {
        "resolved_plan": resolved.model_dump(),
        "remaining_placeholders": remaining,
        "last_stage": "apply_overrides",
    }


# ---------------------------------------------------------------------------
# Enrichment (Stage 3.5)
# ---------------------------------------------------------------------------


def enrichment_node(state: WorkflowState) -> dict:
    from orchestrator.api.services.enrichment.service import EnrichmentError, enrich_plan

    plan = _plan_from_state(state.get("resolved_plan") or state.get("plan"))
    if plan is None:
        return {
            "enrichment_status": "error",
            "enrichment_error": "enrichment: no plan in state",
            "last_stage": "enrichment",
        }
    try:
        result = enrich_plan(plan)
    except EnrichmentError as exc:
        return {
            "enrichment_status": "error",
            "enrichment_error": str(exc),
            "last_stage": "enrichment",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "enrichment_status": "error",
            "enrichment_error": f"enrichment failed: {exc}",
            "last_stage": "enrichment",
        }

    if result.status == "complete":
        return {
            "enrichment_status": "complete",
            "workflow_spec": result.spec,
            "enrichment_question": None,
            "enrichment_pending": None,
            "last_stage": "enrichment",
        }
    return {
        "enrichment_status": "needs_clarification",
        "enrichment_question": result.question,
        "enrichment_pending": {
            "messages": result.messages,
            "clarification_rounds": result.clarification_rounds,
        },
        "last_stage": "enrichment",
    }


def enrichment_interrupt_node(state: WorkflowState) -> dict:
    """Pause for a human answer to an enrichment clarification question."""
    from orchestrator.api.services.enrichment.service import (
        EnrichmentError,
        continue_enrichment,
    )

    answer = interrupt(
        {
            "type": "enrichment_clarification",
            "session_id": state.get("session_id"),
            "question": state.get("enrichment_question"),
        }
    )
    if isinstance(answer, dict):
        answer_text = answer.get("answer") or answer.get("content") or str(answer)
    else:
        answer_text = str(answer)

    plan = _plan_from_state(state.get("resolved_plan") or state.get("plan"))
    pending = state.get("enrichment_pending") or {}
    if plan is None:
        return {
            "enrichment_status": "error",
            "enrichment_error": "enrichment resume: no plan",
            "last_stage": "enrichment_clarify",
        }
    try:
        result = continue_enrichment(
            plan=plan,
            messages=pending.get("messages") or [],
            answer=answer_text,
            clarification_rounds=int(pending.get("clarification_rounds") or 0),
        )
    except EnrichmentError as exc:
        return {
            "enrichment_status": "error",
            "enrichment_error": str(exc),
            "last_stage": "enrichment_clarify",
        }

    if result.status == "complete":
        return {
            "enrichment_status": "complete",
            "workflow_spec": result.spec,
            "enrichment_question": None,
            "enrichment_pending": None,
            "last_stage": "enrichment_clarify",
        }
    return {
        "enrichment_status": "needs_clarification",
        "enrichment_question": result.question,
        "enrichment_pending": {
            "messages": result.messages,
            "clarification_rounds": result.clarification_rounds,
        },
        "last_stage": "enrichment_clarify",
    }


# ---------------------------------------------------------------------------
# ASL generation + validation
# ---------------------------------------------------------------------------


def asl_node(state: WorkflowState) -> dict:
    from orchestrator.api.services.asl_service import (
        AslGenerationError,
        build_asl_from_spec,
        find_asl_placeholders,
    )

    spec = state.get("workflow_spec")
    if not spec:
        return {
            "asl_error": "asl: no workflow_spec in state",
            "last_stage": "asl",
        }

    spec_placeholders = find_asl_placeholders(spec)
    if spec_placeholders:
        return {
            "asl": None,
            "asl_graph_dot": None,
            "asl_validations": None,
            "asl_error": None,
            "spec_placeholders": spec_placeholders,
            "last_stage": "asl",
        }

    try:
        result = build_asl_from_spec(spec)
        validations = (
            result.validations.to_dict()
            if hasattr(result.validations, "to_dict")
            else result.validations
        )
        return {
            "asl": result.asl,
            "asl_graph_dot": result.graph_dot,
            "asl_validations": validations,
            "asl_error": None,
            "spec_placeholders": [],
            "last_stage": "asl",
        }
    except AslGenerationError as exc:
        return {
            "asl": None,
            "asl_error": str(exc),
            "spec_placeholders": [],
            "last_stage": "asl",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "asl": None,
            "asl_error": f"ASL generation failed unexpectedly: {exc}",
            "spec_placeholders": [],
            "last_stage": "asl",
        }


def deploy_ready_node(state: WorkflowState) -> dict:
    """Marker node — actual deploy stays in deploy.service (side-effecting)."""
    return {"last_stage": "deploy_ready"}


def out_of_scope_node(state: WorkflowState) -> dict:
    return {
        "extraction_status": "out_of_scope",
        "last_stage": "out_of_scope",
    }


def error_handler_node(state: WorkflowState) -> dict:
    return {
        "last_stage": "error",
        "error": state.get("error") or "unknown orchestration error",
    }
