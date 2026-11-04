"""Public orchestration service — LangGraph-backed session operations.

Drop-in surface for session_router: same function names and exceptions as
pipeline_analysis.session_service. Automatic stages (analysis / render /
diagnose / enrichment / ASL on approve) run through the StateGraph;
methods that are pure persistence or side-effecting AWS calls delegate
to the legacy session_service so behavior stays identical.

Persistence: session_repository remains source of truth for history,
latest_plan, and locked_spec. Graph checkpointer is keyed by session_id.
"""

from __future__ import annotations

import json
from typing import Any

from orchestrator.api.repositories import session_repository as session_store
from orchestrator.api.repositories.session_repository import Session
from orchestrator.api.schemas.common import Plan
from orchestrator.api.schemas.session import (
    ApproveResponse,
    AslValidateResponse,
    DeployRequest,
    DeployResponse,
    MessageResponse,
)
from orchestrator.api.services.llm_trace import end_request, log_event, start_request

# Re-export domain errors so routers can write:
#   except session_service.SessionNotFoundError
from orchestrator.api.services.pipeline_analysis.session_service import (  # noqa: F401
    NoApprovablePlanError,
    NoAslError,
    NoEnrichmentQuestionError,
    NoSpecPlaceholdersError,
    NoWorkflowSpecError,
    SessionNotFoundError,
    UnresolvedSpecPlaceholdersError,
)
from orchestrator.api.services.pipeline_analysis import session_service as _legacy

from .graph import build_workflow_graph, get_default_graph


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _get_graph():
    return get_default_graph()


def _plan_obj(raw: Any) -> Plan | None:
    if raw is None:
        return None
    if isinstance(raw, Plan):
        return raw
    return Plan.model_validate(raw)


def _message_response_from_state(state: dict) -> MessageResponse:
    status = state.get("extraction_status") or "needs_clarification"
    if status == "ready":
        return MessageResponse(
            status="ready",
            plan=_plan_obj(state.get("plan")),
            summary_markdown=state.get("summary_markdown"),
            diagram_dot=state.get("diagram_dot"),
            placeholders=state.get("placeholders") or [],
            pipeline_analysis=state.get("analysis"),
        )
    if status == "out_of_scope":
        return MessageResponse(
            status="out_of_scope",
            message=state.get("out_of_scope_message")
            or (
                "This assistant is only designed to help you build AWS Step "
                "Functions workflows. Please describe a workflow to orchestrate."
            ),
        )
    return MessageResponse(
        status="needs_clarification",
        question=state.get("clarification_message"),
    )


def _approve_response_from_state(state: dict) -> ApproveResponse:
    from orchestrator.api.services.asl_service import describe_placeholders

    spec = state.get("workflow_spec")
    tokens = list(state.get("spec_placeholders") or [])
    plan = state.get("resolved_plan")
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
    details = state.get("placeholder_details")
    if not isinstance(details, dict) or (tokens and not details):
        details = describe_placeholders(spec, tokens, plan=plan_dict)
    return ApproveResponse(
        status="approved",
        resolved_plan=_plan_obj(state.get("resolved_plan")),
        remaining_placeholders=state.get("remaining_placeholders") or [],
        workflow_spec=spec,
        enrichment_error=state.get("enrichment_error"),
        enrichment_question=state.get("enrichment_question"),
        spec_placeholders=tokens,
        placeholder_details=details or {},
        asl=state.get("asl"),
        asl_graph_dot=state.get("asl_graph_dot"),
        asl_validations=state.get("asl_validations"),
        asl_source=state.get("asl_source"),
        asl_error=state.get("asl_error"),
    )


def _persist_extraction(session_id: str, state: dict) -> None:
    """Mirror session_service side effects for UI compatibility."""
    status = state.get("extraction_status")
    plan = _plan_obj(state.get("plan")) if status == "ready" else None
    session_store.set_latest_plan(session_id, plan)
    if status == "ready" and plan is not None:
        payload = {
            "status": "ready",
            "plan": plan.model_dump(),
            "summary_markdown": state.get("summary_markdown"),
            "diagram_dot": state.get("diagram_dot"),
            "placeholders": state.get("placeholders") or [],
            "pipeline_analysis": state.get("analysis"),
        }
        session_store.append_message(session_id, "assistant", json.dumps(payload))
    elif status == "needs_clarification":
        session_store.append_message(
            session_id, "assistant", state.get("clarification_message") or ""
        )
    elif status == "out_of_scope":
        session_store.append_message(
            session_id, "assistant", state.get("out_of_scope_message") or ""
        )


def _persist_locked(session_id: str, state: dict) -> None:
    from orchestrator.api.services.asl_service import describe_placeholders

    spec = state.get("workflow_spec")
    tokens = list(state.get("spec_placeholders") or [])
    plan = state.get("resolved_plan")
    plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
    details = state.get("placeholder_details")
    if not isinstance(details, dict) or (tokens and not details):
        details = describe_placeholders(spec, tokens, plan=plan_dict)
    locked = {
        "resolved_plan": state.get("resolved_plan"),
        "remaining_placeholders": state.get("remaining_placeholders") or [],
        "workflow_spec": spec,
        "enrichment_error": state.get("enrichment_error"),
        "enrichment_question": state.get("enrichment_question"),
        "spec_placeholders": tokens,
        "placeholder_details": details or {},
        "asl": state.get("asl"),
        "asl_graph_dot": state.get("asl_graph_dot"),
        "asl_validations": state.get("asl_validations"),
        "asl_error": state.get("asl_error"),
    }
    if state.get("enrichment_pending"):
        locked["enrichment_pending"] = state["enrichment_pending"]
    # Preserve any deployment block already on the session
    session = session_store.get_session(session_id)
    if session and session.locked_spec and session.locked_spec.get("deployment"):
        locked["deployment"] = session.locked_spec["deployment"]
    session_store.set_locked_spec(session_id, locked)


# ---------------------------------------------------------------------------
# Public API (router-facing) — must match session_service surface
# ---------------------------------------------------------------------------


def start_session() -> Session:
    return session_store.create_session()


def get_session_or_raise(session_id: str) -> Session:
    session = session_store.get_session(session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    return session


def get_live_status(session_id: str) -> dict:
    """Delegate — same in-memory llm_trace status both paths share."""
    return _legacy.get_live_status(session_id)


def send_message(session_id: str, message: str) -> MessageResponse:
    """User message → turn graph (route → extract | patch_params | revise_plan | …).

    Value-only edits after a plan/ASL exists take the cheap patch path
    (one small LLM + mechanical ASL regen). Structural changes re-run
    pipeline analysis. Brand-new designs use extract.

    Per-session lock allows many chats in parallel while serialising
    concurrent requests *within* the same session_id.
    """
    get_session_or_raise(session_id)
    from orchestrator.api.services.session_lock import session_lock

    with session_lock(session_id):
        start_request(session_id=session_id, user_message=message)
        try:
            from .graph import run_message_turn

            state = run_message_turn(session_id, message)
            end_request(
                status=state.get("extraction_status") or state.get("last_stage") or "ok"
            )
            return _message_response_from_turn_state(state)
        except Exception as exc:  # noqa: BLE001
            end_request(status="error", error=str(exc))
            raise


def reject_plan(session_id: str, feedback: str) -> MessageResponse:
    """Explicit reject — always structural revise (never param patch)."""
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No approvable plan yet -- conversation still needs clarification."
        )
    start_request(session_id=session_id, user_message=feedback)
    try:
        from .graph import run_reject_turn

        state = run_reject_turn(session_id, feedback)
        end_request(status=state.get("extraction_status") or state.get("last_stage") or "ok")
        return _message_response_from_turn_state(state)
    except Exception as exc:  # noqa: BLE001
        end_request(status="error", error=str(exc))
        raise


def _message_response_from_turn_state(state: dict) -> MessageResponse:
    """Map turn-graph state (or nested message_response) to MessageResponse.

    Value-only patches include asl / intent so the UI stays on ASL review.
    """
    nested = state.get("message_response") if isinstance(state.get("message_response"), dict) else {}
    status = (nested.get("status") if nested else None) or state.get("extraction_status")

    if status == "ready":
        intent = (nested.get("intent") if nested else None) or state.get("route")
        phase = (nested.get("phase") if nested else None) or state.get("phase")
        return MessageResponse(
            status="ready",
            plan=_plan_obj((nested.get("plan") if nested else None) or state.get("plan")),
            message=(nested.get("message") if nested else None),
            summary_markdown=(nested.get("summary_markdown") if nested else None)
            or state.get("summary_markdown"),
            diagram_dot=(nested.get("diagram_dot") if nested else None)
            or state.get("diagram_dot"),
            placeholders=(nested.get("placeholders") if nested else None)
            or state.get("placeholders")
            or [],
            pipeline_analysis=(nested.get("pipeline_analysis") if nested else None)
            or state.get("analysis"),
            intent=intent,
            phase=phase,
            workflow_spec=(nested.get("workflow_spec") if nested else None)
            or state.get("workflow_spec"),
            remaining_placeholders=(nested.get("remaining_placeholders") if nested else None),
            spec_placeholders=(nested.get("spec_placeholders") if nested else None)
            or state.get("spec_placeholders"),
            asl=(nested.get("asl") if nested else None) or state.get("asl"),
            asl_graph_dot=(nested.get("asl_graph_dot") if nested else None)
            or state.get("asl_graph_dot"),
            asl_validations=(nested.get("asl_validations") if nested else None)
            or state.get("asl_validations"),
            asl_source=(nested.get("asl_source") if nested else None)
            or state.get("asl_source"),
            asl_error=(nested.get("asl_error") if nested else None)
            or state.get("asl_error"),
        )
    if status == "out_of_scope":
        return MessageResponse(
            status="out_of_scope",
            message=(nested.get("message") if nested else None)
            or state.get("out_of_scope_message")
            or (
                "This assistant is only designed to help you build AWS Step "
                "Functions workflows. Please describe a workflow to orchestrate."
            ),
        )
    if status == "needs_clarification":
        return MessageResponse(
            status="needs_clarification",
            question=(nested.get("question") if nested else None)
            or state.get("clarification_message"),
        )
    return _message_response_from_state(state)


def approve_plan(session_id: str, overrides: dict[str, str] | None = None) -> ApproveResponse:
    """Approve latest plan → overrides → enrichment → ASL."""
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No approvable plan yet -- conversation still needs clarification."
        )

    overrides = overrides or {}
    if (session.locked_spec or {}).get("asl_first"):
        # ASL-first sessions have no enrichment/spec stages to run through
        # the graph -- the legacy service owns that flow.
        return _legacy.approve_plan(session_id, overrides)
    start_request(session_id=session_id)
    graph = _get_graph()
    config = _config(session_id)

    snap = graph.get_state(config)
    values = dict(snap.values) if snap and snap.values else {}
    if not values.get("plan"):
        values["plan"] = session.latest_plan.model_dump()
        values["session_id"] = session_id
        graph.update_state(config, values)

    graph.update_state(
        config,
        {
            "plan_decision": "approve",
            "plan_overrides": overrides,
        },
    )

    from .nodes import apply_overrides_node, asl_node, enrichment_node

    state: dict[str, Any] = dict(graph.get_state(config).values)
    state.update(apply_overrides_node(state))
    state.update(enrichment_node(state))

    if state.get("enrichment_status") == "complete":
        state.update(asl_node(state))

    graph.update_state(config, state)
    _persist_locked(session_id, state)
    end_request(status="approved")
    return _approve_response_from_state(state)


def answer_enrichment_question(session_id: str, answer: str) -> ApproveResponse:
    """Resume enrichment after a clarification question."""
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    pending = locked.get("enrichment_pending")
    if not pending:
        raise NoEnrichmentQuestionError(
            "No enrichment clarification pending for this session."
        )
    if locked.get("asl_first"):
        return _legacy.answer_enrichment_question(session_id, answer)

    start_request(session_id=session_id)
    graph = _get_graph()
    config = _config(session_id)

    from .nodes import asl_node
    from orchestrator.api.services.enrichment.service import (
        EnrichmentError,
        continue_enrichment,
    )

    state: dict[str, Any] = dict(graph.get_state(config).values or {})
    if not state.get("resolved_plan") and locked.get("resolved_plan"):
        state["resolved_plan"] = locked["resolved_plan"]
    if not state.get("enrichment_pending"):
        state["enrichment_pending"] = pending
    if not state.get("enrichment_question"):
        state["enrichment_question"] = locked.get("enrichment_question")
    state["session_id"] = session_id

    plan = _plan_obj(state.get("resolved_plan"))
    try:
        result = continue_enrichment(
            plan=plan,
            messages=pending.get("messages") or [],
            answer=answer,
            clarification_rounds=int(pending.get("clarification_rounds") or 0),
        )
        if result.status == "complete":
            state["enrichment_status"] = "complete"
            state["workflow_spec"] = result.spec
            state["enrichment_question"] = None
            state["enrichment_pending"] = None
            state.update(asl_node(state))
        else:
            state["enrichment_status"] = "needs_clarification"
            state["enrichment_question"] = result.question
            state["enrichment_pending"] = {
                "messages": result.messages,
                "clarification_rounds": result.clarification_rounds,
            }
    except EnrichmentError as exc:
        state["enrichment_status"] = "error"
        state["enrichment_error"] = str(exc)

    graph.update_state(config, state)
    _persist_locked(session_id, state)
    end_request(status="approved")
    return _approve_response_from_state(state)


# ---------------------------------------------------------------------------
# Methods that stay on the legacy path (identical behavior; no graph needed)
# ---------------------------------------------------------------------------


def fill_spec_placeholders(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    return _legacy.fill_spec_placeholders(session_id, overrides)


def generate_asl_for_session(session_id: str) -> AslValidateResponse:
    return _legacy.generate_asl_for_session(session_id)


def revalidate_asl_payload(
    session_id: str | None, asl: dict | None
) -> AslValidateResponse:
    return _legacy.revalidate_asl_payload(session_id, asl)


def get_asl_for_download(session_id: str) -> tuple[dict, str]:
    return _legacy.get_asl_for_download(session_id)


def deploy_workflow(session_id: str, request: DeployRequest) -> DeployResponse:
    return _legacy.deploy_workflow(session_id, request)


def rebuild_graph(*, checkpointer: Any = None):
    """Replace the process-wide default graph (e.g. after swapping checkpointer)."""
    import orchestrator.api.services.orchestration.graph as graph_mod

    graph_mod.default_graph = build_workflow_graph(checkpointer=checkpointer)
    return graph_mod.default_graph