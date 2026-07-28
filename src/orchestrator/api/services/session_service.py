"""Session orchestration used by routers.

Routes stay thin: HTTP in/out only. This module owns the conversation
loop formerly living as _run_extraction_turn inside session_router.
"""

from __future__ import annotations

from orchestrator.api.repositories import session_repository as session_store
from orchestrator.api.repositories.session_repository import Session
from orchestrator.api.schemas.session import ApproveResponse, MessageResponse
from orchestrator.api.services.extraction_service import ExtractionError, extract_turn
from orchestrator.api.services.plan_render_service import (
    apply_overrides,
    find_placeholders,
    render_plan_diagram,
    render_plan_summary,
)


class SessionNotFoundError(Exception):
    """Unknown session_id."""


class NoApprovablePlanError(Exception):
    """Reject/approve called while no ready plan is stored."""


def start_session() -> Session:
    return session_store.create_session()


def get_session_or_raise(session_id: str) -> Session:
    session = session_store.get_session(session_id)
    if session is None:
        raise SessionNotFoundError(session_id)
    return session


def run_extraction_turn(session_id: str, session: Session, message: str) -> MessageResponse:
    """Shared by /message and /reject -- both are "add a user message,
    run extraction again, return the result" with identical rendering
    behavior. /reject is not a different code path, just a different
    HTTP entry point into the same conversation loop.
    """
    session_store.append_message(session_id, "user", message)

    # session.history was fetched before the append above -- the
    # SQLite-backed store returns a snapshot, not a live reference, so
    # it would NOT include the message we just wrote. Build the
    # history actually sent to the LLM locally instead of trusting the
    # pre-fetched snapshot.
    current_history = [*session.history, {"role": "user", "content": message}]

    try:
        result = extract_turn(current_history)
    except ExtractionError:
        # Don't record a broken assistant turn into history -- next
        # retry should see the same clean state, not a garbage reply
        # baked into the conversation. Re-raise for the router to map
        # to HTTP 502 (same as before).
        raise

    # Record the assistant's turn as JSON text, matching what the LLM
    # actually said -- keeps history faithful to the real conversation
    # for the next call's context.
    session_store.append_message(session_id, "assistant", result.model_dump_json())

    # Persist the latest ready plan -- /approve needs to know "is there
    # actually something approvable right now". Cleared (None)
    # whenever the conversation is still clarifying, so a stale plan
    # can't be approved after the user changes something (including a
    # /reject that reopens clarification).
    session_store.set_latest_plan(
        session_id, result.plan if result.status == "ready" else None
    )

    if result.status == "ready":
        return MessageResponse(
            status="ready",
            plan=result.plan,
            summary_markdown=render_plan_summary(result.plan),
            diagram_dot=render_plan_diagram(result.plan),
            placeholders=find_placeholders(result.plan),
        )
    return MessageResponse(
        status="needs_clarification",
        question=result.clarifying_question,
    )


def send_message(session_id: str, message: str) -> MessageResponse:
    session = get_session_or_raise(session_id)
    return run_extraction_turn(session_id, session, message)


def reject_plan(session_id: str, feedback: str) -> MessageResponse:
    """User wants changes to the current plan. Not a separate flow --
    the feedback is just appended as the next user turn, same as
    /message. Reuses the identical extraction/rendering logic so the
    two code paths can't drift apart.
    """
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No plan to reject -- conversation still needs clarification."
        )
    return run_extraction_turn(session_id, session, feedback)


def approve_plan(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    """Human checkpoint (Stage 3). Requires a ready plan to exist --
    can't approve mid-clarification. Substitutes any TODO_ overrides
    the caller provided; anything not covered is left unresolved on
    purpose (see plan_render_service.apply_overrides docstring) rather
    than blocking approval.

    NOTE: this does NOT yet produce a full workflow_spec.json (Stage
    3.5 enrichment -- ASL types, resource ARNs, next_step wiring --
    is still unbuilt). This stores the resolved plan as an
    intermediate "approved" checkpoint.
    """
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No approvable plan yet -- conversation still needs clarification."
        )

    resolved_plan, remaining = apply_overrides(session.latest_plan, overrides)

    session_store.set_locked_spec(
        session_id,
        {
            "resolved_plan": resolved_plan.model_dump(),
            "remaining_placeholders": remaining,
        },
    )

    return ApproveResponse(
        status="approved",
        resolved_plan=resolved_plan,
        remaining_placeholders=remaining,
    )