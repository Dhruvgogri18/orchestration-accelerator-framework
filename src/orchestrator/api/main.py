"""FastAPI app exposing the extraction flow to a client (e.g. the
Streamlit UI). Routes are intentionally thin -- all real logic lives
in extractor.py, plan_render.py, and session_store.py. A route's job
is: validate the HTTP request, call one function, shape the response.
Nothing here should ever construct a prompt or touch conversation
state directly.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from orchestrator.extraction import session_store
from orchestrator.extraction.extractor import ExtractionError, extract_turn
from orchestrator.extraction.models import (
    ApproveRequest,
    ApproveResponse,
    MessageRequest,
    MessageResponse,
    RejectRequest,
    SessionStateResponse,
    StartSessionResponse,
)
from orchestrator.extraction.plan_render import (
    apply_overrides,
    find_placeholders,
    render_plan_diagram,
    render_plan_summary,
)

app = FastAPI(title="Orchestration Accelerator API")


@app.post("/session", response_model=StartSessionResponse)
def start_session() -> StartSessionResponse:
    session = session_store.create_session()
    return StartSessionResponse(session_id=session.session_id)


@app.get("/session/{session_id}", response_model=SessionStateResponse)
def get_session_state(session_id: str) -> SessionStateResponse:
    """Read-only view of a session's full state -- history, the
    latest ready plan (if any), and the approved/resolved spec (if
    /approve has been called). Pure read, no side effects, safe to
    call repeatedly (e.g. on every page load/refresh) -- unlike
    /message, /approve, /reject, which all change state and must stay
    POST.
    """
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return SessionStateResponse(
        session_id=session.session_id,
        history=session.history,
        latest_plan=session.latest_plan,
        locked_spec=session.locked_spec,
    )


def _run_extraction_turn(session_id: str, session, message: str) -> MessageResponse:
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
    except ExtractionError as exc:
        # Don't record a broken assistant turn into history -- next
        # retry should see the same clean state, not a garbage reply
        # baked into the conversation.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Record the assistant's turn as JSON text, matching what the LLM
    # actually said -- keeps history faithful to the real conversation
    # for the next call's context.
    session_store.append_message(session_id, "assistant", result.model_dump_json())

    # Persist the latest ready plan -- /approve needs to know "is there
    # actually something approvable right now". Cleared (None)
    # whenever the conversation is still clarifying, so a stale plan
    # can't be approved after the user changes something (including a
    # /reject that reopens clarification).
    session_store.set_latest_plan(session_id, result.plan if result.status == "ready" else None)

    if result.status == "ready":
        return MessageResponse(
            status="ready",
            plan=result.plan,
            summary_markdown=render_plan_summary(result.plan),
            diagram_dot=render_plan_diagram(result.plan),
            placeholders=find_placeholders(result.plan),
        )
    return MessageResponse(status="needs_clarification", question=result.clarifying_question)


@app.post("/session/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return _run_extraction_turn(session_id, session, request.message)


@app.post("/session/{session_id}/reject", response_model=MessageResponse)
def reject_plan(session_id: str, request: RejectRequest) -> MessageResponse:
    """User wants changes to the current plan. Not a separate flow --
    the feedback is just appended as the next user turn, same as
    /message. Reuses the identical extraction/rendering logic so the
    two code paths can't drift apart.
    """
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if session.latest_plan is None:
        raise HTTPException(
            status_code=409,
            detail="No plan to reject -- conversation still needs clarification.",
        )
    return _run_extraction_turn(session_id, session, request.feedback)


@app.post("/session/{session_id}/approve", response_model=ApproveResponse)
def approve_plan(session_id: str, request: ApproveRequest) -> ApproveResponse:
    """Human checkpoint (Stage 3). Requires a ready plan to exist --
    can't approve mid-clarification. Substitutes any TODO_ overrides
    the caller provided; anything not covered is left unresolved on
    purpose (see plan_render.apply_overrides docstring) rather than
    blocking approval.

    NOTE: this does NOT yet produce a full workflow_spec.json (Stage
    3.5 enrichment -- ASL types, resource ARNs, next_step wiring --
    is still unbuilt). This stores the resolved plan as an
    intermediate "approved" checkpoint.
    """
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    if session.latest_plan is None:
        raise HTTPException(
            status_code=409,
            detail="No approvable plan yet -- conversation still needs clarification.",
        )

    resolved_plan, remaining = apply_overrides(session.latest_plan, request.overrides)

    session_store.set_locked_spec(
        session_id,
        {"resolved_plan": resolved_plan.model_dump(), "remaining_placeholders": remaining},
    )

    return ApproveResponse(
        status="approved",
        resolved_plan=resolved_plan,
        remaining_placeholders=remaining,
    )