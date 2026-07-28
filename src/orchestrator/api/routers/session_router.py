"""All session-related routes.

Routes are intentionally thin -- real logic lives in services/ and
repositories/. A route's job is: validate the HTTP request, call one
service function, map domain errors to HTTP, return the response.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from orchestrator.api.schemas.session import (
    ApproveRequest,
    ApproveResponse,
    MessageRequest,
    MessageResponse,
    RejectRequest,
    SessionStateResponse,
    StartSessionResponse,
)
from orchestrator.api.services.extraction_service import ExtractionError
from orchestrator.api.services import session_service

router = APIRouter(prefix="/session", tags=["session"])


@router.post("", response_model=StartSessionResponse)
def start_session() -> StartSessionResponse:
    session = session_service.start_session()
    return StartSessionResponse(session_id=session.session_id)


@router.get("/{session_id}", response_model=SessionStateResponse)
def get_session_state(session_id: str) -> SessionStateResponse:
    """Read-only view of a session's full state -- history, the
    latest ready plan (if any), and the approved/resolved spec (if
    /approve has been called). Pure read, no side effects, safe to
    call repeatedly (e.g. on every page load/refresh) -- unlike
    /message, /approve, /reject, which all change state and must stay
    POST.
    """
    try:
        session = session_service.get_session_or_raise(session_id)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None

    return SessionStateResponse(
        session_id=session.session_id,
        history=session.history,
        latest_plan=session.latest_plan,
        locked_spec=session.locked_spec,
    )


@router.post("/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
    try:
        return session_service.send_message(session_id, request.message)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{session_id}/reject", response_model=MessageResponse)
def reject_plan(session_id: str, request: RejectRequest) -> MessageResponse:
    """User wants changes to the current plan. Not a separate flow --
    the feedback is just appended as the next user turn, same as
    /message.
    """
    try:
        return session_service.reject_plan(session_id, request.feedback)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoApprovablePlanError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{session_id}/approve", response_model=ApproveResponse)
def approve_plan(session_id: str, request: ApproveRequest) -> ApproveResponse:
    """Human checkpoint (Stage 3). Requires a ready plan to exist --
    can't approve mid-clarification. Substitutes any TODO_ overrides
    the caller provided; anything not covered is left unresolved on
    purpose rather than blocking approval.
    """
    try:
        return session_service.approve_plan(session_id, request.overrides)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoApprovablePlanError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None