"""All session-related routes.

Routes are intentionally thin -- real logic lives in services/ and
repositories/. A route's job is: validate the HTTP request, call one
service function, map domain errors to HTTP, return the response.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from orchestrator.api.schemas.session import (
    ApproveRequest,
    ApproveResponse,
    AslValidateRequest,
    AslValidateResponse,
    DeployRequest,
    DeployResponse,
    EnrichmentAnswerRequest,
    MessageRequest,
    MessageResponse,
    RejectRequest,
    SessionStateResponse,
    SpecPlaceholdersRequest,
    StartSessionResponse,
)
from orchestrator.api.services.deploy.service import DeploymentError, PreDeployCheckError
from orchestrator.api.services.pipeline_analysis.service import ExtractionError
from orchestrator.api.services.plan.render import PlanRenderError, find_placeholders
from orchestrator.api.services.pipeline_analysis import session_service

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
        placeholders=find_placeholders(session.latest_plan) if session.latest_plan else None,
    )


@router.get("/{session_id}/status")
def get_session_status(session_id: str) -> dict:
    """Poll this while /message, /reject, /approve, or
    /enrichment/answer is in flight (on another connection) to show
    the user what the backend is actually doing right now, instead of
    a generic spinner.

    Cheap in-memory read, safe to poll every ~1s. Does not require the
    session to already exist server-side at call time -- always
    returns {"busy": false} rather than 404 so a client can start
    polling immediately after firing the request without a race.
    """
    return session_service.get_live_status(session_id)


@router.post("/{session_id}/message", response_model=MessageResponse)
def send_message(session_id: str, request: MessageRequest) -> MessageResponse:
    try:
        return session_service.send_message(session_id, request.message)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except ExtractionError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except PlanRenderError as exc:
        # Plan came back "ready" but with no structured control flow --
        # an extraction defect (see plan_render_service.py), not a
        # client error. Same 502 treatment as ExtractionError: the
        # client should retry the turn.
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
    except PlanRenderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/{session_id}/approve", response_model=ApproveResponse)
def approve_plan(session_id: str, request: ApproveRequest) -> ApproveResponse:
    """Human checkpoint (Stage 3). Requires a ready plan to exist --
    can't approve mid-clarification. Substitutes any TODO_ overrides
    the caller provided; anything not covered is left unresolved on
    purpose rather than blocking approval.

    When enrichment completes, Stage 4 ASL generation + validation +
    graph rendering runs automatically and is returned on the response.
    """
    try:
        return session_service.approve_plan(session_id, request.overrides)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoApprovablePlanError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/{session_id}/enrichment/answer", response_model=ApproveResponse)
def answer_enrichment_question(
    session_id: str, request: EnrichmentAnswerRequest
) -> ApproveResponse:
    """Answers a clarifying question enrichment asked during /approve
    (see ApproveResponse.enrichment_question). May itself return
    another enrichment_question if further clarification is needed.
    On success, Stage 4 ASL is generated as well.
    """
    try:
        return session_service.answer_enrichment_question(session_id, request.answer)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoEnrichmentQuestionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/{session_id}/spec/placeholders", response_model=ApproveResponse)
def fill_spec_placeholders(
    session_id: str, request: SpecPlaceholdersRequest
) -> ApproveResponse:
    """Fill every TODO_ token enrichment left in workflow_spec, all in
    one call (see ApproveResponse.spec_placeholders for why this exists
    as a separate step from /approve's overrides).

    409 if there's no workflow_spec locked yet, or if there are simply
    no spec-level placeholders pending right now (e.g. because they
    were already all filled, or enrichment never introduced any) --
    use /asl/generate instead if the goal is just to force a
    regeneration. On success with everything resolved, Stage 4 runs
    immediately, same as /approve and /enrichment/answer.
    """
    try:
        return session_service.fill_spec_placeholders(session_id, request.overrides)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoSpecPlaceholdersError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/{session_id}/asl/generate", response_model=AslValidateResponse)
def generate_asl(session_id: str) -> AslValidateResponse:
    """Re-run Stage 4 from the locked workflow_spec (refresh ASL,
    validations, and console-style graph).
    """
    try:
        return session_service.generate_asl_for_session(session_id)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoWorkflowSpecError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except session_service.UnresolvedSpecPlaceholdersError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/{session_id}/asl/validate", response_model=AslValidateResponse)
def validate_asl_endpoint(
    session_id: str, request: AslValidateRequest
) -> AslValidateResponse:
    """Validate ASL (body.asl or the session's stored ASL) against the
    locked workflow_spec when present. Runs all five validation groups.
    """
    try:
        return session_service.revalidate_asl_payload(session_id, request.asl)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoWorkflowSpecError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.get("/{session_id}/asl/download")
def download_asl(session_id: str) -> Response:
    """Download the session's stored ASL definition as a JSON file.

    Serves the locked ASL from the backend (not a client-side dump of
    whatever the UI last rendered). Returns application/json with
    Content-Disposition: attachment.
    """
    try:
        asl, filename = session_service.get_asl_for_download(session_id)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoAslError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None

    body = json.dumps(asl, indent=2, ensure_ascii=False).encode("utf-8")
    return Response(
        content=body,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(body)),
        },
    )


@router.post("/{session_id}/deploy", response_model=DeployResponse)
def deploy_state_machine(session_id: str, request: DeployRequest) -> DeployResponse:
    """Stage 5: deploy the session's locked ASL to AWS Step Functions
    (create if no state machine of this name exists yet, update in
    place if it does).

    Runs a final round of pre-deploy checks first (validation report
    must be ok, AWS's own live validation must not have hard-failed,
    zero leftover TODO_ placeholders anywhere in the ASL, and an IAM
    execution role must be configured) and refuses to deploy — 422,
    with the full list of reasons — unless every one passes. Pass
    `force: true` in the body to skip these checks; there is no
    partial-deploy state, so a failed check never leaves anything
    half-created.
    """
    try:
        return session_service.deploy_workflow(session_id, request)
    except session_service.SessionNotFoundError:
        raise HTTPException(status_code=404, detail="Unknown session_id") from None
    except session_service.NoAslError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    except PreDeployCheckError as exc:
        raise HTTPException(status_code=422, detail={"reasons": exc.reasons}) from None
    except DeploymentError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc