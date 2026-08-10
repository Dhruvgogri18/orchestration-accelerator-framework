"""Session orchestration used by routers.

Routes stay thin: HTTP in/out only. This module owns the conversation
loop formerly living as _run_extraction_turn inside session_router.
"""

from __future__ import annotations

from orchestrator.api.repositories import session_repository as session_store
from orchestrator.api.repositories.session_repository import Session
from orchestrator.api.schemas.common import Plan
from orchestrator.api.schemas.session import (
    ApproveResponse,
    AslValidateResponse,
    MessageResponse,
)
from orchestrator.api.services.asl_service import (
    AslGenerationError,
    build_asl_from_spec,
    render_asl_graph,
    validate_asl,
)
from orchestrator.api.services.enrichment_service import (
    EnrichmentError,
    continue_enrichment,
    enrich_plan,
)
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


class NoEnrichmentQuestionError(Exception):
    """answer_enrichment_question called with no clarification pending."""


class NoWorkflowSpecError(Exception):
    """ASL generation/validation requested but no workflow_spec is locked."""


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
    if result.status == "out_of_scope":
        return MessageResponse(
            status="out_of_scope",
            message=result.message
            or (
                "This assistant is only designed to help you build AWS Step "
                "Functions workflows. Please describe a workflow to orchestrate."
            ),
        )
    return MessageResponse(
        status="needs_clarification",
        question=result.clarifying_question,
    )


def send_message(session_id: str, message: str) -> MessageResponse:
    session = get_session_or_raise(session_id)
    return run_extraction_turn(session_id, session, message)


def reject_plan(session_id: str, feedback: str) -> MessageResponse:
    """User wants changes to the current plan.

    Feedback is still one more user turn in the same extraction loop,
    but it is wrapped with an explicit revision instruction so the
    model cannot treat it as a soft comment and re-emit the previous
    plan unchanged (common with temperature=0 and the prior plan JSON
    still in history).
    """
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No plan to reject -- conversation still needs clarification."
        )
    revision_message = (
        "REJECTED — revise the previous plan. Do not return the same "
        "plan, steps, or structure. Apply this feedback:\n\n"
        f"{feedback.strip()}"
    )
    return run_extraction_turn(session_id, session, revision_message)


def _run_asl_stage(workflow_spec: dict) -> tuple[dict | None, str | None, dict | None, str | None]:
    """Stage 4: workflow_spec -> ASL + validations + graph.

    Returns (asl, graph_dot, validations_dict, error_message).
    Never raises — ASL failures are reported alongside a successful approval.
    """
    try:
        result = build_asl_from_spec(workflow_spec)
        return (
            result.asl,
            result.graph_dot,
            result.validations.to_dict(),
            None,
        )
    except AslGenerationError as exc:
        return None, None, None, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface unexpected translator bugs
        return None, None, None, f"ASL generation failed unexpectedly: {exc}"


def approve_plan(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    """Human checkpoint (Stage 3). Requires a ready plan to exist --
    can't approve mid-clarification. Substitutes any TODO_ overrides
    the caller provided; anything not covered is left unresolved on
    purpose (see plan_render_service.apply_overrides docstring) rather
    than blocking approval.

    Stage 3.5 enrichment (resolved plan -> full workflow_spec.json)
    runs right after -- but a failure there does NOT fail the approve
    call itself. Approval is a human decision about the WORKFLOW; it
    should succeed even if the mechanical translation into ASL
    structure needs another attempt later. enrichment_error is
    returned so the caller knows to retry/investigate, without losing
    the approval that already happened.

    When enrichment completes, Stage 4 (ASL generation + the five
    validation groups + console-style graph) runs immediately and the
    results are stored on the locked_spec and returned on the response.
    """
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No approvable plan yet -- conversation still needs clarification."
        )

    resolved_plan, remaining = apply_overrides(session.latest_plan, overrides)

    workflow_spec: dict | None = None
    enrichment_error: str | None = None
    enrichment_question: str | None = None
    asl: dict | None = None
    asl_graph_dot: str | None = None
    asl_validations: dict | None = None
    asl_error: str | None = None
    try:
        result = enrich_plan(resolved_plan)
        if result.status == "complete":
            workflow_spec = result.spec
            asl, asl_graph_dot, asl_validations, asl_error = _run_asl_stage(workflow_spec)
        else:  # needs_clarification -- pause, persist state for answer endpoint
            enrichment_question = result.question
    except EnrichmentError as exc:
        enrichment_error = str(exc)

    locked = {
        "resolved_plan": resolved_plan.model_dump(),
        "remaining_placeholders": remaining,
        "workflow_spec": workflow_spec,
        "enrichment_error": enrichment_error,
        "enrichment_question": enrichment_question,
        "asl": asl,
        "asl_graph_dot": asl_graph_dot,
        "asl_validations": asl_validations,
        "asl_error": asl_error,
    }
    if enrichment_question is not None:
        locked["enrichment_pending"] = {
            "messages": result.messages,
            "clarification_rounds": result.clarification_rounds,
        }
    session_store.set_locked_spec(session_id, locked)

    return ApproveResponse(
        status="approved",
        resolved_plan=resolved_plan,
        remaining_placeholders=remaining,
        workflow_spec=workflow_spec,
        enrichment_error=enrichment_error,
        enrichment_question=enrichment_question,
        asl=asl,
        asl_graph_dot=asl_graph_dot,
        asl_validations=asl_validations,
        asl_error=asl_error,
    )


def answer_enrichment_question(session_id: str, answer: str) -> ApproveResponse:
    """Resumes an enrichment conversation paused on a clarifying
    question (see approve_plan). Reuses the persisted resolved_plan/
    remaining_placeholders -- only the enrichment outcome changes.
    When enrichment completes, Stage 4 ASL generation runs as well.
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    pending = locked.get("enrichment_pending")
    if not pending:
        raise NoEnrichmentQuestionError("No enrichment clarification pending for this session.")

    plan = Plan.model_validate(locked["resolved_plan"])
    workflow_spec: dict | None = None
    enrichment_error: str | None = None
    enrichment_question: str | None = None
    asl: dict | None = None
    asl_graph_dot: str | None = None
    asl_validations: dict | None = None
    asl_error: str | None = None
    try:
        result = continue_enrichment(
            plan, pending["messages"], answer, pending["clarification_rounds"]
        )
        if result.status == "complete":
            workflow_spec = result.spec
            asl, asl_graph_dot, asl_validations, asl_error = _run_asl_stage(workflow_spec)
        else:
            enrichment_question = result.question
    except EnrichmentError as exc:
        enrichment_error = str(exc)

    locked["workflow_spec"] = workflow_spec
    locked["enrichment_error"] = enrichment_error
    locked["enrichment_question"] = enrichment_question
    locked["asl"] = asl
    locked["asl_graph_dot"] = asl_graph_dot
    locked["asl_validations"] = asl_validations
    locked["asl_error"] = asl_error
    if enrichment_question is not None:
        locked["enrichment_pending"] = {
            "messages": result.messages,
            "clarification_rounds": result.clarification_rounds,
        }
    else:
        locked.pop("enrichment_pending", None)
    session_store.set_locked_spec(session_id, locked)

    return ApproveResponse(
        status="approved",
        resolved_plan=plan,
        remaining_placeholders=locked["remaining_placeholders"],
        workflow_spec=workflow_spec,
        enrichment_error=enrichment_error,
        enrichment_question=enrichment_question,
        asl=asl,
        asl_graph_dot=asl_graph_dot,
        asl_validations=asl_validations,
        asl_error=asl_error,
    )


def generate_asl_for_session(session_id: str) -> AslValidateResponse:
    """Re-run Stage 4 from the locked workflow_spec (e.g. after a
    manual edit of the spec on the client, or to refresh validations).
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    spec = locked.get("workflow_spec")
    if not spec:
        raise NoWorkflowSpecError(
            "No workflow_spec locked for this session — approve a plan first."
        )

    asl, graph_dot, validations, error = _run_asl_stage(spec)
    if error:
        # Still persist whatever we can; validations may be partial.
        locked["asl"] = asl
        locked["asl_graph_dot"] = graph_dot
        locked["asl_validations"] = validations
        locked["asl_error"] = error
        session_store.set_locked_spec(session_id, locked)
        return AslValidateResponse(
            asl=asl,
            validations=validations
            or {
                "ok": False,
                "errors": [{"group": "generation", "code": "error", "message": error}],
            },
            graph_dot=graph_dot,
        )

    locked["asl"] = asl
    locked["asl_graph_dot"] = graph_dot
    locked["asl_validations"] = validations
    locked["asl_error"] = None
    session_store.set_locked_spec(session_id, locked)
    return AslValidateResponse(asl=asl, validations=validations or {}, graph_dot=graph_dot)


def revalidate_asl_payload(
    session_id: str | None, asl: dict | None
) -> AslValidateResponse:
    """Validate a client-supplied ASL (or the session's stored ASL)
    against the locked workflow_spec when available.
    """
    spec: dict | None = None
    stored_asl: dict | None = None
    if session_id:
        session = get_session_or_raise(session_id)
        locked = session.locked_spec or {}
        spec = locked.get("workflow_spec")
        stored_asl = locked.get("asl")

    target = asl if asl is not None else stored_asl
    if target is None:
        raise NoWorkflowSpecError("No ASL provided and none stored on the session.")

    graph_dot = render_asl_graph(target) if isinstance(target, dict) else ""
    report = validate_asl(target, spec=spec, graph_dot=graph_dot)
    return AslValidateResponse(
        asl=target if isinstance(target, dict) else None,
        validations=report.to_dict(),
        graph_dot=graph_dot or None,
    )


class NoAslError(Exception):
    """Download/fetch requested but no ASL is stored on the session."""


def get_asl_for_download(session_id: str) -> tuple[dict, str]:
    """Return the locked ASL definition and a suggested download filename.

    Raises SessionNotFoundError or NoAslError.
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    asl = locked.get("asl")
    if not isinstance(asl, dict) or not asl:
        raise NoAslError(
            "No ASL definition stored for this session — approve a plan "
            "(and complete enrichment) first."
        )

    name = "workflow"
    comment = asl.get("Comment")
    if isinstance(comment, str) and comment.strip():
        # Comment is often "name: summary" from the generator
        name = comment.split(":")[0].strip() or name
    # Keep filename filesystem-safe
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("._")
    filename = f"{safe or 'workflow'}.asl.json"
    return asl, filename

