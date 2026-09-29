"""Session orchestration used by routers.

Routes stay thin: HTTP in/out only. This module owns the conversation
loop formerly living as _run_extraction_turn inside session_router.

Lives inside pipeline_analysis/ (moved from services/session_service.py)
since analyze_pipeline() and ExtractionError -- its two heaviest
dependencies -- both live in the sibling pipeline_analysis/service.py
module; the old extraction/ package (extraction/service.py,
extraction/prompts.py) that ExtractionError previously came from has
been deleted -- its only other still-live pieces (classify_scope,
_chat_completion_structured, _call_llm_json_with_repair,
MAX_STAGE_ATTEMPTS) were folded directly into pipeline_analysis/service.py,
the only module that ever called them.
"""

from __future__ import annotations

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
from orchestrator.api.services.asl_service import (
    AslGenerationError,
    build_asl_from_spec,
    classify_placeholder_kind,
    describe_placeholders,
    find_asl_placeholders,
    render_asl_graph,
    sanitize_asl_literals,
    validate_asl,
    validate_placeholder_value,
)
from orchestrator.api.services.deploy.service import (
    deploy_state_machine,
    slugify_state_machine_name,
)
from orchestrator.api.services.enrichment.service import (
    EnrichmentError,
    continue_enrichment,
    enrich_plan,
)
from orchestrator.api.services.llm_trace import (
    end_request,
    get_status,
    llm_stage,
    log_event,
    start_request,
)
from orchestrator.api.services.plan.matching import match as match_plan
from orchestrator.api.services.pipeline_analysis.service import (
    ExtractionError,
    analyze_pipeline,
)
from orchestrator.api.services.plan.canonicalize import canonicalize_plan
from orchestrator.api.services.plan.render import (
    apply_overrides,
    apply_overrides_to_json,
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


class NoSpecPlaceholdersError(Exception):
    """fill_spec_placeholders called with no spec-level placeholders
    pending -- either no workflow_spec is locked yet, or the locked one
    already has zero TODO_ tokens (use /asl/generate to force a
    regeneration instead)."""


class UnresolvedSpecPlaceholdersError(Exception):
    """ASL generation was requested (directly, via /asl/generate) while
    the locked workflow_spec still has TODO_ tokens in it. Every other
    path into Stage 4 (approve_plan, answer_enrichment_question,
    fill_spec_placeholders) already routes through
    _finalize_workflow_spec, which refuses to build ASL from a spec
    that still has placeholders in it -- this exists so the
    force-regenerate entrypoint can't be used to sidestep that same
    rule. Caller should resolve everything via
    POST /session/{id}/spec/placeholders first."""


def start_session() -> Session:
    return session_store.create_session()


def get_live_status(session_id: str) -> dict:
    """What the backend is doing *right now* for this session, for the
    frontend to poll while /message or /approve is in flight on another
    connection. Cheap in-memory read -- safe to call every second.

    Returns {"busy": False} when nothing is running (already finished,
    not started, or unknown session -- deliberately not a 404 here so
    a client can poll optimistically right after firing a request).
    """
    snap = get_status(session_id)
    if not snap:
        return {"busy": False}
    return {
        "busy": True,
        "stage": snap.get("stage"),
        "label": snap.get("label"),
        "updated_at": snap.get("updated_at"),
    }


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

    Uses analyze_pipeline() -- the paper-aligned Stage 1 Pipeline
    Analysis chain (pipeline_analysis/service.py: environment inference
    -> component identification -> flow structure analysis -> parameter
    extraction -> integration analysis). There is no user-facing pause
    between its five sub-stages: the only human checkpoint in this
    product is plan review below (plain-language approve/reject), not
    an intermediate component list a non-developer user would have no
    way to judge.
    """
    session_store.append_message(session_id, "user", message)
    start_request(session_id=session_id, user_message=message)

    # session.history was fetched before the append above -- the
    # SQLite-backed store returns a snapshot, not a live reference, so
    # it would NOT include the message we just wrote. Build the
    # history actually sent to the LLM locally instead of trusting the
    # pre-fetched snapshot.
    current_history = [*session.history, {"role": "user", "content": message}]

    try:
        with llm_stage("pipeline_analysis"):
            result = analyze_pipeline(current_history)
    except ExtractionError as _ext_err:
        end_request(status="error", error=str(_ext_err))
        # Don't record a broken assistant turn into history -- next
        # retry should see the same clean state, not a garbage reply
        # baked into the conversation. Re-raise for the router to map
        # to HTTP 502 (same as before).
        raise

    # Record the assistant's turn as JSON text, matching what the LLM
    # actually said -- keeps history faithful to the real conversation
    # for the next call's context.
    end_request(status=result.status)

    # Plan matching / canonicalize still run on every ready turn -- match_plan
    # emits the plan_match telemetry event below and (for low/none tiers)
    # writes the curation-queue entry inside match_plan itself via
    # log_unmatched, and canonicalize_plan's output is compared against the
    # raw plan purely to log when it *would* have rewritten the topology.
    # NEITHER is allowed to reassign result.plan any more: what we render
    # (summary_markdown, diagram_dot), return to the client, and persist as
    # `latest_plan` for /approve + downstream ASL generation must always be
    # exactly what flow_structure_analysis produced -- never a template
    # projection (tier == "high"/"medium") and never a regex-heuristic
    # topology rebuild (canonicalize_plan Layer B). See plan/canonicalize.py
    # and plan/matching/matcher.py docstrings for what each would otherwise
    # have done.
    if result.status == "ready" and result.plan is not None:
        match_result = match_plan(
            result.plan,
            session_id=session_id,
            user_message=message,
        )
        log_event(
            "plan_match",
            tier=match_result.tier,
            archetype=getattr(match_result, "archetype", None)
            or getattr(match_result, "template_name", None),
            score=match_result.score,
            signature_hash=getattr(match_result, "signature_hash", None),
        )

        # Diagnostic only, from here down -- `diagnostic_plan` is never
        # assigned back to result.plan. It exists so curation can see when
        # the template/heuristic path disagrees with the raw LLM output,
        # without that disagreement ever reaching the user.
        if match_result.tier in ("high", "medium") and match_result.plan is not None:
            diagnostic_plan = (
                match_result.plan
                if match_result.tier == "high"
                else canonicalize_plan(match_result.plan)
            )
        else:
            diagnostic_plan = canonicalize_plan(result.plan)

        if diagnostic_plan.model_dump() != result.plan.model_dump():
            log_event(
                "plan_canonicalize_would_have_changed",
                tier=match_result.tier,
                archetype=getattr(match_result, "archetype", None)
                or getattr(match_result, "template_name", None),
                signature_hash=getattr(match_result, "signature_hash", None),
            )

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
            # Full Stage 1 Pipeline Analysis artifact (environment,
            # components, flow, parameters, integrations) -- see
            # schemas/pipeline_analysis.py:PipelineAnalysis. `result.plan`
            # above is derived from `analysis["flow"]` for everything
            # downstream that only understands the Plan shape; this field
            # is purely for client-side transparency/audit, matching the
            # paper's "human-readable report and comprehensive JSON
            # artifact" deliverable for Pipeline Analysis.
            pipeline_analysis=result.analysis,
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
    """Route every free-text turn through the LangGraph turn graph.

    Value-only edits (rename bucket/ARN/TODO, etc.) take patch_params
    (one small LLM + mechanical ASL regen). Structural changes take
    revise_plan (full re-extraction). New designs take extract.
    """
    get_session_or_raise(session_id)
    start_request(session_id=session_id, user_message=message)
    try:
        from orchestrator.api.services.orchestration.graph import run_message_turn
        from orchestrator.api.schemas.common import Plan

        state = run_message_turn(session_id, message)
        end_request(status=state.get("extraction_status") or state.get("last_stage") or "ok")
        return _message_response_from_turn_state(state)
    except Exception as exc:  # noqa: BLE001
        end_request(status="error", error=str(exc))
        raise


def reject_plan(session_id: str, feedback: str) -> MessageResponse:
    """Explicit reject — always structural revise (never param patch)."""
    session = get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise NoApprovablePlanError(
            "No plan to reject -- conversation still needs clarification."
        )
    start_request(session_id=session_id, user_message=feedback)
    try:
        from orchestrator.api.services.orchestration.graph import run_reject_turn

        state = run_reject_turn(session_id, feedback)
        end_request(status=state.get("extraction_status") or state.get("last_stage") or "ok")
        return _message_response_from_turn_state(state)
    except Exception as exc:  # noqa: BLE001
        end_request(status="error", error=str(exc))
        raise


def _message_response_from_turn_state(state: dict) -> MessageResponse:
    """Map turn-graph state to MessageResponse.

    For value-only patches, include asl / asl_graph_dot / intent so the UI
    stays on the ASL surface instead of re-showing plan review.
    """
    nested = state.get("message_response") if isinstance(state.get("message_response"), dict) else {}
    status = (nested.get("status") if nested else None) or state.get("extraction_status") or "needs_clarification"

    if status == "ready":
        plan_raw = (nested.get("plan") if nested else None) or state.get("plan")
        plan = None
        if plan_raw is not None:
            plan = plan_raw if isinstance(plan_raw, Plan) else Plan.model_validate(plan_raw)

        intent = (nested.get("intent") if nested else None) or state.get("route")
        phase = (nested.get("phase") if nested else None) or state.get("phase")
        asl = (nested.get("asl") if nested else None) or state.get("asl")
        asl_graph_dot = (nested.get("asl_graph_dot") if nested else None) or state.get(
            "asl_graph_dot"
        )
        asl_source = (nested.get("asl_source") if nested else None) or state.get("asl_source")

        return MessageResponse(
            status="ready",
            plan=plan,
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
            asl=asl,
            asl_graph_dot=asl_graph_dot,
            asl_validations=(nested.get("asl_validations") if nested else None)
            or state.get("asl_validations"),
            asl_source=asl_source,
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
    return MessageResponse(
        status="needs_clarification",
        question=(nested.get("question") if nested else None)
        or state.get("clarification_message"),
    )


def _run_asl_stage(
    workflow_spec: dict,
) -> tuple[dict | None, str | None, dict | None, str | None, str | None]:
    """Stage 4: workflow_spec -> ASL + validations + graph.

    Returns (asl, graph_dot, validations_dict, source, error_message).
    `source` is result.source ("mechanical" / "llm" / "mechanical_fallback"
    -- see asl_service.AslResult) so callers can surface which generator
    actually ran, e.g. after setting ASL_GENERATION_MODE. Never raises —
    ASL failures are reported alongside a successful approval.
    """
    try:
        with llm_stage("asl_generation"):
            result = build_asl_from_spec(workflow_spec)
        return (
            result.asl,
            result.graph_dot,
            result.validations.to_dict(),
            result.source,
            None,
        )
    except AslGenerationError as exc:
        return None, None, None, None, str(exc)
    except Exception as exc:  # noqa: BLE001 — surface unexpected translator bugs
        return None, None, None, None, f"ASL generation failed unexpectedly: {exc}"


def _finalize_workflow_spec(
    workflow_spec: dict,
) -> tuple[dict | None, str | None, dict | None, str | None, str | None, list[str]]:
    """The gate between Stage 3.5 (enrichment) and Stage 4 (ASL
    generation): scan the WHOLE workflow_spec for TODO_ placeholders
    before generating anything from it.

    Enrichment's ARN whitelist deliberately lets TODO_ tokens through
    when it can't resolve something on its own -- correct for Stage
    3.5, since a placeholder is a valid sentinel for "needs a human"
    at that point. But those tokens never appeared on the Stage 2 plan
    (apply_overrides/find_placeholders only ever saw plan text), so
    without this check they'd only surface after ASL generation ran
    once already (via asl_service.find_asl_placeholders inside
    validate_asl) -- meaning the user discovers what to fill in one
    token at a time, across repeated generate-inspect-fix cycles,
    instead of all at once.

    find_asl_placeholders is reused as-is here (not renamed) -- despite
    the name, it's a generic recursive JSON string scanner with no ASL-
    specific logic, so it works identically against a workflow_spec
    dict.

    Returns (asl, graph_dot, validations_dict, source, asl_error,
    spec_placeholders). When spec_placeholders is non-empty, the first
    five are always (None, None, None, None, None) -- ASL generation is
    skipped entirely, not attempted-and-reported.
    """
    spec_placeholders = find_asl_placeholders(workflow_spec)
    if spec_placeholders:
        return None, None, None, None, None, spec_placeholders
    asl, graph_dot, validations, source, asl_error = _run_asl_stage(workflow_spec)
    return asl, graph_dot, validations, source, asl_error, []


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
    asl_source: str | None = None
    asl_error: str | None = None
    spec_placeholders: list[str] = []
    start_request(session_id=session_id)
    try:
        result = enrich_plan(resolved_plan)
        if result.status == "complete":
            workflow_spec = result.spec
            asl, asl_graph_dot, asl_validations, asl_source, asl_error, spec_placeholders = (
                _finalize_workflow_spec(workflow_spec)
            )
        else:  # needs_clarification -- pause, persist state for answer endpoint
            enrichment_question = result.question
    except EnrichmentError as exc:
        enrichment_error = str(exc)
    finally:
        end_request(status="approved")

    # Human-readable "what to fill in" text for each spec_placeholders
    # token, derived from the workflow_spec's own parameters/
    # integrations/components metadata (falls back to plan text, then
    # a humanized token name -- see describe_placeholders docstring).
    placeholder_details = describe_placeholders(
        workflow_spec, spec_placeholders, plan=resolved_plan.model_dump()
    )

    locked = {
        "resolved_plan": resolved_plan.model_dump(),
        "remaining_placeholders": remaining,
        "workflow_spec": workflow_spec,
        "enrichment_error": enrichment_error,
        "enrichment_question": enrichment_question,
        "spec_placeholders": spec_placeholders,
        "placeholder_details": placeholder_details,
        "asl": asl,
        "asl_graph_dot": asl_graph_dot,
        "asl_validations": asl_validations,
        "asl_source": asl_source,
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
        spec_placeholders=spec_placeholders,
        placeholder_details=placeholder_details,
        asl=asl,
        asl_graph_dot=asl_graph_dot,
        asl_validations=asl_validations,
        asl_source=asl_source,
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
    asl_source: str | None = None
    asl_error: str | None = None
    spec_placeholders: list[str] = []
    start_request(session_id=session_id)
    try:
        result = continue_enrichment(
            plan, pending["messages"], answer, pending["clarification_rounds"]
        )
        if result.status == "complete":
            workflow_spec = result.spec
            asl, asl_graph_dot, asl_validations, asl_source, asl_error, spec_placeholders = (
                _finalize_workflow_spec(workflow_spec)
            )
        else:
            enrichment_question = result.question
    except EnrichmentError as exc:
        enrichment_error = str(exc)
    finally:
        end_request(status="approved")

    placeholder_details = describe_placeholders(
        workflow_spec, spec_placeholders, plan=plan.model_dump()
    )

    locked["workflow_spec"] = workflow_spec
    locked["enrichment_error"] = enrichment_error
    locked["enrichment_question"] = enrichment_question
    locked["spec_placeholders"] = spec_placeholders
    locked["placeholder_details"] = placeholder_details
    locked["asl"] = asl
    locked["asl_graph_dot"] = asl_graph_dot
    locked["asl_validations"] = asl_validations
    locked["asl_source"] = asl_source
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
        spec_placeholders=spec_placeholders,
        placeholder_details=placeholder_details,
        asl=asl,
        asl_graph_dot=asl_graph_dot,
        asl_validations=asl_validations,
        asl_source=asl_source,
        asl_error=asl_error,
    )


def fill_spec_placeholders(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    """Stage 3.5 -> Stage 4 gate: apply every value for the tokens
    approve_plan()/answer_enrichment_question() reported in
    spec_placeholders, all in one call, then generate ASL only once
    nothing is left unresolved.

    overrides is applied across the ENTIRE locked workflow_spec (every
    string, at any depth -- see apply_overrides_to_json), not just the
    specific fields ApproveRequest.overrides touches, since enrichment
    can place a TODO_ token anywhere. Whatever wasn't covered by
    overrides is reported back in spec_placeholders on the response --
    calling this with a partial set is allowed (mirrors approve_plan's
    "unresolved is reported, not rejected" stance), it just means ASL
    generation stays skipped until a follow-up call clears the rest.
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    workflow_spec = locked.get("workflow_spec")
    if not isinstance(workflow_spec, dict) or not workflow_spec:
        raise NoSpecPlaceholdersError(
            "No workflow_spec is locked for this session -- approve a plan "
            "(and complete enrichment) first."
        )
    if not locked.get("spec_placeholders"):
        raise NoSpecPlaceholdersError(
            "No spec-level placeholders are pending for this session. "
            "Use POST /session/{id}/asl/generate to force a regeneration."
        )

    resolved_plan = Plan.model_validate(locked["resolved_plan"])
    plan_dict = resolved_plan.model_dump()

    # Reject overrides of the wrong shape (a bare name where an ARN is
    # required, an ARN/URI where a bare name is required, etc.) instead
    # of writing them into the spec -- a malformed value would either
    # fail deploy pre-checks opaquely later or, worse, silently produce
    # a working-looking but wrong ARN/name. Anything rejected here is
    # simply left out of what we apply, so find_asl_placeholders() will
    # still report its token as unresolved afterwards.
    placeholder_errors: dict[str, str] = {}
    valid_overrides: dict[str, str] = {}
    for token, value in overrides.items():
        kind = classify_placeholder_kind(workflow_spec, token, plan=plan_dict)
        error = validate_placeholder_value(kind, value)
        if error:
            placeholder_errors[token] = error
        else:
            valid_overrides[token] = value

    updated_spec = apply_overrides_to_json(workflow_spec, valid_overrides)

    asl: dict | None = None
    asl_graph_dot: str | None = None
    asl_validations: dict | None = None
    asl_source: str | None = None
    asl_error: str | None = None
    start_request(session_id=session_id)
    try:
        asl, asl_graph_dot, asl_validations, asl_source, asl_error, spec_placeholders = (
            _finalize_workflow_spec(updated_spec)
        )
    finally:
        end_request(status="approved")

    placeholder_details = describe_placeholders(
        updated_spec, spec_placeholders, plan=plan_dict
    )

    locked["workflow_spec"] = updated_spec
    locked["spec_placeholders"] = spec_placeholders
    locked["placeholder_details"] = placeholder_details
    locked["placeholder_errors"] = placeholder_errors
    locked["asl"] = asl
    locked["asl_graph_dot"] = asl_graph_dot
    locked["asl_validations"] = asl_validations
    locked["asl_source"] = asl_source
    locked["asl_error"] = asl_error
    session_store.set_locked_spec(session_id, locked)

    return ApproveResponse(
        status="approved",
        resolved_plan=resolved_plan,
        remaining_placeholders=locked.get("remaining_placeholders") or [],
        workflow_spec=updated_spec,
        enrichment_error=locked.get("enrichment_error"),
        enrichment_question=locked.get("enrichment_question"),
        spec_placeholders=spec_placeholders,
        placeholder_details=placeholder_details,
        placeholder_errors=placeholder_errors,
        asl=asl,
        asl_graph_dot=asl_graph_dot,
        asl_validations=asl_validations,
        asl_source=asl_source,
        asl_error=asl_error,
    )


def generate_asl_for_session(session_id: str) -> AslValidateResponse:
    """Re-run Stage 4 from the locked workflow_spec (e.g. after a
    manual edit of the spec on the client, or to refresh validations).

    Subject to the same placeholder gate as every other path into
    Stage 4 (see _finalize_workflow_spec): if the locked spec still
    has TODO_ tokens anywhere in it, generation is refused rather than
    run against unresolved input. This matters here specifically
    because a client-side manual edit could reintroduce a TODO_ token
    that was previously resolved -- without this check that edit would
    sail straight into ASL generation.
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    spec = locked.get("workflow_spec")
    if not spec:
        raise NoWorkflowSpecError(
            "No workflow_spec locked for this session — approve a plan first."
        )

    spec_placeholders = find_asl_placeholders(spec)
    if spec_placeholders:
        locked["spec_placeholders"] = spec_placeholders
        session_store.set_locked_spec(session_id, locked)
        raise UnresolvedSpecPlaceholdersError(
            "workflow_spec still has unresolved TODO_ placeholders: "
            f"{', '.join(spec_placeholders)}. Resolve them via "
            "POST /session/{id}/spec/placeholders before generating ASL."
        )

    asl, graph_dot, validations, source, error = _run_asl_stage(spec)
    if error:
        # Still persist whatever we can; validations may be partial.
        locked["asl"] = asl
        locked["asl_graph_dot"] = graph_dot
        locked["asl_validations"] = validations
        locked["asl_source"] = source
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
            asl_source=source,
        )

    locked["asl"] = asl
    locked["asl_graph_dot"] = graph_dot
    locked["asl_validations"] = validations
    locked["asl_source"] = source
    locked["asl_error"] = None
    session_store.set_locked_spec(session_id, locked)
    return AslValidateResponse(
        asl=asl, validations=validations or {}, graph_dot=graph_dot, asl_source=source
    )


def revalidate_asl_payload(
    session_id: str | None, asl: dict | None
) -> AslValidateResponse:
    """Validate a client-supplied ASL (or the session's stored ASL)
    against the locked workflow_spec when available.
    """
    spec: dict | None = None
    stored_asl: dict | None = None
    stored_source: str | None = None
    if session_id:
        session = get_session_or_raise(session_id)
        locked = session.locked_spec or {}
        spec = locked.get("workflow_spec")
        stored_asl = locked.get("asl")
        stored_source = locked.get("asl_source")

    target = asl if asl is not None else stored_asl
    if target is None:
        raise NoWorkflowSpecError("No ASL provided and none stored on the session.")
    # asl_source only makes sense for the stored ASL (we know what
    # generated it); a client-supplied payload's origin is unknown to us.
    source = stored_source if asl is None else None

    # Client-supplied ASL (the ``asl`` branch above) never passed through
    # generate_asl()'s translator, so it never got the "mistakenly wrapped
    # literal" fix -- e.g. a hand-edited "{% s3://bucket/key %}" that
    # should just be a plain string. Apply the same fix here so a re-
    # validated payload is held to the same bar as a freshly generated
    # one, instead of failing opaquely at AWS's live validator with no
    # actionable message. No-op on values that are already correct
    # (stored_asl, or an asl payload with no mistaken wraps).
    if isinstance(target, dict):
        target = sanitize_asl_literals(target)

    graph_dot = render_asl_graph(target) if isinstance(target, dict) else ""
    report = validate_asl(target, spec=spec, graph_dot=graph_dot)
    return AslValidateResponse(
        asl=target if isinstance(target, dict) else None,
        validations=report.to_dict(),
        graph_dot=graph_dot or None,
        asl_source=source,
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


def deploy_workflow(session_id: str, request: DeployRequest) -> DeployResponse:
    """Stage 5: create-or-update the session's locked ASL as a real
    Step Functions state machine.

    Requires a locked ASL definition (approve a plan, complete
    enrichment, and let Stage 4 generate ASL first — same
    precondition as get_asl_for_download). The actual readiness gate
    (validation report, live AWS validate, leftover placeholders,
    execution role) lives in deploy_state_machine()/
    run_pre_deploy_checks() — this function just resolves what to pass
    it and persists the outcome.
    """
    session = get_session_or_raise(session_id)
    locked = session.locked_spec or {}
    asl = locked.get("asl")
    if not isinstance(asl, dict) or not asl:
        raise NoAslError(
            "No ASL definition stored for this session — approve a plan "
            "(and complete enrichment) first."
        )

    if request.state_machine_name:
        name = slugify_state_machine_name(request.state_machine_name)
    else:
        resolved_plan = locked.get("resolved_plan") or {}
        name = slugify_state_machine_name(resolved_plan.get("name") or "workflow")

    result = deploy_state_machine(
        name=name,
        asl=asl,
        role_arn=request.role_arn,
        deploy_role_arn=request.deploy_role_arn,
        external_id=request.external_id,
        validations=locked.get("asl_validations"),
        tags=request.tags,
        skip_checks=request.force,
    )

    locked["deployment"] = {
        "state_machine_arn": result.state_machine_arn,
        "action": result.action,
        "state_machine_name": result.state_machine_name,
        "region": result.region,
        "role_arn": result.role_arn,
        "account_id": result.account_id,
    }
    session_store.set_locked_spec(session_id, locked)

    return DeployResponse(
        status="deployed",
        state_machine_arn=result.state_machine_arn,
        action=result.action,
        state_machine_name=result.state_machine_name,
        region=result.region,
        role_arn=result.role_arn,
        account_id=result.account_id,
    )