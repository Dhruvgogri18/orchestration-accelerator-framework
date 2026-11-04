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
        describe_placeholders,
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
        plan = state.get("resolved_plan") or state.get("plan")
        plan_dict = plan.model_dump() if hasattr(plan, "model_dump") else plan
        return {
            "asl": None,
            "asl_graph_dot": None,
            "asl_validations": None,
            "asl_source": None,
            "asl_error": None,
            "spec_placeholders": spec_placeholders,
            "placeholder_details": describe_placeholders(
                spec, spec_placeholders, plan=plan_dict
            ),
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
            "asl_source": result.source,
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


# ---------------------------------------------------------------------------
# Session turn graph nodes (message / reject routing)
# ---------------------------------------------------------------------------


def load_context(state: dict) -> dict:
    """Load session artifacts and derive phase for the router."""
    from orchestrator.api.repositories import session_repository as session_store
    from orchestrator.api.services.orchestration.phase import derive_phase, phase_label
    from orchestrator.api.services.plan.render import render_plan_summary

    session_id = state.get("session_id") or ""
    session = session_store.get_session(session_id)
    if session is None:
        return {
            "error": f"Unknown session_id: {session_id}",
            "phase": "collecting",
            "has_plan": False,
            "has_asl": False,
            "last_stage": "load_context",
        }

    locked = session.locked_spec if isinstance(session.locked_spec, dict) else {}
    plan = session.latest_plan
    asl = locked.get("asl")
    has_asl = isinstance(asl, dict) and bool(asl.get("States") or asl)
    phase = derive_phase(session)
    plan_summary = None
    if plan is not None:
        try:
            plan_summary = render_plan_summary(plan)
        except Exception:  # noqa: BLE001
            plan_summary = None

    return {
        "phase": phase.value if hasattr(phase, "value") else str(phase),
        "has_plan": plan is not None,
        "has_asl": has_asl,
        "enrichment_question": locked.get("enrichment_question"),
        "plan_summary": plan_summary,
        "last_stage": "load_context",
    }


def route_turn_node(state: dict) -> dict:
    """LLM (or pre-set) route decision for this turn."""
    # Explicit reject button hard-codes route=revise_plan upstream.
    if state.get("action") == "reject" and state.get("route") == "revise_plan":
        return {
            "route": "revise_plan",
            "route_rationale": "Explicit reject — structural revise.",
            "last_stage": "route_turn",
        }

    from orchestrator.api.services.orchestration.llm_router import route_turn

    decision = route_turn(
        user_message=state.get("user_message") or "",
        phase=state.get("phase") or "collecting",
        has_plan=bool(state.get("has_plan")),
        has_asl=bool(state.get("has_asl")),
        enrichment_question=state.get("enrichment_question"),
        plan_summary=state.get("plan_summary"),
        session_id=state.get("session_id"),
    )
    return {
        "route": decision.route,
        "route_rationale": decision.rationale,
        "last_stage": "route_turn",
    }


def extract_node(state: dict) -> dict:
    """New or continuing workflow design — enterprise RAG+tools plan graph."""
    from orchestrator.api.repositories import session_repository as session_store
    from orchestrator.api.services.pipeline_analysis import session_service as legacy

    session_id = state.get("session_id") or ""
    message = state.get("user_message") or ""
    session = session_store.get_session(session_id)
    if session is None:
        return {
            "extraction_status": "error",
            "error": f"Unknown session_id: {session_id}",
            "last_stage": "extract",
        }

    response = legacy.run_extraction_turn(session_id, session, message)
    update: dict[str, Any] = {
        "extraction_status": response.status,
        "last_stage": "extract",
        "message_response": response.model_dump(),
    }
    if response.status == "ready" and response.plan is not None:
        update["plan"] = response.plan.model_dump()
        update["summary_markdown"] = response.summary_markdown
        update["diagram_dot"] = response.diagram_dot
        update["placeholders"] = response.placeholders
        update["analysis"] = response.pipeline_analysis
        # Snapshot plan history for restore_previous
        try:
            from orchestrator.api.services.orchestration import plan_history

            locked = session_store.get_session(session_id)
            locked_spec = (locked.locked_spec if locked else None) or {}
            updated = plan_history.push_plan(locked_spec, response.plan)
            if updated is not locked_spec:
                session_store.set_locked_spec(session_id, updated)
        except Exception:  # noqa: BLE001
            pass
    elif response.status == "needs_clarification":
        update["clarification_message"] = response.question
    elif response.status == "out_of_scope":
        update["out_of_scope_message"] = response.message
    return update


def revise_plan_node(state: dict) -> dict:
    """Structural redesign — re-run extraction with revision instruction.

    Clears downstream locked artifacts (ASL / spec) but preserves plan_history.
    """
    from orchestrator.api.repositories import session_repository as session_store
    from orchestrator.api.services.orchestration import plan_history
    from orchestrator.api.services.pipeline_analysis import session_service as legacy
    from orchestrator.api.services.pipeline_analysis.session_service import (
        NoApprovablePlanError,
    )

    session_id = state.get("session_id") or ""
    feedback = (state.get("user_message") or "").strip()
    session = session_store.get_session(session_id)
    if session is None:
        return {
            "extraction_status": "error",
            "error": f"Unknown session_id: {session_id}",
            "last_stage": "revise_plan",
        }

    if session.latest_plan is None and state.get("action") == "reject":
        raise NoApprovablePlanError(
            "No plan to reject -- conversation still needs clarification."
        )

    # Preserve history, invalidate downstream (ASL / enrichment / spec).
    locked = session.locked_spec if isinstance(session.locked_spec, dict) else {}
    preserved = plan_history.preserve_history_on_clear(locked)
    if session.latest_plan is not None:
        preserved = plan_history.push_plan(preserved, session.latest_plan)
    # ASL-first: keep the current ASL so the LLM revises it minimally.
    from orchestrator.api.services.asl_first import service as _asl_first

    preserved.update(_asl_first.carry_over(locked))
    session_store.set_locked_spec(session_id, preserved)

    if state.get("action") == "reject":
        revision_message = (
            "REJECTED — revise the previous plan. Do not return the same "
            "plan, steps, or structure. Apply this feedback:\n\n"
            f"{feedback}"
        )
    else:
        revision_message = (
            "REVISE the previous plan. Do not return the same plan unchanged. "
            "Apply this structural feedback:\n\n"
            f"{feedback}"
        )

    # Refresh session after locked_spec clear
    session = session_store.get_session(session_id) or session
    response = legacy.run_extraction_turn(session_id, session, revision_message)
    update: dict[str, Any] = {
        "extraction_status": response.status,
        "last_stage": "revise_plan",
        "message_response": response.model_dump(),
    }
    if response.status == "ready" and response.plan is not None:
        update["plan"] = response.plan.model_dump()
        update["summary_markdown"] = response.summary_markdown
        update["diagram_dot"] = response.diagram_dot
        update["placeholders"] = response.placeholders
        update["analysis"] = response.pipeline_analysis
        try:
            locked2 = session_store.get_session(session_id)
            locked_spec2 = (locked2.locked_spec if locked2 else None) or {}
            updated = plan_history.push_plan(locked_spec2, response.plan)
            session_store.set_locked_spec(session_id, updated)
        except Exception:  # noqa: BLE001
            pass
    elif response.status == "needs_clarification":
        update["clarification_message"] = response.question
    elif response.status == "out_of_scope":
        update["out_of_scope_message"] = response.message
    return update


def patch_params_node(state: dict) -> dict:
    """Value-only parameter patch — one small LLM + deterministic apply + mechanical ASL.

    If the patch extractor decides the change is structural, fall back to revise_plan.
    """
    from orchestrator.api.repositories import session_repository as session_store
    from orchestrator.api.schemas.common import Plan
    from orchestrator.api.services.asl_service import (
        build_asl_from_spec,
        describe_placeholders,
        find_asl_placeholders,
    )
    from orchestrator.api.services.orchestration.patch import (
        apply_overrides_to_plan_and_artifacts,
        collect_known_tokens,
        resolve_overrides,
    )
    from orchestrator.api.services.plan.render import (
        find_placeholders,
        render_plan_diagram,
        render_plan_summary,
    )

    session_id = state.get("session_id") or ""
    message = state.get("user_message") or ""
    session = session_store.get_session(session_id)
    if session is None:
        return {
            "extraction_status": "error",
            "error": f"Unknown session_id: {session_id}",
            "last_stage": "patch_params",
        }

    # ASL-first sessions: no workflow_spec to patch -- patch the ASL itself;
    # anything structural falls through to the revise path.
    if (session.locked_spec or {}).get("asl_first"):
        from orchestrator.api.services.asl_first import service as _asl_first

        patched = _asl_first.patch_values_from_message(state)
        return patched if patched is not None else revise_plan_node(state)

    plan = session.latest_plan
    locked = session.locked_spec if isinstance(session.locked_spec, dict) else {}
    workflow_spec = locked.get("workflow_spec") if isinstance(locked.get("workflow_spec"), dict) else None
    asl = locked.get("asl") if isinstance(locked.get("asl"), dict) else None

    known = collect_known_tokens(plan=plan, workflow_spec=workflow_spec, asl=asl)
    plan_summary = state.get("plan_summary")
    is_value_only, overrides, rationale = resolve_overrides(
        message,
        known_tokens=known,
        workflow_spec=workflow_spec,
        plan_summary=plan_summary,
    )

    if not is_value_only or not overrides:
        # Fall back to structural revise
        return revise_plan_node(state)

    new_plan, new_spec, remaining = apply_overrides_to_plan_and_artifacts(
        plan=plan,
        workflow_spec=workflow_spec,
        overrides=overrides,
    )

    # Persist plan
    if new_plan is not None:
        session_store.set_latest_plan(session_id, new_plan)
        session_store.append_message(session_id, "user", message)

    # Update locked_spec: plan overrides on resolved_plan + spec + mechanical ASL
    new_locked = dict(locked)
    if new_plan is not None:
        new_locked["resolved_plan"] = new_plan.model_dump()
        try:
            from orchestrator.api.services.orchestration import plan_history

            new_locked = plan_history.push_plan(new_locked, new_plan)
        except Exception:  # noqa: BLE001
            pass

    asl_out = None
    asl_graph_dot = None
    asl_validations = None
    asl_source = None
    asl_error = None
    spec_placeholders: list[str] = []

    if isinstance(new_spec, dict):
        new_locked["workflow_spec"] = new_spec
        spec_placeholders = find_asl_placeholders(new_spec)
        new_locked["spec_placeholders"] = spec_placeholders
        if not spec_placeholders:
            try:
                result = build_asl_from_spec(new_spec)
                asl_out = result.asl
                asl_graph_dot = result.graph_dot
                asl_validations = (
                    result.validations.to_dict()
                    if hasattr(result.validations, "to_dict")
                    else result.validations
                )
                asl_source = result.source
                asl_error = None
            except Exception as exc:  # noqa: BLE001
                asl_error = str(exc)
                asl_out = None
                asl_graph_dot = None
                asl_validations = None
                asl_source = None
        else:
            asl_out = None
            asl_graph_dot = None
            asl_validations = None
            asl_source = None
        new_locked["asl"] = asl_out
        new_locked["asl_graph_dot"] = asl_graph_dot
        new_locked["asl_validations"] = asl_validations
        new_locked["asl_source"] = asl_source
        new_locked["asl_error"] = asl_error
        new_locked["placeholder_details"] = (
            describe_placeholders(
                new_spec,
                spec_placeholders,
                plan=new_plan.model_dump() if new_plan is not None else None,
            )
            if spec_placeholders
            else {}
        )
        # Clear enrichment question if we successfully patched
        new_locked.pop("enrichment_question", None)
        new_locked.pop("enrichment_pending", None)

    session_store.set_locked_spec(session_id, new_locked)

    summary = render_plan_summary(new_plan) if new_plan is not None else None
    diagram = render_plan_diagram(new_plan) if new_plan is not None else None
    placeholders = find_placeholders(new_plan) if new_plan is not None else []

    assistant_payload = {
        "status": "ready",
        "plan": new_plan.model_dump() if new_plan is not None else None,
        "summary_markdown": summary,
        "diagram_dot": diagram,
        "placeholders": placeholders,
        "patch_overrides": overrides,
        "patch_rationale": rationale,
    }
    session_store.append_message(
        session_id, "assistant", __import__("json").dumps(assistant_payload)
    )

    msg = (
        "Updated values without redesigning the workflow. ASL refreshed."
        if asl_out
        else (
            "Updated values without redesigning the workflow."
            + (
                " Fill remaining placeholders to regenerate ASL."
                if spec_placeholders
                else ""
            )
        )
    )
    return {
        "extraction_status": "ready",
        "plan": new_plan.model_dump() if new_plan is not None else None,
        "summary_markdown": summary,
        "diagram_dot": diagram,
        "placeholders": placeholders,
        "patch_overrides": overrides,
        "patch_rationale": rationale,
        "is_value_only": True,
        "workflow_spec": new_spec,
        "asl": asl_out,
        "asl_graph_dot": asl_graph_dot,
        "asl_validations": asl_validations,
        "asl_source": asl_source,
        "asl_error": asl_error,
        "spec_placeholders": spec_placeholders,
        "last_stage": "patch_params",
        "route": "patch_params",
        "message_response": {
            "status": "ready",
            "plan": new_plan.model_dump() if new_plan is not None else None,
            "summary_markdown": summary,
            "diagram_dot": diagram,
            "placeholders": placeholders,
            "pipeline_analysis": None,
            "message": msg,
            # Drive UI ASL surface (not plan review)
            "intent": "patch_params",
            "phase": "asl_review" if asl_out else "spec_placeholders",
            "workflow_spec": new_spec,
            "remaining_placeholders": remaining if isinstance(remaining, list) else placeholders,
            "spec_placeholders": spec_placeholders,
            "placeholder_details": (
                describe_placeholders(
                    new_spec,
                    spec_placeholders,
                    plan=new_plan.model_dump() if new_plan is not None else None,
                )
                if spec_placeholders
                else {}
            ),
            "asl": asl_out,
            "asl_graph_dot": asl_graph_dot,
            "asl_validations": asl_validations,
            "asl_source": asl_source,
            "asl_error": asl_error,
        },
    }


def restore_previous_node(state: dict) -> dict:
    """Restore the previous plan snapshot from plan_history (no re-extraction)."""
    from orchestrator.api.repositories import session_repository as session_store
    from orchestrator.api.services.orchestration import plan_history
    from orchestrator.api.services.plan.render import (
        find_placeholders,
        render_plan_diagram,
        render_plan_summary,
    )

    session_id = state.get("session_id") or ""
    message = state.get("user_message") or ""
    session = session_store.get_session(session_id)
    if session is None:
        return {
            "extraction_status": "error",
            "error": f"Unknown session_id: {session_id}",
            "last_stage": "restore_previous",
        }

    locked = session.locked_spec if isinstance(session.locked_spec, dict) else {}
    prev = plan_history.previous_plan(locked, current=session.latest_plan)
    if prev is None:
        # Nothing to restore — fall back to revise so the user gets a response
        return revise_plan_node(state)

    session_store.append_message(session_id, "user", message)
    session_store.set_latest_plan(session_id, prev)

    # Invalidate downstream; keep history
    preserved = plan_history.preserve_history_on_clear(locked)
    session_store.set_locked_spec(session_id, preserved)

    summary = render_plan_summary(prev)
    diagram = render_plan_diagram(prev)
    placeholders = find_placeholders(prev)
    payload = {
        "status": "ready",
        "plan": prev.model_dump(),
        "summary_markdown": summary,
        "diagram_dot": diagram,
        "placeholders": placeholders,
        "restored_previous": True,
    }
    session_store.append_message(
        session_id, "assistant", __import__("json").dumps(payload)
    )

    return {
        "extraction_status": "ready",
        "plan": prev.model_dump(),
        "summary_markdown": summary,
        "diagram_dot": diagram,
        "placeholders": placeholders,
        "last_stage": "restore_previous",
        "message_response": {
            "status": "ready",
            "plan": prev.model_dump(),
            "summary_markdown": summary,
            "diagram_dot": diagram,
            "placeholders": placeholders,
            "pipeline_analysis": None,
        },
    }


def answer_enrichment_node(state: dict) -> dict:
    """User is answering a pending enrichment question."""
    # Domain path — avoid circular import with orchestration.service
    from orchestrator.api.services.pipeline_analysis.session_service import (
        answer_enrichment_question as _answer_enrichment,
    )

    session_id = state.get("session_id") or ""
    answer = state.get("user_message") or ""
    try:
        response = _answer_enrichment(session_id, answer)
        return {
            "last_stage": "answer_enrichment",
            "approve_response": response.model_dump(),
            "enrichment_question": response.enrichment_question,
            "workflow_spec": response.workflow_spec,
            "asl": response.asl,
            "asl_graph_dot": response.asl_graph_dot,
            "asl_validations": response.asl_validations,
            "asl_source": response.asl_source,
            "asl_error": response.asl_error,
            "spec_placeholders": response.spec_placeholders,
            # Surface as ready plan if we still have one
            "extraction_status": "ready",
            "message_response": {
                "status": "ready",
                "plan": (
                    response.resolved_plan.model_dump()
                    if response.resolved_plan is not None
                    else None
                ),
                "summary_markdown": None,
                "diagram_dot": None,
                "placeholders": response.remaining_placeholders or [],
                "pipeline_analysis": None,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "extraction_status": "error",
            "error": str(exc),
            "last_stage": "answer_enrichment",
        }


def approve_plan_node(state: dict) -> dict:
    """User clearly approved the current plan via chat."""
    # Domain path — avoid circular import with orchestration.service
    from orchestrator.api.services.pipeline_analysis.session_service import (
        approve_plan as _approve_plan,
    )

    session_id = state.get("session_id") or ""
    try:
        response = _approve_plan(session_id, {})
        return {
            "last_stage": "approve_plan",
            "approve_response": response.model_dump(),
            "resolved_plan": (
                response.resolved_plan.model_dump()
                if response.resolved_plan is not None
                else None
            ),
            "workflow_spec": response.workflow_spec,
            "asl": response.asl,
            "asl_graph_dot": response.asl_graph_dot,
            "asl_validations": response.asl_validations,
            "asl_source": response.asl_source,
            "asl_error": response.asl_error,
            "spec_placeholders": response.spec_placeholders,
            "extraction_status": "ready",
            "message_response": {
                "status": "ready",
                "plan": (
                    response.resolved_plan.model_dump()
                    if response.resolved_plan is not None
                    else None
                ),
                "summary_markdown": None,
                "diagram_dot": None,
                "placeholders": response.remaining_placeholders or [],
                "pipeline_analysis": None,
            },
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "extraction_status": "error",
            "error": str(exc),
            "last_stage": "approve_plan",
        }