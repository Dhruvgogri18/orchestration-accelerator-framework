"""ASL-first pipeline service (PIPELINE_MODE=asl_first). See __init__.py.

Session storage (all inside the existing ``locked_spec`` JSON blob, so no
DB migration):

  asl_first                 True -- marks the session as ASL-first
  asl_draft                 the working ASL (may contain TODO_ tokens);
                            NEVER shown to the user
  asl_name / asl_summary    LLM-provided workflow name / summary
  asl_assumptions           decisions the LLM defaulted (shown with the plan)
  asl_questions             clarifying questions to ask after plan approval
  enrichment_question /     the question currently being asked -- these are
  enrichment_pending        the SAME keys the staged flow uses, so
                            derive_phase() -> ENRICHING, the router's
                            answer_enrichment route and the UI keep working
  spec_placeholders /       TODO_ tokens still to be filled (-> phase
  placeholder_details       SPEC_PLACEHOLDERS)
  asl, asl_graph_dot, ...   set ONLY once everything is resolved
                            (-> phase ASL_REVIEW)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from orchestrator.api.config.settings import get_api_settings
from orchestrator.api.repositories import session_repository as session_store
from orchestrator.api.schemas.common import ExtractionResult, Plan
from orchestrator.api.schemas.session import (
    ApproveResponse,
    AslValidateResponse,
    MessageResponse,
)
from orchestrator.api.services.asl_first.plan_view import asl_to_plan, humanize_tokens
from orchestrator.api.services.asl_first.prompts import (
    ANSWERS_INSTRUCTION,
    ASL_FIRST_SYSTEM_PROMPT,
    REALIGN_SYSTEM_PROMPT,
)
from orchestrator.api.services.asl_service import (
    _find_key_for_token,
    _humanize_todo_token,
    _kind_from_field_key,
    _scrub_states_intrinsics_in_tree,
    apply_placeholder_overrides,
    classify_placeholder_kind,
    describe_placeholders,
    find_asl_placeholders,
    render_asl_graph,
    sanitize_asl_literals,
    validate_asl,
    validate_placeholder_value,
)
from orchestrator.api.services.llm_trace import end_request, llm_stage, log_event, start_request
from orchestrator.api.services.plan.matching import match as match_plan
from orchestrator.api.services.plan.render import (
    apply_overrides,
    render_plan_diagram,
    render_plan_summary,
)
from orchestrator.indexing.gateway_client import chat_completion

ASL_SOURCE = "asl_first"
MAX_QUESTIONS = 4

# validate_asl() fails ANY document that still contains TODO_ tokens
# (extras/unresolved_placeholders). In this pipeline tokens are expected in
# the draft, so those issues must not drive the repair loop -- otherwise the
# model would be pushed to invent real values to make the validator happy.
_IGNORED_DRAFT_ISSUES = {
    ("extras", "unresolved_placeholders"),
    ("extras", "aws_validate_unavailable"),
}
_MAX_ISSUES_IN_FEEDBACK = 8


class AslFirstError(Exception):
    """The LLM could not produce a usable draft within the attempt budget."""


@dataclass
class DraftResult:
    status: str  # ready | needs_clarification | out_of_scope
    asl: dict | None = None
    name: str | None = None
    summary: str | None = None
    assumptions: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    clarifying_question: str | None = None
    message: str | None = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def is_enabled() -> bool:
    return get_api_settings().pipeline_mode == "asl_first"


def is_asl_first_session(locked: dict | None) -> bool:
    return isinstance(locked, dict) and bool(locked.get("asl_first"))


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _preserved_history(locked: dict | None) -> dict:
    hist = locked.get("plan_history") if isinstance(locked, dict) else None
    return {"plan_history": list(hist)} if isinstance(hist, list) and hist else {}


def carry_over(locked: dict | None) -> dict:
    """What revise_plan_node must keep when it clears locked_spec for a
    structural revise: the current ASL, so the LLM revises it minimally
    instead of redesigning from scratch (and values the user already gave
    are not asked for again)."""
    if not is_asl_first_session(locked):
        return {}
    current = locked.get("asl") or locked.get("asl_draft")
    if isinstance(current, dict) and current:
        return {"asl_first": True, "asl_draft": current}
    return {}


def _assistant_summary(content: str) -> str:
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return (content or "")[:500]
    if not isinstance(data, dict):
        return (content or "")[:500]
    status = data.get("status")
    if status == "needs_clarification":
        return f"(asked) {data.get('clarifying_question')}"
    if status == "out_of_scope":
        return str(data.get("message") or "")
    plan = data.get("plan")
    if isinstance(plan, dict):
        return f"(proposed workflow '{plan.get('name')}': {plan.get('summary')})"
    return ""


def _transcript(history: list[dict]) -> str:
    lines: list[str] = []
    for m in history:
        role, content = m.get("role"), m.get("content") or ""
        if role == "user":
            lines.append(f"USER: {content}")
        elif role == "assistant":
            text = _assistant_summary(content)
            if text:
                lines.append(f"ASSISTANT: {text}")
    return "\n\n".join(lines)


def _blocking_issues(report) -> list:
    return [
        i for i in report.issues if (i.group, i.code) not in _IGNORED_DRAFT_ISSUES
    ]


def _draft_ok(report) -> bool:
    return (
        report.json_ok
        and report.structure_ok
        and report.reachability_ok
        and report.aws_constraints_ok
        and not _blocking_issues(report)
    )


def _feedback(report) -> str:
    issues = _blocking_issues(report)[:_MAX_ISSUES_IN_FEEDBACK]
    if not issues:
        return "the document was rejected but no specific issue was reported"
    return "; ".join(f"[{i.group}/{i.code}] {i.message}" for i in issues)


def _clean_list(value: Any, cap: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    out = [str(x).strip() for x in value if str(x).strip()]
    return out[:cap] if cap else out



def _latest_user_text(history: list[dict]) -> str:
    """Most recent user utterance — used as the retrieval query."""
    for m in reversed(history or []):
        if m.get("role") == "user" and (m.get("content") or "").strip():
            return str(m["content"]).strip()
    return ""


def _fetch_repo_context(query: str) -> tuple[str, dict]:
    """Retrieve repo context for the LLM. Never raises; empty on failure/disabled.

    Returns (context_markdown, stats_dict).
    """
    stats: dict = {"enabled": False, "hits": 0, "error": None}
    q = (query or "").strip()
    if not q:
        return "", stats
    try:
        from orchestrator.config import get_settings
        from orchestrator.indexing.pipeline import default_repo_id
        from orchestrator.retrieval import retrieve

        settings = get_settings()
        # Accept either orchestrator.config or API settings flag (same env var).
        api_on = False
        try:
            from orchestrator.api.config.settings import get_api_settings

            api_on = bool(get_api_settings().retrieval_enabled)
        except Exception:  # noqa: BLE001
            api_on = False
        if not (settings.retrieval_enabled or api_on):
            return "", stats
        repo_id = default_repo_id(settings.repo_path)
        result = retrieve(
            q,
            repo_id=repo_id,
            enabled=True,
        )
        stats.update(
            enabled=True,
            hits=len(result.hits),
            dense_count=result.dense_count,
            structured_count=result.structured_count,
            truncated=result.truncated,
            repo_id=repo_id,
        )
        if result.context:
            try:
                log_event(
                    "asl_first_retrieval",
                    query=q[:200],
                    hits=len(result.hits),
                    dense_count=result.dense_count,
                    structured_count=result.structured_count,
                    truncated=result.truncated,
                    repo_id=repo_id,
                )
            except Exception:  # noqa: BLE001
                pass
        return result.context or "", stats
    except Exception as exc:  # noqa: BLE001 - retrieval must never block drafting
        stats["error"] = f"{type(exc).__name__}: {exc}"
        try:
            log_event("asl_first_retrieval_error", error=stats["error"], query=q[:200])
        except Exception:  # noqa: BLE001
            pass
        return "", stats


# ---------------------------------------------------------------------------
# LLM: draft / revise
# ---------------------------------------------------------------------------


def generate_draft(
    history: list[dict],
    *,
    current_asl: dict | None = None,
    extra: str | None = None,
) -> DraftResult:
    """One LLM turn: conversation (+ optional current ASL) -> DraftResult.

    A candidate ASL is mechanically validated (everything except the
    TODO_-placeholder gate and the live AWS call) and, if it fails, fed
    back its concrete errors -- same repair-loop shape as
    asl_generation/llm_service.py.
    """
    max_attempts = max(1, get_api_settings().asl_llm_max_attempts)

    user = "CONVERSATION:\n" + _transcript(history)
    # Dual retrieval (dense + structured) → re-rank → inject as REPOSITORY CONTEXT.
    # Gated by RETRIEVAL_ENABLED; failures are swallowed so drafting still works.
    repo_ctx, _retrieval_stats = _fetch_repo_context(_latest_user_text(history))
    if repo_ctx:
        user += "\n\nREPOSITORY CONTEXT:\n" + repo_ctx
    if current_asl:
        user += "\n\nCURRENT ASL:\n" + json.dumps(current_asl, separators=(",", ":"))
    if extra:
        user += "\n\n" + extra
    messages = [
        {"role": "system", "content": ASL_FIRST_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]

    last_error = "unknown error"

    def retry(raw: str, error: str) -> None:
        nonlocal last_error
        last_error = error
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Error: {error}. Return the complete corrected JSON "
                    "object only (same shape). TODO_ placeholder tokens are "
                    "expected and must stay."
                ),
            }
        )

    for _ in range(max_attempts):
        raw = chat_completion(messages, temperature=0)
        try:
            data = json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError as exc:
            retry(raw, f"Response was not valid JSON: {exc}")
            continue
        if not isinstance(data, dict):
            retry(raw, "Response must be a single JSON object")
            continue

        status = data.get("status")
        if status == "out_of_scope":
            return DraftResult(status=status, message=data.get("message"))
        if status == "needs_clarification":
            q = str(data.get("clarifying_question") or "").strip()
            if not q:
                retry(raw, "clarifying_question is required when needs_clarification")
                continue
            return DraftResult(status=status, clarifying_question=q)
        if status != "ready":
            retry(raw, "status must be ready, needs_clarification or out_of_scope")
            continue

        asl = data.get("asl")
        if isinstance(asl, str):
            try:
                asl = json.loads(asl)
            except json.JSONDecodeError:
                asl = None
        if not isinstance(asl, dict) or "States" not in asl or "StartAt" not in asl:
            retry(raw, "'asl' must be an object with top-level StartAt and States")
            continue

        candidate = sanitize_asl_literals(asl)
        report = validate_asl(
            candidate,
            spec=None,
            graph_dot=render_asl_graph(candidate),
            include_aws_validate=False,
        )
        if not _draft_ok(report):
            retry(raw, "the ASL failed validation: " + _feedback(report))
            continue

        return DraftResult(
            status="ready",
            asl=candidate,
            name=str(data.get("name") or "").strip() or None,
            summary=str(data.get("summary") or "").strip() or None,
            # Fidelity-first: never surface invented behaviour as assumptions.
            assumptions=[],
            questions=_clean_list(data.get("questions"), MAX_QUESTIONS),
        )

    raise AslFirstError(
        f"ASL-first generation failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# placeholders / finalize
# ---------------------------------------------------------------------------


def _enclosing_state(asl: dict, token: str) -> tuple[str, str] | None:
    pat = re.compile(r"\b" + re.escape(token) + r"\b")

    def walk(states: dict) -> tuple[str, str] | None:
        for name, st in (states or {}).items():
            if not isinstance(st, dict):
                continue
            nested: list[dict] = []
            if st.get("Type") == "Parallel":
                nested = [
                    b.get("States") or {}
                    for b in st.get("Branches") or []
                    if isinstance(b, dict)
                ]
            elif st.get("Type") == "Map":
                proc = st.get("ItemProcessor") or st.get("Iterator") or {}
                nested = [proc.get("States") or {}]
            for n in nested:
                found = walk(n)
                if found:
                    return found
            own = {
                k: v
                for k, v in st.items()
                if k not in ("Branches", "ItemProcessor", "Iterator")
            }
            if pat.search(json.dumps(own)):
                return name, str(st.get("Comment") or "")
        return None

    return walk(asl.get("States") or {})


def _placeholder_details(asl: dict, tokens: list[str], plan: Plan | None) -> dict[str, str]:
    try:
        base = describe_placeholders(
            None, tokens, plan=plan.model_dump() if plan else None
        )
    except Exception:  # noqa: BLE001 -- the fill form must never be bare
        base = {}
    out: dict[str, str] = {}
    for t in tokens:
        text = base.get(t) or _humanize_todo_token(t)
        ctx = _enclosing_state(asl, t)
        if ctx and ctx[0] not in text:
            text += f" -- used in step '{ctx[0]}'"
            if ctx[1].strip():
                text += f" ({humanize_tokens(ctx[1]).rstrip('. ')})"
        out[t] = text
    return out


def _classify(asl: dict, token: str):
    """Value shape expected for `token`: hard token suffix first, then the
    ASL field it sits in (TopicArn -> arn), then the soft-suffix fallback."""
    if not token.upper().endswith(("_ARN", "_URL", "_URI", "_ENDPOINT")):
        key = _find_key_for_token(asl, token)
        kind = _kind_from_field_key(key) if key else None
        if kind:
            return kind
    return classify_placeholder_kind(None, token)


def _finalize(draft: dict) -> tuple[dict, str, dict]:
    """Same final normalization + full validation (incl. the live AWS
    ValidateStateMachineDefinition call) build_asl_from_spec applies."""
    asl = sanitize_asl_literals(draft)
    asl = _scrub_states_intrinsics_in_tree(asl, path_root="$states.result")
    graph_dot = render_asl_graph(asl)
    report = validate_asl(asl, spec=None, graph_dot=graph_dot)
    return asl, graph_dot, report.to_dict()


def _topology(asl: Any) -> Any:
    """Control-flow skeleton only -- used to tell a value-only edit from a
    structural one."""
    if isinstance(asl, dict) and "States" in asl:
        return {"start": asl.get("StartAt"), "states": _topology(asl["States"])}
    if not isinstance(asl, dict):
        return None
    out: dict[str, Any] = {}
    for name, st in asl.items():
        if not isinstance(st, dict):
            continue
        node: dict[str, Any] = {
            "t": st.get("Type"),
            "n": st.get("Next"),
            "e": st.get("End"),
            "d": st.get("Default"),
            "c": [c.get("Next") for c in st.get("Choices") or [] if isinstance(c, dict)],
            "x": [c.get("Next") for c in st.get("Catch") or [] if isinstance(c, dict)],
        }
        if st.get("Type") == "Parallel":
            node["b"] = [_topology(b) for b in st.get("Branches") or []]
        if st.get("Type") == "Map":
            node["b"] = [_topology(st.get("ItemProcessor") or st.get("Iterator") or {})]
        out[name] = node
    return out


# ---------------------------------------------------------------------------
# Structure locking (determinism without the full staged pipeline)
# ---------------------------------------------------------------------------


def _plan_topology_sig(plan: Plan) -> str:
    """Compact structural signature of a Plan: step ids + wiring only.

    Ignores description/service wording so paraphrase-stable role graphs
    compare equal even when Comments differ slightly.
    """
    steps = sorted(plan.steps, key=lambda s: s.id or "")
    parts: list[str] = []
    for s in steps:
        branches = sorted(
            (f"{(b.condition or '').strip().lower()}->{b.next}" for b in (s.branches or [])),
        )
        fails = sorted(s.on_failure or [])
        timeouts = sorted(s.on_timeout or [])
        parts.append(
            "|".join(
                [
                    s.id or "",
                    s.next or "",
                    ",".join(branches),
                    ",".join(fails),
                    ",".join(timeouts),
                    "1" if s.branches else "0",
                ]
            )
        )
    return "\n".join(parts)


def _locked_plan_from(plan: Plan) -> tuple[Plan, str, float]:
    """Optionally snap *plan* for stability without inventing new steps.

    ASL-first is fidelity-first: the user prompt is the sole source of
    behaviour. We deliberately do NOT run canonicalize light-cleanup or
    pattern rebuild here -- those rename notify ids (e.g. SendFailureAlert
    -> AlertOperations) and inject pipeline family states, which forces a
    needless realign and drifts from the draft the model just produced.

    High-confidence template match is accepted only when it does not grow
    the step set. Otherwise the plan derived from ASL is kept as-is.

    Returns (locked_plan, tier, score). Matching uses the pure keyword
    role tagger only -- no extra LLM or embedding calls.
    """
    match_result = match_plan(plan)
    tier = match_result.tier
    score = float(match_result.score or 0.0)
    n = len(plan.steps)

    if (
        tier == "high"
        and match_result.plan is not None
        and len(match_result.plan.steps) <= n
    ):
        return match_result.plan, tier, score

    return plan, tier, score


def _realign_asl_to_plan(current_asl: dict, locked_plan: Plan) -> dict | None:
    """One constrained LLM call: rewrite ASL control-flow to match locked_plan.

    Returns a validated ASL dict, or None if realignment fails (caller keeps
    the original draft). Temperature 0; same mechanical validate gate as
    generate_draft (TODO_ tokens allowed).
    """
    plan_payload = {
        "name": locked_plan.name,
        "summary": locked_plan.summary,
        "steps": [
            {
                "id": s.id,
                "service": s.service,
                "description": s.description,
                "next": s.next,
                "on_failure": list(s.on_failure or []),
                "on_timeout": list(s.on_timeout or []),
                "branches": [
                    {"condition": b.condition, "next": b.next}
                    for b in (s.branches or [])
                ],
                "component_type": s.component_type,
            }
            for s in locked_plan.steps
        ],
    }
    user = (
        "LOCKED PLAN (authoritative control-flow):\n"
        + json.dumps(plan_payload, separators=(",", ":"))
        + "\n\nCURRENT ASL (preserve TODO_ tokens and Task bodies where possible):\n"
        + json.dumps(current_asl, separators=(",", ":"))
    )
    messages = [
        {"role": "system", "content": REALIGN_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    max_attempts = max(1, min(2, get_api_settings().asl_llm_max_attempts))
    last_error = "unknown"

    for _ in range(max_attempts):
        raw = chat_completion(messages, temperature=0)
        try:
            data = json.loads(_strip_code_fences(raw))
        except json.JSONDecodeError as exc:
            last_error = f"not JSON: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Error: {last_error}. Return only the JSON object with key \"asl\".",
                }
            )
            continue
        if not isinstance(data, dict):
            last_error = "response must be a JSON object"
            continue
        asl = data.get("asl", data)
        if isinstance(asl, str):
            try:
                asl = json.loads(asl)
            except json.JSONDecodeError:
                asl = None
        if not isinstance(asl, dict) or "States" not in asl or "StartAt" not in asl:
            last_error = "'asl' must be an object with StartAt and States"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": f"Error: {last_error}. Return the complete corrected JSON object only.",
                }
            )
            continue
        candidate = sanitize_asl_literals(asl)
        report = validate_asl(
            candidate,
            spec=None,
            graph_dot=render_asl_graph(candidate),
            include_aws_validate=False,
        )
        if not _draft_ok(report):
            last_error = "ASL validation failed: " + _feedback(report)
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Error: {last_error}. Return the complete corrected JSON "
                        "object only. TODO_ tokens must stay."
                    ),
                }
            )
            continue
        return candidate

    log_event("asl_first_realign_failed", error=last_error)
    return None


def lock_draft_structure(
    asl: dict,
    *,
    name: str | None = None,
    summary: str | None = None,
) -> tuple[dict, Plan]:
    """Derive plan, snap to stable structure, realign ASL only if topology drifted.

    Keeps ASL as source of truth: when the locked plan differs structurally
    from the free-form draft, one constrained realign call updates the ASL.
    If realign fails, the original ASL and its asl_to_plan view are kept.
    """
    plan = asl_to_plan(asl, name=name, summary=summary)
    locked_plan, tier, score = _locked_plan_from(plan)

    log_event(
        "asl_first_structure_lock",
        tier=tier,
        score=score,
        sig_before=_plan_topology_sig(plan)[:200],
        sig_after=_plan_topology_sig(locked_plan)[:200],
        changed=_plan_topology_sig(plan) != _plan_topology_sig(locked_plan),
    )

    if _plan_topology_sig(plan) == _plan_topology_sig(locked_plan):
        # Light cleanup may have renamed notify ids etc. Prefer locked labels
        # when signatures match at the wiring level after canonicalize.
        return asl, locked_plan

    with llm_stage("asl_first_realign"):
        realigned = _realign_asl_to_plan(asl, locked_plan)

    if realigned is None:
        return asl, plan

    try:
        # Source of truth stays ASL: re-derive plan from realigned document,
        # then re-apply lock so display ids stay canonical when possible.
        view = asl_to_plan(
            realigned,
            name=locked_plan.name or name,
            summary=locked_plan.summary or summary,
        )
        final_plan, _, _ = _locked_plan_from(view)
        # If realign preserved topology of locked_plan, prefer locked_plan
        # for stable user-facing labels.
        if _plan_topology_sig(view) == _plan_topology_sig(locked_plan):
            final_plan = locked_plan
        return realigned, final_plan
    except ValueError:
        return realigned, locked_plan


# ---------------------------------------------------------------------------
# session state -> responses
# ---------------------------------------------------------------------------


def _format_questions(questions: list[str]) -> str:
    body = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    return (
        "Before I finalize the workflow I need a few decisions from you:\n\n"
        f"{body}\n\n"
        "Reply in one message (you can number your answers)."
    )


def _approve_response(locked: dict, plan: Plan, **extra: Any) -> ApproveResponse:
    return ApproveResponse(
        status="approved",
        resolved_plan=plan,
        remaining_placeholders=locked.get("remaining_placeholders") or [],
        workflow_spec=None,
        enrichment_error=locked.get("enrichment_error"),
        enrichment_question=locked.get("enrichment_question"),
        spec_placeholders=locked.get("spec_placeholders") or [],
        placeholder_details=locked.get("placeholder_details") or {},
        placeholder_errors=locked.get("placeholder_errors") or {},
        asl=locked.get("asl"),
        asl_graph_dot=locked.get("asl_graph_dot"),
        asl_validations=locked.get("asl_validations"),
        asl_source=locked.get("asl_source"),
        asl_error=locked.get("asl_error"),
        **extra,
    )


def _advance(session_id: str, locked: dict, plan: Plan) -> ApproveResponse:
    """Questions are done: gate on placeholders, else finalize."""
    draft = locked["asl_draft"]
    tokens = find_asl_placeholders(draft)
    locked.pop("enrichment_question", None)
    locked.pop("enrichment_pending", None)
    locked["enrichment_error"] = None
    locked["spec_placeholders"] = tokens
    locked["placeholder_details"] = _placeholder_details(draft, tokens, plan)

    if tokens:  # hide the ASL until every value is supplied
        locked.update(asl=None, asl_graph_dot=None, asl_validations=None,
                      asl_source=None, asl_error=None)
    else:
        try:
            with llm_stage("asl_validation"):
                asl, graph_dot, validations = _finalize(draft)
            locked.update(asl=asl, asl_graph_dot=graph_dot,
                          asl_validations=validations, asl_source=ASL_SOURCE,
                          asl_error=None)
        except Exception as exc:  # noqa: BLE001 -- report, don't lose the approval
            locked.update(asl=None, asl_graph_dot=None, asl_validations=None,
                          asl_source=None,
                          asl_error=f"ASL finalization failed unexpectedly: {exc}")
    session_store.set_locked_spec(session_id, locked)
    return _approve_response(locked, plan)


def _ask(session_id: str, locked: dict, plan: Plan, questions: list[str], rounds: int) -> ApproveResponse:
    locked["asl_questions"] = questions
    locked["enrichment_question"] = _format_questions(questions)
    locked["enrichment_pending"] = {
        "mode": ASL_SOURCE,
        "questions": questions,
        "clarification_rounds": rounds,
    }
    locked["enrichment_error"] = None
    locked.update(spec_placeholders=[], placeholder_details={}, asl=None,
                  asl_graph_dot=None, asl_validations=None, asl_source=None,
                  asl_error=None)
    session_store.set_locked_spec(session_id, locked)
    return _approve_response(locked, plan)


# ---------------------------------------------------------------------------
# public entry points (hooked from session_service / orchestration.service)
# ---------------------------------------------------------------------------


def run_extraction_turn(session_id: str, session, message: str) -> MessageResponse:
    """Step 1+2: prompt -> hidden ASL draft -> plan view for the user."""
    from orchestrator.api.services.pipeline_analysis.service import ExtractionError

    session_store.append_message(session_id, "user", message)
    start_request(session_id=session_id, user_message=message)
    history = [*session.history, {"role": "user", "content": message}]
    locked_old = session.locked_spec if isinstance(session.locked_spec, dict) else {}
    current = locked_old.get("asl_draft")
    current = current if isinstance(current, dict) and current else None

    try:
        with llm_stage("pipeline_analysis"):
            draft = generate_draft(history, current_asl=current)
    except AslFirstError as exc:
        end_request(status="error", error=str(exc))
        raise ExtractionError(str(exc)) from exc
    end_request(status=draft.status)

    if draft.status == "out_of_scope":
        msg = draft.message or (
            "This assistant is only designed to help you build AWS Step "
            "Functions workflows. Please describe a workflow to orchestrate."
        )
        session_store.append_message(
            session_id, "assistant",
            ExtractionResult(status="out_of_scope", message=msg).model_dump_json(),
        )
        session_store.set_latest_plan(session_id, None)
        return MessageResponse(status="out_of_scope", message=msg)

    if draft.status == "needs_clarification":
        session_store.append_message(
            session_id, "assistant",
            ExtractionResult(
                status="needs_clarification",
                clarifying_question=draft.clarifying_question,
            ).model_dump_json(),
        )
        session_store.set_latest_plan(session_id, None)
        return MessageResponse(
            status="needs_clarification", question=draft.clarifying_question
        )

    try:
        # Snap free-form draft topology to match/canonicalize archetypes so
        # paraphrases of the same intent yield the same Mermaid plan. At most
        # one extra LLM call (realign) when structure actually drifts.
        locked_asl, plan = lock_draft_structure(
            draft.asl, name=draft.name, summary=draft.summary
        )
    except ValueError as exc:
        raise ExtractionError(str(exc)) from exc

    session_store.append_message(
        session_id, "assistant",
        ExtractionResult(status="ready", plan=plan).model_dump_json(),
    )
    session_store.set_latest_plan(session_id, plan)

    locked = _preserved_history(locked_old)
    locked.update(
        asl_first=True,
        asl_draft=locked_asl,
        asl_name=plan.name,
        asl_summary=plan.summary,
        asl_assumptions=draft.assumptions,
        asl_questions=draft.questions,
    )
    session_store.set_locked_spec(session_id, locked)

    summary_md = render_plan_summary(plan)
    if draft.assumptions:
        summary_md += "\n\n**Assumptions I made**\n" + "\n".join(
            f"- {a}" for a in draft.assumptions
        )
    n = len(draft.questions)
    summary_md += (
        "\n\n_After you approve this plan I'll "
        + (f"ask {n} short question{'s' if n != 1 else ''}, then " if n else "")
        + "collect the remaining details (names, ARNs, ...) before showing the final ASL._"
    )
    return MessageResponse(
        status="ready",
        plan=plan,
        summary_markdown=summary_md,
        diagram_dot=render_plan_diagram(plan),
        # Placeholders are asked for AFTER approval, not at plan review.
        placeholders=[],
        pipeline_analysis=None,
    )


def approve(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    """Step 3: plan approved -> ask the LLM's questions (or go straight on)."""
    from orchestrator.api.services.pipeline_analysis import session_service as ss

    session = ss.get_session_or_raise(session_id)
    if session.latest_plan is None:
        raise ss.NoApprovablePlanError(
            "No approvable plan yet -- conversation still needs clarification."
        )
    locked = dict(session.locked_spec or {})
    draft = locked.get("asl_draft")
    if not isinstance(draft, dict) or not draft:
        raise ss.NoApprovablePlanError("No ASL draft exists for this plan yet.")

    resolved_plan, remaining = apply_overrides(session.latest_plan, overrides or {})
    if overrides:
        known = set(find_asl_placeholders(draft))
        valid = {
            t: v for t, v in overrides.items()
            if t in known and not validate_placeholder_value(_classify(draft, t), v)
        }
        if valid:
            draft = apply_placeholder_overrides(draft, valid)
            locked["asl_draft"] = draft

    locked["resolved_plan"] = resolved_plan.model_dump()
    locked["remaining_placeholders"] = remaining

    start_request(session_id=session_id)
    try:
        questions = _clean_list(locked.get("asl_questions"), MAX_QUESTIONS)
        if questions:
            return _ask(session_id, locked, resolved_plan, questions, rounds=0)
        return _advance(session_id, locked, resolved_plan)
    finally:
        end_request(status="approved")


def answer_question(session_id: str, answer: str) -> ApproveResponse:
    """Step 3 (cont.): user answered -> revise the draft; ask more or go on."""
    from orchestrator.api.services.pipeline_analysis import session_service as ss

    session = ss.get_session_or_raise(session_id)
    locked = dict(session.locked_spec or {})
    pending = locked.get("enrichment_pending")
    if not pending or pending.get("mode") != ASL_SOURCE:
        raise ss.NoEnrichmentQuestionError(
            "No enrichment clarification pending for this session."
        )
    plan = Plan.model_validate(locked["resolved_plan"])
    draft = locked["asl_draft"]
    asked = pending.get("questions") or []
    rounds = int(pending.get("clarification_rounds") or 0) + 1

    extra = (
        ANSWERS_INSTRUCTION
        + "\n\nQUESTIONS ASKED:\n"
        + "\n".join(f"{i}. {q}" for i, q in enumerate(asked, start=1))
        + f"\n\nUSER ANSWERS:\n{answer}"
    )
    start_request(session_id=session_id, user_message=answer)
    try:
        with llm_stage("enrichment"):
            result = generate_draft(session.history, current_asl=draft, extra=extra)
        if result.status != "ready" or not result.asl:
            raise AslFirstError("The model did not return an updated workflow.")
    except AslFirstError as exc:
        # Keep the pending question so the user can simply answer again.
        locked["enrichment_error"] = str(exc)
        session_store.set_locked_spec(session_id, locked)
        end_request(status="error", error=str(exc))
        return _approve_response(locked, plan)

    locked["asl_assumptions"] = result.assumptions or locked.get("asl_assumptions", [])
    try:
        locked_asl, plan = lock_draft_structure(
            result.asl,
            name=result.name or plan.name,
            summary=result.summary or plan.summary,
        )
        locked["asl_draft"] = locked_asl
        locked["resolved_plan"] = plan.model_dump()
        session_store.set_latest_plan(session_id, plan)
    except ValueError:
        locked["asl_draft"] = result.asl
        # keep the previous plan view

    try:
        if result.questions and rounds < get_api_settings().asl_first_max_clarification_rounds:
            return _ask(session_id, locked, plan, result.questions, rounds=rounds)
        locked["asl_questions"] = []
        return _advance(session_id, locked, plan)
    finally:
        end_request(status="approved")


def fill_placeholders(session_id: str, overrides: dict[str, str]) -> ApproveResponse:
    """Step 4+5: patch user values into the ASL in code, then finalize.
    No LLM call."""
    from orchestrator.api.services.pipeline_analysis import session_service as ss

    session = ss.get_session_or_raise(session_id)
    locked = dict(session.locked_spec or {})
    draft = locked.get("asl_draft")
    if not isinstance(draft, dict) or not locked.get("spec_placeholders"):
        raise ss.NoSpecPlaceholdersError(
            "No spec-level placeholders are pending for this session."
        )
    plan = Plan.model_validate(locked["resolved_plan"])

    errors: dict[str, str] = {}
    valid: dict[str, str] = {}
    for token, value in (overrides or {}).items():
        err = validate_placeholder_value(_classify(draft, token), value)
        if err:
            errors[token] = err
        elif (value or "").strip():
            valid[token] = value.strip()

    start_request(session_id=session_id)
    try:
        if valid:
            draft = apply_placeholder_overrides(draft, valid)
            locked["asl_draft"] = draft
        locked["placeholder_errors"] = errors
        return _advance(session_id, locked, plan)
    finally:
        end_request(status="approved")


def regenerate(session_id: str) -> AslValidateResponse:
    """POST /asl/generate for an ASL-first session: re-run finalization on
    the current document (no LLM)."""
    from orchestrator.api.services.pipeline_analysis import session_service as ss

    session = ss.get_session_or_raise(session_id)
    locked = dict(session.locked_spec or {})
    doc = locked.get("asl") or locked.get("asl_draft")
    if not isinstance(doc, dict) or not doc:
        raise ss.NoWorkflowSpecError("No ASL exists for this session yet.")
    tokens = find_asl_placeholders(doc)
    if tokens:
        locked["spec_placeholders"] = tokens
        session_store.set_locked_spec(session_id, locked)
        raise ss.UnresolvedSpecPlaceholdersError(
            "ASL still has unresolved TODO_ placeholders: "
            f"{', '.join(tokens)}. Resolve them via "
            "POST /session/{id}/spec/placeholders first."
        )
    asl, graph_dot, validations = _finalize(doc)
    locked.update(asl=asl, asl_graph_dot=graph_dot, asl_validations=validations,
                  asl_source=ASL_SOURCE, asl_error=None)
    session_store.set_locked_spec(session_id, locked)
    return AslValidateResponse(
        asl=asl, validations=validations, graph_dot=graph_dot, asl_source=ASL_SOURCE
    )


def patch_values_from_message(state: dict) -> dict | None:
    """Chat-typed value edits ("the bucket is acme-uploads") without
    restarting the flow.

    One LLM revise of the current ASL; if the control-flow topology is
    unchanged it is treated as a value-only edit and applied in place
    (phase preserved). Returns None when it is NOT value-only (or anything
    goes wrong) so the caller falls back to the structural revise path.
    """
    session_id = state.get("session_id") or ""
    message = (state.get("user_message") or "").strip()
    session = session_store.get_session(session_id)
    if session is None or not message:
        return None
    locked = dict(session.locked_spec or {})
    if not is_asl_first_session(locked):
        return None
    current = locked.get("asl") or locked.get("asl_draft")
    if not isinstance(current, dict) or not current:
        return None

    extra = (
        "VALUE-ONLY EDIT: the user's latest message may supply or change "
        "values (names, ARNs, URLs, numbers). Apply it WITHOUT changing the "
        "workflow structure; do not add, remove or rewire states."
    )
    history = [*session.history, {"role": "user", "content": message}]
    try:
        with llm_stage("pipeline_analysis"):
            result = generate_draft(history, current_asl=current, extra=extra)
    except AslFirstError:
        return None
    if result.status != "ready" or not result.asl:
        return None
    if _topology(result.asl) != _topology(current):
        return None  # structural -> caller runs the full revise path

    phase0 = state.get("phase")
    plan = session.latest_plan or Plan.model_validate(locked["resolved_plan"])
    session_store.append_message(session_id, "user", message)
    locked["asl_draft"] = result.asl
    update_plan = asl_to_plan(result.asl, name=plan.name, summary=plan.summary)
    session_store.set_latest_plan(session_id, update_plan)
    if isinstance(locked.get("resolved_plan"), dict):
        locked["resolved_plan"] = update_plan.model_dump()

    resp: dict[str, Any] = {
        "status": "ready",
        "plan": update_plan.model_dump(),
        "summary_markdown": render_plan_summary(update_plan),
        "diagram_dot": render_plan_diagram(update_plan),
        "placeholders": [],
        "pipeline_analysis": None,
        "intent": "patch_params",
        "message": "Updated values without redesigning the workflow.",
    }
    if phase0 in ("spec_placeholders", "asl_review", "failed"):
        advanced = _advance(session_id, locked, update_plan)
        resp.update(
            phase="asl_review" if advanced.asl else "spec_placeholders",
            spec_placeholders=advanced.spec_placeholders,
            placeholder_details=advanced.placeholder_details,
            asl=advanced.asl,
            asl_graph_dot=advanced.asl_graph_dot,
            asl_validations=advanced.asl_validations,
            asl_source=advanced.asl_source,
            asl_error=advanced.asl_error,
        )
    else:  # plan_review: just refresh the draft + plan
        session_store.set_locked_spec(session_id, locked)
        resp["phase"] = phase0 or "plan_review"

    return {
        "extraction_status": "ready",
        "plan": update_plan.model_dump(),
        "summary_markdown": resp["summary_markdown"],
        "diagram_dot": resp["diagram_dot"],
        "placeholders": [],
        "is_value_only": True,
        "asl": resp.get("asl"),
        "asl_graph_dot": resp.get("asl_graph_dot"),
        "asl_validations": resp.get("asl_validations"),
        "asl_source": resp.get("asl_source"),
        "asl_error": resp.get("asl_error"),
        "spec_placeholders": resp.get("spec_placeholders") or [],
        "last_stage": "patch_params",
        "route": "patch_params",
        "message_response": resp,
    }
