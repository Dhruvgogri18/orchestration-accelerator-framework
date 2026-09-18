"""LLM-based turn router — understands arbitrary user language.

Replaces brittle keyword/regex intent classification. One structured
call decides the next graph edge given phase + short artifact context.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from orchestrator.indexing.gateway_client import chat_completion

RouteName = Literal[
    "extract",
    "revise_plan",
    "patch_params",
    "restore_previous",
    "answer_enrichment",
    "approve",
    "out_of_scope",
]


class RouteDecision(BaseModel):
    route: RouteName = Field(
        description=(
            "extract = new or continuing workflow design from NL; "
            "patch_params = only resource names / TODO values / ARNs change "
            "(no control-flow change); "
            "restore_previous = undo last revision / go back to previous plan; "
            "revise_plan = structural design change to the current plan; "
            "answer_enrichment = user is answering an enrichment question; "
            "approve = user clearly approves the current plan; "
            "out_of_scope = not about AWS Step Functions workflows."
        )
    )
    rationale: str = Field(description="One short sentence.")


ROUTER_SYSTEM = """\
You route turns for an AWS Step Functions workflow assistant.
Pick exactly one route. Do not design the workflow yourself.

Routes:
- extract: brand-new workflow while no plan exists yet.
- restore_previous: user wants the *previous* plan back — any phrasing
  (undo, go back, revert, "the one before", "map one again", "not this
  version", etc.). System reloads a saved snapshot; do NOT use revise_plan.
- patch_params: COSMETIC / VALUE-ONLY edits. Control flow stays identical.
  Examples (any wording): rename a bucket/table/topic/queue/lambda/job,
  change an ARN, fill or replace a TODO_* value, change a timeout number
  or retry count as a parameter, swap a resource name. If in doubt whether
  something is only a value vs structure, prefer patch_params when no steps
  are added/removed and Map/Parallel/Choice shape is unchanged.
- revise_plan: STRUCTURAL redesign only — add/remove/reorder steps, switch
  Map vs Parallel, new branches, different stages, new services in the flow.
- answer_enrichment: answering a pending enrichment question.
- approve: clear approval of the current plan.
- out_of_scope: not about Step Functions workflows.

Rules:
- restore_previous over everything when they ask to undo/go back.
- patch_params over revise_plan for names, ARNs, TODO values, numbers.
- revise_plan ONLY for real structural change.
- enriching phase → answer_enrichment unless they clearly change topic.
- no plan yet → extract.
- Respond with ONLY JSON:
{"route":"<one of the routes>","rationale":"..."}
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def route_turn(
    *,
    user_message: str,
    phase: str,
    has_plan: bool,
    has_asl: bool,
    enrichment_question: str | None,
    plan_summary: str | None = None,
    session_id: str | None = None,
) -> RouteDecision:
    """LLM route decision. Falls back to safe defaults if the call fails."""
    # Bind session for llm_trace so router calls land in the session log
    # (not logs/sessions/_unassigned.log).
    if session_id:
        try:
            from orchestrator.api.services.llm_trace import start_request

            start_request(session_id=session_id, user_message=user_message)
        except Exception:  # noqa: BLE001
            pass

    context = {
        "phase": phase,
        "has_plan": has_plan,
        "has_asl": has_asl,
        "enrichment_question": enrichment_question,
        "plan_summary_excerpt": (plan_summary or "")[:1200],
        "user_message": user_message,
    }
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM},
        {
            "role": "user",
            "content": (
                "Context (JSON):\n"
                + json.dumps(context, ensure_ascii=False)
                + "\n\nChoose the route for this user_message."
            ),
        },
    ]

    try:
        raw = chat_completion(messages, temperature=0)
        cleaned = _strip_fences(raw)
        data = json.loads(cleaned)
        decision = RouteDecision.model_validate(data)
        return decision
    except Exception:  # noqa: BLE001 — never block the turn on router failure
        return _fallback_route(
            user_message=user_message,
            phase=phase,
            has_plan=has_plan,
            enrichment_question=enrichment_question,
        )


def _fallback_route(
    *,
    user_message: str,
    phase: str,
    has_plan: bool,
    enrichment_question: str | None,
) -> RouteDecision:
    text = (user_message or "").strip().lower()
    if phase == "enriching" or enrichment_question:
        return RouteDecision(
            route="answer_enrichment",
            rationale="Fallback: enrichment pending.",
        )
    if re.match(r"^(approve|looks good|lgtm)\b", text):
        return RouteDecision(route="approve", rationale="Fallback: approve phrase.")
    # Router LLM failed — conservative fallbacks only (no phrase lists).
    # Prefer revise over wrong restore; prefer patch only for explicit TODOs.
    from orchestrator.api.services.orchestration.patch import (
        extract_explicit_todo_overrides,
    )
    if has_plan and extract_explicit_todo_overrides(user_message, known_tokens=[]):
        return RouteDecision(
            route="patch_params",
            rationale="Fallback: explicit TODO assignment.",
        )
    if has_plan and phase in (
        "plan_review",
        "asl_review",
        "deployed",
        "spec_placeholders",
        "failed",
    ):
        return RouteDecision(
            route="revise_plan",
            rationale="Fallback: plan/ASL exists — treat as revision.",
        )
    return RouteDecision(route="extract", rationale="Fallback: extract.")
