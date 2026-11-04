"""Budgeted tool-calling for accurate, low-latency resource resolution.

Design goals
------------
* **Accurate** — prefer deterministic ``resolve_identifier`` (strict
  confidence + ambiguity rules) over free-form model invention.
* **Low tokens** — tiny prompts; at most one short LLM hop, and only when
  deterministic resolution is ambiguous or service is unknown.
* **Fast** — hard caps: max 2 rounds, max 4 tool calls, early exit on
  success; parallel-friendly sequential policy that stops when done.

There is no long ReAct loop and no result caching (live AWS only).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .auto_resolve import (
    infer_service_from_token,
    suggest_overrides_for_tokens,
)
from .registry import execute_tool, list_tool_schemas

logger = logging.getLogger(__name__)

# Hard budgets (product policy)
MAX_ROUNDS = 2
MAX_TOOL_CALLS = 4
MAX_LLM_TOKENS_HINT = 400  # soft; we keep prompts tiny


def tool_schemas_for_llm() -> list[dict[str, Any]]:
    """OpenAI-compatible tool definitions (subset useful for resolution)."""
    allow = {
        "list_resources",
        "resolve_identifier",
        "search_resources",
        "get_resource",
        "validate_resource_reference",
    }
    return [s for s in list_tool_schemas() if s.get("function", {}).get("name") in allow]


def resolve_placeholders_interactive(
    tokens: list[str],
    *,
    session_id: str | None = None,
    region: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Primary entry — deterministic suggestions (fast path)."""
    return suggest_overrides_for_tokens(
        tokens,
        region=region,
        session_id=session_id,
        context=context,
    )


def run_budgeted_resolution(
    tokens: list[str],
    *,
    session_id: str | None = None,
    region: str | None = None,
    context: str | None = None,
    use_llm_fallback: bool = True,
) -> dict[str, Any]:
    """Resolve TODO_ tokens with a budgeted multi-step strategy.

    Round 1 (always): deterministic ``suggest_overrides_for_tokens``.
    Round 2 (optional, only for unresolved tokens): one constrained LLM
    tool-calling hop *or* a second deterministic search pass with broader
    hints. Never exceeds MAX_TOOL_CALLS.

    Returns the same shape as ``suggest_overrides_for_tokens`` plus
    ``meta`` with rounds / tool_calls / mode.
    """
    tokens = [t for t in (tokens or []) if t]
    meta: dict[str, Any] = {
        "rounds": 0,
        "tool_calls": 0,
        "mode": "deterministic",
        "llm_used": False,
    }
    if not tokens:
        return {
            "suggestions": {},
            "auto_applicable": {},
            "needs_user": [],
            "meta": meta,
        }

    # --- Round 1: deterministic ---
    result = suggest_overrides_for_tokens(
        tokens,
        region=region,
        session_id=session_id,
        context=context,
    )
    meta["rounds"] = 1
    # Each token roughly 1 resolve_identifier call
    meta["tool_calls"] = len(tokens)

    needs = list(result.get("needs_user") or [])
    if not needs or meta["tool_calls"] >= MAX_TOOL_CALLS:
        meta["done_reason"] = "all_matched" if not needs else "budget_exhausted"
        result["meta"] = meta
        return result

    if not use_llm_fallback:
        meta["done_reason"] = "llm_disabled"
        result["meta"] = meta
        return result

    # --- Round 2: refine unresolved only ---
    remaining_budget = MAX_TOOL_CALLS - meta["tool_calls"]
    if remaining_budget <= 0:
        meta["done_reason"] = "budget_exhausted"
        result["meta"] = meta
        return result

    refined = _round_two_refine(
        needs,
        suggestions=result.get("suggestions") or {},
        region=region,
        session_id=session_id,
        context=context,
        max_calls=remaining_budget,
    )
    meta["rounds"] = 2
    meta["tool_calls"] += int(refined.get("tool_calls") or 0)
    meta["llm_used"] = bool(refined.get("llm_used"))
    meta["mode"] = "deterministic+llm" if meta["llm_used"] else "deterministic+search"
    meta["done_reason"] = refined.get("done_reason") or "round2_complete"

    # Merge refined matches into result
    suggestions = dict(result.get("suggestions") or {})
    auto = dict(result.get("auto_applicable") or {})
    for token, entry in (refined.get("updates") or {}).items():
        suggestions[token] = entry
        if entry.get("matched") and entry.get("value"):
            auto[token] = entry["value"]

    still_need = [
        t
        for t in tokens
        if not (isinstance(suggestions.get(t), dict) and suggestions[t].get("matched"))
    ]

    return {
        "suggestions": suggestions,
        "auto_applicable": auto,
        "needs_user": still_need,
        "meta": meta,
    }


def _round_two_refine(
    tokens: list[str],
    *,
    suggestions: dict[str, Any],
    region: str | None,
    session_id: str | None,
    context: str | None,
    max_calls: int,
) -> dict[str, Any]:
    """Second pass: broader search, then optional single LLM tool hop."""
    updates: dict[str, Any] = {}
    calls = 0
    llm_used = False

    for token in tokens:
        if calls >= max_calls:
            break
        prev = suggestions.get(token) or {}
        # If already ambiguous with alternatives, don't burn more calls —
        # surface alternatives to the user (accurate > forced pick).
        if prev.get("ambiguous") and prev.get("alternatives"):
            continue

        svc = prev.get("service") or infer_service_from_token(token) or "lambda"
        hint = prev.get("hint") or re.sub(r"^TODO_", "", token).replace("_", " ")

        # Broader search_resources (1 call)
        search = execute_tool(
            "search_resources",
            {
                "query": hint,
                "services": [svc],
                "region": region,
                "max_results": 5,
            },
            session_id=session_id,
        )
        calls += 1
        hits = (search.data or {}).get("results") or []
        if len(hits) == 1 and float(hits[0].get("confidence") or 0) >= 0.75:
            res = hits[0].get("resource") or {}
            value = res.get("arn") if "ARN" in token.upper() else (res.get("name") or res.get("id"))
            if value:
                updates[token] = {
                    "value": value,
                    "confidence": hits[0].get("confidence"),
                    "matched": True,
                    "ambiguous": False,
                    "service": svc,
                    "resource": res,
                    "alternatives": [],
                    "reason": "search_unique_hit",
                    "hint": hint,
                }
                continue

        # If multiple hits, attach as alternatives (user picks — no LLM cost)
        if len(hits) > 1:
            alts = [
                {
                    "confidence": h.get("confidence"),
                    "resource": h.get("resource"),
                }
                for h in hits[:5]
            ]
            updates[token] = {
                "value": None,
                "confidence": hits[0].get("confidence"),
                "matched": False,
                "ambiguous": True,
                "service": svc,
                "resource": None,
                "alternatives": alts,
                "reason": "search_multiple_hits",
                "hint": hint,
            }
            continue

        # Optional single LLM tool-call hop for truly unknown service tokens
        if calls < max_calls and prev.get("reason") == "unknown_service_for_token":
            llm_result = _llm_pick_service_and_resolve(
                token, hint, region=region, session_id=session_id
            )
            if llm_result:
                calls += int(llm_result.get("tool_calls") or 1)
                llm_used = True
                if llm_result.get("entry"):
                    updates[token] = llm_result["entry"]

    return {
        "updates": updates,
        "tool_calls": calls,
        "llm_used": llm_used,
        "done_reason": "round2_complete",
    }


def _llm_pick_service_and_resolve(
    token: str,
    hint: str,
    *,
    region: str | None,
    session_id: str | None,
) -> dict[str, Any] | None:
    """One constrained LLM call: pick service id only, then deterministic resolve.

    Avoids multi-tool agent chatter. Prompt is ~100 tokens.
    """
    try:
        from orchestrator.indexing.gateway_client import chat_completion
    except Exception:
        return None

    services = [
        "lambda",
        "s3",
        "dynamodb",
        "sqs",
        "sns",
        "states",
        "glue",
        "ecs",
        "events",
        "kinesis",
        "batch",
        "sagemaker",
        "iam",
    ]
    messages = [
        {
            "role": "system",
            "content": (
                "Pick the single most likely AWS service id for this workflow "
                "placeholder. Reply with ONLY one word from the list, nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"token={token}\nhint={hint}\nservices={','.join(services)}",
        },
    ]
    try:
        raw = chat_completion(messages, temperature=0, max_tokens=16)
    except TypeError:
        try:
            raw = chat_completion(messages, temperature=0)
        except Exception:
            return None
    except Exception:
        return None

    svc = (raw or "").strip().lower().split()[0] if raw else ""
    svc = re.sub(r"[^a-z0-9-]", "", svc)
    if svc not in services:
        return None

    result = execute_tool(
        "resolve_identifier",
        {"service": svc, "hint": hint, "region": region},
        session_id=session_id,
    )
    data = result.data or {}
    entry = {
        "value": None,
        "confidence": float(data.get("confidence") or 0),
        "matched": bool(data.get("matched")),
        "ambiguous": bool(data.get("ambiguous")),
        "service": svc,
        "resource": data.get("resource"),
        "alternatives": data.get("alternatives") or [],
        "reason": data.get("reason") or "llm_service_pick",
        "hint": hint,
    }
    if entry["matched"] and entry["resource"]:
        res = entry["resource"]
        entry["value"] = (
            res.get("arn") if "ARN" in token.upper() else (res.get("name") or res.get("id"))
        )
    return {"entry": entry, "tool_calls": 1}


def run_targeted_resolution(
    hints: list[dict[str, str]],
    *,
    session_id: str | None = None,
    region: str | None = None,
    max_calls: int = MAX_TOOL_CALLS,
) -> dict[str, Any]:
    """Resolve a list of {service?, hint, token?} without an LLM."""
    calls = 0
    results: list[dict[str, Any]] = []
    for item in hints:
        if calls >= max_calls:
            break
        token = item.get("token") or ""
        hint = item.get("hint") or token
        service = item.get("service") or infer_service_from_token(token) or "lambda"
        result = execute_tool(
            "resolve_identifier",
            {"service": service, "hint": hint, "region": region},
            session_id=session_id,
        )
        calls += 1
        results.append(
            {
                "input": item,
                "status": result.status.value,
                "data": result.data,
                "telemetry_id": result.telemetry_id,
            }
        )
    return {"results": results, "tool_calls": calls}
