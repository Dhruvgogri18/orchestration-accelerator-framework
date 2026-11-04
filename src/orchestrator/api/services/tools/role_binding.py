"""Structure-first resource binding.

Flow:
  1. LLM designs control flow + TODO_ role tokens (no full-account scan).
  2. For each TODO_ we infer service + role keywords.
  3. Tools list only those services and rank candidates.
  4. UI always gets ranked candidates to pick from; high-confidence
     unique matches are marked matched=True for prefill.

General rules (not prompt-specific):
  - Always surface every listed candidate of the relevant service as an
    alternative (capped), even when name similarity is weak — the user
    still needs a dropdown when the account only has a few resources.
  - If a service has exactly one resource, treat it as a strong default
    suggestion for every TODO of that service (user can still change it).
  - matched=True only when ranking is confident enough; alternatives
    always carry top-level name + arn for the UI.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .auto_resolve import (
    _hint_from_token,
    _value_from_resource,
    infer_service_from_token,
)
from .registry import execute_tool

logger = logging.getLogger(__name__)

_STOP = {
    "TODO", "ARN", "NAME", "THE", "A", "AN", "AND", "OR", "FOR", "TO", "OF",
    "LAMBDA", "FUNCTION", "TOPIC", "QUEUE", "BUCKET", "TABLE", "SNS", "SQS",
    "S3", "DYNAMODB", "STATE", "MACHINE", "JOB", "ROLE", "USED", "STEP",
    "IN", "REQUEST",  # very common filler in business prompts — still allow
}


def _keywords(*texts: str) -> set[str]:
    out: set[str] = set()
    for text in texts:
        if not text:
            continue
        for part in re.split(r"[^a-zA-Z0-9]+", text):
            p = part.strip().lower()
            if len(p) < 3:
                continue
            if p.upper() in _STOP and p not in ("leave", "approval", "reject"):
                # keep domain words even if short stop-ish
                if p.upper() in {"REQUEST"}:
                    continue
                continue
            out.add(p)
    return out


def _score_candidate(
    keywords: set[str],
    resource: dict[str, Any],
    detail_summary: dict[str, Any] | None = None,
    *,
    hint: str = "",
) -> float:
    """Score 0..1 how well a resource matches role keywords / hint."""
    name = (resource.get("name") or resource.get("id") or "").lower()
    compact_n = re.sub(r"[^a-z0-9]", "", name)
    blob = name
    if detail_summary:
        blob += " " + " ".join(str(v) for v in detail_summary.values() if v)
        tags = detail_summary.get("tags") or {}
        if isinstance(tags, dict):
            blob += " " + " ".join(f"{k} {v}" for k, v in tags.items())
    blob_l = blob.lower()
    compact_blob = re.sub(r"[^a-z0-9]", "", blob_l)

    score = 0.0
    if keywords:
        hits = 0
        for kw in keywords:
            if not kw:
                continue
            if kw in blob_l or kw in compact_n or kw in compact_blob:
                hits += 1
            elif len(kw) >= 4 and (kw in compact_n or compact_n in kw):
                hits += 0.7
        score = hits / max(len(keywords), 1)
        if hits >= max(2, len(keywords) - 1):
            score = min(1.0, score + 0.12)

    # Compact hint vs name (e.g. recordapproval vs approvalrecord)
    compact_h = re.sub(r"[^a-z0-9]", "", (hint or "").lower())
    if compact_h and compact_n:
        if compact_h == compact_n:
            score = max(score, 0.95)
        elif compact_h in compact_n or compact_n in compact_h:
            score = max(score, 0.85)
        else:
            # token-sort similarity: shared character coverage
            shared = sum(1 for c in set(compact_h) if c in compact_n)
            if shared:
                score = max(score, 0.35 * shared / max(len(set(compact_h)), 1))
            # ordered subsequence of keyword chunks
            for kw in sorted(keywords, key=len, reverse=True):
                ck = re.sub(r"[^a-z0-9]", "", kw)
                if len(ck) >= 4 and ck in compact_n:
                    score = max(score, 0.7)

    return round(min(1.0, float(score)), 4)


def _lambda_signals(identifier: str, region: str | None, session_id: str | None) -> dict[str, Any]:
    result = execute_tool(
        "get_resource",
        {
            "service": "lambda",
            "identifier": identifier,
            "region": region,
            "include": ["config", "tags"],
        },
        session_id=session_id,
    )
    if not (result.data or {}).get("found"):
        return {}
    res = (result.data or {}).get("resource") or {}
    cfg = res.get("config") or {}
    summary = res.get("summary") or {}
    env = {}
    if isinstance(cfg.get("Environment"), dict):
        env = (cfg.get("Environment") or {}).get("Variables") or {}
    return {
        "description": cfg.get("Description") or "",
        "handler": cfg.get("Handler") or summary.get("handler") or "",
        "runtime": cfg.get("Runtime") or summary.get("runtime") or "",
        "env_keys": sorted(env.keys())[:20] if isinstance(env, dict) else [],
        "tags": res.get("tags") or {},
    }


def _alt(score: float, resource: dict[str, Any], detail: dict | None = None) -> dict[str, Any]:
    """Stable alternative shape for API/UI (top-level name + arn)."""
    return {
        "confidence": score,
        "name": resource.get("name") or resource.get("id"),
        "arn": resource.get("arn"),
        "id": resource.get("id"),
        "service": resource.get("service"),
        "signals": detail or None,
    }


def bind_placeholders_to_account(
    tokens: list[str],
    *,
    step_context: dict[str, str] | None = None,
    region: str | None = None,
    session_id: str | None = None,
    max_list: int = 30,
    enrich_lambda_config: bool = True,
) -> dict[str, Any]:
    step_context = step_context or {}
    suggestions: dict[str, Any] = {}
    auto_applicable: dict[str, str] = {}
    needs_user: list[str] = []
    tool_calls = 0

    by_svc: dict[str, list[str]] = {}
    for token in tokens or []:
        if not token or not str(token).startswith("TODO_"):
            continue
        svc = infer_service_from_token(token) or "lambda"
        by_svc.setdefault(svc, []).append(token)

    listed: dict[str, list[dict]] = {}
    for svc in by_svc:
        result = execute_tool(
            "list_resources",
            {"service": svc, "region": region, "max_results": max_list, "filters": {}},
            session_id=session_id,
        )
        tool_calls += 1
        listed[svc] = list((result.data or {}).get("resources") or [])

    for svc, svc_tokens in by_svc.items():
        resources = listed.get(svc) or []
        detail_cache: dict[str, dict] = {}
        if svc == "lambda" and enrich_lambda_config and resources:
            for r in resources[:12]:
                rid = r.get("id") or r.get("name")
                if not rid:
                    continue
                detail_cache[str(rid)] = _lambda_signals(str(rid), region, session_id)
                tool_calls += 1

        sole = len(resources) == 1

        for token in svc_tokens:
            hint = _hint_from_token(token)
            ctx = step_context.get(token) or ""
            kws = _keywords(hint, ctx, token)

            scored: list[tuple[float, dict, dict]] = []
            for r in resources:
                rid = str(r.get("id") or r.get("name") or "")
                detail = detail_cache.get(rid) or {}
                s = _score_candidate(kws, r, detail, hint=hint)
                # Always keep candidates when the account set is small so the
                # UI can still offer a dropdown (general policy, not prompt-specific).
                if s < 0.15 and len(resources) > 8:
                    continue
                # Floor so every listed resource appears when few exist
                if len(resources) <= 8:
                    s = max(s, 0.2)
                if sole:
                    s = max(s, 0.8)
                scored.append((s, r, detail))

            scored.sort(key=lambda x: (-x[0], x[1].get("name") or ""))

            entry: dict[str, Any] = {
                "value": None,
                "confidence": 0.0,
                "matched": False,
                "ambiguous": False,
                "service": svc,
                "resource": None,
                "alternatives": [],
                "reason": None,
                "hint": hint,
                "role_keywords": sorted(kws),
            }

            if not scored:
                entry["reason"] = "no_candidates_in_account"
                needs_user.append(token)
                suggestions[token] = entry
                continue

            best_s, best_r, best_d = scored[0]
            second_s = scored[1][0] if len(scored) > 1 else 0.0

            # Full ranked list for the picker (best first)
            entry["alternatives"] = [
                _alt(s, r, d) for s, r, d in scored[:10]
            ]

            ambiguous = (
                len(scored) > 1
                and best_s < 0.95
                and (best_s - second_s) < 0.1
                and second_s >= 0.45
            )

            # Auto-prefill rules (general):
            #  - unique high score
            #  - OR sole resource of that service in account
            can_match = (not ambiguous and best_s >= 0.75) or (sole and best_s >= 0.75)

            if ambiguous:
                entry["ambiguous"] = True
                entry["confidence"] = best_s
                entry["reason"] = "ambiguous_role_match"
                needs_user.append(token)
            elif can_match:
                value = _value_from_resource(best_r, token)
                entry["matched"] = True
                entry["confidence"] = best_s
                entry["resource"] = best_r
                entry["value"] = value
                entry["reason"] = "sole_resource" if sole else "role_bind"
                if value:
                    auto_applicable[token] = value
                else:
                    needs_user.append(token)
            else:
                entry["confidence"] = best_s
                entry["reason"] = "low_confidence_role_match"
                needs_user.append(token)

            suggestions[token] = entry

    return {
        "suggestions": suggestions,
        "auto_applicable": auto_applicable,
        "needs_user": needs_user,
        "meta": {
            "mode": "structure_first_role_binding",
            "services_queried": sorted(by_svc.keys()),
            "tool_calls": tool_calls,
            "candidates": {k: len(v) for k, v in listed.items()},
        },
    }
