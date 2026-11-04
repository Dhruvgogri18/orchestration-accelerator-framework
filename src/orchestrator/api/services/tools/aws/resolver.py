"""Resolve human / partial identifiers to canonical resources.

Deterministic ranking (exact > prefix > contains > fuzzy).  Strict
ambiguity policy: if the top two scores are close, we refuse to auto-pick
and return alternatives instead — callers must not embed a low-confidence
or ambiguous match into a plan/ASL.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any

from ..schemas import ResourceSummary
from .allowlist import canonical_service
from .client import get_default_region
from .resource_lister import list_resources

# Confidence thresholds (product policy)
CONFIDENT_MATCH = 0.75          # auto-pick allowed at or above this
MIN_CANDIDATE_SCORE = 0.40      # below this, candidate is dropped
AMBIGUITY_GAP = 0.08            # if best - second < gap, treat as ambiguous


def _score(hint: str, candidate: ResourceSummary, context: str | None = None) -> float:
    """Deterministic relevance score in [0, 1]."""
    h = hint.lower().strip()
    name = (candidate.name or candidate.id or "").lower()
    arn = (candidate.arn or "").lower()

    if not h:
        return 0.0

    if name == h or (candidate.id or "").lower() == h:
        return 1.0
    if arn and (arn == h or arn.endswith(":" + h) or arn.endswith("/" + h)):
        return 1.0

    score = 0.0

    if name.startswith(h) or h.startswith(name):
        score = max(score, 0.88)

    if h in name or name in h:
        score = max(score, 0.72)

    h_tokens = set(re.split(r"[^a-z0-9]+", h)) - {""}
    n_tokens = set(re.split(r"[^a-z0-9]+", name)) - {""}
    if h_tokens and n_tokens:
        overlap = len(h_tokens & n_tokens) / max(len(h_tokens), len(n_tokens))
        score = max(score, 0.45 + 0.4 * overlap)

    # Compact form: "review leave request" vs "reviewleaverequest"
    compact_h = re.sub(r"[^a-z0-9]", "", h)
    compact_n = re.sub(r"[^a-z0-9]", "", name)
    if compact_h and compact_n:
        if compact_h == compact_n:
            score = max(score, 0.95)
        elif compact_h in compact_n or compact_n in compact_h:
            score = max(score, 0.82)
        else:
            score = max(score, SequenceMatcher(None, compact_h, compact_n).ratio() * 0.75)

    ratio = SequenceMatcher(None, h, name).ratio()
    score = max(score, ratio * 0.68)

    if context:
        ctx_tokens = set(re.split(r"[^a-z0-9]+", context.lower())) - {""}
        if ctx_tokens & n_tokens:
            score = min(1.0, score + 0.06)

    if candidate.arn and score >= 0.5:
        score = min(1.0, score + 0.02)

    return round(score, 4)


def resolve_identifier(
    service: str,
    hint: str,
    *,
    region: str | None = None,
    context: str | None = None,
    min_score: float = MIN_CANDIDATE_SCORE,
    max_alternatives: int = 5,
) -> dict[str, Any]:
    """Resolve a hint to the best matching resource under strict policy.

    Return shape:
      {
        "matched": bool,          # True only when confident AND unambiguous
        "ambiguous": bool,
        "confidence": float,
        "resource": dict | null,  # set only when matched
        "alternatives": [...],
        "hint": str,
        "service": str,
        "reason": str | null,
        "candidates_examined": int,
      }
    """
    svc = canonical_service(service)
    region = region or get_default_region()
    hint = (hint or "").strip()

    empty = {
        "matched": False,
        "ambiguous": False,
        "confidence": 0.0,
        "resource": None,
        "alternatives": [],
        "hint": hint,
        "service": svc,
        "candidates_examined": 0,
    }

    if not hint:
        return {**empty, "reason": "empty_hint"}

    # Prefer exact name filter first (cheap path)
    candidates: list[ResourceSummary] = []
    try:
        exact = list_resources(
            svc, region=region, filters={"name": hint}, max_results=10
        )
        candidates.extend(exact)
    except Exception:
        pass

    token = re.split(r"[^a-zA-Z0-9]+", hint)[0]
    filters: dict[str, Any] = {}
    if token and len(token) >= 2:
        filters["name_contains"] = token

    try:
        broad = list_resources(svc, region=region, filters=filters, max_results=50)
        seen = {c.id for c in candidates}
        for c in broad:
            if c.id not in seen:
                candidates.append(c)
    except Exception:
        pass

    if len(candidates) < 3:
        try:
            all_listed = list_resources(svc, region=region, max_results=50)
            seen = {c.id for c in candidates}
            for c in all_listed:
                if c.id not in seen:
                    candidates.append(c)
        except Exception:
            pass

    scored: list[tuple[float, ResourceSummary]] = []
    for c in candidates:
        s = _score(hint, c, context)
        if s >= min_score:
            scored.append((s, c))

    scored.sort(key=lambda x: (-x[0], x[1].name or x[1].id))

    if not scored:
        return {
            **empty,
            "reason": "no_candidates_above_threshold",
            "candidates_examined": len(candidates),
        }

    best_score, best = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    alternatives = [
        {"confidence": s, "resource": r.model_dump()}
        for s, r in scored[1 : max_alternatives + 1]
    ]

    # Ambiguity: top two are close and both reasonably high
    ambiguous = (
        len(scored) > 1
        and best_score < 1.0
        and (best_score - second_score) < AMBIGUITY_GAP
        and second_score >= MIN_CANDIDATE_SCORE
    )

    if ambiguous:
        return {
            "matched": False,
            "ambiguous": True,
            "confidence": best_score,
            "resource": None,  # refuse to auto-pick
            "alternatives": [
                {"confidence": best_score, "resource": best.model_dump()},
                *alternatives,
            ],
            "hint": hint,
            "service": svc,
            "reason": "ambiguous_matches",
            "candidates_examined": len(candidates),
        }

    if best_score < CONFIDENT_MATCH:
        return {
            "matched": False,
            "ambiguous": False,
            "confidence": best_score,
            "resource": None,
            "alternatives": [
                {"confidence": best_score, "resource": best.model_dump()},
                *alternatives,
            ],
            "hint": hint,
            "service": svc,
            "reason": "low_confidence",
            "candidates_examined": len(candidates),
        }

    return {
        "matched": True,
        "ambiguous": False,
        "confidence": best_score,
        "resource": best.model_dump(),
        "alternatives": alternatives,
        "hint": hint,
        "service": svc,
        "reason": None,
        "candidates_examined": len(candidates),
    }
