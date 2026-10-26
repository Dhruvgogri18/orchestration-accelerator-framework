"""Rule-based re-ranker (v1): blend dense + exact scores with status/kind priors.

No extra model call. A later v2 can swap in a cross-encoder while keeping the
same ScoredHit interface.
"""

from __future__ import annotations

from orchestrator.retrieval.models import ScoredHit

# Prefer workflow-relevant kinds slightly over generic windows / docs
_KIND_PRIOR = {
    "job_definition": 0.08,
    "lambda_handler": 0.08,
    "glue_script": 0.07,
    "pipeline": 0.06,
    "env_resources": 0.05,
    "config_entry": 0.04,
    "python_module": 0.03,
    "doc_section": 0.02,
    "config_section": 0.02,
    "text_window": 0.0,
}


def rerank(
    hits: list[ScoredHit],
    *,
    alpha: float = 0.55,
    beta: float = 0.35,
    min_score: float = 0.15,
    top_n: int = 12,
) -> list[ScoredHit]:
    """Score, filter, and truncate.

    final = alpha * dense + beta * exact + kind_prior + status_adj
    Hits present in both strategies get a small multi-source bonus.
    """
    alpha = max(0.0, alpha)
    beta = max(0.0, beta)
    scored: list[ScoredHit] = []

    for hit in hits:
        kind = hit.chunk_kind
        prior = _KIND_PRIOR.get(kind, 0.01)
        status_adj = 0.0
        status = (hit.status or "").lower()
        if status == "deprecated":
            status_adj -= 0.35  # strong demotion so active peers win
        elif status == "active":
            status_adj += 0.03
        if hit.metadata.get("mentions_deprecated"):
            status_adj -= 0.08

        multi = 0.05 if len(hit.sources) > 1 else 0.0
        final = alpha * hit.dense_score + beta * hit.exact_score + prior + status_adj + multi
        hit.final_score = max(0.0, min(1.0, final))
        scored.append(hit)

    scored.sort(key=lambda h: (h.final_score, h.dense_score, h.exact_score), reverse=True)
    kept = [h for h in scored if h.final_score >= min_score]
    return kept[: max(1, top_n)] if kept else scored[: max(1, top_n)]
