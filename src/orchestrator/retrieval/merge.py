"""Merge dense + structured hit lists by chunk id."""

from __future__ import annotations

from orchestrator.retrieval.models import ScoredHit


def merge_hits(
    dense: list[ScoredHit],
    structured: list[ScoredHit],
) -> list[ScoredHit]:
    """Union by id; keep both score channels and combine source labels."""
    by_id: dict[str, ScoredHit] = {}

    for hit in dense:
        by_id[hit.id] = ScoredHit(
            id=hit.id,
            text=hit.text,
            metadata=dict(hit.metadata),
            dense_score=hit.dense_score,
            exact_score=hit.exact_score,
            sources=list(hit.sources),
        )

    for hit in structured:
        existing = by_id.get(hit.id)
        if existing is None:
            by_id[hit.id] = ScoredHit(
                id=hit.id,
                text=hit.text,
                metadata=dict(hit.metadata),
                dense_score=hit.dense_score,
                exact_score=hit.exact_score,
                sources=list(hit.sources),
            )
            continue
        existing.exact_score = max(existing.exact_score, hit.exact_score)
        for src in hit.sources:
            if src not in existing.sources:
                existing.sources.append(src)
        # Prefer longer text if one side returned empty (should not happen)
        if len(hit.text) > len(existing.text):
            existing.text = hit.text
            existing.metadata = dict(hit.metadata)

    return list(by_id.values())
