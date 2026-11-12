"""Strategy A — dense (semantic) retrieval over pgvector."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from orchestrator.retrieval.models import ScoredHit


def _distance_to_score(distance: float | None) -> float:
    """pgvector cosine distance (`<=>`) is in ~[0, 2]; map to similarity [0, 1]."""
    if distance is None:
        return 0.0
    # cosine distance 0 => identical; 1 => orthogonal; 2 => opposite
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve_dense(
    query: str,
    *,
    collection: Any,
    n_results: int = 10,
    filters: dict[str, Any] | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[ScoredHit]:
    """Embed ``query`` and return the top-N cosine hits as ScoredHits."""
    if not (query or "").strip():
        return []
    from orchestrator.indexing.vectorstore import query as vector_query

    hits = vector_query(
        query.strip(),
        collection=collection,
        n_results=max(1, n_results),
        filters=filters,
        embed=embed,
    )
    out: list[ScoredHit] = []
    for h in hits:
        out.append(
            ScoredHit(
                id=h.id,
                text=h.text or "",
                metadata=dict(h.metadata or {}),
                dense_score=_distance_to_score(h.distance),
                sources=["dense"],
            )
        )
    return out
