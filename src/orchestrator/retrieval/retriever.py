"""Retrieval layer used by workflow synthesis.

Dual strategy (dense semantic + structured metadata), merge by chunk id,
rule-based re-rank, then format under a token budget.

Public entry points:
  retrieve(...)         -> RetrievalResult (hits + formatted context)
  retrieve_context(...) -> str context block only (empty when disabled / no hits)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from orchestrator.retrieval.dense import retrieve_dense
from orchestrator.retrieval.format import format_context
from orchestrator.retrieval.merge import merge_hits
from orchestrator.retrieval.models import RetrievalResult, ScoredHit
from orchestrator.retrieval.rerank import rerank
from orchestrator.retrieval.structured import retrieve_structured

# Defaults (overridable via kwargs; settings used when callers omit them)
DEFAULT_DENSE_TOP_K = 10
DEFAULT_STRUCTURED_MAX = 20
DEFAULT_RERANK_TOP_N = 12
DEFAULT_MIN_SCORE = 0.15
DEFAULT_TOKEN_BUDGET = 3000
DEFAULT_ALPHA = 0.55  # dense weight
DEFAULT_BETA = 0.35  # exact weight


def _settings_or_none():
    try:
        from orchestrator.config import get_settings

        return get_settings()
    except Exception:  # noqa: BLE001
        return None


def _retrieval_enabled() -> bool:
    s = _settings_or_none()
    return bool(s.retrieval_enabled) if s is not None else False


def _defaults_from_settings() -> dict[str, Any]:
    s = _settings_or_none()
    if s is None:
        return {}
    return {
        "dense_top_k": s.retrieval_dense_top_k,
        "structured_max": s.retrieval_structured_max,
        "rerank_top_n": s.retrieval_rerank_top_n,
        "min_score": s.retrieval_min_score,
        "token_budget": s.retrieval_token_budget,
        "alpha": s.retrieval_alpha,
        "beta": s.retrieval_beta,
    }


def _open_collection(
    *,
    collection: Any | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
):
    if collection is not None:
        return collection
    from orchestrator.indexing import vectorstore

    client = vectorstore.get_client(persist_dir) if persist_dir else vectorstore.get_client()
    # create=False: retrieval must not invent an empty collection mid-request
    return vectorstore.get_collection(
        client=client,
        name=collection_name,
        create=False,
    )


def retrieve(
    query: str,
    *,
    repo_id: str | None = None,
    filters: dict[str, Any] | None = None,
    token_budget: int | None = None,
    dense_top_k: int | None = None,
    structured_max: int | None = None,
    rerank_top_n: int | None = None,
    min_score: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    collection: Any | None = None,
    persist_dir: str | None = None,
    collection_name: str | None = None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
    enabled: bool | None = None,
) -> RetrievalResult:
    """Run dense + structured retrieval, merge, re-rank, and format context.

    When retrieval is disabled (RETRIEVAL_ENABLED=false and ``enabled`` is not
    forced True), returns an empty result with ``disabled=True`` and no
    embedding / Chroma calls.
    """
    query = (query or "").strip()
    is_enabled = _retrieval_enabled() if enabled is None else bool(enabled)
    if not is_enabled:
        return RetrievalResult(query=query, hits=[], context="", disabled=True)
    if not query:
        return RetrievalResult(query=query, hits=[], context="")

    cfg = _defaults_from_settings()
    dense_top_k = dense_top_k if dense_top_k is not None else cfg.get("dense_top_k", DEFAULT_DENSE_TOP_K)
    structured_max = (
        structured_max if structured_max is not None else cfg.get("structured_max", DEFAULT_STRUCTURED_MAX)
    )
    rerank_top_n = (
        rerank_top_n if rerank_top_n is not None else cfg.get("rerank_top_n", DEFAULT_RERANK_TOP_N)
    )
    min_score = min_score if min_score is not None else cfg.get("min_score", DEFAULT_MIN_SCORE)
    token_budget = (
        token_budget if token_budget is not None else cfg.get("token_budget", DEFAULT_TOKEN_BUDGET)
    )
    alpha = alpha if alpha is not None else cfg.get("alpha", DEFAULT_ALPHA)
    beta = beta if beta is not None else cfg.get("beta", DEFAULT_BETA)

    coll = _open_collection(
        collection=collection,
        persist_dir=persist_dir,
        collection_name=collection_name,
    )
    if coll is None:
        return RetrievalResult(query=query, hits=[], context="", disabled=False)

    # Scope dense filters to repo when requested
    dense_filters = dict(filters or {})
    if repo_id and "repo_id" not in dense_filters:
        dense_filters["repo_id"] = repo_id

    dense_hits = retrieve_dense(
        query,
        collection=coll,
        n_results=dense_top_k,
        filters=dense_filters or None,
        embed=embed,
    )
    structured_hits = retrieve_structured(
        query,
        collection=coll,
        repo_id=repo_id,
        max_hits=structured_max,
    )

    merged = merge_hits(dense_hits, structured_hits)
    ranked = rerank(
        merged,
        alpha=alpha,
        beta=beta,
        min_score=min_score,
        top_n=rerank_top_n,
    )
    context, included, truncated = format_context(
        ranked,
        token_budget=token_budget,
    )
    return RetrievalResult(
        query=query,
        hits=included,
        context=context,
        dense_count=len(dense_hits),
        structured_count=len(structured_hits),
        truncated=truncated,
        disabled=False,
    )


def retrieve_context(
    query: str,
    *,
    repo_id: str | None = None,
    filters: dict[str, Any] | None = None,
    token_budget: int | None = None,
    **kwargs: Any,
) -> str:
    """Convenience wrapper: return only the formatted context string."""
    result = retrieve(
        query,
        repo_id=repo_id,
        filters=filters,
        token_budget=token_budget,
        **kwargs,
    )
    return result.context


def hits_as_dicts(hits: list[ScoredHit]) -> list[dict[str, Any]]:
    """Serialize hits for logging / API responses."""
    return [
        {
            "id": h.id,
            "path": h.path,
            "resource_name": h.resource_name,
            "chunk_kind": h.chunk_kind,
            "status": h.status,
            "dense_score": round(h.dense_score, 4),
            "exact_score": round(h.exact_score, 4),
            "final_score": round(h.final_score, 4),
            "sources": list(h.sources),
        }
        for h in hits
    ]
