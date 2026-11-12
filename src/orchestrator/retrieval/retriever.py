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
from orchestrator.retrieval.structured import (
    extract_environment,
    extract_type_groups,
    filter_by_environment,
    hit_types,
    retrieve_structured,
)

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


def ensure_type_coverage(
    ranked: list[ScoredHit],
    pool: list[ScoredHit],
    groups: list[tuple[str, ...]],
    *,
    top_n: int,
) -> list[ScoredHit]:
    """Make sure every resource type the user named is represented in ``ranked``.

    Re-ranking is a single global top-N, so a kind of resource with one good chunk
    (the one SNS topic) can lose to ten mediocre chunks of another kind. For each
    type group in the query that has no hit in ``ranked``, bring in the best-scored
    chunk of that type from ``pool`` (all merged, scored hits), evicting the lowest
    scored hit that is not the only representative of some group. At most one chunk
    per group, so noise stays bounded.
    """
    if not groups or not ranked:
        return ranked
    kept = list(ranked)
    kept_ids = {h.id for h in kept}

    def covers(h: ScoredHit, group: tuple[str, ...]) -> bool:
        return bool(hit_types(h.metadata) & set(group))

    for group in groups:
        if any(covers(h, group) for h in kept):
            continue
        candidates = [h for h in pool if h.id not in kept_ids and covers(h, group)]
        if not candidates:
            continue
        best = max(candidates, key=lambda h: (h.final_score, h.exact_score, h.dense_score))
        if len(kept) >= top_n:
            evictable = [
                h
                for h in kept
                if not any(
                    covers(h, g) and sum(1 for k in kept if covers(k, g)) == 1 for g in groups
                )
            ]
            if not evictable:
                continue
            victim = min(evictable, key=lambda h: (h.final_score, h.dense_score, h.exact_score))
            kept.remove(victim)
            kept_ids.discard(victim.id)
        kept.append(best)
        kept_ids.add(best.id)
    kept.sort(key=lambda h: (h.final_score, h.dense_score, h.exact_score), reverse=True)
    return kept


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
    environment: str | None = None,
) -> RetrievalResult:
    """Run dense + structured retrieval, merge, re-rank, and format context.

    When retrieval is disabled (RETRIEVAL_ENABLED=false and ``enabled`` is not
    forced True), returns an empty result with ``disabled=True`` and no
    embedding / Chroma calls.

    ``environment`` scopes results to one environment (``"dev"``, ``"prod"``...).
    When omitted it is inferred from the query if exactly one environment is named.
    Chunks of other environments are dropped; environment-neutral chunks are kept.
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

    env = (environment or extract_environment(query) or "").lower() or None

    # When an environment is requested, over-fetch so the other environment's
    # near-duplicate chunks cannot use up the dense slots, then trim back.
    dense_hits = retrieve_dense(
        query,
        collection=coll,
        n_results=dense_top_k * 2 if env else dense_top_k,
        filters=dense_filters or None,
        embed=embed,
    )
    dense_hits = filter_by_environment(dense_hits, env)[:dense_top_k]
    structured_hits = retrieve_structured(
        query,
        collection=coll,
        repo_id=repo_id,
        max_hits=structured_max,
        environment=env,
    )

    merged = merge_hits(dense_hits, structured_hits)
    ranked = rerank(
        merged,
        alpha=alpha,
        beta=beta,
        min_score=min_score,
        top_n=rerank_top_n,
    )
    ranked = ensure_type_coverage(
        ranked, merged, extract_type_groups(query), top_n=rerank_top_n
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
        environment=env,
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