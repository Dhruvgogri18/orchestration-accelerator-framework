"""Strategy B — structured / exact retrieval via metadata and light keyword cues.

Extracts resource-like tokens from the query (kebab/snake names, common AWS
service words) and looks them up with vectorstore.get_by_metadata. No embedding
call; complements dense search when the user names concrete jobs, Lambdas, or
topics.
"""

from __future__ import annotations

import re
from typing import Any

from orchestrator.retrieval.models import ScoredHit

# kebab-case or snake_case identifiers long enough to be resource names
_NAME_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)\b",
    re.IGNORECASE,
)

# Common service / kind cues that map to chunk metadata resource_type values
_SERVICE_CUES: dict[str, list[str]] = {
    "lambda": ["lambda"],
    "lambdas": ["lambda"],
    "function": ["lambda"],
    "glue": ["glue", "glue_job"],
    "sns": ["sns_topic"],
    "topic": ["sns_topic"],
    "topics": ["sns_topic"],
    "sqs": ["sqs_queue"],
    "queue": ["sqs_queue"],
    "queues": ["sqs_queue"],
    "s3": ["s3_bucket"],
    "bucket": ["s3_bucket"],
    "buckets": ["s3_bucket"],
    "dynamodb": ["dynamodb_table"],
    "dynamo": ["dynamodb_table"],
    "table": ["dynamodb_table"],
    "pipeline": ["pipeline"],
    "pipelines": ["pipeline"],
    "job": ["glue", "glue_job", "lambda"],
    "jobs": ["glue", "glue_job", "lambda"],
}

# Prefer these chunk kinds when scoring exact hits
_KIND_BOOST = {
    "job_definition": 1.0,
    "lambda_handler": 0.95,
    "glue_script": 0.95,
    "pipeline": 0.9,
    "env_resources": 0.85,
    "config_entry": 0.8,
    "python_module": 0.7,
    "doc_section": 0.6,
    "config_section": 0.55,
}


def extract_candidates(query: str) -> tuple[list[str], list[str]]:
    """Return (resource_name_tokens, resource_type_hints) from free text."""
    text = query or ""
    names = sorted({m.group(1).lower() for m in _NAME_RE.finditer(text)})
    # Drop pure noise like "step-functions" service marketing terms if alone —
    # keep them; they still help filter.
    types: list[str] = []
    lower = text.lower()
    for cue, mapped in _SERVICE_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            types.extend(mapped)
    # de-dupe, preserve order
    seen: set[str] = set()
    uniq_types: list[str] = []
    for t in types:
        if t not in seen:
            seen.add(t)
            uniq_types.append(t)
    return names, uniq_types


def _score_hit(hit: Any, names: list[str], types: list[str]) -> float:
    meta = hit.metadata or {}
    score = 0.35  # base for being returned at all
    rname = str(meta.get("resource_name") or "").lower()
    rnames = [str(x).lower() for x in (meta.get("resource_names") or [])]
    rtype = str(meta.get("resource_type") or "").lower()
    rtypes = [str(x).lower() for x in (meta.get("resource_types") or [])]

    for n in names:
        if rname == n or n in rnames:
            score += 0.45
            break
        if n and (n in rname or any(n in x for x in rnames)):
            score += 0.25
            break

    if types:
        type_set = {rtype, *rtypes}
        if type_set & set(types):
            score += 0.15

    kind = str(meta.get("chunk_kind") or "")
    score += 0.1 * _KIND_BOOST.get(kind, 0.4)

    status = str(meta.get("status") or "").lower()
    if status == "deprecated":
        score -= 0.25
    elif status == "active":
        score += 0.05

    if meta.get("mentions_deprecated"):
        score -= 0.1

    return max(0.0, min(1.0, score))


def retrieve_structured(
    query: str,
    *,
    collection: Any,
    repo_id: str | None = None,
    limit_per_lookup: int = 8,
    max_hits: int = 20,
) -> list[ScoredHit]:
    """Look up chunks by extracted resource names and type hints."""
    names, types = extract_candidates(query)
    if not names and not types:
        return []

    from orchestrator.indexing.vectorstore import get_by_metadata

    found: dict[str, ScoredHit] = {}

    def _ingest(hits: list[Any]) -> None:
        for h in hits:
            s = _score_hit(h, names, types)
            existing = found.get(h.id)
            if existing is None or s > existing.exact_score:
                found[h.id] = ScoredHit(
                    id=h.id,
                    text=h.text or "",
                    metadata=dict(h.metadata or {}),
                    exact_score=s,
                    sources=["structured"],
                )

    base: dict[str, Any] = {}
    if repo_id:
        base["repo_id"] = repo_id

    # Exact resource_name matches first (strongest signal)
    for name in names:
        where = {**base, "resource_name": name}
        try:
            _ingest(get_by_metadata(where, collection=collection, limit=limit_per_lookup))
        except Exception:  # noqa: BLE001 - empty / unsupported where must not break retrieval
            pass
        # Array membership on resource_names when present
        try:
            _ingest(
                get_by_metadata(
                    {**base, "resource_names__contains": name},
                    collection=collection,
                    limit=limit_per_lookup,
                )
            )
        except Exception:  # noqa: BLE001
            pass

    # Type-only lookups when we have type cues (and optionally no names)
    if types:
        try:
            _ingest(
                get_by_metadata(
                    {**base, "resource_type": types},
                    collection=collection,
                    limit=limit_per_lookup * 2,
                )
            )
        except Exception:  # noqa: BLE001
            pass

    ranked = sorted(found.values(), key=lambda h: h.exact_score, reverse=True)
    return ranked[: max(1, max_hits)]
