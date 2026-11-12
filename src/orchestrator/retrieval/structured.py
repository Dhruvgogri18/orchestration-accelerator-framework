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
    # "job" means a Glue job. It must NOT imply "lambda": doing so pulled every
    # Lambda definition in the repo into any prompt that said "Glue job" and
    # crowded out the resources the user actually asked for (e.g. the SNS topic).
    "job": ["glue", "glue_job"],
    "jobs": ["glue", "glue_job"],
}

# Environment cues. Only an unambiguous single environment in the query is used.
_ENV_ALIASES = {
    "dev": "dev",
    "prod": "prod",
    "production": "prod",
    "staging": "staging",
    "qa": "qa",
    "uat": "uat",
}
_ENV_RE = re.compile(r"\b(dev|prod|production|staging|qa|uat)\b", re.IGNORECASE)
_DEV_LONG_RE = re.compile(r"\bdevelopment\s+(?:env|environment|account|stack)\b", re.IGNORECASE)

# Words that carry no retrieval signal for the lexical "purpose" match below.
_STOPWORDS = frozenset(
    """a an and are as at be by for from if in into is it its of on or that the then this
    to up using via when where which while with without create build make workflow
    workflows step steps state machine send sends sent load loads run runs runs file files
    use uses need needs want wants should would could please also each every after before
    else otherwise""".split()
)

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


def extract_type_groups(query: str) -> list[tuple[str, ...]]:
    """Resource-type groups named by the query, one group per cue (e.g. ``("sns_topic",)``).

    A group is satisfied when a hit has *any* type in it, so ``glue`` and
    ``glue_job`` count as one need. Used to guarantee each kind of resource the
    user mentioned is represented in the final context.
    """
    lower = (query or "").lower()
    groups: list[tuple[str, ...]] = []
    for cue, mapped in _SERVICE_CUES.items():
        if re.search(rf"\b{re.escape(cue)}\b", lower):
            group = tuple(mapped)
            if group not in groups and not any(set(group) <= set(g) for g in groups):
                groups = [g for g in groups if not set(g) <= set(group)]
                groups.append(group)
    return groups


def extract_environment(query: str) -> str | None:
    """Return ``"dev"`` / ``"prod"`` / ... when the query names exactly one environment."""
    text = query or ""
    found = {_ENV_ALIASES[m.group(1).lower()] for m in _ENV_RE.finditer(text)}
    if _DEV_LONG_RE.search(text):
        found.add("dev")
    return next(iter(found)) if len(found) == 1 else None


def hit_types(metadata: dict[str, Any]) -> set[str]:
    """All resource types a chunk is tagged with (single ``resource_type`` + ``resource_types``)."""
    out: set[str] = set()
    rtype = metadata.get("resource_type")
    if rtype:
        out.add(str(rtype).lower())
    for t in metadata.get("resource_types") or []:
        out.add(str(t).lower())
    return out


def filter_by_environment(hits: list[Any], environment: str | None) -> list[Any]:
    """Drop chunks that belong to a *different* environment than the one requested.

    Chunks with no environment (job catalog, docs, code) are always kept. If no
    chunk carries the requested environment at all (the repo does not label its
    environments that way), nothing is dropped, so a stray word like "dev" can
    never wipe out the context.
    """
    if not environment:
        return hits
    env = environment.lower()

    def _env(h: Any) -> str:
        return str((h.metadata or {}).get("environment") or "").lower()

    if not any(_env(h) == env for h in hits):
        return hits
    return [h for h in hits if not _env(h) or _env(h) == env]


def _stem(word: str) -> str:
    for suffix in ("ing", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z][a-z0-9]{2,}", (text or "").lower().replace("_", " "))]


def extract_terms(query: str, names: list[str]) -> list[str]:
    """Content words in the query that describe *purpose* (e.g. ``ops``, ``alert``, ``fail``).

    Resource names, service cue words, environment words and filler are removed,
    so what is left can be matched against chunk text to tell apart resources of
    the same type (``ops-alerts`` vs ``order-summary`` SNS topics).
    """
    skip: set[str] = set(_STOPWORDS) | set(_SERVICE_CUES) | set(_ENV_ALIASES) | {"development"}
    for name in names:
        skip.update(re.split(r"[-_]", name))
    terms: list[str] = []
    for raw in re.findall(r"[a-z][a-z0-9]{2,}", (query or "").lower().replace("-", " ").replace("_", " ")):
        if raw in skip:
            continue
        stem = _stem(raw)
        if stem not in terms:
            terms.append(stem)
    return terms


def _lexical_bonus(hit: Any, terms: list[str]) -> float:
    """+0.06 per query term found in the chunk text (capped at 0.18)."""
    if not terms:
        return 0.0
    words = _words(hit.text or "")
    matched = 0
    for term in terms:
        if any(w == term or (len(term) >= 4 and (w.startswith(term) or term.startswith(w)) and len(w) >= 4) for w in words):
            matched += 1
    return min(0.18, 0.06 * matched)


def _score_hit(
    hit: Any,
    names: list[str],
    types: list[str],
    environment: str | None = None,
    terms: list[str] | None = None,
) -> float:
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

    if environment and str(meta.get("environment") or "").lower() == environment:
        score += 0.10  # chunk is for exactly the environment the user asked about

    score += _lexical_bonus(hit, terms or [])

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
    environment: str | None = None,
) -> list[ScoredHit]:
    """Look up chunks by extracted resource names and type hints.

    ``environment`` defaults to the single environment named in the query (if
    any). Chunks of other environments are dropped; environment-neutral chunks
    (job catalog, docs, code) are kept.
    """
    names, types = extract_candidates(query)
    if not names and not types:
        return []
    env = (environment or extract_environment(query) or "").lower() or None
    terms = extract_terms(query, names)

    from orchestrator.indexing.vectorstore import get_by_metadata

    found: dict[str, ScoredHit] = {}

    def _ingest(hits: list[Any]) -> None:
        for h in hits:
            s = _score_hit(h, names, types, env, terms)
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

    # Type-only lookups when we have type cues (and optionally no names).
    # Environment-scoped first, so a dev question reaches the dev chunks even
    # when many same-typed chunks exist; then unscoped for environment-neutral
    # chunks (job catalog, code).
    if types:
        scopes: list[dict[str, Any]] = [{"environment": env}, {}] if env else [{}]
        for scope in scopes:
            try:
                _ingest(
                    get_by_metadata(
                        {**base, **scope, "resource_type": types},
                        collection=collection,
                        limit=limit_per_lookup * 2,
                    )
                )
            except Exception:  # noqa: BLE001
                pass
            # Multi-resource section chunks (e.g. buckets+tables) are tagged with
            # the plural ``resource_types`` array, which the lookup above never saw.
            for rtype in types:
                try:
                    _ingest(
                        get_by_metadata(
                            {**base, **scope, "resource_types__contains": rtype},
                            collection=collection,
                            limit=limit_per_lookup,
                        )
                    )
                except Exception:  # noqa: BLE001
                    pass

    candidates = filter_by_environment(list(found.values()), env)
    ranked = sorted(candidates, key=lambda h: h.exact_score, reverse=True)
    return ranked[: max(1, max_hits)]