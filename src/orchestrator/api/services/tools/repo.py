"""Repository-context tools (Part B RAG surface).

These tools query the local Chroma index built by ``orchestrator index``.
They are the RAG counterpart of the live-AWS tools in ``aws/``:

  * search_repo_context  — dense + structured retrieval for a free-text query
  * resolve_repo_resource — exact/fuzzy match of a job, Lambda, Glue script, etc.
  * list_repo_artifacts   — enumerate known resource names by kind

Design rules (same as the AWS tools layer):
  * Deterministic output shapes (sorted, stable keys)
  * Never invent resources — empty result when index is cold or retrieval is off
  * No LLM calls; pure retrieval + metadata filters
  * Safe to call every turn; failures return structured empty results
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_NAME_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)\b",
    re.IGNORECASE,
)

# Map free-text kind hints to chunk metadata values produced by the chunker.
_KIND_ALIASES: dict[str, list[str]] = {
    "lambda": ["lambda", "lambda_handler"],
    "lambda_handler": ["lambda_handler", "lambda"],
    "glue": ["glue", "glue_job", "glue_script"],
    "glue_job": ["glue_job", "glue", "glue_script"],
    "glue_script": ["glue_script", "glue_job"],
    "job": ["glue_job", "glue_script", "job_definition", "lambda_handler"],
    "pipeline": ["pipeline"],
    "sns": ["sns_topic"],
    "sns_topic": ["sns_topic"],
    "sqs": ["sqs_queue"],
    "sqs_queue": ["sqs_queue"],
    "s3": ["s3_bucket"],
    "s3_bucket": ["s3_bucket"],
    "dynamodb": ["dynamodb_table"],
    "dynamodb_table": ["dynamodb_table"],
    "config": ["config_entry", "config_section", "env_resources"],
    "doc": ["doc_section"],
}


def _retrieval_available() -> bool:
    try:
        from orchestrator.config import get_settings

        s = get_settings()
        if getattr(s, "retrieval_enabled", False):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from orchestrator.api.config.settings import get_api_settings

        return bool(get_api_settings().retrieval_enabled)
    except Exception:  # noqa: BLE001
        return False


def _default_repo_id() -> str | None:
    try:
        from orchestrator.config import get_settings
        from orchestrator.indexing.pipeline import default_repo_id

        return default_repo_id(get_settings().repo_path)
    except Exception:  # noqa: BLE001
        return None


def _hit_to_artifact(hit: Any) -> dict[str, Any]:
    """Normalise a ScoredHit / Hit into a stable artifact dict."""
    meta = getattr(hit, "metadata", None) or {}
    if not isinstance(meta, dict):
        meta = {}
    text = getattr(hit, "text", "") or ""
    score = getattr(hit, "score", None)
    if score is None:
        dist = getattr(hit, "distance", None)
        score = (1.0 - float(dist)) if dist is not None else None

    resource_name = meta.get("resource_name") or ""
    resource_names = meta.get("resource_names") or []
    if isinstance(resource_names, str):
        resource_names = [resource_names]
    resource_type = meta.get("resource_type") or meta.get("kind") or ""
    entry_point = meta.get("entry_point") or ""
    path = meta.get("path") or meta.get("source_path") or ""

    return {
        "id": getattr(hit, "id", None) or meta.get("id") or "",
        "resource_name": resource_name or (resource_names[0] if resource_names else ""),
        "resource_names": sorted({str(n) for n in resource_names if n}),
        "resource_type": str(resource_type) if resource_type else "",
        "kind": str(meta.get("kind") or resource_type or ""),
        "entry_point": str(entry_point) if entry_point else "",
        "path": str(path),
        "score": round(float(score), 4) if score is not None else None,
        "snippet": (text[:400] + ("…" if len(text) > 400 else "")).strip(),
        "metadata": {
            k: v
            for k, v in meta.items()
            if k
            in (
                "repo_id",
                "file_type",
                "resource_type",
                "resource_types",
                "kind",
                "entry_point",
                "script",
                "status",
                "env",
            )
            and v not in (None, "", [], {})
        },
    }


def handle_search_repo_context(args: dict) -> dict:
    """Dense + structured retrieval over the indexed repository.

    Input:
      query: str (required)
      repo_id: str | None
      top_k: int (default 8)
      min_score: float | None

    Output:
      enabled, hits (list of artifacts), context (markdown block), stats
    """
    query = (args.get("query") or "").strip()
    if not query:
        return {
            "enabled": _retrieval_available(),
            "hits": [],
            "context": "",
            "stats": {"error": "empty_query"},
        }

    if not _retrieval_available():
        return {
            "enabled": False,
            "hits": [],
            "context": "",
            "stats": {"reason": "RETRIEVAL_ENABLED is false"},
        }

    top_k = int(args.get("top_k") or 8)
    top_k = max(1, min(top_k, 30))
    min_score = args.get("min_score")
    repo_id = args.get("repo_id") or _default_repo_id()

    try:
        from orchestrator.retrieval import retrieve

        kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "enabled": True,
            "rerank_top_n": top_k,
        }
        if min_score is not None:
            kwargs["min_score"] = float(min_score)

        result = retrieve(query, **kwargs)
        artifacts = [_hit_to_artifact(h) for h in (result.hits or [])][:top_k]
        # Stable order: score desc, then resource_name, then id
        artifacts.sort(
            key=lambda a: (
                -(a["score"] or 0.0),
                a.get("resource_name") or "",
                a.get("id") or "",
            )
        )
        return {
            "enabled": True,
            "hits": artifacts,
            "context": result.context or "",
            "stats": {
                "dense_count": getattr(result, "dense_count", 0),
                "structured_count": getattr(result, "structured_count", 0),
                "truncated": getattr(result, "truncated", False),
                "repo_id": repo_id,
                "hit_count": len(artifacts),
            },
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("search_repo_context failed: %s", exc)
        return {
            "enabled": True,
            "hits": [],
            "context": "",
            "stats": {"error": f"{type(exc).__name__}: {exc}", "repo_id": repo_id},
        }


def handle_resolve_repo_resource(args: dict) -> dict:
    """Resolve a human name / partial name to a repo-indexed artifact.

    Input:
      hint: str (required) — job name, Lambda name, script path fragment
      kind: str | None — lambda, glue, pipeline, sns, ...
      repo_id: str | None

    Output:
      matched: bool
      artifact: dict | None
      alternatives: list[dict]
      confidence: float
    """
    hint = (args.get("hint") or "").strip()
    if not hint:
        return {
            "matched": False,
            "artifact": None,
            "alternatives": [],
            "confidence": 0.0,
            "reason": "empty_hint",
        }

    if not _retrieval_available():
        return {
            "matched": False,
            "artifact": None,
            "alternatives": [],
            "confidence": 0.0,
            "reason": "retrieval_disabled",
        }

    kind = (args.get("kind") or "").strip().lower() or None
    repo_id = args.get("repo_id") or _default_repo_id()
    kind_filters = _KIND_ALIASES.get(kind, [kind] if kind else [])

    # Prefer structured exact match via retrieve_structured path inside retrieve
    search = handle_search_repo_context(
        {"query": hint, "repo_id": repo_id, "top_k": 15, "min_score": 0.05}
    )
    hits = list(search.get("hits") or [])

    if kind_filters:
        filtered = [
            h
            for h in hits
            if (h.get("resource_type") or "").lower() in kind_filters
            or (h.get("kind") or "").lower() in kind_filters
            or any(
                t in kind_filters
                for t in (h.get("metadata") or {}).get("resource_types") or []
            )
        ]
        if filtered:
            hits = filtered

    hint_l = hint.lower()
    exact: list[dict] = []
    partial: list[dict] = []
    for h in hits:
        names = {hint_l}
        rn = (h.get("resource_name") or "").lower()
        rns = [str(x).lower() for x in (h.get("resource_names") or [])]
        path = (h.get("path") or "").lower()
        if rn == hint_l or hint_l in rns:
            exact.append(h)
        elif hint_l in rn or any(hint_l in n for n in rns) or hint_l in path:
            partial.append(h)
        elif rn and (rn in hint_l or any(n in hint_l for n in rns if n)):
            partial.append(h)

    ranked = exact + partial
    if not ranked:
        return {
            "matched": False,
            "artifact": None,
            "alternatives": hits[:5],
            "confidence": 0.0,
            "reason": "no_match",
            "repo_id": repo_id,
        }

    best = ranked[0]
    confidence = 0.95 if exact else 0.7
    # Ambiguity: multiple exact matches → do not auto-pick
    if len(exact) > 1:
        return {
            "matched": False,
            "artifact": None,
            "alternatives": exact[:8],
            "confidence": 0.5,
            "reason": "ambiguous",
            "repo_id": repo_id,
        }

    return {
        "matched": True,
        "artifact": best,
        "alternatives": ranked[1:6],
        "confidence": confidence,
        "repo_id": repo_id,
    }


def handle_list_repo_artifacts(args: dict) -> dict:
    """List indexed artifacts, optionally filtered by kind.

    Input:
      kind: str | None
      repo_id: str | None
      max_results: int (default 40)

    Uses structured metadata scan when possible; falls back to a broad
    semantic query over kind keywords.
    """
    if not _retrieval_available():
        return {
            "enabled": False,
            "artifacts": [],
            "count": 0,
            "reason": "retrieval_disabled",
        }

    kind = (args.get("kind") or "").strip().lower() or None
    repo_id = args.get("repo_id") or _default_repo_id()
    max_results = max(1, min(int(args.get("max_results") or 40), 100))

    # Seed query from kind so dense+structured both fire
    if kind:
        query = f"{kind} job handler script resource definition"
    else:
        query = "lambda glue pipeline job handler sns sqs workflow"

    search = handle_search_repo_context(
        {"query": query, "repo_id": repo_id, "top_k": max_results, "min_score": 0.05}
    )
    artifacts = list(search.get("hits") or [])

    if kind:
        kind_filters = set(_KIND_ALIASES.get(kind, [kind]))
        artifacts = [
            a
            for a in artifacts
            if (a.get("resource_type") or "").lower() in kind_filters
            or (a.get("kind") or "").lower() in kind_filters
        ]

    # Deduplicate by resource_name (prefer higher score)
    seen: set[str] = set()
    unique: list[dict] = []
    for a in artifacts:
        key = (a.get("resource_name") or a.get("id") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(a)
        if len(unique) >= max_results:
            break

    unique.sort(key=lambda a: (a.get("resource_name") or "", a.get("id") or ""))
    return {
        "enabled": True,
        "artifacts": unique,
        "count": len(unique),
        "repo_id": repo_id,
        "kind": kind,
    }


def extract_repo_name_tokens(text: str) -> list[str]:
    """Deterministic name-token extraction for grounding (shared with discovery)."""
    return sorted({m.group(1).lower() for m in _NAME_RE.finditer(text or "")})


def handle_validate_plan_against_repo(args: dict) -> dict:
    """Ground plan steps against the repository index (tool surface).

    Input:
      plan: dict (required)
      repo_id: str | None
      require_all: bool (default False)

    Output: grounding report dict (ok, hits, unresolved, ...).
    """
    plan = args.get("plan")
    if not isinstance(plan, dict):
        return {
            "ok": False,
            "enabled": _retrieval_available(),
            "resolved_count": 0,
            "unresolved": [],
            "notes": ["plan must be a dict"],
            "hits": [],
        }
    try:
        from orchestrator.synthesis.grounding import ground_plan_against_repo

        report = ground_plan_against_repo(
            plan,
            repo_id=args.get("repo_id"),
            require_all=bool(args.get("require_all")),
        )
        return report.to_dict()
    except Exception as exc:  # noqa: BLE001
        logger.warning("validate_plan_against_repo failed: %s", exc)
        return {
            "ok": False,
            "enabled": _retrieval_available(),
            "resolved_count": 0,
            "unresolved": [],
            "notes": [f"{type(exc).__name__}: {exc}"],
            "hits": [],
        }


def build_repo_context_bundle(user_text: str, *, repo_id: str | None = None) -> dict[str, Any]:
    """One-shot bundle for injection into discovery / pipeline analysis.

    Always safe: returns structured empty payload when retrieval is off.
    """
    text = (user_text or "").strip()
    base: dict[str, Any] = {
        "enabled": _retrieval_available(),
        "repo_id": repo_id or _default_repo_id(),
        "hits": [],
        "context": "",
        "resolved": [],
        "name_tokens": extract_repo_name_tokens(text),
    }
    if not text or not base["enabled"]:
        return base

    search = handle_search_repo_context(
        {"query": text, "repo_id": base["repo_id"], "top_k": 10}
    )
    base["hits"] = search.get("hits") or []
    base["context"] = search.get("context") or ""
    base["stats"] = search.get("stats") or {}

    # Resolve explicit name tokens against the index
    resolved = []
    for token in base["name_tokens"][:12]:
        res = handle_resolve_repo_resource(
            {"hint": token, "repo_id": base["repo_id"]}
        )
        if res.get("matched") and res.get("artifact"):
            resolved.append(
                {
                    "token": token,
                    "artifact": res["artifact"],
                    "confidence": res.get("confidence"),
                }
            )
    base["resolved"] = resolved
    return base
