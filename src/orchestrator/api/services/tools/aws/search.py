"""Cross-service resource search.

Strategy (in order):
  1. If Resource Groups Tagging API is available and tags are supplied → use it.
  2. Parallel list_resources calls across requested (or priority) services,
     rank by name/token overlap with the query.
  3. Return a unified, scored list.

Designed so a future RAG / Resource Explorer backend can replace the
body without changing the tool contract.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..schemas import ResourceSummary
from .allowlist import PRIORITY_SERVICES, canonical_service
from .client import get_client, get_default_region
from .resource_lister import list_resources, supported_list_services
from .resolver import _score

logger = logging.getLogger(__name__)


def _tokenise(text: str) -> set[str]:
    return set(re.split(r"[^a-z0-9]+", (text or "").lower())) - {""}


def search_resources(
    query: str,
    *,
    services: list[str] | None = None,
    tags: dict[str, str] | None = None,
    region: str | None = None,
    max_results: int = 20,
) -> dict[str, Any]:
    region = region or get_default_region()
    query = (query or "").strip()
    tags = tags or {}
    max_results = max(1, min(int(max_results), 100))

    if services:
        svc_list = [canonical_service(s) for s in services]
    else:
        # Default to high-value services for orchestration
        supported = set(supported_list_services())
        svc_list = [s for s in PRIORITY_SERVICES if s in supported]

    # --- Tag-based path via Resource Groups Tagging API ---
    tag_hits: list[ResourceSummary] = []
    if tags:
        try:
            tag_hits = _search_by_tags(tags, region=region, services=svc_list, max_results=max_results)
        except Exception as exc:
            logger.debug("Tag search failed: %s", exc)

    # --- Name / query path ---
    name_hits: list[tuple[float, ResourceSummary]] = []
    if query:
        token = re.split(r"[^a-zA-Z0-9]+", query)[0] if query else ""
        filters: dict[str, Any] = {}
        if token and len(token) >= 2:
            filters["name_contains"] = token
        if tags:
            filters["tags"] = tags

        def _fetch(svc: str) -> list[ResourceSummary]:
            try:
                return list_resources(svc, region=region, filters=filters, max_results=30)
            except Exception as exc:
                logger.debug("list_resources(%s) during search failed: %s", svc, exc)
                return []

        with ThreadPoolExecutor(max_workers=min(8, len(svc_list) or 1)) as pool:
            futures = {pool.submit(_fetch, s): s for s in svc_list}
            for fut in as_completed(futures):
                for r in fut.result():
                    score = _score(query, r, context=None)
                    if score >= 0.3:
                        name_hits.append((score, r))

    # Merge: prefer higher scores; de-dupe by (service, id)
    seen: set[tuple[str, str]] = set()
    merged: list[dict[str, Any]] = []

    # Tag hits first (exact tag match is high confidence)
    for r in tag_hits:
        key = (r.service, r.id)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"confidence": 0.9, "resource": r.model_dump(), "match_reason": "tags"})

    name_hits.sort(key=lambda x: (-x[0], x[1].service, x[1].name or x[1].id))
    for score, r in name_hits:
        key = (r.service, r.id)
        if key in seen:
            continue
        seen.add(key)
        merged.append({"confidence": score, "resource": r.model_dump(), "match_reason": "name"})
        if len(merged) >= max_results:
            break

    return {
        "query": query,
        "services_searched": svc_list,
        "tags": tags,
        "region": region,
        "total": len(merged),
        "results": merged[:max_results],
    }


def _search_by_tags(
    tags: dict[str, str],
    *,
    region: str,
    services: list[str],
    max_results: int,
) -> list[ResourceSummary]:
    """Use resourcegroupstaggingapi.get_resources when possible."""
    client = get_client("resourcegroupstaggingapi", region=region)
    tag_filters = [{"Key": k, "Values": [v]} for k, v in tags.items()]

    # Map our service ids to AWS service namespaces used by the tagging API
    service_map = {
        "lambda": "lambda",
        "s3": "s3",
        "dynamodb": "dynamodb",
        "sqs": "sqs",
        "sns": "sns",
        "states": "states",
        "glue": "glue",
        "ecs": "ecs",
        "events": "events",
    }
    type_filters = []
    for s in services:
        ns = service_map.get(s)
        if ns:
            type_filters.append(ns)

    kwargs: dict[str, Any] = {
        "TagFilters": tag_filters,
        "ResourcesPerPage": min(50, max_results),
    }
    # ResourceTypeFilters expects forms like "lambda:function"
    # We keep it broad and filter client-side.

    out: list[ResourceSummary] = []
    pagination_token = None
    while len(out) < max_results:
        if pagination_token:
            kwargs["PaginationToken"] = pagination_token
        resp = client.get_resources(**kwargs)
        for m in resp.get("ResourceTagMappingList") or []:
            arn = m.get("ResourceARN") or ""
            tag_list = m.get("Tags") or []
            tag_dict = {t["Key"]: t["Value"] for t in tag_list if "Key" in t}
            # Parse service from ARN
            parts = arn.split(":")
            svc = parts[2] if len(parts) > 2 else "unknown"
            if svc == "stepfunctions":
                svc = "states"
            name = parts[-1].split("/")[-1] if parts else arn
            out.append(
                ResourceSummary(
                    service=svc,
                    resource_type="resource",
                    id=name,
                    name=name,
                    arn=arn,
                    region=region,
                    tags=tag_dict,
                    summary={},
                )
            )
            if len(out) >= max_results:
                break
        pagination_token = resp.get("PaginationToken") or None
        if not pagination_token:
            break
    return out
