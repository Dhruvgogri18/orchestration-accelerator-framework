"""Format discovered AWS resources for injection into LLM prompts.

Keeps the payload compact (token-budget aware) and deterministic so
repeated runs against the same account state produce the same context
block. Discovery results are never cached — each turn hits live AWS.
"""

from __future__ import annotations

import json
import time
from typing import Any


def _compact_resource(r: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "service": r.get("service"),
        "type": r.get("resource_type"),
        "id": r.get("id") or r.get("name"),
        "name": r.get("name") or r.get("id"),
    }
    if r.get("arn"):
        out["arn"] = r["arn"]
    summary = r.get("summary") or {}
    for k in ("runtime", "handler", "status", "fifo", "type", "state", "event_bus"):
        if summary.get(k) is not None:
            out[k] = summary[k]
    tags = r.get("tags") or {}
    if tags:
        out["tags"] = dict(list(tags.items())[:8])
    return out


def run_deterministic_context(user_text: str) -> dict[str, Any]:
    """Always-on, zero-LLM tools: normalize, extract entities, classify pattern.

    Safe to call every turn. No AWS, no cache, no model calls.
    """
    from .deterministic import (
        handle_classify_workflow_pattern,
        handle_extract_entities,
        handle_normalize_prompt,
    )

    text = user_text or ""
    norm = handle_normalize_prompt({"text": text})
    ents = handle_extract_entities({"text": text})
    # Prefer normalized text for pattern classification (more stable)
    pattern = handle_classify_workflow_pattern(
        {"text": norm.get("normalized") or text}
    )
    return {
        "normalized": norm.get("normalized") or "",
        "services_inferred": list(norm.get("services") or []),
        "name_tokens": list(norm.get("name_tokens") or []),
        "entities": list(ents.get("entities") or []),
        "pattern": pattern.get("pattern") or "unknown",
        "pattern_confidence": pattern.get("confidence") or 0.0,
        "pattern_candidates": list(pattern.get("candidates") or []),
        "source": "deterministic",
    }


def format_discovery_for_prompt(
    discovery: dict[str, Any] | None,
    *,
    max_resources_per_service: int = 12,
    max_total_chars: int = 12_000,
) -> str | None:
    """Return a system-context string, or None if there is nothing useful.

    Includes zero-LLM deterministic fields (pattern, services, entities)
    even when live AWS listing is empty or deferred.
    """
    if not discovery:
        return None

    by_svc = discovery.get("resources_by_service") or {}
    search_hits = discovery.get("search_hits") or []
    services = discovery.get("services_inferred") or []
    account_id = discovery.get("account_id")
    pattern = discovery.get("pattern")
    name_tokens = discovery.get("name_tokens") or []
    entities = discovery.get("entities") or []
    normalized = discovery.get("normalized") or ""
    bindings = discovery.get("bindings") or {}

    has_aws = bool(by_svc or search_hits or bindings)
    has_det = bool(pattern or services or name_tokens or entities or normalized)
    if not has_aws and not has_det:
        return None

    lines: list[str] = [
        "WORKFLOW CONTEXT (deterministic pre-processing — no invention. "
        "Prefer real names/ARNs below when they match intent; otherwise TODO_ "
        "placeholders. Do not invent ARNs or account IDs.)",
    ]

    # Compact deterministic block (cheap tokens, high signal)
    if pattern or normalized or services:
        lines.append("## Intent signals (rule-based, not LLM)")
        if pattern:
            conf = discovery.get("pattern_confidence")
            lines.append(
                f"- pattern: {pattern}"
                + (f" (confidence={conf})" if conf is not None else "")
            )
        if services:
            lines.append(f"- services: {', '.join(services)}")
        if name_tokens:
            lines.append(f"- name_tokens: {', '.join(str(t) for t in name_tokens[:12])}")
        if normalized and len(normalized) <= 400:
            lines.append(f"- normalized: {normalized}")
        lines.append("")

    if entities:
        lines.append("## Extracted entities (explicit only)")
        for e in entities[:15]:
            if not isinstance(e, dict):
                continue
            lines.append(
                f"- {e.get('kind')}: {e.get('value')}"
                + (f" [{e.get('service_hint')}]" if e.get("service_hint") else "")
            )
        lines.append("")

    repo_hits = discovery.get("repo_hits") or []
    repo_resolved = discovery.get("repo_resolved") or []
    if repo_resolved or repo_hits:
        lines.append("## Repository artifacts (from project index — prefer these)")
        for r in repo_resolved[:10]:
            art = r.get("artifact") or {}
            lines.append(
                f"- resolved {r.get('token')}: {art.get('resource_name')} "
                f"({art.get('resource_type') or art.get('kind') or '?'}) "
                f"path={art.get('path') or '-'}"
            )
        if not repo_resolved:
            for h in repo_hits[:8]:
                lines.append(
                    f"- {h.get('resource_name') or h.get('id')}: "
                    f"{h.get('resource_type') or h.get('kind') or '?'} "
                    f"path={h.get('path') or '-'}"
                )
        lines.append("")


    if bindings:
        lines.append("## Grounded bindings (live resolve)")
        for key in sorted(bindings.keys())[:20]:
            b = bindings[key]
            if not isinstance(b, dict):
                continue
            lines.append(
                f"- {key}: matched={b.get('matched')} arn={b.get('arn') or ''} "
                f"name={b.get('name') or ''}"
            )
        lines.append("")

    if has_aws:
        lines.append(
            "## AWS account resources (live, read-only, single account)"
        )
        if account_id:
            lines.append(f"Account: {account_id}")
        lines.append(
            f"Inferred services: {', '.join(services) if services else '(none)'}"
        )
        lines.append("")

        for svc in sorted(by_svc.keys()):
            resources = by_svc[svc] or []
            lines.append(f"## {svc} ({len(resources)} listed)")
            for r in resources[:max_resources_per_service]:
                compact = _compact_resource(r if isinstance(r, dict) else {})
                lines.append(f"- {json.dumps(compact, separators=(',', ':'))}")
            if len(resources) > max_resources_per_service:
                lines.append(
                    f"  … +{len(resources) - max_resources_per_service} more not shown"
                )
            lines.append("")

        if search_hits:
            lines.append("## Top search matches")
            for hit in search_hits[:10]:
                conf = hit.get("confidence")
                res = hit.get("resource") or {}
                compact = _compact_resource(res if isinstance(res, dict) else {})
                lines.append(
                    f"- conf={conf}: {json.dumps(compact, separators=(',', ':'))}"
                )
            lines.append("")

    text = "\n".join(lines).strip()
    if len(text) > max_total_chars:
        text = text[: max_total_chars - 20] + "\n… [truncated]"
    return text


def inject_discovery_into_history(
    history: list[dict],
    discovery: dict[str, Any] | None,
) -> list[dict]:
    """Return a shallow copy of history with a discovery system message inserted."""
    block = format_discovery_for_prompt(discovery)
    if not block:
        return history

    out: list[dict] = []
    inserted = False
    for msg in history:
        role = (msg.get("role") or "").lower()
        if not inserted and role not in ("system",):
            out.append({"role": "system", "content": block})
            inserted = True
        out.append(msg)
    if not inserted:
        out.append({"role": "system", "content": block})
    return out


def run_discovery_safe(
    user_text: str,
    *,
    session_id: str | None = None,
    region: str | None = None,
) -> dict[str, Any] | None:
    """Run deterministic tools always; optionally live AWS discovery.

    - Zero-LLM normalize / extract / classify run on every turn (when tools
      are enabled). No model cost.
    - Live AWS listing still respects TOOLS_DISCOVERY_MODE:
        * deferred / off → skip AWS scan, still return deterministic bundle
        * eager / on     → full discover_for_workflow merge
    Results are never cached.
    """
    try:
        from orchestrator.api.config.settings import get_api_settings

        settings = get_api_settings()
        if not getattr(settings, "tools_enabled", True):
            return None
        mode = (getattr(settings, "tools_discovery_mode", "deferred") or "deferred").lower()
        region = region or getattr(settings, "aws_region", None)
        budget_ms = int(getattr(settings, "tools_discovery_budget_ms", 12_000) or 12_000)
    except Exception:
        mode = "deferred"
        budget_ms = 12_000
        settings = None

    # --- Always: deterministic context (CPU only) ---
    try:
        bundle: dict[str, Any] = run_deterministic_context(user_text)
    except Exception:
        bundle = {
            "normalized": (user_text or "")[:500],
            "services_inferred": [],
            "name_tokens": [],
            "entities": [],
            "pattern": "unknown",
            "source": "deterministic_fallback",
        }

    try:
        from orchestrator.api.services.llm_trace import log_event

        log_event(
            "deterministic_context",
            pattern=bundle.get("pattern"),
            services=bundle.get("services_inferred"),
            entity_count=len(bundle.get("entities") or []),
            name_token_count=len(bundle.get("name_tokens") or []),
        )
    except Exception:
        pass

    # --- Repo RAG context (Part B; no-op when RETRIEVAL_ENABLED=false) ---
    try:
        from orchestrator.api.services.tools.repo import build_repo_context_bundle

        repo_bundle = build_repo_context_bundle(user_text) or {}
        if repo_bundle.get("enabled"):
            bundle["repo_enabled"] = True
            bundle["repo_id"] = repo_bundle.get("repo_id")
            bundle["repo_hits"] = repo_bundle.get("hits") or []
            bundle["repo_resolved"] = repo_bundle.get("resolved") or []
            bundle["repo_context"] = repo_bundle.get("context") or ""
            # Promote resolved repo names into name_tokens for downstream grounding
            extra_tokens = [
                (r.get("artifact") or {}).get("resource_name")
                for r in (repo_bundle.get("resolved") or [])
                if (r.get("artifact") or {}).get("resource_name")
            ]
            if extra_tokens:
                existing = list(bundle.get("name_tokens") or [])
                bundle["name_tokens"] = list(dict.fromkeys(existing + extra_tokens))
            try:
                from orchestrator.api.services.llm_trace import log_event as _le

                _le(
                    "repo_context",
                    hit_count=len(bundle.get("repo_hits") or []),
                    resolved_count=len(bundle.get("repo_resolved") or []),
                    repo_id=bundle.get("repo_id"),
                )
            except Exception:
                pass
        else:
            bundle["repo_enabled"] = False
    except Exception:
        bundle["repo_enabled"] = False

    if mode in ("off", "false", "0"):
        return bundle

    # deferred: no eager AWS scan (role_binding may still run later)
    if mode == "deferred":
        bundle["aws_discovery"] = "deferred"
        return bundle

    # --- Eager AWS discovery (optional) ---
    started = time.perf_counter()
    try:
        from orchestrator.api.services.tools.discovery import discover_for_workflow
        from orchestrator.api.services.llm_trace import log_event, llm_stage
        from orchestrator.api.services.tools.aws.client import require_identity

        with llm_stage("aws_resource_discovery"):
            try:
                identity = require_identity(region)
                account_id = identity.account_id
            except Exception as exc:
                log_event(
                    "discovery_skipped",
                    reason="identity_unavailable",
                    error=str(exc),
                )
                bundle["aws_discovery"] = "identity_unavailable"
                return bundle

            aws_bundle = discover_for_workflow(
                user_text,
                region=region,
                session_id=session_id,
            )
            # Merge: deterministic fields win for pattern/normalized;
            # AWS fields fill resources. Prefer union of services.
            det_services = list(bundle.get("services_inferred") or [])
            aws_services = list(aws_bundle.get("services_inferred") or [])
            merged_services = list(dict.fromkeys(det_services + aws_services))
            bundle.update(
                {
                    "resources_by_service": aws_bundle.get("resources_by_service") or {},
                    "search_hits": aws_bundle.get("search_hits") or [],
                    "tool_calls": aws_bundle.get("tool_calls") or [],
                    "picker_resources": aws_bundle.get("picker_resources") or [],
                    "name_hints": aws_bundle.get("name_hints")
                    or bundle.get("name_tokens")
                    or [],
                    "account_id": account_id,
                    "region": region,
                    "services_inferred": merged_services,
                    "aws_discovery": "ok",
                }
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        n = sum(
            len(v or [])
            for v in (bundle.get("resources_by_service") or {}).values()
        )
        log_event(
            "discovery_complete",
            services=bundle.get("services_inferred"),
            resource_count=n,
            search_hits=len(bundle.get("search_hits") or []),
            tool_calls=len(bundle.get("tool_calls") or []),
            account_id=bundle.get("account_id"),
            pattern=bundle.get("pattern"),
            latency_ms=round(elapsed_ms, 2),
            budget_ms=budget_ms,
            budget_exceeded=elapsed_ms > budget_ms,
        )
        return bundle
    except Exception as exc:  # noqa: BLE001
        try:
            from orchestrator.api.services.llm_trace import log_event

            log_event(
                "discovery_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except Exception:
            pass
        bundle["aws_discovery"] = "failed"
        return bundle
