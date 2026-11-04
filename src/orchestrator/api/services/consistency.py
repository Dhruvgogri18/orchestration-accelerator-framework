"""Generic request normalisation helpers (domain-agnostic).

IMPORTANT
---------
This module does **not** cache plans or prompts, and does **not** encode
product-domain knowledge (order/payment/retail/…). Accuracy for arbitrary
prompts comes from RAG + tools + the LLM, not from synonym tables.

What remains here is only:
  * light whitespace / filler cleanup for logging fingerprints
  * structural plan hash (topology only — ids, services, edges)

Fingerprint is for observability correlation, not for response reuse.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

_WHITESPACE = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s\-./_]")

# Generic English filler only — never domain action names.
_FILLER = re.compile(
    r"\b("
    r"please|kindly|can you|could you|would you|i want to|i need to|"
    r"i'd like to|i would like to|help me|for me|"
    r"a|an|the|this|that|these|those"
    r")\b",
    re.I,
)


def canonicalize_request(text: str) -> str:
    """Light, domain-agnostic normalisation for logging / correlation only.

    Not used as a cache key for plans or LLM responses.
    """
    if not text:
        return ""
    s = text.strip().lower()
    s = _FILLER.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s


def request_fingerprint(text: str) -> str:
    """SHA-256 of the normalised request (first 32 hex chars) — observability only."""
    canon = canonicalize_request(text)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:32]


def _step_skeleton(step: dict[str, Any]) -> dict[str, Any]:
    branches = step.get("branches") or []
    branch_targets = sorted(
        {
            (str(b.get("condition") or "").strip().lower(), str(b.get("next") or ""))
            for b in branches
            if isinstance(b, dict)
        }
    )
    return {
        "id": str(step.get("id") or ""),
        "service": str(step.get("service") or "").strip().lower(),
        "next": step.get("next"),
        "on_failure": sorted(step.get("on_failure") or []),
        "on_timeout": sorted(step.get("on_timeout") or []),
        "branches": branch_targets,
        "component_type": step.get("component_type"),
    }


def plan_structural_hash(plan: dict[str, Any] | None) -> str:
    """Stable hash of plan topology (ids, services, edges) — not prose."""
    if not plan:
        return "empty"
    steps = plan.get("steps") or []
    skeleton = {
        "name": _normalize_workflow_name(plan.get("name") or ""),
        "steps": [_step_skeleton(s) for s in steps if isinstance(s, dict)],
    }
    skeleton["steps"].sort(key=lambda s: s["id"])
    payload = json.dumps(skeleton, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def _normalize_workflow_name(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", (name or "").strip().lower())
    return s.strip("-") or "workflow"


@dataclass
class ConsistencyReport:
    request_fingerprint: str
    canonical_request: str
    plan_structural_hash: str
    pattern: str | None = None
    services: list[str] = field(default_factory=list)
    name_tokens: list[str] = field(default_factory=list)
    matched_template: str | None = None
    match_tier: str | None = None
    self_consistent: bool | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_fingerprint": self.request_fingerprint,
            "canonical_request": self.canonical_request,
            "plan_structural_hash": self.plan_structural_hash,
            "pattern": self.pattern,
            "services": list(self.services),
            "name_tokens": list(self.name_tokens),
            "matched_template": self.matched_template,
            "match_tier": self.match_tier,
            "self_consistent": self.self_consistent,
            "notes": list(self.notes),
        }


def build_consistency_report(
    user_text: str,
    plan: dict[str, Any] | None = None,
    *,
    discovery: dict[str, Any] | None = None,
    match_tier: str | None = None,
    matched_template: str | None = None,
    alternate_plan: dict[str, Any] | None = None,
) -> ConsistencyReport:
    """Observability report only — never used to short-circuit generation."""
    canon = canonicalize_request(user_text)
    fp = request_fingerprint(user_text)
    struct = plan_structural_hash(plan)

    pattern = None
    services: list[str] = []
    name_tokens: list[str] = []
    if discovery:
        pattern = discovery.get("pattern")
        services = list(
            discovery.get("services_inferred") or discovery.get("services") or []
        )
        name_tokens = list(discovery.get("name_tokens") or [])

    self_ok: bool | None = None
    notes: list[str] = []
    if alternate_plan is not None and plan is not None:
        alt_hash = plan_structural_hash(alternate_plan)
        self_ok = alt_hash == struct
        if not self_ok:
            notes.append(
                f"self-consistency mismatch: primary={struct} alternate={alt_hash}"
            )

    return ConsistencyReport(
        request_fingerprint=fp,
        canonical_request=canon,
        plan_structural_hash=struct,
        pattern=pattern,
        services=sorted(set(str(x) for x in services)),
        name_tokens=sorted(set(str(x) for x in name_tokens)),
        matched_template=matched_template,
        match_tier=match_tier,
        self_consistent=self_ok,
        notes=notes,
    )


def cache_get_plan(user_text: str) -> dict[str, Any] | None:
    """Always None — plan/prompt caching is disabled."""
    return None


def cache_put_plan(user_text: str, plan: dict[str, Any]) -> str:
    """No-op. Returns fingerprint for logging only."""
    return request_fingerprint(user_text or "")


def cache_clear() -> None:
    """No-op."""
    return None
