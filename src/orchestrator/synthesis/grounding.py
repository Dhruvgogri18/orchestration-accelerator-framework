"""Ground a workflow plan against repository context.

Ensures generated plans reference real jobs / Lambdas / Glue scripts from the
indexed repo when retrieval is enabled. Unmatched resources become explicit
TODO_ placeholders rather than invented ARNs — deploy is blocked until resolved.

This is the enterprise counterpart of live-AWS grounding: repo is the source of
truth for *what exists in the project*; AWS tools remain the source of truth
for *what is deployed in the account*.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:[_-][a-z0-9]+)+)\b",
    re.IGNORECASE,
)

# Tokens that look like resources but are generic / not project-specific
_STOP_TOKENS = {
    "step_functions",
    "state_machine",
    "step-functions",
    "state-machine",
    "aws_lambda",
    "aws-lambda",
    "glue_job",
    "glue-job",
    "sns_topic",
    "sqs_queue",
    "s3_bucket",
    "dynamodb_table",
    "entry_point",
    "handler_py",
}


@dataclass
class GroundingHit:
    step_id: str
    token: str
    matched: bool
    artifact: dict[str, Any] | None = None
    confidence: float = 0.0
    alternatives: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class GroundingReport:
    ok: bool
    hits: list[GroundingHit] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    resolved_count: int = 0
    notes: list[str] = field(default_factory=list)
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "enabled": self.enabled,
            "resolved_count": self.resolved_count,
            "unresolved": list(self.unresolved),
            "notes": list(self.notes),
            "hits": [
                {
                    "step_id": h.step_id,
                    "token": h.token,
                    "matched": h.matched,
                    "confidence": h.confidence,
                    "artifact": h.artifact,
                    "alternatives": h.alternatives[:5],
                }
                for h in self.hits
            ],
        }


def _retrieval_on() -> bool:
    try:
        from orchestrator.config import get_settings

        if get_settings().retrieval_enabled:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from orchestrator.api.config.settings import get_api_settings

        return bool(get_api_settings().retrieval_enabled)
    except Exception:  # noqa: BLE001
        return False


def _tokens_from_step(step: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    for key in ("id", "service", "description", "resource", "resource_name", "name"):
        val = step.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(val)
    params = step.get("parameters") or step.get("Parameters") or {}
    if isinstance(params, dict):
        for v in params.values():
            if isinstance(v, str):
                parts.append(v)
    text = " ".join(parts)
    found = []
    seen: set[str] = set()
    for m in _TOKEN_RE.finditer(text):
        tok = m.group(1).lower()
        if tok in _STOP_TOKENS or len(tok) < 5:
            continue
        if tok not in seen:
            seen.add(tok)
            found.append(tok)
    return found


def ground_plan_against_repo(
    plan: dict[str, Any] | None,
    *,
    repo_id: str | None = None,
    require_all: bool = False,
) -> GroundingReport:
    """Match plan step resource tokens against the Chroma repo index.

    When retrieval is disabled, returns ``ok=True, enabled=False`` (no-op).
    When ``require_all`` is True, any unmatched token makes ``ok=False``.
    """
    if not plan or not isinstance(plan, dict):
        return GroundingReport(ok=False, notes=["empty_plan"], enabled=_retrieval_on())

    if not _retrieval_on():
        return GroundingReport(
            ok=True,
            enabled=False,
            notes=["RETRIEVAL_ENABLED is false — repo grounding skipped"],
        )

    try:
        from orchestrator.api.services.tools.repo import handle_resolve_repo_resource
    except ImportError:
        return GroundingReport(
            ok=True,
            enabled=False,
            notes=["repo tools unavailable"],
        )

    steps = plan.get("steps") or []
    hits: list[GroundingHit] = []
    unresolved: list[str] = []
    resolved = 0

    for step in steps:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("id") or "")
        tokens = _tokens_from_step(step)
        if not tokens:
            continue
        for tok in tokens:
            try:
                result = handle_resolve_repo_resource(
                    {"hint": tok, "repo_id": repo_id}
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("resolve_repo_resource(%s) failed: %s", tok, exc)
                hits.append(
                    GroundingHit(step_id=sid, token=tok, matched=False, confidence=0.0)
                )
                unresolved.append(f"{sid}:{tok}")
                continue

            matched = bool(result.get("matched"))
            conf = float(result.get("confidence") or 0.0)
            artifact = result.get("artifact") if matched else None
            alts = list(result.get("alternatives") or [])
            hits.append(
                GroundingHit(
                    step_id=sid,
                    token=tok,
                    matched=matched,
                    artifact=artifact if isinstance(artifact, dict) else None,
                    confidence=conf,
                    alternatives=[a for a in alts if isinstance(a, dict)],
                )
            )
            if matched:
                resolved += 1
            else:
                unresolved.append(f"{sid}:{tok}")

    notes: list[str] = []
    if not hits:
        notes.append("No resource-like tokens found in plan steps to ground.")
    elif unresolved:
        notes.append(
            f"{len(unresolved)} token(s) not found in repo index; "
            "prefer TODO_ placeholders or name real jobs from the repository."
        )

    ok = True
    if require_all and unresolved:
        ok = False
    # Soft fail: if we found tokens but resolved none, flag as not ok when require_all
    if require_all and hits and resolved == 0:
        ok = False

    return GroundingReport(
        ok=ok,
        hits=hits,
        unresolved=unresolved,
        resolved_count=resolved,
        notes=notes,
        enabled=True,
    )


def apply_grounding_to_plan(
    plan: dict[str, Any],
    report: GroundingReport,
) -> dict[str, Any]:
    """Annotate plan steps with resolved repo artifacts (non-destructive copy)."""
    import copy

    out = copy.deepcopy(plan)
    if not report.enabled or not report.hits:
        return out

    by_step: dict[str, list[GroundingHit]] = {}
    for h in report.hits:
        by_step.setdefault(h.step_id, []).append(h)

    for step in out.get("steps") or []:
        if not isinstance(step, dict):
            continue
        sid = str(step.get("id") or "")
        step_hits = by_step.get(sid) or []
        matched = [h for h in step_hits if h.matched and h.artifact]
        if not matched:
            continue
        best = max(matched, key=lambda h: h.confidence)
        art = best.artifact or {}
        step.setdefault("grounding", {})
        step["grounding"] = {
            "resource_name": art.get("resource_name"),
            "resource_type": art.get("resource_type") or art.get("kind"),
            "entry_point": art.get("entry_point"),
            "path": art.get("path"),
            "confidence": best.confidence,
            "token": best.token,
        }
        # Prefer real entry points over free-text when empty
        if art.get("entry_point") and not step.get("entry_point"):
            step["entry_point"] = art["entry_point"]
        if art.get("resource_name") and not step.get("resource_name"):
            step["resource_name"] = art["resource_name"]
    return out
