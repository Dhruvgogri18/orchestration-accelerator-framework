"""Lightweight parameter / resource patches — no full pipeline re-run.

Flow:
  1. Optional fast path for explicit ``TODO_X = value`` (no LLM).
  2. Otherwise one small structured LLM call extracts replacements from
     *any* natural-language phrasing (bucket renames, ARN updates, etc.).
  3. Overrides are applied deterministically to plan + workflow_spec;
     ASL is regenerated mechanically when the spec is present.

Structural redesigns are *not* handled here — those stay on revise_plan.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from orchestrator.api.schemas.common import Plan
from orchestrator.api.services.asl_service import find_asl_placeholders
from orchestrator.api.services.plan.render import (
    apply_overrides,
    apply_overrides_to_json,
    find_placeholders,
)
from orchestrator.indexing.gateway_client import chat_completion

# Explicit TODO assignment still handled without LLM (cheap + reliable).
_TODO_ASSIGN = re.compile(
    r"\b(TODO_[A-Z0-9_]+)\b\s*(?:=|:|to|as)\s*[\"']?([^\s\"',;]+(?:\s+[^\s\"',;]+)*)[\"']?",
    re.IGNORECASE,
)

_RESTORE_PREV = re.compile(
    r"\b("
    r"undo(?:\s+the)?\s+last\s+(?:revision|change|edit)|"
    r"undo\s+that|"
    r"go\s+back(?:\s+to\s+(?:the\s+)?previous(?:\s+plan)?)?|"
    r"restore\s+(?:the\s+)?previous(?:\s+plan)?|"
    r"bring\s+back\s+(?:the\s+)?(?:previous|last)(?:\s+plan)?|"
    r"revert(?:\s+to)?\s+(?:the\s+)?previous(?:\s+plan)?|"
    r"previous\s+plan\s+please|"
    r"back\s+to\s+(?:the\s+)?map\s+one|"
    r"use\s+(?:the\s+)?previous\s+plan"
    r")\b",
    re.I,
)


class PatchExtraction(BaseModel):
    """Structured result of the lightweight patch LLM."""

    is_value_only: bool = Field(
        description=(
            "True if the user only changes resource names, ARNs, TODO values, "
            "or other parameters — control flow stays the same. "
            "False if they change steps, parallel/map structure, branching, etc."
        )
    )
    overrides: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of exact placeholder token (e.g. TODO_S3_BUCKET_NAME) or "
            "exact existing string value found in the spec → new value. "
            "Empty if is_value_only is false or nothing to change."
        ),
    )
    rationale: str = Field(default="", description="One short sentence.")


PATCH_SYSTEM = """\
You extract parameter/value edits for an AWS Step Functions workflow assistant.
You do NOT redesign the workflow.

Given the user message and the list of known TODO_* tokens / current values,
decide:

1. is_value_only=true when the user only renames or replaces values
   (bucket, table, topic, queue, Lambda name, ARN, job name, prefix, etc.)
   regardless of how they phrase it.
2. is_value_only=false when they change structure (add/remove steps, Map vs
   Parallel, branching, ordering, new stages).

When is_value_only=true, fill overrides:
- Prefer keys that are exact TODO_* tokens from the known list.
- If the user renames something that already has a concrete value in
  current_values, use that exact old string as the key and the new string
  as the value (so the replacement can be applied by search-and-replace).
- Never invent TODO tokens that are not in known_tokens unless the user
  typed that exact TODO_* token in the message.

Respond with ONLY JSON:
{"is_value_only": true, "overrides": {"TODO_S3_BUCKET_NAME": "my-bucket"}, "rationale": "..."}
"""


def looks_like_restore_previous(message: str) -> bool:
    return bool(_RESTORE_PREV.search(message or ""))


def looks_like_param_only_change(message: str) -> bool:
    """Soft pre-filter only — real decision is LLM-based in resolve_overrides.

    Returns True when the message *might* be value-only so the graph can
    prefer the patch node; the patch node may still fall back to revise.
    """
    text = (message or "").strip()
    if not text:
        return False
    if looks_like_restore_previous(text):
        return False
    if _TODO_ASSIGN.search(text):
        return True
    if re.search(
        r"\b(change|set|update|rename|replace|use|switch|point)\b",
        text,
        re.I,
    ):
        return True
    return False


def collect_known_tokens(
    *,
    plan: Plan | None,
    workflow_spec: dict | None,
    asl: dict | None,
) -> list[str]:
    tokens: list[str] = []
    if plan is not None:
        for t in find_placeholders(plan):
            if t not in tokens:
                tokens.append(t)
    if isinstance(workflow_spec, dict):
        for t in find_asl_placeholders(workflow_spec):
            if t not in tokens:
                tokens.append(t)
    if isinstance(asl, dict):
        for t in find_asl_placeholders(asl):
            if t not in tokens:
                tokens.append(t)
    return tokens


def _collect_string_values(node: Any, out: list[str], *, limit: int = 80) -> None:
    """Gather concrete string values from spec for old→new replacements."""
    if len(out) >= limit:
        return
    if isinstance(node, str):
        s = node.strip()
        if (
            s
            and not s.startswith("TODO_")
            and len(s) < 200
            and s not in out
            and not s.startswith("{")
        ):
            out.append(s)
        return
    if isinstance(node, dict):
        for v in node.values():
            _collect_string_values(v, out, limit=limit)
    elif isinstance(node, list):
        for v in node:
            _collect_string_values(v, out, limit=limit)


def extract_explicit_todo_overrides(message: str, known_tokens: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for m in _TODO_ASSIGN.finditer(message or ""):
        raw = m.group(1)
        token = raw if raw.startswith("TODO_") else f"TODO_{raw}"
        token_u = token.upper()
        matched = next((t for t in known_tokens if t.upper() == token_u), token)
        val = m.group(2).strip().strip("\"'")
        if matched and val:
            overrides[matched] = val
    return overrides


def extract_patch_with_llm(
    message: str,
    *,
    known_tokens: list[str],
    current_values: list[str] | None = None,
    plan_summary: str | None = None,
) -> PatchExtraction:
    """One small structured LLM call — not a full pipeline re-run."""
    payload = {
        "user_message": message,
        "known_tokens": known_tokens,
        "current_values_sample": (current_values or [])[:40],
        "plan_summary_excerpt": (plan_summary or "")[:800],
    }
    messages = [
        {"role": "system", "content": PATCH_SYSTEM},
        {
            "role": "user",
            "content": (
                "Extract value overrides from this context (JSON):\n"
                + json.dumps(payload, ensure_ascii=False)
            ),
        },
    ]
    try:
        raw = chat_completion(messages, temperature=0)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        data = json.loads(text)
        return PatchExtraction.model_validate(data)
    except Exception:  # noqa: BLE001 — caller may fall back to revise
        return PatchExtraction(
            is_value_only=False,
            overrides={},
            rationale="Patch extraction failed; treat as structural revise.",
        )


def resolve_overrides(
    message: str,
    *,
    known_tokens: list[str],
    workflow_spec: dict | None = None,
    plan_summary: str | None = None,
) -> tuple[bool, dict[str, str], str]:
    """Return (is_value_only, overrides, rationale).

    Tries explicit TODO assignments first (no LLM), then one patch LLM call.
    """
    explicit = extract_explicit_todo_overrides(message, known_tokens)
    if explicit and not re.search(
        r"\b(parallel|map over|for each|add step|remove step|branch)\b",
        message or "",
        re.I,
    ):
        return True, explicit, "Explicit TODO assignment(s)."

    current_values: list[str] = []
    if isinstance(workflow_spec, dict):
        _collect_string_values(workflow_spec, current_values)

    result = extract_patch_with_llm(
        message,
        known_tokens=known_tokens,
        current_values=current_values,
        plan_summary=plan_summary,
    )
    normalized: dict[str, str] = {}
    for k, v in (result.overrides or {}).items():
        if not k or v is None or str(v).strip() == "":
            continue
        key = str(k).strip()
        val = str(v).strip()
        match = next((t for t in known_tokens if t.upper() == key.upper()), None)
        normalized[match or key] = val
    return result.is_value_only, normalized, result.rationale or ""


def apply_overrides_to_plan_and_artifacts(
    *,
    plan: Plan | None,
    workflow_spec: dict | None,
    overrides: dict[str, str],
) -> tuple[Plan | None, dict | None, list[str]]:
    """Apply overrides to plan + workflow_spec (token and literal string replace)."""
    new_plan = plan
    if plan is not None and overrides:
        new_plan, _ = apply_overrides(plan, overrides)
        dump = new_plan.model_dump()
        dump = apply_overrides_to_json(dump, overrides)
        try:
            new_plan = Plan.model_validate(dump)
        except Exception:  # noqa: BLE001
            pass

    new_spec = workflow_spec
    remaining: list[str] = []
    if isinstance(workflow_spec, dict):
        new_spec = (
            apply_overrides_to_json(workflow_spec, overrides) if overrides else workflow_spec
        )
        remaining = find_asl_placeholders(new_spec)
    return new_plan, new_spec if isinstance(new_spec, dict) else None, remaining
