"""Shared models for the synthesis package (plan → ASL → validation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ExtractionStatus = Literal["ready", "needs_clarification", "out_of_scope", "error"]


@dataclass
class SynthesisContext:
    """Everything a synthesis turn needs before calling the LLM."""

    user_message: str
    canonical_request: str = ""
    request_fingerprint: str = ""
    history: list[dict] = field(default_factory=list)
    # Deterministic + AWS + repo bundles
    discovery: dict[str, Any] = field(default_factory=dict)
    repo_context: dict[str, Any] = field(default_factory=dict)
    repo_context_markdown: str = ""
    # Prior session state
    current_plan: dict[str, Any] | None = None
    session_id: str | None = None


@dataclass
class PlanResult:
    status: ExtractionStatus
    plan: dict[str, Any] | None = None
    analysis: dict[str, Any] | None = None
    clarifying_question: str | None = None
    message: str | None = None
    match_tier: str | None = None
    matched_template: str | None = None
    consistency: dict[str, Any] | None = None
    from_cache: bool = False
    error: str | None = None


@dataclass
class AslResult:
    asl: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    validation: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    report: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)
    groups: dict[str, Any] = field(default_factory=dict)
