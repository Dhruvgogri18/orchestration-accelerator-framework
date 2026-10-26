"""Shared types for the dual-strategy retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoredHit:
    """One chunk candidate with scores from one or both strategies."""

    id: str
    text: str
    metadata: dict[str, Any]
    dense_score: float = 0.0  # 1 - cosine_distance, clamped to [0, 1]
    exact_score: float = 0.0  # 0..1 boost from structured matches
    final_score: float = 0.0  # after re-rank blend
    sources: list[str] = field(default_factory=list)  # "dense" | "structured"

    @property
    def chunk_kind(self) -> str:
        return str(self.metadata.get("chunk_kind") or "")

    @property
    def path(self) -> str:
        return str(self.metadata.get("path") or "")

    @property
    def resource_name(self) -> str | None:
        name = self.metadata.get("resource_name")
        return str(name) if name else None

    @property
    def status(self) -> str | None:
        status = self.metadata.get("status")
        return str(status) if status else None


@dataclass
class RetrievalResult:
    """Output of retrieve_context / retrieve."""

    query: str
    hits: list[ScoredHit]
    context: str  # formatted block for the LLM (may be empty)
    dense_count: int = 0
    structured_count: int = 0
    truncated: bool = False  # True if token budget cut the list short
    disabled: bool = False  # True when RETRIEVAL_ENABLED is false
