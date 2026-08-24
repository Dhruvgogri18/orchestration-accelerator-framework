"""The confidence-gated lookup that ties everything together.

    HIGH   (exact signature match)      -> deterministic template snap,
                                            no LLM call needed for structure.
    MEDIUM (score >= MEDIUM_THRESHOLD)  -> nearest template rendered, but
                                            flagged tentative -- in the real
                                            system this is where you'd spend
                                            ONE cheap LLM confirmation call
                                            ("does this fixed structure match
                                            what was described? yes/adjust")
                                            instead of free generation.
    LOW / NONE                          -> no render. Caller MUST fall back
                                            to the existing self-consistency
                                            + plan_canonicalize pipeline.
                                            Logged to the curation queue so
                                            a human can promote it to a new
                                            template later.

Never silently force a low-confidence fit -- that produces a confident,
consistent, WRONG graph, which is worse than falling back.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from . import templates
from .curation_queue import log_unmatched
from orchestrator.api.schemas.common import Plan
from .roles import EmbedFn, LlmClassifyFn
from .signature import plan_signature, signature_hash

MEDIUM_THRESHOLD = 0.70

Tier = Literal["high", "medium", "low", "none"]


@dataclass
class MatchResult:
    tier: Tier
    template_name: str | None
    score: float
    signature_hash: str
    plan: Plan | None  # None means: caller must use its own fallback pipeline

    @property
    def archetype(self) -> str | None:
        """Alias for template_name -- same value, other callers expect this name."""
        return self.template_name


def match(
    plan: Plan,
    embed_fn: EmbedFn = None,
    llm_classify_fn: LlmClassifyFn = None,
    session_id: str | None = None,
    user_message: str | None = None,
) -> MatchResult:
    sig = plan_signature(plan, embed_fn, llm_classify_fn)
    h = signature_hash(plan, embed_fn, llm_classify_fn)

    exact = templates.lookup_exact(sig)
    if exact is not None:
        rendered = templates.render(plan, exact, embed_fn, llm_classify_fn)
        return MatchResult("high", exact.name, 1.0, h, rendered)

    best, score = templates.best_match(sig)
    if best is not None and score >= MEDIUM_THRESHOLD:
        rendered = templates.render(plan, best, embed_fn, llm_classify_fn)
        return MatchResult("medium", best.name, score, h, rendered)

    log_unmatched(
        plan, signature_hash=h, nearest_template=best.name if best else None, score=score,
        session_id=session_id, user_message=user_message,
    )
    return MatchResult("low" if best else "none", best.name if best else None, score, h, None)