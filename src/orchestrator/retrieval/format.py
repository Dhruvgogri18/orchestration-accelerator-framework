"""Turn ranked hits into a token-budgeted context block for the LLM."""

from __future__ import annotations

from collections.abc import Callable

from orchestrator.retrieval.models import ScoredHit

TokenCounter = Callable[[str], int]


def _default_counter() -> TokenCounter:
    try:
        from orchestrator.api.services.llm_trace import count_tokens

        return count_tokens
    except Exception:  # pragma: no cover
        return lambda text: max(1, len(text or "") // 4)


def _hit_header(hit: ScoredHit) -> str:
    parts = [hit.path or hit.id]
    if hit.resource_name:
        parts.append(hit.resource_name)
    kind = hit.chunk_kind
    if kind:
        parts.append(kind)
    status = hit.status
    if status and status != "active":
        parts.append(f"status={status}")
    src = "+".join(hit.sources) if hit.sources else "?"
    parts.append(f"score={hit.final_score:.2f}/{src}")
    return " | ".join(parts)


def format_context(
    hits: list[ScoredHit],
    *,
    token_budget: int = 3000,
    count_tokens: TokenCounter | None = None,
) -> tuple[str, list[ScoredHit], bool]:
    """Build a markdown context string.

    Returns (context, hits_included, truncated).
    """
    if not hits:
        return "", [], False

    count = count_tokens or _default_counter()
    preamble = (
        "Repository context (retrieved; prefer these real names/ARNs over invented ones):\n"
    )
    used = count(preamble)
    blocks: list[str] = []
    included: list[ScoredHit] = []
    truncated = False

    for hit in hits:
        body = (hit.text or "").strip()
        if not body:
            continue
        block = f"### {_hit_header(hit)}\n```\n{body}\n```\n"
        cost = count(block)
        if included and used + cost > token_budget:
            truncated = True
            break
        # Always allow at least one hit even if it alone exceeds the budget
        blocks.append(block)
        included.append(hit)
        used += cost

    if not blocks:
        return "", [], False
    return preamble + "\n".join(blocks), included, truncated
