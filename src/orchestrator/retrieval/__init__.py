"""Repository-context retrieval (Part B).

Dual strategy: dense semantic search + structured metadata lookup, merged and
re-ranked, then formatted under a token budget for synthesis / ASL-first.
"""

from orchestrator.retrieval.models import RetrievalResult, ScoredHit
from orchestrator.retrieval.retriever import (
    hits_as_dicts,
    retrieve,
    retrieve_context,
)

__all__ = [
    "RetrievalResult",
    "ScoredHit",
    "hits_as_dicts",
    "retrieve",
    "retrieve_context",
]
