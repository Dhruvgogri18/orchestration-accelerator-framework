"""Shared LangGraph state for the workflow orchestration graph.

Industry pattern: one typed state object flows through every node.
Mutable fields are updated by returning partial dicts from nodes;
messages use the add_messages reducer so turns accumulate safely.

Plan / analysis are stored as plain dicts (not live Pydantic models)
so checkpointers can serialize state without custom encoders.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Optional

from typing_extensions import TypedDict

try:
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover - older langgraph
    def add_messages(left: list, right: list) -> list:  # type: ignore[misc]
        return (left or []) + (right or [])


ExtractionStatus = Literal["ready", "needs_clarification", "out_of_scope", "error"]
EnrichmentStatus = Literal["complete", "needs_clarification", "error"]
PlanDecision = Literal["approve", "reject", "pending"]


class WorkflowState(TypedDict, total=False):
    """Full graph state. All keys optional so partial updates are valid."""

    # --- session / conversation ---
    session_id: str
    messages: Annotated[list, add_messages]
    user_message: str

    # --- Stage 1: scope + pipeline analysis ---
    scope_status: Optional[Literal["in_scope", "out_of_scope"]]
    extraction_status: Optional[ExtractionStatus]
    analysis: Optional[dict[str, Any]]
    plan: Optional[dict[str, Any]]  # Plan.model_dump()
    clarification_message: Optional[str]
    out_of_scope_message: Optional[str]

    # --- deterministic presentation ---
    summary_markdown: Optional[str]
    diagram_dot: Optional[str]
    placeholders: Optional[list[str]]

    # --- matching diagnostics (never mutates plan) ---
    match_tier: Optional[str]
    match_archetype: Optional[str]
    match_score: Optional[float]
    signature_hash: Optional[str]

    # --- human plan review (HITL) ---
    plan_decision: Optional[PlanDecision]
    plan_overrides: Optional[dict[str, str]]
    reject_feedback: Optional[str]

    # --- Stage 3.5: enrichment ---
    resolved_plan: Optional[dict[str, Any]]
    remaining_placeholders: Optional[list[str]]
    workflow_spec: Optional[dict[str, Any]]
    enrichment_status: Optional[EnrichmentStatus]
    enrichment_question: Optional[str]
    enrichment_error: Optional[str]
    enrichment_pending: Optional[dict[str, Any]]

    # --- Stage 4: ASL ---
    asl: Optional[dict[str, Any]]
    asl_graph_dot: Optional[str]
    asl_validations: Optional[dict[str, Any]]
    asl_error: Optional[str]
    spec_placeholders: Optional[list[str]]

    # --- deploy (thin handoff) ---
    deploy_request: Optional[dict[str, Any]]
    deploy_result: Optional[dict[str, Any]]

    # --- control ---
    error: Optional[str]
    last_stage: Optional[str]
