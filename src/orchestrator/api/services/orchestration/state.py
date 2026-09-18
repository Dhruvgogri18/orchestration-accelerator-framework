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


# ---------------------------------------------------------------------------
# Turn-graph state (message / reject routing)
# ---------------------------------------------------------------------------


class GraphState(TypedDict, total=False):
    """State for the session turn graph (route → extract|patch|revise|…)."""

    session_id: str
    user_message: str
    action: Optional[str]  # "message" | "reject"
    route: Optional[str]
    route_rationale: Optional[str]

    # Loaded context
    phase: Optional[str]
    has_plan: Optional[bool]
    has_asl: Optional[bool]
    enrichment_question: Optional[str]
    plan_summary: Optional[str]

    # Extraction / revision outputs (mirror WorkflowState subset)
    extraction_status: Optional[ExtractionStatus]
    analysis: Optional[dict[str, Any]]
    plan: Optional[dict[str, Any]]
    clarification_message: Optional[str]
    out_of_scope_message: Optional[str]
    summary_markdown: Optional[str]
    diagram_dot: Optional[str]
    placeholders: Optional[list[str]]

    # Patch / restore
    patch_overrides: Optional[dict[str, str]]
    patch_rationale: Optional[str]
    is_value_only: Optional[bool]

    # Approve / enrichment side effects (when routed here)
    resolved_plan: Optional[dict[str, Any]]
    remaining_placeholders: Optional[list[str]]
    workflow_spec: Optional[dict[str, Any]]
    enrichment_status: Optional[EnrichmentStatus]
    enrichment_error: Optional[str]
    enrichment_pending: Optional[dict[str, Any]]
    asl: Optional[dict[str, Any]]
    asl_graph_dot: Optional[str]
    asl_validations: Optional[dict[str, Any]]
    asl_error: Optional[str]
    spec_placeholders: Optional[list[str]]

    error: Optional[str]
    last_stage: Optional[str]
    # Carries a MessageResponse-shaped dict for the service adapter
    message_response: Optional[dict[str, Any]]
    approve_response: Optional[dict[str, Any]]
