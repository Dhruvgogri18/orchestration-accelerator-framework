"""LangGraph StateGraph for the Prompt2DAG-aligned workflow product.

Control flow (happy path):

  START
    → classify_scope
        ├─ out_of_scope → END
        └─ pipeline_analysis
              ├─ needs_clarification → END   (client sends another /message)
              ├─ out_of_scope → END
              ├─ error → error_handler → END
              └─ ready → render_plan → diagnose_match → plan_review (HITL)
                    ├─ reject → pipeline_analysis   (feedback in messages)
                    ├─ pending → END
                    └─ approve → apply_overrides → enrichment
                          ├─ needs_clarification → enrichment_interrupt (HITL) → …
                          ├─ error → error_handler → END
                          └─ complete → asl
                                ├─ ready → deploy_ready → END
                                └─ failed → error_handler → END

Human-in-the-loop uses interrupt(); the HTTP adapter resumes with
Command(resume=payload) and the same thread_id (session_id).
"""

from __future__ import annotations

from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import (
    apply_overrides_node,
    asl_node,
    classify_scope_node,
    deploy_ready_node,
    diagnose_match_node,
    enrichment_interrupt_node,
    enrichment_node,
    error_handler_node,
    out_of_scope_node,
    pipeline_analysis_node,
    plan_review_interrupt_node,
    render_plan_node,
)
from .state import WorkflowState

# ---------------------------------------------------------------------------
# Pure routing helpers (unit-testable without running the graph)
# ---------------------------------------------------------------------------


def route_after_scope(
    state: WorkflowState,
) -> Literal["pipeline_analysis", "out_of_scope"]:
    if state.get("scope_status") == "out_of_scope":
        return "out_of_scope"
    return "pipeline_analysis"


def route_after_analysis(
    state: WorkflowState,
) -> Literal["render_plan", "needs_clarification", "out_of_scope", "error"]:
    status = state.get("extraction_status")
    if status == "ready":
        return "render_plan"
    if status == "needs_clarification":
        return "needs_clarification"
    if status == "out_of_scope":
        return "out_of_scope"
    return "error"


def route_after_plan_review(
    state: WorkflowState,
) -> Literal["apply_overrides", "reject_revise", "wait_for_human"]:
    decision = state.get("plan_decision")
    if decision == "approve":
        return "apply_overrides"
    if decision == "reject":
        return "reject_revise"
    return "wait_for_human"


def route_after_enrichment(
    state: WorkflowState,
) -> Literal["asl", "enrichment_clarify", "error"]:
    status = state.get("enrichment_status")
    if status == "complete":
        return "asl"
    if status == "needs_clarification":
        return "enrichment_clarify"
    return "error"


def route_after_asl(
    state: WorkflowState,
) -> Literal["ready_to_deploy", "asl_blocked", "asl_failed"]:
    """spec_placeholders block deploy but are not hard errors (user must fill)."""
    if state.get("spec_placeholders"):
        return "asl_blocked"
    if state.get("asl") and not state.get("asl_error"):
        return "ready_to_deploy"
    return "asl_failed"


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------


def build_workflow_graph(*, checkpointer: Any = None):
    """Compile the orchestration graph.

    Parameters
    ----------
    checkpointer:
        LangGraph checkpointer. Defaults to in-memory MemorySaver (dev).
        Production should pass a Postgres/Redis saver keyed by session_id.
    """
    g: StateGraph = StateGraph(WorkflowState)

    # Nodes
    g.add_node("classify_scope", classify_scope_node)
    g.add_node("pipeline_analysis", pipeline_analysis_node)
    g.add_node("render_plan", render_plan_node)
    g.add_node("diagnose_match", diagnose_match_node)
    g.add_node("plan_review", plan_review_interrupt_node)
    g.add_node("apply_overrides", apply_overrides_node)
    g.add_node("enrichment", enrichment_node)
    g.add_node("enrichment_clarify", enrichment_interrupt_node)
    g.add_node("asl", asl_node)
    g.add_node("deploy_ready", deploy_ready_node)
    g.add_node("out_of_scope", out_of_scope_node)
    g.add_node("error_handler", error_handler_node)

    # Entry
    g.add_edge(START, "classify_scope")

    g.add_conditional_edges(
        "classify_scope",
        route_after_scope,
        {
            "pipeline_analysis": "pipeline_analysis",
            "out_of_scope": "out_of_scope",
        },
    )

    g.add_conditional_edges(
        "pipeline_analysis",
        route_after_analysis,
        {
            "render_plan": "render_plan",
            "needs_clarification": END,
            "out_of_scope": "out_of_scope",
            "error": "error_handler",
        },
    )

    g.add_edge("render_plan", "diagnose_match")
    g.add_edge("diagnose_match", "plan_review")

    g.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {
            "apply_overrides": "apply_overrides",
            "reject_revise": "pipeline_analysis",
            "wait_for_human": END,
        },
    )

    g.add_edge("apply_overrides", "enrichment")

    g.add_conditional_edges(
        "enrichment",
        route_after_enrichment,
        {
            "asl": "asl",
            "enrichment_clarify": "enrichment_clarify",
            "error": "error_handler",
        },
    )

    # After answering a clarification, re-enter enrichment routing via a
    # small re-dispatch: enrichment_clarify already wrote enrichment_status.
    g.add_conditional_edges(
        "enrichment_clarify",
        route_after_enrichment,
        {
            "asl": "asl",
            "enrichment_clarify": "enrichment_clarify",  # another question
            "error": "error_handler",
        },
    )

    g.add_conditional_edges(
        "asl",
        route_after_asl,
        {
            "ready_to_deploy": "deploy_ready",
            "asl_blocked": END,  # placeholders remain — client fills then re-runs ASL
            "asl_failed": "error_handler",
        },
    )

    g.add_edge("deploy_ready", END)
    g.add_edge("out_of_scope", END)
    g.add_edge("error_handler", END)

    return g.compile(checkpointer=checkpointer if checkpointer is not None else MemorySaver())


# Module-level default graph for simple imports / tests.
# Routers should prefer build_workflow_graph() with an explicit checkpointer.
default_graph = None


def get_default_graph():
    global default_graph
    if default_graph is None:
        default_graph = build_workflow_graph()
    return default_graph
