"""LangGraph turn orchestration for the workflow assistant.

Nodes call existing services (pipeline_analysis, enrichment, ASL, deploy).
The graph decides *which* path to run for arbitrary natural language;
it does not replace mechanical ASL generation or validators.

Routes (from llm_router):
  extract            — new / continuing design from NL
  revise_plan        — structural redesign of current plan
  patch_params       — value-only edits (names, ARNs, TODOs); mechanical ASL regen
  restore_previous   — undo last revision from plan_history
  answer_enrichment  — answer pending enrichment question
  approve            — clear approval of current plan
  out_of_scope       — not about Step Functions workflows
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from orchestrator.api.services.orchestration.state import GraphState
from orchestrator.api.services.orchestration import nodes


def _route_after_router(state: GraphState) -> str:
    return state.get("route") or "extract"


def build_turn_graph():
    """Compile the session turn graph (no checkpointer — session DB is source of truth)."""
    g: StateGraph = StateGraph(GraphState)

    g.add_node("load_context", nodes.load_context)
    g.add_node("route_turn", nodes.route_turn_node)
    g.add_node("extract", nodes.extract_node)
    g.add_node("revise_plan", nodes.revise_plan_node)
    g.add_node("patch_params", nodes.patch_params_node)
    g.add_node("restore_previous", nodes.restore_previous_node)
    g.add_node("answer_enrichment", nodes.answer_enrichment_node)
    g.add_node("approve_plan", nodes.approve_plan_node)

    g.add_edge(START, "load_context")
    g.add_edge("load_context", "route_turn")
    g.add_conditional_edges(
        "route_turn",
        _route_after_router,
        {
            "extract": "extract",
            "revise_plan": "revise_plan",
            "patch_params": "patch_params",
            "restore_previous": "restore_previous",
            "answer_enrichment": "answer_enrichment",
            "approve": "approve_plan",
            "out_of_scope": "extract",
        },
    )
    g.add_edge("extract", END)
    g.add_edge("revise_plan", END)
    g.add_edge("patch_params", END)
    g.add_edge("restore_previous", END)
    g.add_edge("answer_enrichment", END)
    g.add_edge("approve_plan", END)

    return g.compile()


_compiled = None


def get_turn_graph():
    global _compiled
    if _compiled is None:
        _compiled = build_turn_graph()
    return _compiled


# Aliases used by older imports / service layer that expected graph.py to
# expose the pipeline graph factory. Pipeline graph lives in grph.py.
def build_workflow_graph(*, checkpointer: Any = None):
    from orchestrator.api.services.orchestration.grph import (
        build_workflow_graph as _build,
    )

    return _build(checkpointer=checkpointer)


def get_default_graph():
    from orchestrator.api.services.orchestration.grph import get_default_graph as _get

    return _get()


def run_message_turn(session_id: str, message: str) -> dict[str, Any]:
    """Run one user message through the turn graph. Returns GraphState dict."""
    graph = get_turn_graph()
    return graph.invoke(
        {
            "session_id": session_id,
            "user_message": message,
            "action": "message",
        }
    )


def run_reject_turn(session_id: str, feedback: str) -> dict[str, Any]:
    """Explicit reject button — still structural revise (not param patch)."""
    graph = get_turn_graph()
    return graph.invoke(
        {
            "session_id": session_id,
            "user_message": feedback,
            "action": "reject",
            "route": "revise_plan",
        }
    )
