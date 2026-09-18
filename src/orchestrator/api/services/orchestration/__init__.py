"""LangGraph-based orchestration for the Prompt2DAG workflow product.

Public surface:
  - build_workflow_graph / get_default_graph  — pipeline graph factory (grph)
  - build_turn_graph / get_turn_graph / run_message_turn / run_reject_turn
  - WorkflowState / GraphState
  - service.send_message / approve_plan / …  — router-facing adapter

session_service.send_message always routes through the turn graph
(patch_params for value-only edits, revise_plan for structural changes).
"""

from .graph import (
    build_turn_graph,
    build_workflow_graph,
    get_default_graph,
    get_turn_graph,
    run_message_turn,
    run_reject_turn,
)
from .state import GraphState, WorkflowState

__all__ = [
    "WorkflowState",
    "GraphState",
    "build_workflow_graph",
    "get_default_graph",
    "build_turn_graph",
    "get_turn_graph",
    "run_message_turn",
    "run_reject_turn",
]
