"""LangGraph-based orchestration for the Prompt2DAG workflow product.

Public surface:
  - build_workflow_graph / get_default_graph  — graph factory
  - WorkflowState                            — shared state type
  - service.send_message / approve_plan / …  — router-facing adapter

The existing pipeline_analysis.session_service remains the default wired
into session_router. Switch the router imports to this package when ready
to cut over; both share session_repository for persistence.
"""

from .graph import build_workflow_graph, get_default_graph
from .state import WorkflowState

__all__ = [
    "WorkflowState",
    "build_workflow_graph",
    "get_default_graph",
]
