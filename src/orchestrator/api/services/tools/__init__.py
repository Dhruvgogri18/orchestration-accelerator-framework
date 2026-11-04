"""Dynamic AWS tooling layer for the Orchestration Accelerator.

This package exposes a small set of *service-agnostic* tools that the LLM
(or any graph node) can call to discover, resolve, and validate AWS
resources.  The design goals are:

  * **Dynamic** – one tool surface covers Lambda, S3, DynamoDB, SQS, SNS,
    Step Functions, Glue, ECS, EventBridge, and any future service added
    to the allow-list.  No per-service tool explosion.
  * **Deterministic** – temperature-0 friendly; aggressive caching;
    stable JSON shapes; allow-listed operations only.
  * **Observable** – every call is traced through the existing llm_trace
    infrastructure plus structured tool metrics (latency,
    error taxonomy, resource counts).
  * **Safe** – read-only by default; explicit allow-lists for services
    and operation prefixes; no IAM mutation, no destructive calls.
  * **RAG-ready** – the same tool contracts will later be backed by a
    code/index store with live AWS as fallback.  Callers do not change.

Public entry points
-------------------
  from orchestrator.api.services.tools import (
      get_tool_registry,
      execute_tool,
      list_tool_schemas,
      discover_for_workflow,
      run_discovery_safe,
      ToolResult,
      get_tool_metrics,
  )

See ``registry.py`` for the full tool catalogue and JSON Schema definitions.
See ``ARCHITECTURE.md`` for the leadership-oriented design brief.
"""

from __future__ import annotations

from .context import (
    format_discovery_for_prompt,
    inject_discovery_into_history,
    run_deterministic_context,
    run_discovery_safe,
)
from .discovery import discover_for_workflow, infer_services_from_text, resolve_many
from .registry import (
    ToolResult,
    clear_tool_cache,
    execute_tool,
    get_tool_registry,
    list_tool_schemas,
    tool_names,
)
from .telemetry import get_tool_metrics
from .auto_resolve import suggest_overrides_for_tokens, suggest_from_plan_and_spec
from .agent import tool_schemas_for_llm, resolve_placeholders_interactive
from .deterministic import (
    handle_assign_canonical_step_ids,
    handle_classify_workflow_pattern,
    handle_extract_entities,
    handle_get_service_integration_contract,
    handle_normalize_prompt,
    handle_suggest_placeholder_names,
)

__all__ = [
    "ToolResult",
    "execute_tool",
    "get_tool_registry",
    "list_tool_schemas",
    "tool_names",
    "clear_tool_cache",
    "discover_for_workflow",
    "infer_services_from_text",
    "resolve_many",
    "get_tool_metrics",
    "format_discovery_for_prompt",
    "inject_discovery_into_history",
    "run_deterministic_context",
    "run_discovery_safe",
    "suggest_overrides_for_tokens",
    "suggest_from_plan_and_spec",
    "tool_schemas_for_llm",
    "resolve_placeholders_interactive",
    "handle_normalize_prompt",
    "handle_extract_entities",
    "handle_classify_workflow_pattern",
    "handle_assign_canonical_step_ids",
    "handle_suggest_placeholder_names",
    "handle_get_service_integration_contract",
]
