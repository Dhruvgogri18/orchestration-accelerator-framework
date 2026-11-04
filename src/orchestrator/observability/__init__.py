"""Observability: OpenTelemetry tracing for GenAI pipeline stages."""

from orchestrator.observability.otel import (
    get_tracer,
    span,
    trace_llm_call,
    trace_retrieval,
    trace_tool,
    configure_otel,
)

__all__ = [
    "configure_otel",
    "get_tracer",
    "span",
    "trace_llm_call",
    "trace_retrieval",
    "trace_tool",
]
