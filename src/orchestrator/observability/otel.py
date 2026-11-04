"""OpenTelemetry instrumentation for the Orchestration Accelerator.

Every AI call, retrieval call, tool call, and LangGraph node should open a
span so latency, token usage, and errors are visible in any OTel backend
(Jaeger, Grafana Tempo, Honeycomb, Cloud Trace, …).

Configuration (all optional — safe no-ops when OTel SDK is absent):

  OTEL_SERVICE_NAME=orchestration-accelerator
  OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
  OTEL_TRACES_EXPORTER=otlp|console|none   (default: none if unset)
  OTEL_ENABLED=1                          (default: 1 when SDK installed)

Design:
  * Never raises into product code — tracing failures are swallowed.
  * Works without the opentelemetry packages (noop tracer).
  * Attributes follow GenAI semantic conventions where applicable:
      gen_ai.system, gen_ai.request.model, gen_ai.usage.input_tokens, …
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger(__name__)

_CONFIGURED = False
_TRACER = None


def _otel_enabled() -> bool:
    flag = os.getenv("OTEL_ENABLED", "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return False
    return True


def configure_otel(*, force: bool = False) -> None:
    """Idempotent setup of TracerProvider + exporter."""
    global _CONFIGURED, _TRACER
    if _CONFIGURED and not force:
        return
    _CONFIGURED = True

    if not _otel_enabled():
        _TRACER = _NoopTracer()
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    except ImportError:
        logger.info("opentelemetry not installed — using noop tracer")
        _TRACER = _NoopTracer()
        return

    service = os.getenv("OTEL_SERVICE_NAME", "orchestration-accelerator")
    resource = Resource.create(
        {
            "service.name": service,
            "service.namespace": "oaf",
        }
    )
    provider = TracerProvider(resource=resource)

    exporter_kind = (os.getenv("OTEL_TRACES_EXPORTER") or "none").strip().lower()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()

    if exporter_kind == "otlp" or endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )

            exp = OTLPSpanExporter(endpoint=endpoint or None)
            provider.add_span_processor(BatchSpanProcessor(exp))
        except Exception as exc:  # noqa: BLE001
            logger.warning("OTLP exporter setup failed: %s — falling back to console", exc)
            provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    elif exporter_kind == "console":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
    # "none" → no processor (spans exist in-process only)

    trace.set_tracer_provider(provider)
    _TRACER = trace.get_tracer("orchestrator", "1.0.0")


def get_tracer():
    if _TRACER is None:
        configure_otel()
    return _TRACER


class _NoopSpan:
    def set_attribute(self, *args, **kwargs):
        return None

    def set_status(self, *args, **kwargs):
        return None

    def record_exception(self, *args, **kwargs):
        return None

    def add_event(self, *args, **kwargs):
        return None

    def end(self, *args, **kwargs):
        return None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _NoopTracer:
    def start_as_current_span(self, name: str, **kwargs):
        return _NoopSpan()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    """Open a named span; attributes are set best-effort."""
    tracer = get_tracer()
    try:
        with tracer.start_as_current_span(name) as s:
            for k, v in attributes.items():
                if v is None:
                    continue
                try:
                    if isinstance(v, (str, int, float, bool)):
                        s.set_attribute(k, v)
                    else:
                        s.set_attribute(k, str(v)[:512])
                except Exception:  # noqa: BLE001
                    pass
            t0 = time.perf_counter()
            try:
                yield s
            except Exception as exc:
                try:
                    s.record_exception(exc)
                    from opentelemetry.trace import Status, StatusCode

                    s.set_status(Status(StatusCode.ERROR, str(exc)[:200]))
                except Exception:  # noqa: BLE001
                    pass
                raise
            finally:
                try:
                    s.set_attribute("duration_ms", int((time.perf_counter() - t0) * 1000))
                except Exception:  # noqa: BLE001
                    pass
    except Exception:
        # Never break product flow because of tracing
        yield _NoopSpan()


@contextmanager
def trace_llm_call(
    *,
    model: str | None = None,
    stage: str = "llm",
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> Iterator[Any]:
    attrs: dict[str, Any] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.system": "openai_compatible_gateway",
        "oaf.stage": stage,
    }
    if model:
        attrs["gen_ai.request.model"] = model
    if input_tokens is not None:
        attrs["gen_ai.usage.input_tokens"] = input_tokens
    if output_tokens is not None:
        attrs["gen_ai.usage.output_tokens"] = output_tokens
    with span(f"llm.{stage}", **attrs) as s:
        yield s


@contextmanager
def trace_retrieval(*, query: str = "", hit_count: int | None = None) -> Iterator[Any]:
    attrs: dict[str, Any] = {"oaf.stage": "retrieval"}
    if query:
        attrs["retrieval.query"] = query[:200]
    if hit_count is not None:
        attrs["retrieval.hit_count"] = hit_count
    with span("retrieval.repo", **attrs) as s:
        yield s


@contextmanager
def trace_tool(name: str, **attributes: Any) -> Iterator[Any]:
    attrs = {"oaf.stage": "tool", "tool.name": name, **attributes}
    with span(f"tool.{name}", **attrs) as s:
        yield s
