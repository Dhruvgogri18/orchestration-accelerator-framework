"""High-level discovery helpers for LangGraph nodes and services.

These functions sit *above* the raw tools.  A graph node can call
``discover_for_workflow(user_text)`` once and receive a structured
bundle of candidate resources that later stages (component identification,
parameter extraction, enrichment) can inject into prompts — without each
stage inventing its own AWS calls.
"""

from __future__ import annotations

import re
from typing import Any

from .registry import execute_tool
from .schemas import ToolResult


# Lightweight keyword → service hints.  Deterministic; not an ML model.
# NOTE: bare "workflow" must NOT map to states — almost every user prompt
# says "create a workflow" and that was causing us to list state machines
# while skipping Lambda (the actual Task targets).
_SERVICE_HINTS: list[tuple[re.Pattern[str], list[str]]] = [
    (re.compile(r"\b(lambda|function|serverless|handler)\b", re.I), ["lambda"]),
    (re.compile(r"\b(s3|bucket|object storage)\b", re.I), ["s3"]),
    (re.compile(r"\b(dynamo|dynamodb|nosql|table)\b", re.I), ["dynamodb"]),
    (re.compile(r"\b(sqs|queue|fifo)\b", re.I), ["sqs"]),
    (re.compile(r"\b(sns|topic|notification|notify|pub/?sub|email alert)\b", re.I), ["sns"]),
    (re.compile(r"\b(step\s*functions?|state\s*machine)\b", re.I), ["states"]),
    (re.compile(r"\b(glue|etl|crawler)\b", re.I), ["glue"]),
    (re.compile(r"\b(ecs|fargate|container|task definition)\b", re.I), ["ecs"]),
    (re.compile(r"\b(eventbridge|event\s*bridge|cron|schedule)\b", re.I), ["events"]),
    (re.compile(r"\b(kinesis|stream)\b", re.I), ["kinesis"]),
    (re.compile(r"\b(batch\s+job|aws batch)\b", re.I), ["batch"]),
    (re.compile(r"\b(sagemaker|ml endpoint|model endpoint)\b", re.I), ["sagemaker"]),
]

# Action verbs that almost always imply a Lambda (or other compute) Task
_COMPUTE_VERBS = re.compile(
    r"\b(submit|record|process|validate|review|check|approve|reject|"
    r"create|update|delete|fetch|load|transform|send|invoke|call|run)\b",
    re.I,
)

# Always listed for orchestration accelerator (cheap, high value)
_BASELINE_SERVICES = ("lambda", "sns", "sqs")


def infer_services_from_text(text: str) -> list[str]:
    """Return allow-listed service ids mentioned or implied by free text.

    Always includes baseline services (lambda, sns, sqs) so Task targets
    are discovered even when the user never says the word "Lambda".
    """
    found: list[str] = []
    for s in _BASELINE_SERVICES:
        found.append(s)
    for pattern, services in _SERVICE_HINTS:
        if pattern.search(text or ""):
            for s in services:
                if s not in found:
                    found.append(s)
    if _COMPUTE_VERBS.search(text or "") and "lambda" not in found:
        found.insert(0, "lambda")
    return found


def discover_for_workflow(
    user_text: str,
    *,
    region: str | None = None,
    session_id: str | None = None,
    max_per_service: int = 15,
) -> dict[str, Any]:
    """Run a focused discovery pass for a workflow description.

    Uses zero-LLM normalize_prompt for service/name hints when available.

    Returns a bundle suitable for injection into graph state:

      {
        "services_inferred": [...],
        "resources_by_service": { "lambda": [...], ... },
        "search_hits": [...],
        "tool_calls": [ {tool, status, telemetry_id}, ... ],
      }
    """
    name_hints: list[str] = []
    try:
        from .deterministic import handle_normalize_prompt

        norm = handle_normalize_prompt({"text": user_text or ""})
        services = list(norm.get("services") or []) or infer_services_from_text(
            user_text
        )
        name_hints = list(norm.get("name_tokens") or [])[:8]
    except Exception:
        services = infer_services_from_text(user_text)
        tokens = re.findall(
            r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b|\b[a-z]+(?:_[a-z]+){1,}\b|\b[a-z]+(?:-[a-z]+){1,}\b",
            user_text or "",
        )
        name_hints = list(dict.fromkeys(tokens))[:8]

    tool_calls: list[dict[str, Any]] = []
    resources_by_service: dict[str, list[dict]] = {}

    for svc in services:
        filters: dict[str, Any] = {}
        # If we have a strong name hint, bias the list
        if name_hints:
            filters["name_contains"] = name_hints[0][:40]

        result: ToolResult = execute_tool(
            "list_resources",
            {
                "service": svc,
                "region": region,
                "filters": filters,
                "max_results": max_per_service,
            },
            session_id=session_id,
        )
        tool_calls.append(
            {
                "tool": result.tool,
                "status": result.status.value,
                "telemetry_id": result.telemetry_id,
                "service": svc,
            }
        )
        if result.status.value in ("ok", "partial") and result.data.get("resources"):
            resources_by_service[svc] = result.data["resources"]
        elif name_hints:
            # Retry without name filter so we still surface *something*
            result2 = execute_tool(
                "list_resources",
                {
                    "service": svc,
                    "region": region,
                    "max_results": max_per_service,
                },
                session_id=session_id,
            )
            tool_calls.append(
                {
                    "tool": result2.tool,
                    "status": result2.status.value,
                    "telemetry_id": result2.telemetry_id,
                    "service": svc,
                    "note": "unfiltered_fallback",
                }
            )
            if result2.data.get("resources"):
                resources_by_service[svc] = result2.data["resources"]

    # Cross-service search with the raw user text (short queries work best)
    query = (user_text or "").strip()
    if len(query) > 120:
        query = " ".join(query.split()[:18])
    search_result = execute_tool(
        "search_resources",
        {
            "query": query,
            "services": services,
            "region": region,
            "max_results": 15,
        },
        session_id=session_id,
    )
    tool_calls.append(
        {
            "tool": search_result.tool,
            "status": search_result.status.value,
            "telemetry_id": search_result.telemetry_id,
        }
    )

    # Compact list for UI pickers (name + arn + service), capped
    picker_resources: list[dict] = []
    for svc, resources in resources_by_service.items():
        for r in resources or []:
            if not isinstance(r, dict):
                continue
            picker_resources.append(
                {
                    "service": r.get("service") or svc,
                    "name": r.get("name") or r.get("id"),
                    "id": r.get("id"),
                    "arn": r.get("arn"),
                    "type": r.get("resource_type"),
                }
            )
            if len(picker_resources) >= 40:
                break
        if len(picker_resources) >= 40:
            break

    return {
        "services_inferred": services,
        "name_hints": name_hints,
        "resources_by_service": resources_by_service,
        "search_hits": search_result.data.get("results") or [],
        "tool_calls": tool_calls,
        "region": region,
        "picker_resources": picker_resources,
    }


def resolve_many(
    refs: list[dict[str, str]],
    *,
    region: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Resolve a list of {service, hint} pairs. Useful at enrichment time."""
    out: list[dict[str, Any]] = []
    for ref in refs:
        svc = ref.get("service") or ""
        hint = ref.get("hint") or ref.get("name") or ref.get("identifier") or ""
        result = execute_tool(
            "resolve_identifier",
            {"service": svc, "hint": hint, "region": region, "context": ref.get("context")},
            session_id=session_id,
        )
        out.append(
            {
                "input": ref,
                "status": result.status.value,
                "data": result.data,
                "telemetry_id": result.telemetry_id,
            }
        )
    return out
