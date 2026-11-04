"""Tool registry, JSON Schema export, and unified executor.

No result caching — every call hits live AWS (product requirement).
Single-account scope — STS identity is pinned on first use.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from .schemas import (
    AssignCanonicalStepIdsInput,
    ClassifyWorkflowPatternInput,
    DescribeServiceInput,
    ExtractEntitiesInput,
    GetResourceInput,
    GetServiceIntegrationContractInput,
    GroundResourcesInput,
    InvokeAwsOperationInput,
    ListAwsServicesInput,
    ListResourcesInput,
    NormalizePromptInput,
    ResolveIdentifierInput,
    SearchResourcesInput,
    SuggestPlaceholderNamesInput,
    ToolResult,
    ToolStatus,
    ValidatePlanAgainstAwsInput,
    ValidatePlanAgainstRepoInput,
    ValidateResourceReferenceInput,
    SearchRepoContextInput,
    ResolveRepoResourceInput,
    ListRepoArtifactsInput,
)
from .telemetry.metrics import trace_tool_call

logger = logging.getLogger(__name__)

# Sensitive value patterns redacted from telemetry input summaries
_REDACT_KEYS = re.compile(
    r"(secret|password|token|credential|authorization|private_key|access_key)",
    re.I,
)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _REDACT_KEYS.search(str(k)):
                out[k] = "***"
            else:
                out[k] = _redact(v)
        return out
    if isinstance(obj, list):
        return [_redact(x) for x in obj[:50]]
    if isinstance(obj, str) and len(obj) > 200:
        return obj[:200] + "…"
    return obj


def _ensure_single_account(region: str | None) -> dict[str, Any]:
    """Pin / verify the single AWS account. Returns identity meta."""
    from .aws.client import require_identity

    identity = require_identity(region)
    return {
        "account_id": identity.account_id,
        "identity_arn": identity.arn,
        "region": identity.region,
    }


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handle_list_aws_services(args: dict) -> dict:
    from .aws.catalogue import list_catalogue

    inp = ListAwsServicesInput.model_validate(args)
    services = list_catalogue(category=inp.category)
    return {
        "services": [s.model_dump() for s in services],
        "count": len(services),
        "category_filter": inp.category,
        "scope": "single_account",
    }


def _handle_describe_service(args: dict) -> dict:
    from .aws.catalogue import get_service_info
    from .aws.resource_lister import supported_list_services

    inp = DescribeServiceInput.model_validate(args)
    info = get_service_info(inp.service)
    if info is None:
        return {
            "found": False,
            "service": inp.service,
            "supported_list_services": supported_list_services(),
        }
    data = info.model_dump()
    data["found"] = True
    data["list_adapter_available"] = info.id in supported_list_services()
    return data


def _handle_list_resources(args: dict) -> dict:
    from .aws.resource_lister import list_resources_detailed

    inp = ListResourcesInput.model_validate(args)
    detailed = list_resources_detailed(
        inp.service,
        resource_type=inp.resource_type,
        region=inp.region,
        filters=inp.filters,
        max_results=inp.max_results,
    )
    return {
        "service": inp.service,
        "region": inp.region,
        "count": len(detailed.resources),
        "resources": [r.model_dump() for r in detailed.resources],
        "truncated": len(detailed.resources) >= inp.max_results,
        "adapter_available": detailed.adapter_available,
        "warning": detailed.warning,
    }


def _handle_get_resource(args: dict) -> dict:
    from .aws.resource_getter import get_resource

    inp = GetResourceInput.model_validate(args)
    detail = get_resource(
        inp.service,
        inp.identifier,
        region=inp.region,
        include=inp.include,
    )
    if detail is None:
        return {
            "found": False,
            "service": inp.service,
            "identifier": inp.identifier,
        }
    return {
        "found": True,
        "resource": detail.model_dump(),
    }


def _handle_search_resources(args: dict) -> dict:
    from .aws.search import search_resources

    inp = SearchResourcesInput.model_validate(args)
    return search_resources(
        inp.query,
        services=inp.services,
        tags=inp.tags,
        region=inp.region,
        max_results=inp.max_results,
    )


def _handle_resolve_identifier(args: dict) -> dict:
    from .aws.resolver import resolve_identifier

    inp = ResolveIdentifierInput.model_validate(args)
    return resolve_identifier(
        inp.service,
        inp.hint,
        region=inp.region,
        context=inp.context,
    )


def _handle_validate_resource_reference(args: dict) -> dict:
    from .validation.resource_validator import validate_resource_reference

    inp = ValidateResourceReferenceInput.model_validate(args)
    return validate_resource_reference(
        inp.service,
        inp.identifier,
        region=inp.region,
        expected_type=inp.expected_type,
    )


def _handle_invoke_aws_operation(args: dict) -> dict:
    """Controlled escape hatch — allow-listed read-only operations only."""
    from .aws.allowlist import is_operation_allowed, is_service_allowed
    from .aws.client import get_client, get_default_region

    inp = InvokeAwsOperationInput.model_validate(args)
    if not is_service_allowed(inp.service):
        raise PermissionError(f"Service '{inp.service}' is not allow-listed")
    if not is_operation_allowed(inp.operation):
        raise PermissionError(
            f"Operation '{inp.operation}' is not allow-listed "
            "(only list*/describe*/get*/search* are permitted)"
        )

    region = inp.region or get_default_region()
    client = get_client(inp.service, region=region)
    op = getattr(client, inp.operation, None)
    if op is None or not callable(op):
        raise ValueError(
            f"Operation '{inp.operation}' not found on boto3 client for '{inp.service}'"
        )

    raw = op(**(inp.parameters or {}))
    if isinstance(raw, dict):
        raw = {k: v for k, v in raw.items() if k != "ResponseMetadata"}
    serialised = json.loads(json.dumps(raw, default=str))
    text = json.dumps(serialised)
    truncated = False
    if len(text) > 60_000:
        truncated = True
        serialised = {
            "_truncated": True,
            "_original_keys": list(serialised.keys()) if isinstance(serialised, dict) else None,
            "preview": text[:20_000],
        }

    return {
        "service": inp.service,
        "operation": inp.operation,
        "region": region,
        "result": serialised,
        "truncated": truncated,
    }


def _handle_normalize_prompt(args: dict) -> dict:
    from .deterministic import handle_normalize_prompt

    return handle_normalize_prompt(args)


def _handle_extract_entities(args: dict) -> dict:
    from .deterministic import handle_extract_entities

    return handle_extract_entities(args)


def _handle_classify_workflow_pattern(args: dict) -> dict:
    from .deterministic import handle_classify_workflow_pattern

    return handle_classify_workflow_pattern(args)


def _handle_assign_canonical_step_ids(args: dict) -> dict:
    from .deterministic import handle_assign_canonical_step_ids

    return handle_assign_canonical_step_ids(args)


def _handle_suggest_placeholder_names(args: dict) -> dict:
    from .deterministic import handle_suggest_placeholder_names

    return handle_suggest_placeholder_names(args)


def _handle_get_service_integration_contract(args: dict) -> dict:
    from .deterministic import handle_get_service_integration_contract

    return handle_get_service_integration_contract(args)


def _handle_ground_resources(args: dict) -> dict:
    from .deterministic import handle_ground_resources

    return handle_ground_resources(args)


def _handle_validate_plan_against_aws(args: dict) -> dict:
    from .deterministic import handle_validate_plan_against_aws

    return handle_validate_plan_against_aws(args)


def _handle_search_repo_context(args: dict) -> dict:
    from .repo import handle_search_repo_context

    return handle_search_repo_context(args)


def _handle_resolve_repo_resource(args: dict) -> dict:
    from .repo import handle_resolve_repo_resource

    return handle_resolve_repo_resource(args)


def _handle_list_repo_artifacts(args: dict) -> dict:
    from .repo import handle_list_repo_artifacts

    return handle_list_repo_artifacts(args)


def _handle_validate_plan_against_repo(args: dict) -> dict:
    from .repo import handle_validate_plan_against_repo

    return handle_validate_plan_against_repo(args)




# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ToolHandler = Callable[[dict], dict]

_REGISTRY: dict[str, dict[str, Any]] = {
    "list_aws_services": {
        "handler": _handle_list_aws_services,
        "description": (
            "List AWS services known to the orchestration tools layer, "
            "optionally filtered by category (compute, storage, messaging, "
            "orchestration, analytics, integration, security). "
            "Single-account scope."
        ),
        "parameters_model": ListAwsServicesInput,
        "requires_aws": False,
    },
    "describe_service": {
        "handler": _handle_describe_service,
        "description": (
            "Describe a single AWS service: resource types, common operations, "
            "identifier fields, and whether a live list adapter is available."
        ),
        "parameters_model": DescribeServiceInput,
        "requires_aws": False,
    },
    "list_resources": {
        "handler": _handle_list_resources,
        "description": (
            "List resources for ANY allow-listed AWS service (lambda, s3, dynamodb, "
            "sqs, sns, states, glue, ecs, events, kinesis, batch, sagemaker, iam, ...). "
            "Supports filters: name_contains, name_prefix, name, tags, runtime. "
            "Always on-the-fly (no cache). Single AWS account only."
        ),
        "parameters_model": ListResourcesInput,
        "requires_aws": True,
    },
    "get_resource": {
        "handler": _handle_get_resource,
        "description": (
            "Fetch full configuration for one resource by name or ARN. "
            "Works for any service that has a list adapter. Always live."
        ),
        "parameters_model": GetResourceInput,
        "requires_aws": True,
    },
    "search_resources": {
        "handler": _handle_search_resources,
        "description": (
            "Cross-service search by free-text query and/or tags. "
            "Searches high-value orchestration services by default. "
            "Returns scored matches; does not auto-pick ambiguous results."
        ),
        "parameters_model": SearchResourcesInput,
        "requires_aws": True,
    },
    "resolve_identifier": {
        "handler": _handle_resolve_identifier,
        "description": (
            "Resolve a human name, partial name, or ARN fragment to the best "
            "matching live resource. Only returns matched=true when confidence "
            "is high AND unambiguous; otherwise returns alternatives for the "
            "caller to choose. Prefer this over asking the user for exact names."
        ),
        "parameters_model": ResolveIdentifierInput,
        "requires_aws": True,
    },
    "validate_resource_reference": {
        "handler": _handle_validate_resource_reference,
        "description": (
            "Validate that a resource name or ARN exists and is usable. "
            "Returns valid/invalid, resolved ARN, issues, and suggestions. "
            "Call before embedding a resource into a workflow plan or ASL."
        ),
        "parameters_model": ValidateResourceReferenceInput,
        "requires_aws": True,
    },
    "invoke_aws_operation": {
        "handler": _handle_invoke_aws_operation,
        "description": (
            "Escape hatch: call a specific read-only boto3 operation "
            "(list*, describe*, get*, search*) on an allow-listed service. "
            "Destructive operations are blocked. Single-account only."
        ),
        "parameters_model": InvokeAwsOperationInput,
        "requires_aws": True,
    },
    # --- Zero-LLM deterministic tools (no cache; stable shapes) ---
    "normalize_prompt": {
        "handler": _handle_normalize_prompt,
        "description": (
            "Deterministically normalize a workflow description: strip filler, "
            "expand service synonyms, extract name tokens and service hints. "
            "No LLM. Same input always yields the same output."
        ),
        "parameters_model": NormalizePromptInput,
        "requires_aws": False,
    },
    "extract_entities": {
        "handler": _handle_extract_entities,
        "description": (
            "Extract explicit resource names and ARNs from text. Does not invent "
            "values. Pure rules; no LLM."
        ),
        "parameters_model": ExtractEntitiesInput,
        "requires_aws": False,
    },
    "classify_workflow_pattern": {
        "handler": _handle_classify_workflow_pattern,
        "description": (
            "Classify text into a closed workflow pattern set "
            "(validate_parallel_quality_load, map_over_items, choice_notify, "
            "fan_out_parallel, sequential_tasks, unknown). Pure rules; no LLM."
        ),
        "parameters_model": ClassifyWorkflowPatternInput,
        "requires_aws": False,
    },
    "assign_canonical_step_ids": {
        "handler": _handle_assign_canonical_step_ids,
        "description": (
            "Rewrite plan step ids to role-based canonical ids (VALIDATE_1, "
            "WORKER_1, ...). Uses structural + keyword role tagging only — no LLM."
        ),
        "parameters_model": AssignCanonicalStepIdsInput,
        "requires_aws": False,
    },
    "suggest_placeholder_names": {
        "handler": _handle_suggest_placeholder_names,
        "description": (
            "Suggest deterministic TODO_* placeholder names from entities, roles, "
            "and services. No LLM."
        ),
        "parameters_model": SuggestPlaceholderNamesInput,
        "requires_aws": False,
    },
    "get_service_integration_contract": {
        "handler": _handle_get_service_integration_contract,
        "description": (
            "Return static Step Functions integration contract for a service "
            "(ARN pattern, Task Resource, required params). No live AWS, no LLM. "
            "RAG can later back the same contract shape."
        ),
        "parameters_model": GetServiceIntegrationContractInput,
        "requires_aws": False,
    },
    "ground_resources": {
        "handler": _handle_ground_resources,
        "description": (
            "One-shot grounding: normalize text, extract entities, list/search "
            "live resources, resolve name hints. Returns sorted bindings and "
            "unresolved list. Uses AWS read APIs only; no LLM; no cache."
        ),
        "parameters_model": GroundResourcesInput,
        "requires_aws": True,
    },
    "validate_plan_against_aws": {
        "handler": _handle_validate_plan_against_aws,
        "description": (
            "Validate plan step resource references against live AWS. Returns "
            "ordered issues. No LLM."
        ),
        "parameters_model": ValidatePlanAgainstAwsInput,
        "requires_aws": True,
    },
    "search_repo_context": {
        "handler": _handle_search_repo_context,
        "description": (
            "Search the indexed project repository (Chroma RAG) for jobs, "
            "Lambda handlers, Glue scripts, config, and docs relevant to a "
            "query. Prefer these real names over invented ones. No LLM."
        ),
        "parameters_model": SearchRepoContextInput,
        "requires_aws": False,
    },
    "resolve_repo_resource": {
        "handler": _handle_resolve_repo_resource,
        "description": (
            "Resolve a human/partial name to a repository-indexed artifact "
            "(job, Lambda, Glue script, ...). Returns matched=false when "
            "ambiguous. No LLM."
        ),
        "parameters_model": ResolveRepoResourceInput,
        "requires_aws": False,
    },
    "list_repo_artifacts": {
        "handler": _handle_list_repo_artifacts,
        "description": (
            "List repository-indexed artifacts, optionally filtered by kind "
            "(lambda, glue, pipeline, sns, ...). No LLM."
        ),
        "parameters_model": ListRepoArtifactsInput,
        "requires_aws": False,
    },
    "validate_plan_against_repo": {
        "handler": _handle_validate_plan_against_repo,
        "description": (
            "Ground a workflow plan against the repository index: match step "
            "resource tokens to real jobs, Lambdas, Glue scripts. Prefer repo "
            "names over invented ones. No LLM."
        ),
        "parameters_model": ValidatePlanAgainstRepoInput,
        "requires_aws": False,
    },
}


def get_tool_registry() -> dict[str, dict[str, Any]]:
    return dict(_REGISTRY)


def tool_names() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_tool_schemas() -> list[dict[str, Any]]:
    """OpenAI / LangGraph compatible tool definitions."""
    schemas = []
    for name, entry in _REGISTRY.items():
        model = entry["parameters_model"]
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": entry["description"],
                    "parameters": model.model_json_schema(),
                },
            }
        )
    return schemas


def execute_tool(
    name: str,
    arguments: dict[str, Any] | str | None = None,
    *,
    session_id: str | None = None,
) -> ToolResult:
    """Execute a tool by name and return a uniform ToolResult.

    Results are never cached. AWS-backed tools pin a single account identity
    on first use and refuse to continue if the account changes mid-process.
    """
    if name not in _REGISTRY:
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            error=f"Unknown tool '{name}'. Available: {tool_names()}",
            error_code="unknown_tool",
        )

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolResult(
                tool=name,
                status=ToolStatus.ERROR,
                error=f"Invalid JSON arguments: {exc}",
                error_code="invalid_arguments",
            )
    arguments = arguments or {}

    entry = _REGISTRY[name]
    handler: ToolHandler = entry["handler"]
    requires_aws = bool(entry.get("requires_aws", True))

    input_summary = _redact(
        {
            k: (str(v)[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
            for k, v in list(arguments.items())[:12]
        }
    )

    identity_meta: dict[str, Any] = {}
    try:
        with trace_tool_call(name, session_id=session_id, input_summary=input_summary) as ctx:
            if requires_aws:
                try:
                    identity_meta = _ensure_single_account(arguments.get("region"))
                    ctx.meta["account_id"] = identity_meta.get("account_id")
                except Exception as exc:
                    return ToolResult(
                        tool=name,
                        status=ToolStatus.ERROR,
                        error=f"AWS identity unavailable (single-account mode): {exc}",
                        error_code="identity_unavailable",
                        meta={"latency_ms": ctx.meta.get("latency_ms")},
                        telemetry_id=ctx.telemetry_id,
                    )

            data = handler(arguments)

            if isinstance(data, dict) and "count" in data:
                ctx.meta["result_count"] = data["count"]
            elif isinstance(data, dict) and "total" in data:
                ctx.meta["result_count"] = data["total"]
            elif isinstance(data, dict) and "found" in data:
                ctx.meta["found"] = data["found"]
            elif isinstance(data, dict) and "matched" in data:
                ctx.meta["matched"] = data["matched"]
                ctx.meta["ambiguous"] = data.get("ambiguous")

            status = ToolStatus.OK
            if isinstance(data, dict) and data.get("found") is False:
                status = ToolStatus.NOT_FOUND
            if isinstance(data, dict) and data.get("matched") is False:
                status = ToolStatus.NOT_FOUND
            if isinstance(data, dict) and data.get("warning") and not data.get("resources"):
                status = ToolStatus.PARTIAL

            return ToolResult(
                tool=name,
                status=status,
                data=data if isinstance(data, dict) else {"result": data},
                meta={
                    "latency_ms": ctx.meta.get("latency_ms"),
                    "cache_hit": False,  # explicitly never cached
                    "region": arguments.get("region") or identity_meta.get("region"),
                    "account_id": identity_meta.get("account_id"),
                },
                telemetry_id=ctx.telemetry_id,
            )
    except PermissionError as exc:
        return ToolResult(
            tool=name,
            status=ToolStatus.FORBIDDEN,
            error=str(exc),
            error_code="forbidden",
        )
    except ValueError as exc:
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            error=str(exc),
            error_code="bad_request",
        )
    except Exception as exc:
        logger.exception("Tool %s failed", name)
        return ToolResult(
            tool=name,
            status=ToolStatus.ERROR,
            error=str(exc),
            error_code=type(exc).__name__,
        )


def clear_tool_cache() -> None:
    """No-op retained for API compatibility. Results are never cached."""
    return None
