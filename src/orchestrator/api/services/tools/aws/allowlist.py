"""Safety allow-lists for services and operations.

Only services and operation prefixes listed here may be invoked by the
dynamic tools.  This is the primary hard control that keeps the tooling
layer read-only and predictable for enterprise review.

Scope: **single AWS account** (the account of the runtime credentials).
Multi-account / cross-account role assumption is intentionally out of scope.
"""

from __future__ import annotations

from typing import Final

ALLOWED_SERVICES: Final[frozenset[str]] = frozenset(
    {
        "lambda",
        "s3",
        "dynamodb",
        "sqs",
        "sns",
        "states",
        "glue",
        "ecs",
        "events",
        "kinesis",
        "firehose",
        "apigateway",
        "apigatewayv2",
        "iam",
        "sts",
        "resourcegroupstaggingapi",
        "resource-explorer-2",
        "cloudformation",
        "logs",
        "secretsmanager",
        "ssm",
        "rds",
        "elasticache",
        "opensearch",
        "athena",
        "redshift",
        "batch",
        "ecr",
        "eks",
        "sagemaker",
    }
)

SERVICE_ALIASES: Final[dict[str, str]] = {
    "stepfunctions": "states",
    "step-functions": "states",
    "step_functions": "states",
    "eventbridge": "events",
    "event-bridge": "events",
    "api_gateway": "apigateway",
    "api-gateway": "apigateway",
    "resource_explorer": "resource-explorer-2",
    "resourceexplorer": "resource-explorer-2",
    "tagging": "resourcegroupstaggingapi",
}

SAFE_OPERATION_PREFIXES: Final[tuple[str, ...]] = (
    "list",
    "describe",
    "get",
    "search",
    "batch_get",
    "select",
)

DENIED_OPERATIONS: Final[frozenset[str]] = frozenset(
    {
        "delete_function",
        "delete_bucket",
        "delete_table",
        "delete_queue",
        "delete_topic",
        "delete_state_machine",
        "delete_job",
        "put_object",
        "put_item",
        "update_item",
        "create_function",
        "create_bucket",
        "create_table",
        "create_queue",
        "create_topic",
        "create_state_machine",
        "update_state_machine",
        "start_execution",
        "stop_execution",
        "invoke",
        "publish",
        "send_message",
        "put_events",
        "put_record",
        "put_records",
        "update_function_code",
        "update_function_configuration",
        "tag_resource",
        "untag_resource",
    }
)

PRIORITY_SERVICES: Final[tuple[str, ...]] = (
    "lambda",
    "s3",
    "dynamodb",
    "sqs",
    "sns",
    "states",
    "glue",
    "ecs",
    "events",
)

_DANGEROUS_STARTS: Final[tuple[str, ...]] = (
    "update",
    "delete",
    "create",
    "put",
    "send",
    "publish",
    "invoke",
    "remove",
    "detach",
    "attach",
)


def canonical_service(service: str) -> str:
    """Normalise user/LLM-provided service name to our canonical id."""
    s = (service or "").strip().lower().replace("_", "-")
    if s in ("states", "stepfunctions", "step-functions"):
        return "states"
    if s in SERVICE_ALIASES:
        return SERVICE_ALIASES[s]
    return s


def is_service_allowed(service: str) -> bool:
    c = canonical_service(service)
    if c in ALLOWED_SERVICES:
        return True
    raw = (service or "").strip().lower()
    return raw in ALLOWED_SERVICES or raw in SERVICE_ALIASES


def is_operation_allowed(operation: str) -> bool:
    op = (operation or "").strip()
    if not op:
        return False
    op_lower = op.lower()
    if op_lower in DENIED_OPERATIONS:
        return False
    if any(op_lower.startswith(p) for p in _DANGEROUS_STARTS):
        # Still allow get/list/describe/search even if name is weird
        if not any(op_lower.startswith(p) for p in SAFE_OPERATION_PREFIXES):
            return False
    return any(op_lower.startswith(p) for p in SAFE_OPERATION_PREFIXES)


def boto3_service_name(service: str) -> str:
    """Map our canonical id to the name boto3.client() expects."""
    c = canonical_service(service)
    if c == "states":
        return "stepfunctions"
    if c == "resource-explorer-2":
        return "resource-explorer-2"
    return c
