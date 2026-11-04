"""Normalise heterogeneous AWS API responses into ResourceSummary / ResourceDetail.

Keeping a single shape is what makes the tools dynamic: the LLM (and our
downstream stages) never have to learn per-service response formats.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ..schemas import ResourceDetail, ResourceSummary


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _tags_from_list(tag_list: list | None) -> dict[str, str]:
    if not tag_list:
        return {}
    out: dict[str, str] = {}
    for t in tag_list:
        if isinstance(t, dict):
            k = t.get("Key") or t.get("key")
            v = t.get("Value") or t.get("value")
            if k is not None:
                out[str(k)] = str(v) if v is not None else ""
    return out


def _tags_from_map(tag_map: dict | None) -> dict[str, str]:
    if not tag_map:
        return {}
    return {str(k): str(v) for k, v in tag_map.items()}


# ---------------------------------------------------------------------------
# Per-service normalisers (list path)
# ---------------------------------------------------------------------------


def normalise_lambda_function(raw: dict, region: str, account_id: str | None = None) -> ResourceSummary:
    name = raw.get("FunctionName") or raw.get("functionName") or ""
    arn = raw.get("FunctionArn") or raw.get("FunctionARN")
    return ResourceSummary(
        service="lambda",
        resource_type="function",
        id=name,
        name=name,
        arn=arn,
        region=region,
        account_id=account_id,
        tags=_tags_from_map(raw.get("Tags")) or _tags_from_list(raw.get("tags")),
        summary={
            "runtime": raw.get("Runtime"),
            "handler": raw.get("Handler"),
            "memory_mb": raw.get("MemorySize"),
            "timeout_s": raw.get("Timeout"),
            "package_type": raw.get("PackageType"),
            "architectures": raw.get("Architectures"),
            "last_modified": _iso(raw.get("LastModified")),
            "role": raw.get("Role"),
        },
        last_modified=_iso(raw.get("LastModified")),
    )


def normalise_s3_bucket(raw: dict, region: str, account_id: str | None = None) -> ResourceSummary:
    name = raw.get("Name") or ""
    return ResourceSummary(
        service="s3",
        resource_type="bucket",
        id=name,
        name=name,
        arn=f"arn:aws:s3:::{name}" if name else None,
        region=region,
        account_id=account_id,
        tags={},
        summary={"creation_date": _iso(raw.get("CreationDate"))},
        last_modified=_iso(raw.get("CreationDate")),
    )


def normalise_dynamodb_table(name: str, region: str, account_id: str | None = None, extra: dict | None = None) -> ResourceSummary:
    extra = extra or {}
    arn = extra.get("TableArn")
    return ResourceSummary(
        service="dynamodb",
        resource_type="table",
        id=name,
        name=name,
        arn=arn,
        region=region,
        account_id=account_id,
        tags=_tags_from_list(extra.get("Tags")),
        summary={
            "status": extra.get("TableStatus"),
            "item_count": extra.get("ItemCount"),
            "billing_mode": (extra.get("BillingModeSummary") or {}).get("BillingMode"),
        },
        last_modified=_iso(extra.get("CreationDateTime")),
    )


def normalise_sqs_queue(url: str, region: str, account_id: str | None = None, attrs: dict | None = None) -> ResourceSummary:
    attrs = attrs or {}
    # Queue name is the last path segment of the URL
    name = url.rstrip("/").split("/")[-1] if url else ""
    arn = attrs.get("QueueArn")
    return ResourceSummary(
        service="sqs",
        resource_type="queue",
        id=name or url,
        name=name or url,
        arn=arn,
        region=region,
        account_id=account_id,
        tags={},
        summary={
            "url": url,
            "visibility_timeout": attrs.get("VisibilityTimeout"),
            "approx_messages": attrs.get("ApproximateNumberOfMessages"),
            "fifo": str(attrs.get("FifoQueue", "false")).lower() == "true",
        },
    )


def normalise_sns_topic(arn: str, region: str, account_id: str | None = None) -> ResourceSummary:
    name = arn.split(":")[-1] if arn else ""
    return ResourceSummary(
        service="sns",
        resource_type="topic",
        id=name,
        name=name,
        arn=arn,
        region=region,
        account_id=account_id,
        tags={},
        summary={},
    )


def normalise_state_machine(raw: dict, region: str, account_id: str | None = None) -> ResourceSummary:
    name = raw.get("name") or ""
    arn = raw.get("stateMachineArn")
    return ResourceSummary(
        service="states",
        resource_type="stateMachine",
        id=name,
        name=name,
        arn=arn,
        region=region,
        account_id=account_id,
        tags={},
        summary={
            "type": raw.get("type"),
            "creation_date": _iso(raw.get("creationDate")),
        },
        last_modified=_iso(raw.get("creationDate")),
    )


def normalise_glue_job(raw: dict, region: str, account_id: str | None = None) -> ResourceSummary:
    name = raw.get("Name") or ""
    return ResourceSummary(
        service="glue",
        resource_type="job",
        id=name,
        name=name,
        arn=None,
        region=region,
        account_id=account_id,
        tags={},
        summary={
            "glue_version": raw.get("GlueVersion"),
            "command": (raw.get("Command") or {}).get("Name"),
            "worker_type": raw.get("WorkerType"),
        },
        last_modified=_iso(raw.get("LastModifiedOn") or raw.get("CreatedOn")),
    )


def normalise_ecs_cluster(arn_or_name: str, region: str, account_id: str | None = None) -> ResourceSummary:
    name = arn_or_name.split("/")[-1] if "/" in arn_or_name else arn_or_name
    return ResourceSummary(
        service="ecs",
        resource_type="cluster",
        id=name,
        name=name,
        arn=arn_or_name if arn_or_name.startswith("arn:") else None,
        region=region,
        account_id=account_id,
        tags={},
        summary={},
    )


def normalise_eventbridge_rule(raw: dict, region: str, account_id: str | None = None) -> ResourceSummary:
    name = raw.get("Name") or ""
    return ResourceSummary(
        service="events",
        resource_type="rule",
        id=name,
        name=name,
        arn=raw.get("Arn"),
        region=region,
        account_id=account_id,
        tags={},
        summary={
            "state": raw.get("State"),
            "event_bus": raw.get("EventBusName"),
            "schedule": raw.get("ScheduleExpression"),
        },
    )


def normalise_generic(
    service: str,
    resource_type: str,
    id_: str,
    region: str,
    *,
    name: str | None = None,
    arn: str | None = None,
    account_id: str | None = None,
    tags: dict | None = None,
    summary: dict | None = None,
    last_modified: str | None = None,
) -> ResourceSummary:
    return ResourceSummary(
        service=service,
        resource_type=resource_type,
        id=id_,
        name=name or id_,
        arn=arn,
        region=region,
        account_id=account_id,
        tags=tags or {},
        summary=summary or {},
        last_modified=last_modified,
    )


def to_detail(summary: ResourceSummary, config: dict | None = None, code_location: dict | None = None) -> ResourceDetail:
    return ResourceDetail(
        **summary.model_dump(),
        config=config or {},
        code_location=code_location,
        related=[],
    )
