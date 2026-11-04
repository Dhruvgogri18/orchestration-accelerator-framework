"""Fetch full detail for a single resource (get_resource tool)."""

from __future__ import annotations

import logging
from typing import Any

from ..schemas import ResourceDetail, ResourceSummary
from . import normalizer as norm
from .allowlist import canonical_service
from .client import get_account_id, get_client, get_default_region
from .resource_lister import list_resources

logger = logging.getLogger(__name__)


def _find_by_hint(service: str, identifier: str, region: str) -> ResourceSummary | None:
    """Resolve a name / ARN / partial to a ResourceSummary via list + match."""
    ident = (identifier or "").strip()
    if not ident:
        return None

    # Exact ARN match path
    if ident.startswith("arn:"):
        # Extract name-ish portion for filtering
        name_part = ident.split(":")[-1].split("/")[-1]
        candidates = list_resources(
            service, region=region, filters={"name_contains": name_part}, max_results=20
        )
        for c in candidates:
            if c.arn == ident or c.id == name_part or c.name == name_part:
                return c
        return None

    # Exact name first
    exact = list_resources(
        service, region=region, filters={"name": ident}, max_results=5
    )
    if exact:
        return exact[0]

    # Prefix / contains
    candidates = list_resources(
        service, region=region, filters={"name_contains": ident}, max_results=20
    )
    for c in candidates:
        if c.id == ident or c.name == ident:
            return c
    if len(candidates) == 1:
        return candidates[0]
    return None


def get_resource(
    service: str,
    identifier: str,
    *,
    region: str | None = None,
    include: list[str] | None = None,
) -> ResourceDetail | None:
    """Return rich detail for one resource, or None if not found."""
    svc = canonical_service(service)
    region = region or get_default_region()
    include = include or ["config", "tags"]
    account_id = get_account_id(region)

    summary = _find_by_hint(svc, identifier, region)
    if summary is None:
        return None

    config: dict[str, Any] = {}
    code_location: dict[str, Any] | None = None

    try:
        if svc == "lambda":
            client = get_client("lambda", region=region)
            if "config" in include or "code_location" in include:
                resp = client.get_function(FunctionName=summary.id)
                config = resp.get("Configuration") or {}
                code = resp.get("Code") or {}
                if "code_location" in include:
                    code_location = {
                        "repository_type": code.get("RepositoryType"),
                        "location": code.get("Location"),
                        "image_uri": code.get("ImageUri"),
                    }
            if "tags" in include and not summary.tags:
                try:
                    tag_resp = client.list_tags(Resource=summary.arn or "")
                    summary.tags = tag_resp.get("Tags") or {}
                except Exception:
                    pass

        elif svc == "s3":
            client = get_client("s3", region=region)
            if "config" in include:
                try:
                    loc = client.get_bucket_location(Bucket=summary.id)
                    config["location"] = loc.get("LocationConstraint") or "us-east-1"
                except Exception as exc:
                    config["location_error"] = str(exc)
            if "tags" in include:
                try:
                    tag_resp = client.get_bucket_tagging(Bucket=summary.id)
                    summary.tags = {
                        t["Key"]: t["Value"] for t in (tag_resp.get("TagSet") or [])
                    }
                except Exception:
                    pass

        elif svc == "dynamodb":
            client = get_client("dynamodb", region=region)
            if "config" in include:
                resp = client.describe_table(TableName=summary.id)
                table = resp.get("Table") or {}
                config = {
                    "table_status": table.get("TableStatus"),
                    "key_schema": table.get("KeySchema"),
                    "attribute_definitions": table.get("AttributeDefinitions"),
                    "billing_mode": (table.get("BillingModeSummary") or {}).get("BillingMode"),
                    "item_count": table.get("ItemCount"),
                    "table_arn": table.get("TableArn"),
                }
                summary.arn = summary.arn or table.get("TableArn")

        elif svc == "sqs":
            client = get_client("sqs", region=region)
            url = (summary.summary or {}).get("url") or summary.id
            if "config" in include:
                resp = client.get_queue_attributes(
                    QueueUrl=url, AttributeNames=["All"]
                )
                config = resp.get("Attributes") or {}
                summary.arn = summary.arn or config.get("QueueArn")
            if "tags" in include:
                try:
                    tag_resp = client.list_queue_tags(QueueUrl=url)
                    summary.tags = tag_resp.get("Tags") or {}
                except Exception:
                    pass

        elif svc == "sns":
            client = get_client("sns", region=region)
            arn = summary.arn or identifier
            if "config" in include:
                resp = client.get_topic_attributes(TopicArn=arn)
                config = resp.get("Attributes") or {}

        elif svc == "states":
            client = get_client("states", region=region)
            arn = summary.arn
            if not arn:
                # Try to construct or re-list
                machines = list_resources("states", region=region, filters={"name": summary.id}, max_results=1)
                if machines:
                    arn = machines[0].arn
            if arn and "config" in include:
                resp = client.describe_state_machine(stateMachineArn=arn)
                config = {
                    "name": resp.get("name"),
                    "status": resp.get("status"),
                    "type": resp.get("type"),
                    "role_arn": resp.get("roleArn"),
                    "creation_date": str(resp.get("creationDate") or ""),
                    # Definition can be large; include only when explicitly requested
                    "definition_length": len(resp.get("definition") or ""),
                }
                if "definition" in include:
                    config["definition"] = resp.get("definition")

        elif svc == "glue":
            client = get_client("glue", region=region)
            if "config" in include:
                resp = client.get_job(JobName=summary.id)
                job = resp.get("Job") or {}
                config = {
                    "glue_version": job.get("GlueVersion"),
                    "command": job.get("Command"),
                    "role": job.get("Role"),
                    "worker_type": job.get("WorkerType"),
                    "number_of_workers": job.get("NumberOfWorkers"),
                    "timeout": job.get("Timeout"),
                }

        elif svc == "iam":
            client = get_client("iam", region=region)
            if "config" in include:
                resp = client.get_role(RoleName=summary.id)
                role = resp.get("Role") or {}
                config = {
                    "arn": role.get("Arn"),
                    "path": role.get("Path"),
                    "assume_role_policy": role.get("AssumeRolePolicyDocument"),
                    "description": role.get("Description"),
                }
                summary.arn = summary.arn or role.get("Arn")

        else:
            # Generic: we only have what list returned
            config = dict(summary.summary or {})

    except Exception as exc:
        logger.warning("get_resource detail fetch failed for %s/%s: %s", svc, identifier, exc)
        config["_detail_error"] = str(exc)

    return norm.to_detail(summary, config=config, code_location=code_location)
