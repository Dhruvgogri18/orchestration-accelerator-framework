"""Dynamic resource listing for any allow-listed AWS service.

Each service has a small adapter that knows how to call the right boto3
paginator / list API and how to normalise the response.  Adding a new
service is a single function + registration; the public tool surface
does not change.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable

from ..schemas import ResourceSummary
from . import normalizer as norm
from .allowlist import canonical_service
from .client import get_account_id, get_client, get_default_region

logger = logging.getLogger(__name__)

# Type of a list adapter: (region, filters, max_results) -> list[ResourceSummary]
ListAdapter = Callable[[str, dict[str, Any], int], list[ResourceSummary]]


def _match_name(name: str, filters: dict[str, Any]) -> bool:
    if not name:
        return False
    contains = filters.get("name_contains")
    prefix = filters.get("name_prefix")
    exact = filters.get("name") or filters.get("name_exact")
    if exact and name != exact:
        return False
    if contains and contains.lower() not in name.lower():
        return False
    if prefix and not name.lower().startswith(str(prefix).lower()):
        return False
    return True


def _match_tags(resource_tags: dict[str, str], required: dict[str, str] | None) -> bool:
    if not required:
        return True
    for k, v in required.items():
        if resource_tags.get(k) != v:
            return False
    return True


# ---------------------------------------------------------------------------
# Service adapters
# ---------------------------------------------------------------------------


def _list_lambda(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("lambda", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("list_functions")
    for page in paginator.paginate():
        for fn in page.get("Functions") or []:
            name = fn.get("FunctionName") or ""
            if not _match_name(name, filters):
                continue
            # Tags require a second call; only fetch when tag filter is present
            # or when we still have budget. For list we keep it light.
            summary = norm.normalise_lambda_function(fn, region, account_id)
            if filters.get("runtime") and summary.summary.get("runtime") != filters["runtime"]:
                continue
            out.append(summary)
            if len(out) >= max_results:
                return out
    return out


def _list_s3(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("s3", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.list_buckets()
    for b in resp.get("Buckets") or []:
        name = b.get("Name") or ""
        if not _match_name(name, filters):
            continue
        # Best-effort region; list_buckets does not return it cheaply
        out.append(norm.normalise_s3_bucket(b, region, account_id))
        if len(out) >= max_results:
            break
    return out


def _list_dynamodb(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("dynamodb", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("list_tables")
    for page in paginator.paginate():
        for name in page.get("TableNames") or []:
            if not _match_name(name, filters):
                continue
            out.append(norm.normalise_dynamodb_table(name, region, account_id))
            if len(out) >= max_results:
                return out
    return out


def _list_sqs(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("sqs", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    # list_queues supports QueueNamePrefix
    kwargs: dict[str, Any] = {}
    if filters.get("name_prefix"):
        kwargs["QueueNamePrefix"] = str(filters["name_prefix"])
    resp = client.list_queues(**kwargs) if kwargs else client.list_queues()
    urls = resp.get("QueueUrls") or []
    for url in urls:
        name = url.rstrip("/").split("/")[-1]
        if not _match_name(name, filters):
            continue
        out.append(norm.normalise_sqs_queue(url, region, account_id))
        if len(out) >= max_results:
            break
    return out


def _list_sns(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("sns", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("list_topics")
    for page in paginator.paginate():
        for t in page.get("Topics") or []:
            arn = t.get("TopicArn") or ""
            name = arn.split(":")[-1]
            if not _match_name(name, filters):
                continue
            out.append(norm.normalise_sns_topic(arn, region, account_id))
            if len(out) >= max_results:
                return out
    return out


def _list_states(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("states", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("list_state_machines")
    for page in paginator.paginate():
        for sm in page.get("stateMachines") or []:
            name = sm.get("name") or ""
            if not _match_name(name, filters):
                continue
            out.append(norm.normalise_state_machine(sm, region, account_id))
            if len(out) >= max_results:
                return out
    return out


def _list_glue(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("glue", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    # Glue list_jobs uses NextToken manually
    next_token = None
    while len(out) < max_results:
        kwargs: dict[str, Any] = {"MaxResults": min(100, max_results - len(out))}
        if next_token:
            kwargs["NextToken"] = next_token
        resp = client.list_jobs(**kwargs)
        names = resp.get("JobNames") or []
        for name in names:
            if not _match_name(name, filters):
                continue
            out.append(norm.normalise_glue_job({"Name": name}, region, account_id))
            if len(out) >= max_results:
                break
        next_token = resp.get("NextToken")
        if not next_token:
            break
    return out


def _list_ecs_clusters(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("ecs", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.list_clusters(maxResults=min(100, max_results))
    for arn in resp.get("clusterArns") or []:
        name = arn.split("/")[-1]
        if not _match_name(name, filters):
            continue
        out.append(norm.normalise_ecs_cluster(arn, region, account_id))
        if len(out) >= max_results:
            break
    return out


def _list_events_rules(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("events", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    kwargs: dict[str, Any] = {"Limit": min(100, max_results)}
    if filters.get("name_prefix"):
        kwargs["NamePrefix"] = str(filters["name_prefix"])
    resp = client.list_rules(**kwargs)
    for rule in resp.get("Rules") or []:
        name = rule.get("Name") or ""
        if not _match_name(name, filters):
            continue
        out.append(norm.normalise_eventbridge_rule(rule, region, account_id))
        if len(out) >= max_results:
            break
    return out


def _list_kinesis(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("kinesis", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.list_streams(Limit=min(100, max_results))
    for name in resp.get("StreamNames") or []:
        if not _match_name(name, filters):
            continue
        out.append(
            norm.normalise_generic(
                "kinesis", "stream", name, region, account_id=account_id
            )
        )
        if len(out) >= max_results:
            break
    return out


def _list_batch_queues(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("batch", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.describe_job_queues()
    for q in resp.get("jobQueues") or []:
        name = q.get("jobQueueName") or ""
        if not _match_name(name, filters):
            continue
        out.append(
            norm.normalise_generic(
                "batch",
                "job_queue",
                name,
                region,
                arn=q.get("jobQueueArn"),
                account_id=account_id,
                summary={"state": q.get("state"), "status": q.get("status")},
            )
        )
        if len(out) >= max_results:
            break
    return out


def _list_sagemaker_endpoints(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("sagemaker", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.list_endpoints(MaxResults=min(100, max_results))
    for ep in resp.get("Endpoints") or []:
        name = ep.get("EndpointName") or ""
        if not _match_name(name, filters):
            continue
        out.append(
            norm.normalise_generic(
                "sagemaker",
                "endpoint",
                name,
                region,
                account_id=account_id,
                summary={"status": ep.get("EndpointStatus")},
                last_modified=str(ep.get("LastModifiedTime") or ""),
            )
        )
        if len(out) >= max_results:
            break
    return out


def _list_iam_roles(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    # IAM is global; region is ignored by the service
    client = get_client("iam", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("list_roles")
    for page in paginator.paginate():
        for role in page.get("Roles") or []:
            name = role.get("RoleName") or ""
            if not _match_name(name, filters):
                continue
            out.append(
                norm.normalise_generic(
                    "iam",
                    "role",
                    name,
                    region,
                    arn=role.get("Arn"),
                    account_id=account_id,
                    summary={"path": role.get("Path")},
                    last_modified=str(role.get("CreateDate") or ""),
                )
            )
            if len(out) >= max_results:
                return out
    return out



def _list_firehose(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("firehose", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    resp = client.list_delivery_streams(Limit=min(100, max_results))
    for name in resp.get("DeliveryStreamNames") or []:
        if not _match_name(name, filters):
            continue
        out.append(
            norm.normalise_generic(
                "firehose", "delivery_stream", name, region, account_id=account_id
            )
        )
        if len(out) >= max_results:
            break
    return out


def _list_ecr(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("ecr", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    paginator = client.get_paginator("describe_repositories")
    for page in paginator.paginate():
        for repo in page.get("repositories") or []:
            name = repo.get("repositoryName") or ""
            if not _match_name(name, filters):
                continue
            out.append(
                norm.normalise_generic(
                    "ecr",
                    "repository",
                    name,
                    region,
                    arn=repo.get("repositoryArn"),
                    account_id=account_id,
                    summary={"uri": repo.get("repositoryUri")},
                )
            )
            if len(out) >= max_results:
                return out
    return out


def _list_logs(region: str, filters: dict, max_results: int) -> list[ResourceSummary]:
    client = get_client("logs", region=region)
    account_id = get_account_id(region)
    out: list[ResourceSummary] = []
    kwargs: dict[str, Any] = {"limit": min(50, max_results)}
    if filters.get("name_prefix"):
        kwargs["logGroupNamePrefix"] = str(filters["name_prefix"])
    resp = client.describe_log_groups(**kwargs)
    for g in resp.get("logGroups") or []:
        name = g.get("logGroupName") or ""
        if not _match_name(name, filters):
            continue
        out.append(
            norm.normalise_generic(
                "logs",
                "log_group",
                name,
                region,
                arn=g.get("arn"),
                account_id=account_id,
            )
        )
        if len(out) >= max_results:
            break
    return out


# Registry of adapters. Key = canonical service id.
_ADAPTERS: dict[str, ListAdapter] = {
    "lambda": _list_lambda,
    "s3": _list_s3,
    "dynamodb": _list_dynamodb,
    "sqs": _list_sqs,
    "sns": _list_sns,
    "states": _list_states,
    "glue": _list_glue,
    "ecs": _list_ecs_clusters,
    "events": _list_events_rules,
    "kinesis": _list_kinesis,
    "batch": _list_batch_queues,
    "sagemaker": _list_sagemaker_endpoints,
    "iam": _list_iam_roles,
    "firehose": _list_firehose,
    "ecr": _list_ecr,
    "logs": _list_logs,
}


class ListResourcesResult:
    """Internal result carrying resources plus soft-failure metadata."""

    __slots__ = ("resources", "warning", "adapter_available")

    def __init__(
        self,
        resources: list[ResourceSummary],
        *,
        warning: str | None = None,
        adapter_available: bool = True,
    ):
        self.resources = resources
        self.warning = warning
        self.adapter_available = adapter_available


def list_resources(
    service: str,
    *,
    resource_type: str | None = None,
    region: str | None = None,
    filters: dict[str, Any] | None = None,
    max_results: int = 50,
) -> list[ResourceSummary]:
    """List resources for any registered service.

    Unknown services on the allow-list but without an adapter return an
    empty list (soft-fail) rather than raising — the registry layer surfaces
    the warning to the caller.  Truly disallowed services still raise.
    """
    result = list_resources_detailed(
        service,
        resource_type=resource_type,
        region=region,
        filters=filters,
        max_results=max_results,
    )
    return result.resources


def list_resources_detailed(
    service: str,
    *,
    resource_type: str | None = None,
    region: str | None = None,
    filters: dict[str, Any] | None = None,
    max_results: int = 50,
) -> ListResourcesResult:
    """Like list_resources but includes adapter warning metadata."""
    from .allowlist import is_service_allowed

    svc = canonical_service(service)
    if not is_service_allowed(svc):
        raise ValueError(
            f"Service '{service}' is not on the tools allow-list. "
            f"Allowed: see tools.aws.allowlist.ALLOWED_SERVICES"
        )

    region = region or get_default_region()
    filters = filters or {}
    max_results = max(1, min(int(max_results), 200))

    adapter = _ADAPTERS.get(svc)
    if adapter is None:
        return ListResourcesResult(
            [],
            warning=(
                f"No list adapter for service '{svc}'. "
                f"Adapters available for: {sorted(_ADAPTERS.keys())}. "
                "Use invoke_aws_operation for a raw read-only call if needed."
            ),
            adapter_available=False,
        )

    try:
        results = adapter(region, filters, max_results)
    except Exception as exc:
        logger.warning("list adapter for %s failed: %s", svc, exc)
        return ListResourcesResult(
            [],
            warning=f"list adapter error for '{svc}': {exc}",
            adapter_available=True,
        )

    if resource_type:
        rt = resource_type.lower()
        results = [r for r in results if r.resource_type.lower() == rt]

    tag_filter = filters.get("tags")
    if isinstance(tag_filter, dict) and tag_filter:
        results = [r for r in results if _match_tags(r.tags, tag_filter)]

    return ListResourcesResult(results[:max_results], adapter_available=True)


def supported_list_services() -> list[str]:
    return sorted(_ADAPTERS.keys())
