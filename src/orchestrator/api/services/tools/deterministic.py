"""Zero-LLM deterministic tools for prompt normalization, pattern
classification, entity extraction, canonical step ids, placeholder
naming, integration contracts, resource grounding orchestration, and
plan validation.

Design constraints (product policy):
  * No result caching of any kind.
  * No LLM calls — pure Python / existing AWS tools only.
  * Stable, sorted JSON shapes so the same input always yields the same
    output (temperature-0 friendly, paraphrase-oriented).
  * RAG-ready contracts: later index backends can replace static
    knowledge without changing tool names or schemas.
"""

from __future__ import annotations

import re
from typing import Any

from .schemas import (
    AssignCanonicalStepIdsInput,
    ClassifyWorkflowPatternInput,
    ExtractEntitiesInput,
    GetServiceIntegrationContractInput,
    GroundResourcesInput,
    NormalizePromptInput,
    SuggestPlaceholderNamesInput,
    ValidatePlanAgainstAwsInput,
)

# ---------------------------------------------------------------------------
# Synonyms & stopwords (shared by normalize + entity + pattern tools)
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"\b("
    r"please|kindly|can you|could you|would you|i want to|i need to|"
    r"i'd like to|i would like to|help me|create a|build a|make a|"
    r"design a|set up a|implement a|write a|generate a|"
    r"workflow that|a workflow|the workflow|this workflow"
    r")\b",
    re.I,
)

_SYNONYMS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bstep\s*functions?\b", re.I), "states"),
    (re.compile(r"\bstate\s*machines?\b", re.I), "states"),
    (re.compile(r"\bevent\s*bridges?\b", re.I), "events"),
    (re.compile(r"\bapi\s*gateways?\b", re.I), "apigateway"),
    (re.compile(r"\b(sqs\s+)?queues?\b", re.I), "sqs"),
    (re.compile(r"\b(sns\s+)?topics?\b", re.I), "sns"),
    (re.compile(r"\b(s3\s+)?buckets?\b", re.I), "s3"),
    (re.compile(r"\bdynamo\s*db\b", re.I), "dynamodb"),
    (re.compile(r"\blambda\s+functions?\b", re.I), "lambda"),
    (re.compile(r"\bserverless\s+functions?\b", re.I), "lambda"),
    (re.compile(r"\bglue\s+jobs?\b", re.I), "glue"),
    (re.compile(r"\betl\s+jobs?\b", re.I), "glue"),
    (re.compile(r"\bfargate\b", re.I), "ecs"),
    (re.compile(r"\bcontainers?\b", re.I), "ecs"),
    (re.compile(r"\bkinesis\s+streams?\b", re.I), "kinesis"),
    (re.compile(r"\bsecret\s*manager\b", re.I), "secretsmanager"),
]

_SERVICE_TOKEN_MAP: dict[str, str] = {
    "lambda": "lambda",
    "s3": "s3",
    "dynamodb": "dynamodb",
    "sqs": "sqs",
    "sns": "sns",
    "states": "states",
    "glue": "glue",
    "ecs": "ecs",
    "events": "events",
    "kinesis": "kinesis",
    "batch": "batch",
    "sagemaker": "sagemaker",
    "apigateway": "apigateway",
    "secretsmanager": "secretsmanager",
    "iam": "iam",
    "rds": "rds",
    "athena": "athena",
    "redshift": "redshift",
    "firehose": "firehose",
    "logs": "logs",
    "ssm": "ssm",
}

_NAME_TOKEN_RE = re.compile(
    r"\b[A-Z][a-z]+(?:[A-Z][a-z0-9]+)+\b"  # CamelCase
    r"|\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,}\b"  # snake_case
    r"|\b[a-z][a-z0-9]*(?:-[a-z0-9]+){1,}\b"  # kebab-case
    r"|\barn:aws:[a-z0-9-]+:[^:\s]*:\d{0,12}:[^\s]+"  # ARN
)

_ARN_RE = re.compile(r"arn:aws:[a-z0-9-]+:[^:\s]*:\d{0,12}:[^\s]+", re.I)

# Pattern classifiers (ordered; first match wins when exclusive)
_PATTERN_RULES: list[tuple[str, re.Pattern[str]]] = [
    (
        "map_over_items",
        re.compile(
            r"\b(for each|for every|each (file|item|record|element|object)|"
            r"map over|iterate over|per (file|item|record))\b",
            re.I,
        ),
    ),
    (
        "validate_parallel_quality_load",
        re.compile(
            r"(?=.*\bvalidat)(?=.*\b(parallel|concurrent|simultaneously|at the same time))"
            r"(?=.*\b(quality|dq|load|dynamo|database|table|write|persist))",
            re.I,
        ),
    ),
    (
        "choice_notify",
        re.compile(
            r"(?=.*\b(if|when|choice|decision|approve|reject|route|branch))"
            r"(?=.*\b(notif|sns|alert|email|publish))",
            re.I,
        ),
    ),
    (
        "fan_out_parallel",
        re.compile(
            r"\b(in parallel|concurrently|at the same time|simultaneously|"
            r"two independent|run both|fan.?out)\b",
            re.I,
        ),
    ),
    (
        "sequential_tasks",
        re.compile(
            r"\b(then|after that|next|followed by|step \d|sequentially)\b",
            re.I,
        ),
    ),
]

_STOP_NAME_TOKENS = frozenset(
    {
        "aws",
        "amazon",
        "workflow",
        "step",
        "function",
        "functions",
        "state",
        "machine",
        "please",
        "using",
        "with",
        "from",
        "into",
        "that",
        "this",
        "should",
        "would",
        "could",
        "must",
        "will",
        "when",
        "after",
        "before",
        "success",
        "failure",
        "error",
        "failed",
        "complete",
        "completed",
    }
)


def _apply_synonyms(text: str) -> str:
    out = text or ""
    for pat, repl in _SYNONYMS:
        out = pat.sub(repl, out)
    return out


def _unique_sorted(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        k = x.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return sorted(out, key=lambda s: s.lower())


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_normalize_prompt(args: dict) -> dict:
    """Deterministic text cleanup — zero LLM cost."""
    inp = NormalizePromptInput.model_validate(args)
    raw = (inp.text or "").strip()
    lowered = raw.lower()
    no_filler = _FILLER_RE.sub(" ", lowered)
    normalized = _apply_synonyms(no_filler)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    services: list[str] = []
    for tok, svc in _SERVICE_TOKEN_MAP.items():
        if re.search(rf"\b{re.escape(tok)}\b", normalized, re.I):
            if svc not in services:
                services.append(svc)
    # Baseline high-value orchestration targets when any compute verb present
    if re.search(
        r"\b(process\w*|validat\w*|transform\w*|loads?\b|invoke\w*|runs?\b|submit\w*|check\w*)",
        normalized,
        re.I,
    ):
        for baseline in ("lambda", "sns", "sqs"):
            if baseline not in services:
                services.append(baseline)

    tokens = [
        t
        for t in _NAME_TOKEN_RE.findall(raw)
        if t.lower() not in _STOP_NAME_TOKENS and not t.lower() in _SERVICE_TOKEN_MAP
    ]
    # Stable unique, preserve first-seen order then cap
    seen: set[str] = set()
    name_tokens: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            name_tokens.append(t)
        if len(name_tokens) >= 16:
            break

    return {
        "normalized": normalized,
        "services": services,
        "name_tokens": name_tokens,
        "original_length": len(raw),
        "normalized_length": len(normalized),
    }


def handle_extract_entities(args: dict) -> dict:
    """Rule-based entity extraction — no invention, no LLM."""
    inp = ExtractEntitiesInput.model_validate(args)
    text = inp.text or ""
    entities: list[dict[str, Any]] = []

    for m in _ARN_RE.finditer(text):
        arn = m.group(0)
        svc = "unknown"
        parts = arn.split(":")
        if len(parts) >= 3:
            svc = parts[2]
        entities.append(
            {
                "kind": "arn",
                "value": arn,
                "service_hint": svc if svc in _SERVICE_TOKEN_MAP else svc,
                "source": "explicit",
            }
        )

    for tok in _NAME_TOKEN_RE.findall(text):
        if tok.lower() in _STOP_NAME_TOKENS:
            continue
        if tok.startswith("arn:"):
            continue
        kind = "name"
        service_hint = None
        low = tok.lower()
        if "queue" in low:
            service_hint = "sqs"
        elif "topic" in low or "notif" in low:
            service_hint = "sns"
        elif "bucket" in low:
            service_hint = "s3"
        elif "table" in low or "dynamo" in low:
            service_hint = "dynamodb"
        elif "lambda" in low or "function" in low or "handler" in low:
            service_hint = "lambda"
        elif "job" in low or "glue" in low:
            service_hint = "glue"
        entities.append(
            {
                "kind": kind,
                "value": tok,
                "service_hint": service_hint,
                "source": "mention",
            }
        )

    # Dedupe by value (case-sensitive for ARNs, case-insensitive for names)
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for e in entities:
        key = e["value"] if e["kind"] == "arn" else e["value"].lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(e)

    # Stable sort: arns first, then names by value
    deduped.sort(key=lambda e: (0 if e["kind"] == "arn" else 1, e["value"].lower()))
    return {"entities": deduped, "count": len(deduped)}


def handle_classify_workflow_pattern(args: dict) -> dict:
    """Closed-set pattern classification via regex — zero LLM cost."""
    inp = ClassifyWorkflowPatternInput.model_validate(args)
    text = _apply_synonyms((inp.text or "").strip())
    matched: list[str] = []
    for name, pat in _PATTERN_RULES:
        if pat.search(text):
            matched.append(name)

    if not matched:
        pattern = "unknown"
        confidence = 0.0
    elif len(matched) == 1:
        pattern = matched[0]
        confidence = 0.85
    else:
        # Prefer more specific patterns when multiple fire
        priority = [
            "validate_parallel_quality_load",
            "map_over_items",
            "choice_notify",
            "fan_out_parallel",
            "sequential_tasks",
        ]
        pattern = next((p for p in priority if p in matched), matched[0])
        confidence = 0.7

    return {
        "pattern": pattern,
        "confidence": confidence,
        "candidates": matched,
        "closed_set": [p for p, _ in _PATTERN_RULES] + ["unknown"],
    }


def handle_assign_canonical_step_ids(args: dict) -> dict:
    """Role-based canonical ids from plan topology — keyword tagger only (no LLM)."""
    inp = AssignCanonicalStepIdsInput.model_validate(args)
    plan = inp.plan or {}
    steps = plan.get("steps") or []
    if not isinstance(steps, list) or not steps:
        return {"id_map": {}, "steps": [], "warning": "empty_plan"}

    try:
        from orchestrator.api.schemas.common import Plan
        from orchestrator.api.services.plan.matching.signature import tag_plan
        from orchestrator.api.services.plan.matching.signature import _canonical_ids

        plan_obj = Plan.model_validate(plan)
        # Explicitly no embed_fn / llm_classify_fn → pure keyword + structure
        tags = tag_plan(plan_obj, embed_fn=None, llm_classify_fn=None)
        canon = _canonical_ids(plan_obj, tags)
    except Exception as exc:  # noqa: BLE001 — soft fail, still useful
        # Fallback: ROLE from service heuristics + index
        canon = {}
        tags = {}
        counters: dict[str, int] = {}
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                continue
            sid = str(s.get("id") or f"step_{i}")
            role = "ACTION"
            blob = f"{sid} {s.get('service') or ''} {s.get('description') or ''}".lower()
            if s.get("branches"):
                role = "DECISION" if len(s.get("branches") or []) >= 2 else "FAN_OUT"
            elif any(k in blob for k in ("validat", "verify", "check")):
                role = "VALIDATE"
            elif any(k in blob for k in ("parallel", "concurrent")):
                role = "FAN_OUT"
            elif any(k in blob for k in ("load", "write", "persist", "dynamo")):
                role = "PERSIST"
            elif any(k in blob for k in ("notif", "sns", "alert")):
                role = "NOTIFY_FAILURE" if "fail" in blob else "NOTIFY_SUCCESS"
            counters[role] = counters.get(role, 0) + 1
            canon[sid] = f"{role}_{counters[role]}"
            tags[sid] = role
        return {
            "id_map": canon,
            "role_tags": tags,
            "steps": _rewrite_steps(steps, canon),
            "warning": f"fallback_tagger: {exc}",
        }

    return {
        "id_map": canon,
        "role_tags": tags,
        "steps": _rewrite_steps(steps, canon),
    }


def _rewrite_steps(steps: list, id_map: dict[str, str]) -> list[dict]:
    """Return steps with ids and edge targets rewritten; stable order."""
    out: list[dict] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        old_id = str(s.get("id") or "")
        new = dict(s)
        new["id"] = id_map.get(old_id, old_id)
        if s.get("next"):
            new["next"] = id_map.get(str(s["next"]), s["next"])
        for edge_key in ("on_failure", "on_timeout"):
            if isinstance(s.get(edge_key), list):
                new[edge_key] = [id_map.get(str(x), x) for x in s[edge_key]]
        if isinstance(s.get("branches"), list):
            new_branches = []
            for b in s["branches"]:
                if not isinstance(b, dict):
                    continue
                nb = dict(b)
                if b.get("next"):
                    nb["next"] = id_map.get(str(b["next"]), b["next"])
                new_branches.append(nb)
            new["branches"] = new_branches
        out.append(new)
    return out


def handle_suggest_placeholder_names(args: dict) -> dict:
    """Deterministic TODO_ names from entities / roles — zero LLM."""
    inp = SuggestPlaceholderNamesInput.model_validate(args)
    suggestions: list[dict[str, str]] = []
    seen: set[str] = set()

    def _emit(token: str, reason: str) -> None:
        token = re.sub(r"[^A-Z0-9_]", "_", token.upper())
        token = re.sub(r"_+", "_", token).strip("_")
        if not token.startswith("TODO_"):
            token = f"TODO_{token}"
        if token in seen or token == "TODO_":
            return
        seen.add(token)
        suggestions.append({"placeholder": token, "reason": reason})

    for e in inp.entities or []:
        if not isinstance(e, dict):
            continue
        val = str(e.get("value") or "")
        svc = e.get("service_hint") or "RESOURCE"
        if e.get("kind") == "arn":
            continue  # ARNs are concrete, not placeholders
        body = re.sub(r"[^A-Za-z0-9]+", "_", val).strip("_")
        if not body:
            continue
        _emit(f"{body}_{svc}", "entity")

    for role in inp.roles or []:
        role_s = re.sub(r"[^A-Za-z0-9]+", "_", str(role)).strip("_")
        if role_s:
            _emit(f"{role_s}_RESOURCE", "role")

    for svc in inp.services or []:
        svc_s = re.sub(r"[^A-Za-z0-9]+", "_", str(svc)).strip("_")
        if svc_s:
            _emit(f"{svc_s}_NAME", "service")

    suggestions.sort(key=lambda x: x["placeholder"])
    return {"placeholders": suggestions, "count": len(suggestions)}


# Static Step Functions integration knowledge (RAG can later replace body)
_INTEGRATION_CONTRACTS: dict[str, dict[str, Any]] = {
    "lambda": {
        "service": "lambda",
        "resource_arn_pattern": "arn:aws:lambda:REGION:ACCOUNT:function:FUNCTION_NAME",
        "common_task_resource": "arn:aws:states:::lambda:invoke",
        "required_params": ["FunctionName"],
        "optional_params": ["Payload", "Qualifier"],
        "notes": "Prefer FunctionName or full function ARN in Parameters.",
    },
    "s3": {
        "service": "s3",
        "resource_arn_pattern": "arn:aws:s3:::BUCKET_NAME",
        "common_task_resource": "arn:aws:states:::aws-sdk:s3:getObject",
        "required_params": ["Bucket"],
        "optional_params": ["Key"],
        "notes": "Buckets are global; Key often comes from input.",
    },
    "dynamodb": {
        "service": "dynamodb",
        "resource_arn_pattern": "arn:aws:dynamodb:REGION:ACCOUNT:table/TABLE_NAME",
        "common_task_resource": "arn:aws:states:::dynamodb:putItem",
        "required_params": ["TableName"],
        "optional_params": ["Item", "Key", "ConditionExpression"],
        "notes": "TableName or table ARN; Item keys must match table schema.",
    },
    "sqs": {
        "service": "sqs",
        "resource_arn_pattern": "arn:aws:sqs:REGION:ACCOUNT:QUEUE_NAME",
        "common_task_resource": "arn:aws:states:::sqs:sendMessage",
        "required_params": ["QueueUrl"],
        "optional_params": ["MessageBody", "MessageGroupId"],
        "notes": "QueueUrl preferred over ARN for sendMessage.",
    },
    "sns": {
        "service": "sns",
        "resource_arn_pattern": "arn:aws:sns:REGION:ACCOUNT:TOPIC_NAME",
        "common_task_resource": "arn:aws:states:::sns:publish",
        "required_params": ["TopicArn"],
        "optional_params": ["Message", "Subject"],
        "notes": "TopicArn required for publish.",
    },
    "glue": {
        "service": "glue",
        "resource_arn_pattern": "arn:aws:glue:REGION:ACCOUNT:job/JOB_NAME",
        "common_task_resource": "arn:aws:states:::glue:startJobRun.sync",
        "required_params": ["JobName"],
        "optional_params": ["Arguments", "WorkerType", "NumberOfWorkers"],
        "notes": "Use .sync for job completion wait.",
    },
    "ecs": {
        "service": "ecs",
        "resource_arn_pattern": "arn:aws:ecs:REGION:ACCOUNT:task-definition/FAMILY",
        "common_task_resource": "arn:aws:states:::ecs:runTask.sync",
        "required_params": ["LaunchType", "Cluster", "TaskDefinition"],
        "optional_params": ["Overrides", "NetworkConfiguration"],
        "notes": "Fargate needs NetworkConfiguration subnets.",
    },
    "events": {
        "service": "events",
        "resource_arn_pattern": "arn:aws:events:REGION:ACCOUNT:rule/RULE_NAME",
        "common_task_resource": "arn:aws:states:::events:putEvents",
        "required_params": ["Entries"],
        "optional_params": [],
        "notes": "Often used as workflow trigger, not Task target.",
    },
    "states": {
        "service": "states",
        "resource_arn_pattern": "arn:aws:states:REGION:ACCOUNT:stateMachine:NAME",
        "common_task_resource": "arn:aws:states:::states:startExecution.sync:2",
        "required_params": ["StateMachineArn"],
        "optional_params": ["Input", "Name"],
        "notes": "Nested state machine executions.",
    },
    "batch": {
        "service": "batch",
        "resource_arn_pattern": "arn:aws:batch:REGION:ACCOUNT:job-definition/NAME",
        "common_task_resource": "arn:aws:states:::batch:submitJob.sync",
        "required_params": ["JobName", "JobDefinition", "JobQueue"],
        "optional_params": ["Parameters"],
        "notes": "Use .sync to wait for job completion.",
    },
}


def handle_get_service_integration_contract(args: dict) -> dict:
    """Static integration knowledge — free, RAG can replace later."""
    inp = GetServiceIntegrationContractInput.model_validate(args)
    svc = (inp.service or "").strip().lower()
    # Aliases
    aliases = {
        "stepfunctions": "states",
        "step_functions": "states",
        "step-functions": "states",
        "eventbridge": "events",
        "dynamo": "dynamodb",
    }
    svc = aliases.get(svc, svc)
    if svc:
        contract = _INTEGRATION_CONTRACTS.get(svc)
        if contract is None:
            return {
                "found": False,
                "service": svc,
                "known_services": sorted(_INTEGRATION_CONTRACTS.keys()),
            }
        return {"found": True, **contract}

    # All contracts, sorted by service id
    return {
        "found": True,
        "services": [dict(_INTEGRATION_CONTRACTS[k]) for k in sorted(_INTEGRATION_CONTRACTS)],
        "count": len(_INTEGRATION_CONTRACTS),
    }


def handle_ground_resources(args: dict) -> dict:
    """Orchestrate normalize → extract → list/search/resolve. AWS only, no LLM."""
    from .registry import execute_tool

    inp = GroundResourcesInput.model_validate(args)
    text = inp.text or ""
    region = inp.region
    session_id = inp.session_id

    norm = handle_normalize_prompt({"text": text})
    ents = handle_extract_entities({"text": text})

    services = list(inp.services or norm.get("services") or [])
    if not services:
        services = ["lambda", "sns", "sqs"]
    # stable unique order
    services = list(dict.fromkeys(services))

    name_tokens = list(norm.get("name_tokens") or [])
    for e in ents.get("entities") or []:
        if e.get("kind") == "name" and e.get("value") not in name_tokens:
            name_tokens.append(e["value"])
    name_tokens = name_tokens[:12]

    tool_calls: list[dict[str, Any]] = []
    resources_by_service: dict[str, list] = {}
    bindings: dict[str, Any] = {}
    unresolved: list[dict[str, Any]] = []

    for svc in services:
        filters: dict[str, Any] = {}
        if name_tokens:
            filters["name_contains"] = str(name_tokens[0])[:40]
        result = execute_tool(
            "list_resources",
            {
                "service": svc,
                "region": region,
                "filters": filters,
                "max_results": inp.max_per_service,
            },
            session_id=session_id,
        )
        tool_calls.append(
            {"tool": result.tool, "status": result.status.value, "service": svc}
        )
        resources = []
        if result.status.value in ("ok", "partial") and result.data.get("resources"):
            resources = result.data["resources"]
        elif name_tokens:
            result2 = execute_tool(
                "list_resources",
                {
                    "service": svc,
                    "region": region,
                    "max_results": inp.max_per_service,
                },
                session_id=session_id,
            )
            tool_calls.append(
                {
                    "tool": result2.tool,
                    "status": result2.status.value,
                    "service": svc,
                    "note": "unfiltered_fallback",
                }
            )
            resources = result2.data.get("resources") or []
        resources_by_service[svc] = resources

    # Resolve explicit name tokens per inferred service_hint
    for e in ents.get("entities") or []:
        if e.get("kind") != "name":
            continue
        hint = e["value"]
        svc = e.get("service_hint") or (services[0] if services else "lambda")
        res = execute_tool(
            "resolve_identifier",
            {"service": svc, "hint": hint, "region": region, "context": text[:200]},
            session_id=session_id,
        )
        tool_calls.append(
            {"tool": res.tool, "status": res.status.value, "hint": hint, "service": svc}
        )
        data = res.data or {}
        key = f"{svc}:{hint}"
        if data.get("matched"):
            bindings[key] = {
                "service": svc,
                "hint": hint,
                "matched": True,
                "arn": data.get("arn") or (data.get("resource") or {}).get("arn"),
                "name": (data.get("resource") or {}).get("name") or data.get("name"),
                "confidence": data.get("confidence"),
            }
        else:
            unresolved.append(
                {
                    "service": svc,
                    "hint": hint,
                    "alternatives": data.get("alternatives") or [],
                }
            )

    # ARNs are already concrete bindings
    for e in ents.get("entities") or []:
        if e.get("kind") != "arn":
            continue
        arn = e["value"]
        svc = e.get("service_hint") or "unknown"
        key = f"{svc}:{arn}"
        bindings[key] = {
            "service": svc,
            "hint": arn,
            "matched": True,
            "arn": arn,
            "name": arn.split(":")[-1] if ":" in arn else arn,
            "confidence": 1.0,
        }

    # Sorted bindings for stability
    sorted_bindings = {k: bindings[k] for k in sorted(bindings.keys())}
    unresolved.sort(key=lambda u: (u.get("service") or "", u.get("hint") or ""))

    return {
        "normalized": norm.get("normalized"),
        "services": services,
        "name_tokens": name_tokens,
        "entities": ents.get("entities") or [],
        "resources_by_service": {
            k: resources_by_service[k] for k in sorted(resources_by_service.keys())
        },
        "bindings": sorted_bindings,
        "unresolved": unresolved,
        "tool_calls": tool_calls,
        "region": region,
    }


def handle_validate_plan_against_aws(args: dict) -> dict:
    """Validate plan step resource hints against live AWS — no LLM."""
    from .registry import execute_tool

    inp = ValidatePlanAgainstAwsInput.model_validate(args)
    plan = inp.plan or {}
    steps = plan.get("steps") or []
    region = inp.region
    session_id = inp.session_id

    issues: list[dict[str, Any]] = []
    checked: list[dict[str, Any]] = []

    for s in steps:
        if not isinstance(s, dict):
            continue
        sid = str(s.get("id") or "")
        service = (s.get("service") or "").strip().lower()
        if not service or service in ("none", "n/a", "null", "control", "pass", "terminal"):
            continue
        # Prefer explicit resource fields, else description tokens
        identifier = (
            s.get("resource")
            or s.get("resource_arn")
            or s.get("function_name")
            or s.get("arn")
        )
        if not identifier:
            # Skip pure descriptive steps without a concrete ref
            desc = s.get("description") or ""
            if "TODO_" in str(desc) or "TODO_" in sid:
                issues.append(
                    {
                        "step_id": sid,
                        "service": service,
                        "severity": "warning",
                        "code": "unresolved_placeholder",
                        "message": "Step still contains a TODO_ placeholder",
                    }
                )
            continue

        identifier = str(identifier)
        if identifier.startswith("TODO_"):
            issues.append(
                {
                    "step_id": sid,
                    "service": service,
                    "severity": "warning",
                    "code": "unresolved_placeholder",
                    "message": f"Placeholder not resolved: {identifier}",
                }
            )
            continue

        result = execute_tool(
            "validate_resource_reference",
            {
                "service": service,
                "identifier": identifier,
                "region": region,
                "expected_type": s.get("resource_type"),
            },
            session_id=session_id,
        )
        data = result.data or {}
        entry = {
            "step_id": sid,
            "service": service,
            "identifier": identifier,
            "valid": bool(data.get("valid")),
            "arn": data.get("arn") or data.get("resolved_arn"),
        }
        checked.append(entry)
        if not entry["valid"]:
            issues.append(
                {
                    "step_id": sid,
                    "service": service,
                    "severity": "error",
                    "code": "invalid_reference",
                    "message": data.get("message")
                    or data.get("error")
                    or "Resource reference did not validate",
                    "suggestions": data.get("suggestions") or [],
                }
            )

    issues.sort(key=lambda i: (i.get("severity") or "", i.get("step_id") or ""))
    checked.sort(key=lambda c: c.get("step_id") or "")
    return {
        "valid": not any(i.get("severity") == "error" for i in issues),
        "checked": checked,
        "issues": issues,
        "checked_count": len(checked),
        "issue_count": len(issues),
    }
