"""Derive a user-facing Plan from a draft ASL document -- in code, no LLM.

The ASL is the source of truth in the ASL-first pipeline; the Plan is only
a *view* of it, so the plan the user approves can never drift from the
ASL it describes. The result is the same flat Plan/PlanStep shape the
staged pipeline produces, so render_plan_summary / render_plan_diagram
and the UI work unchanged:

  * Parallel / Map containers get ``branches`` pointing at the first step
    of each (flattened) nested chain, matching enrichment's convention.
  * Choice states get one ``PlanBranch`` per rule plus an "otherwise" arm.
  * Catch handlers become ``on_failure`` (or ``on_timeout`` for
    States.Timeout-only catches).
  * TODO_ tokens never leak into descriptions (they are asked for later,
    in the placeholder step).
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any

from orchestrator.api.schemas.common import Plan, PlanBranch, PlanStep
from orchestrator.api.services.asl_service import (
    _PLACEHOLDER_PATTERN,
    _service_key_from_resource,
)

# Short service names, matching what the staged plan uses ("Lambda", "SNS").
_SERVICE_NAMES = {
    "lambda": "Lambda", "sns": "SNS", "sqs": "SQS", "s3": "S3",
    "dynamodb": "DynamoDB", "glue": "Glue", "ecs": "ECS", "eks": "EKS",
    "athena": "Athena", "sagemaker": "SageMaker", "bedrock": "Bedrock",
    "events": "EventBridge", "eventbridge": "EventBridge", "ses": "SES",
    "sesv2": "SES", "batch": "Batch", "kinesis": "Kinesis",
    "firehose": "Firehose", "states": "Step Functions",
    "stepfunctions": "Step Functions", "emr": "EMR", "codebuild": "CodeBuild",
    "apigateway": "API Gateway", "secretsmanager": "Secrets Manager",
    "ssm": "Systems Manager", "textract": "Textract", "comprehend": "Comprehend",
    "rekognition": "Rekognition", "sfn": "Step Functions",
}

_COMPONENT_BY_TYPE = {
    "Choice": "DecisionGate",
    "Parallel": "Orchestrator",
    "Map": "Orchestrator",
    "Succeed": "Terminal",
    "Fail": "Terminal",
}


def humanize_tokens(text: str) -> str:
    """TODO_OPS_SNS_TOPIC -> <ops sns topic> (display only)."""
    return _PLACEHOLDER_PATTERN.sub(
        lambda m: "<" + m.group(0)[len("TODO_"):].lower().replace("_", " ") + ">",
        text or "",
    )


def _short_condition(cond: Any) -> str:
    text = str(cond or "").replace("{%", " ").replace("%}", " ")
    text = re.sub(r"\s+", " ", humanize_tokens(text)).strip()
    if not text:
        return "condition"
    return text if len(text) <= 48 else text[:47] + "…"


def _service_label(state: dict) -> str:
    stype = state.get("Type")
    if stype == "Task":
        key = _service_key_from_resource(str(state.get("Resource") or ""))
        if not key:
            return "AWS"
        return _SERVICE_NAMES.get(key, key.replace("-", " ").title())
    if stype == "Choice":
        return "None"
    return "Step Functions"


def _targets(state: dict) -> list[str]:
    out: list[str] = []
    for key in ("Next", "Default"):
        if isinstance(state.get(key), str):
            out.append(state[key])
    for ch in state.get("Choices") or []:
        if isinstance(ch, dict) and isinstance(ch.get("Next"), str):
            out.append(ch["Next"])
    for c in state.get("Catch") or []:
        if isinstance(c, dict) and isinstance(c.get("Next"), str):
            out.append(c["Next"])
    return out


def _order(states: dict, start_at: str | None) -> list[str]:
    """Reading order: breadth-first from StartAt, then anything unreached."""
    seen: list[str] = []
    queue: deque[str] = deque([start_at] if start_at in states else [])
    while queue:
        n = queue.popleft()
        if n in seen or n not in states:
            continue
        seen.append(n)
        queue.extend(t for t in _targets(states[n]) if t not in seen)
    seen.extend(n for n in states if n not in seen)
    return seen


def _make_step(pid: str, state: dict, idmap: dict[str, str]) -> PlanStep:
    stype = state.get("Type") or "Task"
    desc = humanize_tokens(str(state.get("Comment") or "")).strip() or f"{stype} step"

    nxt = state.get("Next")
    next_id = idmap.get(nxt) if isinstance(nxt, str) else None

    on_failure: list[str] = []
    on_timeout: list[str] = []
    for c in state.get("Catch") or []:
        if not isinstance(c, dict):
            continue
        target = idmap.get(c.get("Next"))
        if not target:
            continue
        errs = c.get("ErrorEquals") or []
        bucket = (
            on_timeout
            if errs == ["States.Timeout"]
            else on_failure
        )
        if target not in bucket:
            bucket.append(target)

    branches: list[PlanBranch] = []
    if stype == "Choice":
        for ch in state.get("Choices") or []:
            if not isinstance(ch, dict):
                continue
            target = idmap.get(ch.get("Next"))
            if target:
                label = humanize_tokens(str(ch.get("Comment") or "")).strip()
                branches.append(
                    PlanBranch(
                        condition=label or _short_condition(ch.get("Condition")),
                        next=target,
                    )
                )
        default = idmap.get(state.get("Default"))
        if default:
            branches.append(PlanBranch(condition="otherwise", next=default))

    return PlanStep(
        id=pid,
        service=_service_label(state),
        description=desc,
        next=next_id,
        on_failure=on_failure,
        on_timeout=on_timeout,
        branches=branches,
        component_type=_COMPONENT_BY_TYPE.get(stype, "Other"),
    )


def _emit_scope(
    states: dict,
    start_at: str | None,
    owner: str,
    out: list[PlanStep],
    used: set[str],
) -> str | None:
    """Flatten one States map (and, recursively, any nested ones) into
    ``out``. Returns the plan id of this scope's start state."""
    order = _order(states, start_at)
    idmap: dict[str, str] = {}
    for name in order:
        pid = name if name not in used else f"{owner}_{name}"
        base, i = pid, 2
        while pid in used:
            pid, i = f"{base}_{i}", i + 1
        used.add(pid)
        idmap[name] = pid

    for name in order:
        state = states[name]
        if not isinstance(state, dict):
            continue
        step = _make_step(idmap[name], state, idmap)
        out.append(step)
        stype = state.get("Type")
        if stype == "Parallel":
            for i, branch in enumerate(state.get("Branches") or [], start=1):
                if not isinstance(branch, dict):
                    continue
                first = _emit_scope(
                    branch.get("States") or {},
                    branch.get("StartAt"),
                    idmap[name],
                    out,
                    used,
                )
                if first:
                    step.branches.append(PlanBranch(condition=f"branch {i}", next=first))
        elif stype == "Map":
            proc = state.get("ItemProcessor") or state.get("Iterator") or {}
            first = _emit_scope(
                proc.get("States") or {},
                proc.get("StartAt"),
                idmap[name],
                out,
                used,
            )
            if first:
                step.branches.append(PlanBranch(condition="each item", next=first))
    return idmap.get(start_at or "")


def asl_to_plan(asl: dict, *, name: str | None = None, summary: str | None = None) -> Plan:
    """Build the Plan view of ``asl``. Raises ValueError if it has no states."""
    states = asl.get("States") if isinstance(asl, dict) else None
    if not isinstance(states, dict) or not states:
        raise ValueError("ASL has no States to build a plan from.")

    steps: list[PlanStep] = []
    _emit_scope(states, asl.get("StartAt"), "Sub", steps, set())
    if not steps:
        raise ValueError("ASL has no usable states to build a plan from.")

    comment = str(asl.get("Comment") or "")
    derived_name, _, derived_summary = comment.partition(":")
    return Plan(
        name=(name or derived_name or "workflow").strip() or "workflow",
        summary=humanize_tokens((summary or derived_summary or comment or name or "Workflow").strip()),
        steps=steps,
    )
