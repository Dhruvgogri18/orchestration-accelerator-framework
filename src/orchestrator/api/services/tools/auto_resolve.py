"""Auto-resolve TODO_ placeholders against live AWS resources.

Policy (single-account, no cache):
  - Only propose a value when resolve_identifier returns matched=True
    (high confidence AND unambiguous).
  - Never invent ARNs; never pick among ambiguous candidates.
  - Caller decides whether to auto-apply or only surface suggestions
    to the UI (recommended default: suggest only).
"""

from __future__ import annotations

import re
from typing import Any

from .aws.allowlist import canonical_service
from .aws.resolver import resolve_identifier
from .registry import execute_tool

# Map TODO token fragments → likely AWS service
_TOKEN_SERVICE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"LAMBDA|FUNCTION", re.I), "lambda"),
    (re.compile(r"S3|BUCKET", re.I), "s3"),
    (re.compile(r"DYNAMO|DDB|TABLE", re.I), "dynamodb"),
    (re.compile(r"SQS|QUEUE", re.I), "sqs"),
    (re.compile(r"SNS|TOPIC", re.I), "sns"),
    (re.compile(r"STATE.?MACHINE|STEP.?FUNCTION|SFN", re.I), "states"),
    (re.compile(r"GLUE|ETL.?JOB", re.I), "glue"),
    (re.compile(r"ECS|CLUSTER|TASK.?DEF", re.I), "ecs"),
    (re.compile(r"EVENT.?BRIDGE|EVENTBUS|RULE", re.I), "events"),
    (re.compile(r"KINESIS|STREAM", re.I), "kinesis"),
    (re.compile(r"BATCH", re.I), "batch"),
    (re.compile(r"SAGEMAKER|ENDPOINT", re.I), "sagemaker"),
    (re.compile(r"ROLE|IAM", re.I), "iam"),
]

_TODO_RE = re.compile(r"\bTODO_[A-Z0-9_]+\b")


def infer_service_from_token(token: str) -> str | None:
    for pattern, svc in _TOKEN_SERVICE_HINTS:
        if pattern.search(token):
            return svc
    return None


def _hint_from_token(token: str) -> str:
    """Turn TODO_ORDER_PROCESSOR_LAMBDA into a search hint.

    Also produces a compact form without spaces (reviewleaverequest) so
    fuzzy match can hit Lambda names that drop separators.
    """
    body = re.sub(r"^TODO_", "", token or "")
    body = re.sub(
        r"_(LAMBDA|FUNCTION|BUCKET|TABLE|QUEUE|TOPIC|ARN|NAME|JOB|ROLE)$",
        "",
        body,
        flags=re.I,
    )
    parts = [p for p in body.split("_") if p and p.upper() not in ("ARN", "NAME", "ID")]
    return " ".join(parts) if parts else body


def _value_from_resource(resource: dict[str, Any], token: str) -> str | None:
    """Prefer ARN when token implies ARN; else name/id."""
    wants_arn = bool(re.search(r"ARN", token or "", re.I))
    arn = resource.get("arn")
    name = resource.get("name") or resource.get("id")
    if wants_arn:
        return arn or name
    # SQS often needs URL in some contexts — prefer arn/name we have
    return name or arn


def suggest_overrides_for_tokens(
    tokens: list[str],
    *,
    region: str | None = None,
    session_id: str | None = None,
    context: str | None = None,
) -> dict[str, Any]:
    """Return suggestions for each TODO_ token.

    Shape:
      {
        "suggestions": {
          "TODO_X": {
            "value": "...",          # only if matched
            "confidence": 0.9,
            "matched": true,
            "ambiguous": false,
            "service": "lambda",
            "resource": {...} | null,
            "alternatives": [...],
            "reason": null | str,
          },
          ...
        },
        "auto_applicable": {"TODO_X": "value", ...},  # matched only
        "needs_user": ["TODO_Y", ...],
      }
    """
    suggestions: dict[str, Any] = {}
    auto_applicable: dict[str, str] = {}
    needs_user: list[str] = []

    for token in tokens or []:
        if not token or not str(token).startswith("TODO_"):
            continue
        svc = infer_service_from_token(token)
        hint = _hint_from_token(token)
        entry: dict[str, Any] = {
            "value": None,
            "confidence": 0.0,
            "matched": False,
            "ambiguous": False,
            "service": svc,
            "resource": None,
            "alternatives": [],
            "reason": None,
            "hint": hint,
        }

        if not svc:
            entry["reason"] = "unknown_service_for_token"
            suggestions[token] = entry
            needs_user.append(token)
            continue

        result = execute_tool(
            "resolve_identifier",
            {
                "service": svc,
                "hint": hint,
                "region": region,
                "context": context,
            },
            session_id=session_id,
        )
        data = result.data or {}
        # Retry with compacted hint (no spaces) — matches names like reviewleaverequest
        if not data.get("matched") and not data.get("ambiguous") and hint:
            compact = re.sub(r"[^a-zA-Z0-9]", "", hint)
            if compact and compact.lower() != hint.lower().replace(" ", ""):
                pass  # same
            if compact and len(compact) >= 4:
                result2 = execute_tool(
                    "resolve_identifier",
                    {
                        "service": svc,
                        "hint": compact,
                        "region": region,
                        "context": context,
                    },
                    session_id=session_id,
                )
                data2 = result2.data or {}
                if data2.get("matched") or (data2.get("alternatives") and not data.get("alternatives")):
                    data = data2
                    result = result2
        entry["confidence"] = float(data.get("confidence") or 0.0)
        entry["matched"] = bool(data.get("matched"))
        entry["ambiguous"] = bool(data.get("ambiguous"))
        entry["resource"] = data.get("resource")
        entry["alternatives"] = data.get("alternatives") or []
        entry["reason"] = data.get("reason")

        if entry["matched"] and entry["resource"]:
            value = _value_from_resource(entry["resource"], token)
            if value:
                entry["value"] = value
                auto_applicable[token] = value
            else:
                entry["matched"] = False
                entry["reason"] = "resource_missing_value_field"
                needs_user.append(token)
        else:
            needs_user.append(token)

        suggestions[token] = entry

    return {
        "suggestions": suggestions,
        "auto_applicable": auto_applicable,
        "needs_user": needs_user,
    }


def extract_todo_tokens(*texts: str | None) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for text in texts:
        if not text:
            continue
        for m in _TODO_RE.findall(text):
            if m not in seen:
                seen.add(m)
                found.append(m)
    return found


def suggest_from_plan_and_spec(
    plan: dict | None,
    workflow_spec: dict | None,
    placeholders: list[str] | None,
    *,
    region: str | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    tokens = list(placeholders or [])
    if not tokens:
        import json

        blob = json.dumps(plan or {}) + json.dumps(workflow_spec or {})
        tokens = extract_todo_tokens(blob)
    context = None
    if isinstance(plan, dict):
        context = (plan.get("summary") or plan.get("name") or "")[:200]
    return suggest_overrides_for_tokens(
        tokens, region=region, session_id=session_id, context=context
    )
