"""Stage 4: workflow_spec.json -> Amazon States Language (ASL) + validation.

The locked workflow_spec produced by Stage 3.5 enrichment is the single
source of truth. This module:

  1. Translates it into a deployable ASL state machine definition.
  2. Runs the agreed five validation groups (structural, reachability,
     spec consistency, AWS constraints, high-value extras).
  3. Renders a Graphviz DOT graph in the style of the Step Functions
     console so the UI can show the same topology.

Generation is deterministic and mechanical -- no LLM. If the spec is
valid, ASL generation should always succeed; any failure is a bug in
the translator or an incomplete mapping for a new construct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

SUPPORTED_STATE_TYPES = {
    "Task",
    "Choice",
    "Wait",
    "Parallel",
    "Map",
    "Succeed",
    "Fail",
    "Pass",
}

# Soft limits that keep definitions console/API-friendly.
MAX_STATES_SOFT = 250
MAX_NESTING_DEPTH = 8
MAX_STATE_NAME_LEN = 80
STATE_NAME_RE = re.compile(r"^[A-Za-z0-9_]{1,80}$")

# Same sentinel pattern plan/render.py uses for the thin Plan
# (service/description text). Defined again here because this one
# scans a completely different, later-stage document -- the full ASL
# tree, at any depth, in any field -- not just two fields on a Plan
# step. See find_asl_placeholders() below.
_PLACEHOLDER_PATTERN = re.compile(r"\bTODO_[A-Z0-9_]+\b")


def find_asl_placeholders(node: Any, _seen: list[str] | None = None) -> list[str]:
    """Recursively scan the FULL ASL document -- including nested
    Parallel Branches and Map ItemProcessor/Iterator bodies, since
    those are just nested dicts/lists within the same JSON tree -- for
    any TODO_ token in any string value **or dict key** at any depth.

    Dict keys matter: enrichment often invents DynamoDB attribute names
    like ``TODO_PARTITION_KEY`` as an Item key. Those never appear as
    string values, so a values-only scan would leave them invisible to
    the fill-placeholders UI and to deploy readiness checks.

    Deliberately broader than the aws_constraints group's TODO_ check
    below, which only ever looks at a Task's `Resource`: enrichment
    can just as easily leave a TODO_ token in a Parameters value, a
    Comment, an InputPath, or an Item attribute key, and a workflow
    with any of those is exactly as undeployable as one with a
    placeholder Resource.
    """
    out = _seen if _seen is not None else []
    if isinstance(node, str):
        for m in _PLACEHOLDER_PATTERN.findall(node):
            if m not in out:
                out.append(m)
    elif isinstance(node, dict):
        for k, v in node.items():
            # Keys themselves may be TODO_ tokens (e.g. DynamoDB Item attrs).
            if isinstance(k, str):
                for m in _PLACEHOLDER_PATTERN.findall(k):
                    if m not in out:
                        out.append(m)
            find_asl_placeholders(v, out)
    elif isinstance(node, list):
        for v in node:
            find_asl_placeholders(v, out)
    return out


def _humanize_todo_token(token: str) -> str:
    """TODO_SUCCESS_TOPIC_ARN → 'Success topic ARN'."""
    raw = (token or "").strip()
    if raw.upper().startswith("TODO_"):
        raw = raw[5:]
    words = [w for w in raw.replace("-", "_").split("_") if w]
    if not words:
        return token
    # Keep common acronyms uppercase.
    acronyms = {
        "ARN",
        "AWS",
        "S3",
        "SNS",
        "SQS",
        "IAM",
        "API",
        "URL",
        "URI",
        "ID",
        "ETL",
        "CSV",
        "JSON",
        "DB",
        "RDS",
        "ECS",
        "EKS",
        "VPC",
        "KMS",
        "SES",
        "SFN",
    }
    out: list[str] = []
    for i, w in enumerate(words):
        up = w.upper()
        if up in acronyms:
            out.append(up)
        elif i == 0:
            out.append(w.capitalize())
        else:
            out.append(w.lower())
    return " ".join(out)


# Optional friendly labels + examples for well-known AWS services.
# Detection itself is generic (see _service_key_from_resource /
# _detect_service_hint): any service string works; this map only improves
# wording when we recognize a service. Unknown services fall back to a
# title-cased service name with a generic ARN example.
_SERVICE_LABELS: dict[str, tuple[str, str, str]] = {
    # key -> (human label, example ARN, example bare name)
    "lambda": (
        "Lambda function",
        "arn:aws:lambda:us-east-1:123456789012:function:MyFunction",
        "MyFunction",
    ),
    "dynamodb": (
        "DynamoDB table",
        "arn:aws:dynamodb:us-east-1:123456789012:table/MyTable",
        "MyTable",
    ),
    "sns": (
        "SNS topic",
        "arn:aws:sns:us-east-1:123456789012:MyTopic",
        "MyTopic",
    ),
    "sqs": (
        "SQS queue",
        "arn:aws:sqs:us-east-1:123456789012:MyQueue",
        "MyQueue",
    ),
    "s3": (
        "S3 bucket",
        "arn:aws:s3:::my-bucket",
        "my-bucket",
    ),
    "ecs": (
        "ECS task definition",
        "arn:aws:ecs:us-east-1:123456789012:task-definition/MyTask",
        "MyTask",
    ),
    "ses": (
        "SES identity",
        "arn:aws:ses:us-east-1:123456789012:identity/example.com",
        "example.com",
    ),
    "glue": (
        "Glue job",
        "arn:aws:glue:us-east-1:123456789012:job/MyJob",
        "MyJob",
    ),
    "states": (
        "Step Functions state machine",
        "arn:aws:states:us-east-1:123456789012:stateMachine:MyStateMachine",
        "MyStateMachine",
    ),
    "kms": (
        "KMS key",
        "arn:aws:kms:us-east-1:123456789012:key/abcd1234-a123-4567-b890-1234567890ab",
        "abcd1234-a123-4567-b890-1234567890ab",
    ),
    "rds": (
        "RDS database instance",
        "arn:aws:rds:us-east-1:123456789012:db:my-database",
        "my-database",
    ),
    "athena": (
        "Athena workgroup",
        "arn:aws:athena:us-east-1:123456789012:workgroup/primary",
        "primary",
    ),
    "batch": (
        "Batch job queue",
        "arn:aws:batch:us-east-1:123456789012:job-queue/MyQueue",
        "MyQueue",
    ),
    "events": (
        "EventBridge bus",
        "arn:aws:events:us-east-1:123456789012:event-bus/default",
        "default",
    ),
    "eventbridge": (
        "EventBridge bus",
        "arn:aws:events:us-east-1:123456789012:event-bus/default",
        "default",
    ),
    "stepfunctions": (
        "Step Functions state machine",
        "arn:aws:states:us-east-1:123456789012:stateMachine:MyStateMachine",
        "MyStateMachine",
    ),
}

# Step Functions optimized / SDK integration resource patterns:
#   arn:aws:states:::lambda:invoke
#   arn:aws:states:::glue:startJobRun.sync
#   arn:aws:states:::aws-sdk:s3:getObject
#   arn:aws:states:::sns:publish
_SFN_RESOURCE_SERVICE_RE = re.compile(
    r"arn:aws:states:::(?:aws-sdk:)?([a-z0-9\-]+):",
    re.IGNORECASE,
)
# Ordinary resource ARNs: arn:aws:glue:region:acct:job/MyJob
_PLAIN_ARN_SERVICE_RE = re.compile(
    r"^arn:(?:aws|aws-cn|aws-us-gov):([a-z0-9\-]+):",
    re.IGNORECASE,
)


def _service_key_from_resource(resource: str) -> str | None:
    """Extract the AWS service key from a Step Functions Resource string.

    Works for any service -- not limited to a fixed allow-list -- so a
    newly supported integration still gets a correct service key for
    labeling/validation.
    """
    text = (resource or "").strip()
    if not text:
        return None
    m = _SFN_RESOURCE_SERVICE_RE.search(text)
    if m:
        key = m.group(1).lower()
        # aws-sdk path already captured the real service; "states" alone
        # means the resource is itself a state machine ARN pattern.
        return key
    m = _PLAIN_ARN_SERVICE_RE.match(text)
    if m:
        return m.group(1).lower()
    return None


def _label_for_service_key(key: str) -> tuple[str, str, str]:
    """(label, example_arn, example_name) for a service key."""
    key = (key or "").lower().strip()
    if key in _SERVICE_LABELS:
        return _SERVICE_LABELS[key]
    # Generic fallback: title-case the service key so unknown services
    # still get a readable label instead of "AWS resource".
    pretty = key.replace("-", " ").title() if key else "AWS resource"
    example_arn = f"arn:aws:{key or 'service'}:us-east-1:123456789012:resource/MyResource"
    example_name = "MyResource"
    return pretty, example_arn, example_name


def _detect_service_hint(
    *texts: str,
    prefer_resource: str | None = None,
) -> tuple[str, str, str] | None:
    """Best-effort (label, example ARN, example bare name) for a step.

    Preference order (generic, not service-list-dependent):
      1. Explicit Step Functions Resource ARN / integration string
         (``prefer_resource`` or any text that looks like one).
      2. Longest whole-word service-key match against remaining text
         (description / service field), so ``glue`` is not beaten by a
         coincidental ``s3`` substring inside a Glue job argument key
         like ``--SOURCE_S3_LOCATION``.

    Strips TODO_ tokens before keyword matching so a placeholder name
    cannot match against itself.
    """
    # 1. Resource-first (always wins when present)
    candidates = [prefer_resource] if prefer_resource else []
    candidates.extend(texts)
    for c in candidates:
        if not c:
            continue
        key = _service_key_from_resource(str(c))
        if key and key not in ("aws-sdk",):
            return _label_for_service_key(key)

    # 2. Keyword match on non-resource text only — never scan parameter
    #    key noise (SOURCE_S3_LOCATION, etc.). Callers should pass
    #    resource/service/description, not params JSON.
    blob = _PLACEHOLDER_PATTERN.sub("", " ".join(str(t) for t in texts if t)).lower()
    if not blob.strip():
        return None

    # Prefer longer keys first so "dynamodb" beats a hypothetical "db"
    # and multi-word keys are stable.
    known_keys = sorted(_SERVICE_LABELS.keys(), key=len, reverse=True)
    for key in known_keys:
        # Word-boundary-ish: avoid matching "s3" inside "SOURCE_S3_LOCATION"
        # by requiring the key as a standalone token or after a separator.
        if re.search(rf"(?<![a-z0-9]){re.escape(key)}(?![a-z0-9])", blob):
            return _label_for_service_key(key)
    return None


# ---------------------------------------------------------------------------
# Placeholder value "shape" -- is this token supposed to be filled with a
# full ARN, a bare resource name, a URL/endpoint, or is there no strong
# signal either way? Used both to word descriptions correctly
# (describe_placeholders) and to reject a value of the wrong shape
# (validate_placeholder_value) -- e.g. an S3 bucket-name field filled
# with an s3:// URI or a full ARN, or an ARN field filled with a bare
# name.
# ---------------------------------------------------------------------------

PlaceholderKind = str  # "arn" | "name" | "url" | "generic"

# General AWS ARN shape: arn:partition:service:region:account-id:resource.
# Region and account-id are legitimately empty for some services (e.g.
# S3 bucket ARNs: "arn:aws:s3:::my-bucket"), so those two segments are
# optional -- but all six colon-separated fields, and the "arn:" prefix
# plus a real partition/service, must be present.
_ARN_STRUCTURE_RE = re.compile(
    r"^arn:(aws|aws-cn|aws-us-gov):[a-zA-Z0-9\-]+:[a-z0-9\-]*:(\d{12})?:.+$"
)
_S3_URI_RE = re.compile(r"^s3://", re.IGNORECASE)
# Parameter/field-name suffixes that tell us what shape of value AWS
# expects at that spot, independent of which specific service it is.
_ARN_FIELD_SUFFIX_RE = re.compile(r"arn$", re.IGNORECASE)
_URL_FIELD_SUFFIX_RE = re.compile(r"(url|uri|endpoint)$", re.IGNORECASE)
# Deliberately narrow to fields that name a top-level AWS resource
# (bucket/table/queue/topic/function/job/cluster/role, or anything
# ending "...Name"). Fields like "Key"/"Prefix"/"Path"/"Value" are
# intentionally excluded -- an S3 object key, a DynamoDB item value, a
# JSON path, etc. legitimately contain "/" or ":" and must NOT be
# forced through the bare-name ("no slash, no colon") check below, or
# a perfectly correct value gets rejected.
_NAME_FIELD_SUFFIX_RE = re.compile(
    r"(bucket|table|queue|topic|function|job|cluster|role|name)$", re.IGNORECASE
)


def _find_key_for_token(node: Any, token: str, key: str | None = None) -> str | None:
    """Depth-first search through a JSON-ish structure for the nearest
    dict key whose value contains `token`. Used to recover the actual
    Parameters field name (e.g. "Bucket" vs "BucketArn") a placeholder
    was plugged into, since that field name -- not the AWS service
    alone -- is what determines whether AWS wants an ARN or a bare name
    there.
    """
    if isinstance(node, str):
        return key if token in node else None
    if isinstance(node, dict):
        for k, v in node.items():
            found = _find_key_for_token(v, token, k)
            if found:
                return found
        return None
    if isinstance(node, list):
        for v in node:
            found = _find_key_for_token(v, token, key)
            if found:
                return found
    return None


def _token_is_dict_key(node: Any, token: str) -> bool:
    """True when `token` appears as a dict key (e.g. a DynamoDB attribute
    name inside Item) rather than as a field *value*. Keys are identifiers
    the user must supply, but they must NOT inherit the step's primary
    resource label (table/bucket/topic).
    """
    if isinstance(node, dict):
        for k, v in node.items():
            if str(k) == token or token in str(k):
                return True
            if _token_is_dict_key(v, token):
                return True
    elif isinstance(node, list):
        for v in node:
            if _token_is_dict_key(v, token):
                return True
    return False


def _kind_from_field_key(field_key: str) -> PlaceholderKind | None:
    """Infer value shape from an ASL/Parameters field name.

    Field context is more reliable than a soft token suffix: e.g.
    ``TopicArn`` is always an ARN even when the token is named
    ``TODO_OPS_SNS_TOPIC``. Returns None when the field name gives no
    strong signal (Key, Value, Message, Body, …).
    """
    if not field_key:
        return None
    # Strip path noise: "Arguments.--SOURCE_S3_LOCATION" → last segment
    leaf = field_key.split(".")[-1].strip()
    if _ARN_FIELD_SUFFIX_RE.search(leaf):
        return "arn"
    if _URL_FIELD_SUFFIX_RE.search(leaf):
        return "url"
    if _NAME_FIELD_SUFFIX_RE.search(leaf):
        return "name"
    return None


def classify_placeholder_kind(
    workflow_spec: dict | None,
    token: str,
    *,
    plan: dict | None = None,
) -> PlaceholderKind:
    """Best-effort guess at what SHAPE of value `token` should be
    filled with: "arn" (full ARN), "name" (bare resource name/id, no
    "arn:" prefix, no "s3://" scheme, no "/"), "url" (an endpoint/URI),
    or "generic" (no strong signal -- accept anything non-empty).

    Preference order (accurate for any service):
      1. Hard token suffix: ``_ARN`` / ``_URL`` / ``_URI`` / ``_ENDPOINT``
         (unambiguous intent from the extraction stage).
      2. Structural field context from workflow_spec parameters
         (``TopicArn``, ``JobName``, ``Bucket``, …) -- this overrides
         soft token suffixes like ``_TOPIC`` / ``_NAME`` so a token
         sitting under ``TopicArn`` is treated as an ARN even if named
         ``TODO_OPS_SNS_TOPIC``.
      3. Soft token suffix: ``_NAME`` / ``_BUCKET`` / ``_TABLE`` /
         ``_QUEUE`` / ``_TOPIC`` / ``_JOB``.
      4. Parameter constraints text mentioning "arn".
      5. ``generic``.
    """
    upper = (token or "").upper()
    if upper.endswith("_ARN"):
        return "arn"
    if upper.endswith(("_URL", "_URI", "_ENDPOINT")):
        return "url"

    # Structural context BEFORE soft token suffixes -- field name is the
    # source of truth for what AWS actually expects at that slot.
    if isinstance(workflow_spec, dict):
        components = workflow_spec.get("components") or workflow_spec.get("steps") or []
        if isinstance(components, list):
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                resource = str(comp.get("resource") or comp.get("service") or "")
                if token == resource.strip():
                    return "arn"  # a Task's own Resource is always an ARN
                params = comp.get("parameters")
                if isinstance(params, (dict, list)):
                    field_key = _find_key_for_token(params, token)
                    if field_key:
                        from_field = _kind_from_field_key(field_key)
                        if from_field is not None:
                            return from_field
                        return "generic"

        for param in workflow_spec.get("parameters") or []:
            if isinstance(param, dict) and str(param.get("name") or "") == token:
                constraints = str(param.get("constraints") or "").lower()
                if "arn" in constraints:
                    return "arn"
                name_l = str(param.get("name") or "").lower()
                if name_l.endswith("_arn") or "arn" in str(param.get("description") or "").lower():
                    return "arn"

    # Soft suffixes -- only when no field context decided the shape.
    # NOTE: deliberately does NOT include "_KEY" -- an object/file key
    # is often a path ("incoming/data.csv"), not a bare resource name.
    if upper.endswith(("_NAME", "_BUCKET", "_TABLE", "_QUEUE", "_TOPIC", "_JOB")):
        return "name"

    return "generic"


def validate_placeholder_value(kind: PlaceholderKind, value: str) -> str | None:
    """Check a user-supplied override against the shape `kind` implies.
    Returns None when the value is acceptable, else a short user-facing
    error string. Blank values are left to the caller (a still-empty
    field is "unresolved", not "invalid").
    """
    v = (value or "").strip()
    if not v:
        return None
    if _PLACEHOLDER_PATTERN.search(v):
        return "Still contains a TODO_ placeholder — enter the actual value."

    if kind == "arn":
        if not _ARN_STRUCTURE_RE.match(v):
            return (
                "Must be a full AWS ARN, formatted "
                "arn:aws:<service>:<region>:<account-id>:<resource> "
                "(e.g. arn:aws:sns:us-east-1:123456789012:MyTopic) — "
                "not a bare name or a URI."
            )
        return None

    if kind == "url":
        if v.lower().startswith("arn:"):
            return "Must be a URL/endpoint, not an ARN."
        return None

    if kind == "name":
        if v.lower().startswith("arn:"):
            return "Must be just the resource name, not the full ARN."
        if _S3_URI_RE.match(v):
            return "Must be just the bucket name, not an s3:// URI."
        if "/" in v or ":" in v:
            return "Must be a bare resource name (no '/' or ':')."
        return None

    return None


def _as_clause(desc: str) -> str:
    """'Submit and capture the request.' -> 'submit and capture the request'.

    Folds an existing step description into a "used ... to <do the
    thing>" sentence instead of repeating it verbatim. TODO_ tokens in
    the description are humanized so the fill form never echoes the
    raw placeholder name inside help text.
    """
    d = (desc or "").strip().rstrip(".")
    if not d:
        return d
    d = _PLACEHOLDER_PATTERN.sub(
        lambda m: _humanize_todo_token(m.group(0)), d
    )
    return d[0].lower() + d[1:]


def describe_placeholders(
    workflow_spec: dict | None,
    tokens: list[str],
    *,
    plan: dict | None = None,
) -> dict[str, str]:
    """Map each TODO_ token to a short user-facing description.

    Prefer enrichment metadata on workflow_spec (parameters,
    integrations, components). Fall back to plan step text, then a
    humanized form of the token name so the fill form is never bare.
    """
    details: dict[str, str] = {}
    if not tokens:
        return details

    token_set = {t for t in tokens if t}

    # --- parameters: name is often the TODO_ token itself ---
    if isinstance(workflow_spec, dict):
        for param in workflow_spec.get("parameters") or []:
            if not isinstance(param, dict):
                continue
            name = str(param.get("name") or "")
            if name not in token_set:
                continue
            desc = (param.get("description") or "").strip()
            scope = param.get("scope")
            component_id = param.get("component_id")
            bits: list[str] = []
            if desc:
                bits.append(desc)
            if component_id:
                bits.append(f"Used by step `{component_id}`.")
            elif scope and scope != "global":
                bits.append(f"Scope: {scope}.")
            details[name] = " ".join(bits) if bits else _humanize_todo_token(name)

        # --- integrations: connection_details may be the token ---
        for integ in workflow_spec.get("integrations") or []:
            if not isinstance(integ, dict):
                continue
            conn = str(integ.get("connection_details") or "")
            matched = [t for t in token_set if t in conn]
            if not matched:
                continue
            label = (integ.get("name") or "").strip()
            service = (integ.get("service") or "").strip()
            kind = (integ.get("kind") or "").strip()
            comps = integ.get("components") or []
            parts: list[str] = []
            if label:
                parts.append(label)
            elif service:
                parts.append(f"{service} resource")
            if service and label and service.lower() not in label.lower():
                parts.append(f"({service})")
            if kind:
                parts.append(f"— {kind.replace('_', ' ')}")
            if isinstance(comps, list) and comps:
                parts.append(
                    "Used by: " + ", ".join(f"`{c}`" for c in comps[:4])
                    + ("…" if len(comps) > 4 else "")
                    + "."
                )
            text = " ".join(parts).strip() or _humanize_todo_token(matched[0])
            for t in matched:
                details.setdefault(t, text)

        # --- components / steps: resource or description may embed the token ---
        components = (
            workflow_spec.get("components")
            or workflow_spec.get("steps")
            or []
        )
        if isinstance(components, list):
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                cid = str(comp.get("id") or comp.get("name") or "")
                cdesc = (comp.get("description") or "").strip()
                resource = str(comp.get("resource") or comp.get("service") or "")
                service_field = str(comp.get("service") or "")
                params_blob = json.dumps(comp.get("parameters") or {}, default=str)
                step_ref = f"step `{cid}`" if cid else "this step"

                # A token can play two different roles on the same step,
                # and they need different wording or they collapse to the
                # same generic sentence:
                #   - it IS the whole resource/service value -> the
                #     integration itself is unresolved (unknown ARN).
                #   - it's embedded inside Parameters/service alongside an
                #     otherwise-identifiable integration (e.g. Resource is
                #     the Lambda-invoke pattern, Parameters.FunctionName is
                #     the placeholder) -> we know the service, just not
                #     which specific resource to call.
                for t in token_set:
                    if t in details or not t:
                        continue
                    is_whole_resource = t == resource.strip()
                    is_embedded = not is_whole_resource and (
                        t in params_blob or t in service_field or t in cdesc
                    )
                    if not (is_whole_resource or is_embedded):
                        continue

                    # Service detection: Resource ARN first, then service /
                    # description only — never params JSON (keys like
                    # --SOURCE_S3_LOCATION must not re-label a Glue step
                    # as S3).
                    hint = _detect_service_hint(
                        service_field,
                        cdesc,
                        prefer_resource=("" if is_whole_resource else resource) or None,
                    )
                    kind = classify_placeholder_kind(workflow_spec, t, plan=plan)
                    params_obj = comp.get("parameters")
                    field_key = (
                        _find_key_for_token(params_obj, t)
                        if isinstance(params_obj, (dict, list))
                        else None
                    )
                    # Token used as a dict key (DynamoDB attribute name,
                    # map key, etc.) is not the step's primary resource.
                    as_key_only = (
                        isinstance(params_obj, (dict, list))
                        and _token_is_dict_key(params_obj, t)
                        and field_key is None
                    )
                    if is_whole_resource:
                        if hint:
                            label, arn_example, _name_example = hint
                            details[t] = (
                                f"AWS ARN for the {label} used by {step_ref}"
                                + (f" — {cdesc}." if cdesc else ".")
                                + f" Must be the full ARN, e.g. `{arn_example}`."
                            )
                        else:
                            details[t] = (
                                f"AWS resource ARN required by {step_ref} "
                                "(integration type wasn't identified — e.g. "
                                "Lambda, SNS, or DynamoDB ARN)."
                            )
                    else:
                        clause = _as_clause(cdesc)
                        leaf = (field_key or "").split(".")[-1] if field_key else ""
                        if as_key_only:
                            # e.g. TODO_ATTRIBUTE_NAME as a DynamoDB Item key
                            human = _humanize_todo_token(t)
                            details[t] = (
                                f"{human} used as a field/attribute key by {step_ref}"
                                + (f" to {clause}." if clause else ".")
                            )
                        # Field-aware label: JobName + Glue → "Glue job name";
                        # TopicArn + SNS → "SNS topic ARN". Generic for any
                        # service via resource-derived hint + field leaf.
                        elif hint and kind in ("arn", "name"):
                            label, arn_example, name_example = hint
                            if kind == "arn":
                                field_bit = f" ({leaf})" if leaf and not leaf.lower().endswith("arn") else ""
                                details[t] = (
                                    f"{label} ARN{field_bit} used by {step_ref}"
                                    + (f" to {clause}." if clause else ".")
                                    + f" Must be the full ARN, e.g. `{arn_example}`."
                                )
                            else:
                                # Prefer field leaf when it already names the
                                # resource role (JobName, TableName, Bucket).
                                if leaf and re.search(
                                    r"(bucket|table|queue|topic|function|job|cluster|role|name)$",
                                    leaf,
                                    re.IGNORECASE,
                                ):
                                    role = leaf
                                    # JobName → "job name"; TableName → "table name"
                                    role_h = re.sub(
                                        r"([a-z])([A-Z])", r"\1 \2", role
                                    ).replace("_", " ").lower()
                                    details[t] = (
                                        f"{label.split()[0]} {role_h} used by {step_ref}"
                                        + (f" to {clause}." if clause else ".")
                                        + f" Just the name, not the ARN — e.g. `{name_example}`."
                                    )
                                else:
                                    details[t] = (
                                        f"{label} name used by {step_ref}"
                                        + (f" to {clause}." if clause else ".")
                                        + f" Just the name, not the ARN — e.g. `{name_example}`."
                                    )
                        elif field_key and kind == "arn":
                            details[t] = (
                                f"`{leaf}` (ARN) for {step_ref}"
                                + (f" — {clause}." if clause else ".")
                                + " Must be a full AWS ARN."
                            )
                        elif field_key and kind == "name":
                            details[t] = (
                                f"`{leaf}` for {step_ref}"
                                + (f" — {clause}." if clause else ".")
                                + " Just the resource name, not an ARN."
                            )
                        elif cdesc:
                            clean = _PLACEHOLDER_PATTERN.sub(
                                lambda m: _humanize_todo_token(m.group(0)),
                                cdesc,
                            ).strip().rstrip(".")
                            details[t] = (
                                f"{clean} (step `{cid}`)." if cid else clean
                            )
                        elif hint:
                            label, _arn_example, _name_example = hint
                            details[t] = (
                                f"Value related to the {label} used by {step_ref}."
                            )
                        elif cid:
                            details[t] = (
                                f"Resource for step `{cid}` "
                                f"({_humanize_todo_token(t)})."
                            )

    # --- plan steps (service / description) ---
    if isinstance(plan, dict):
        for step in plan.get("steps") or []:
            if not isinstance(step, dict):
                continue
            sid = str(step.get("id") or "")
            sdesc = (step.get("description") or "").strip()
            service_field = str(step.get("service") or "")
            step_ref = f"step `{sid}`" if sid else "this step"
            for t in token_set:
                if t in details or not t:
                    continue
                # At this early stage (before enrichment/ASL) there's no
                # resource/parameters split yet -- `service` is the only
                # field that can hold the placeholder. When the token IS
                # that whole field, extraction couldn't even name the AWS
                # service; give the same "unresolved integration" phrasing
                # a later ARN-only placeholder would get.
                if t != service_field.strip():
                    continue
                hint = _detect_service_hint(sdesc)
                clause = _as_clause(sdesc)
                kind = classify_placeholder_kind(workflow_spec, t, plan=plan)
                if hint:
                    label, arn_example, name_example = hint
                    if kind == "arn":
                        details[t] = (
                            f"{label} ARN used by {step_ref}"
                            + (f" to {clause}." if clause else ".")
                            + f" Must be the full ARN, e.g. `{arn_example}`."
                        )
                    elif kind == "name":
                        details[t] = (
                            f"{label} name used by {step_ref}"
                            + (f" to {clause}." if clause else ".")
                            + f" Just the name, not the ARN — e.g. `{name_example}`."
                        )
                    else:
                        details[t] = (
                            f"{label} name or ARN used by {step_ref}"
                            + (f" to {clause}." if clause else ".")
                            + f" e.g. `{arn_example}`."
                        )
                elif sdesc:
                    details[t] = (
                        f"AWS service + resource identifier (e.g. a Lambda "
                        f"function name/ARN or SNS topic ARN) used by "
                        f"{step_ref} to {clause}."
                    )
                else:
                    details[t] = (
                        f"AWS service + resource identifier needed for "
                        f"{step_ref} ({_humanize_todo_token(t)})."
                    )

    # --- fallback: humanize any still-missing tokens ---
    for t in tokens:
        if t and t not in details:
            details[t] = _humanize_todo_token(t)

    # --- guarantee no two tokens ever render identical text ---
    # Every branch above tries to produce a token-specific description,
    # but they rely on heuristics (a recognized service keyword, a step
    # description that differs per role, etc.) that can fail to fire on
    # real-world data -- e.g. a step whose resource text has no keyword
    # we know about. Rather than trying to anticipate every case that
    # could cause two tokens on the same step to collide, enforce
    # uniqueness structurally as a last pass: if two tokens ended up with
    # the exact same description, tag each with its own token name so
    # the fill-placeholders form never shows the same text twice.
    by_text: dict[str, list[str]] = {}
    for t, d in details.items():
        by_text.setdefault(d, []).append(t)
    for text, toks in by_text.items():
        if len(toks) < 2:
            continue
        for t in toks:
            details[t] = f"{text} ({_humanize_todo_token(t)})"

    return details


def apply_placeholder_overrides(node: Any, overrides: dict[str, str]) -> Any:
    """Deep-replace TODO_ tokens in a JSON tree (string values AND dict keys).

    Used after Stage 4 so user-supplied values from /approve overrides
    land in the locked ASL / workflow_spec — not only in the thin Plan
    service/description fields. Enrichment routinely re-emits TODOs in
    Parameters, Resource, and as DynamoDB Item attribute *keys*; without
    key rewriting, Apply placeholders would leave those keys unresolved.
    """
    if not overrides:
        return node
    if isinstance(node, str):
        out = node
        for token, value in overrides.items():
            if token and value is not None:
                out = out.replace(token, value)
        return out
    if isinstance(node, dict):
        new_dict: dict[Any, Any] = {}
        for k, v in node.items():
            new_k = k
            if isinstance(k, str):
                for token, value in overrides.items():
                    if token and value is not None and token in k:
                        new_k = k.replace(token, value)
            new_dict[new_k] = apply_placeholder_overrides(v, overrides)
        return new_dict
    if isinstance(node, list):
        return [apply_placeholder_overrides(v, overrides) for v in node]
    return node


@dataclass
class ValidationIssue:
    """One concrete problem found during validation."""

    group: str  # e.g. "structural", "reachability", "spec_consistency", ...
    code: str  # short machine-readable id
    message: str
    path: str | None = None  # JSON-path-ish location when useful


@dataclass
class ValidationReport:
    """Outcome of running all five validation groups on an ASL definition."""

    json_ok: bool = False
    structure_ok: bool = False
    reachability_ok: bool = False
    spec_consistency_ok: bool = False
    aws_constraints_ok: bool = False
    extras_ok: bool = False
    aws_validate_ok: bool | None = None  # None = not attempted
    issues: list[ValidationIssue] = field(default_factory=list)
    aws_details: dict | None = None
    # Every TODO_ token still present anywhere in the ASL document.
    # Kept as its own field (not just buried in `issues`) because it's
    # the one thing a UI needs to answer "what do I still have to fill
    # in", separately from "why did validation fail". A non-empty list
    # here always means extras_ok (and therefore ok) is False -- see
    # _validate_extras()/validate_asl() below. This is what closes the
    # gap where a TODO_ resource -- explicitly allowed through by the
    # aws_constraints group and by enrichment's ARN whitelist, correctly,
    # since a placeholder is a well-formed sentinel at those stages --
    # could otherwise reach `ok: True` here despite not being
    # deployable. TODO_ is valid ASL *syntax*; it is not a deployable
    # *value*, and this report's `ok` is documented (see
    # build_asl_from_spec below) as the deploy signal, so it has to
    # reflect that.
    unresolved_placeholders: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        local_ok = (
            self.json_ok
            and self.structure_ok
            and self.reachability_ok
            and self.spec_consistency_ok
            and self.aws_constraints_ok
            and self.extras_ok
        )
        # Live AWS failure fails the overall report; skipped (None) does not.
        if self.aws_validate_ok is False:
            return False
        return local_ok

    def to_dict(self) -> dict:
        return {
            "json_ok": self.json_ok,
            "structure_ok": self.structure_ok,
            "reachability_ok": self.reachability_ok,
            "spec_consistency_ok": self.spec_consistency_ok,
            "aws_constraints_ok": self.aws_constraints_ok,
            "extras_ok": self.extras_ok,
            "aws_validate_ok": self.aws_validate_ok,
            "ok": self.ok,
            "unresolved_placeholders": self.unresolved_placeholders,
            "errors": [
                {
                    "group": i.group,
                    "code": i.code,
                    "message": i.message,
                    "path": i.path,
                }
                for i in self.issues
            ],
            "aws_details": self.aws_details,
        }


@dataclass
class AslResult:
    """Full Stage-4 outcome: ASL definition + validation + graph."""

    asl: dict
    validations: ValidationReport
    graph_dot: str
    generation_notes: list[str] = field(default_factory=list)


class AslGenerationError(Exception):
    """Raised when workflow_spec cannot be translated (bug or unsupported)."""


# ---------------------------------------------------------------------------
# Spec tree helpers (mirror enrichment_service recursion)
# ---------------------------------------------------------------------------

def _iter_spec_steps(steps: list[dict] | None) -> Iterator[dict]:
    for step in steps or []:
        yield step
        for branch in step.get("branches") or []:
            yield from _iter_spec_steps(branch.get("steps"))
        iterator = step.get("iterator")
        if iterator:
            yield from _iter_spec_steps(iterator.get("steps"))


def _collect_spec_ids(spec: dict) -> set[str]:
    return {s["id"] for s in _iter_spec_steps(spec.get("steps")) if s.get("id")}


# ---------------------------------------------------------------------------
# JSONata helpers (pure JSONata ASL — no JSONPath fields)
# ---------------------------------------------------------------------------
# Generated state machines always set top-level QueryLanguage: "JSONata".
# JSONPath-only fields (Parameters, ItemsPath, Variable+comparator, InputPath,
# ResultPath, OutputPath, ResultSelector) are never emitted. Spec strings that
# still arrive in JSONPath form (from enrichment) are converted here.

# Path-like token: $.foo, $.Payload.status, $.items[0].id, $$, $$.Execution.Id
_VAR = r"(\$(?:\$)?(?:\.[A-Za-z0-9_.\[\]'\"-]+)?)"

# Strict / preferred forms (enrichment is instructed to emit these).
_COND_BOOL = re.compile(
    rf"^\s*{_VAR}\s*(?:==|=|is)\s*(true|false)\s*$",
    re.IGNORECASE,
)
_COND_NUM = re.compile(
    rf"^\s*{_VAR}\s*(>=|<=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$",
)
_COND_EQ = re.compile(
    rf"^\s*{_VAR}\s*(?:==|=|equals)\s*['\"]?([^'\"]+?)['\"]?\s*$",
    re.IGNORECASE,
)
_COND_NEQ = re.compile(
    rf"^\s*{_VAR}\s*(?:!=|<>|not\s+equals?)\s*['\"]?([^'\"]+?)['\"]?\s*$",
    re.IGNORECASE,
)
_COND_PRESENT = re.compile(
    rf"^\s*{_VAR}\s+(?:is\s+present|exists|is\s+not\s+null)\s*$",
    re.IGNORECASE,
)
_COND_NOT_PRESENT = re.compile(
    rf"^\s*{_VAR}\s+(?:is\s+not\s+present|does\s+not\s+exist|is\s+null|is\s+absent)\s*$",
    re.IGNORECASE,
)
_COND_CONTAINS = re.compile(
    rf"^\s*{_VAR}\s+(?:contains|includes)\s+['\"]?([^'\"]+?)['\"]?\s*$",
    re.IGNORECASE,
)

# Softer natural-language forms still seen from older plans / free text.
_COND_SOFT_STATUS = re.compile(
    r"^\s*(?:status\s*(?:==|=|is|equals)\s*)?['\"]?(approved|rejected|success|failed|"
    r"ok|error|true|false|yes|no)['\"]?\s*$",
    re.IGNORECASE,
)
_COND_SOFT_PATH_STATUS = re.compile(
    rf"^\s*{_VAR}\s*$",
)


# AWS Step Functions JSONata rules (docs: transforming-data.html):
# - Top-level or per-state QueryLanguage: "JSONata"
# - No Parameters / InputPath / ResultPath / OutputPath / ResultSelector /
#   ItemsPath / Variable+comparator / key.$ suffixes
# - Use Arguments, Output, Items, Condition, Assign instead
# - Dynamic strings: "{% <expr> %}" — must start with "{%" and end with "%}"
# - Reserved: $states.input | $states.result | $states.errorOutput | $states.context
# - Workflow variables: $varName via Assign; reference as {% $varName %}
# - Static constants stay plain JSON literals (no {% %})

_TODO_TOKEN = re.compile(r"^TODO_[A-Z0-9_]+$")
# "$varName" already a variable reference (not JSONPath $. or $$.)
_VAR_REF = re.compile(r"^\$[A-Za-z_][A-Za-z0-9_]*$")

# Workflow-variable names introduced by Catch Assign (and similar) during
# the current generate_asl() call. Paths like $.errorInfo must become
# $errorInfo (Assign variable), NOT $states.input.errorInfo — the latter
# is empty because Assign does not write into state input.
_ASSIGN_VARS: set[str] = set()


def _jsonpath_to_jsonata_expr(path: str) -> str:
    """Map a JSONPath / path-like reference to a JSONata expression.

    AWS reserved variable shapes:
      $.foo.bar     → $states.input.foo.bar
      $             → $states.input
      $$.Exec.Id    → $states.context.Exec.Id
      $customerName → $customerName  (workflow variable; unchanged)

    When ``foo`` is in ``_ASSIGN_VARS`` (Catch Assign target), ``$.foo``
    and ``$.foo.bar`` become ``$foo`` / ``$foo.bar`` so downstream states
    read the assigned workflow variable instead of missing input fields.
    """
    text = (path or "").strip()
    if not text:
        return "$states.input"
    if text.startswith("{%") and text.endswith("%}"):
        return text[2:-2].strip()
    if text.startswith("$states"):
        return text
    if text == "$":
        return "$states.input"
    if text == "$$":
        return "$states.context"
    if text.startswith("$."):
        rest = text[2:]
        head = rest.split(".", 1)[0]
        if head in _ASSIGN_VARS:
            return "$" + rest
        return "$states.input." + rest
    if text.startswith("$$."):
        return "$states.context." + text[3:]
    # Workflow variable already: $bucketName
    if _VAR_REF.match(text):
        return text
    # Bare identifier → state input field (or Assign var when known)
    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", text):
        if text in _ASSIGN_VARS:
            return f"${text}"
        return f"$states.input.{text}"
    return text


def _as_jsonata_string(expr: str) -> str:
    """Wrap a JSONata expression in ASL ``{% ... %}`` delimiters (AWS form)."""
    text = (expr or "").strip()
    if text.startswith("{%") and text.endswith("%}"):
        return text
    return "{% " + text + " %}"


def _todo_to_input_expr(token: str) -> str:
    """Unresolved TODO_TOKEN → bind from execution input (JSONata).

    Example: TODO_S3_BUCKET → {% $states.input.TODO_S3_BUCKET %}
    Callers can pass these fields on execution start; filled concrete values
    stay plain literals and are not rewritten.
    """
    return _as_jsonata_string(f"$states.input.{token}")


def _convert_value_to_jsonata(value: Any) -> Any:
    """Recursively apply AWS JSONata value rules to Arguments / Output trees.

    Nested dicts are converted with the same key rules as top-level
    Parameters → Arguments (strip ``.$`` suffixes, path-ify string
    values). Without that, enrichment shapes like::

        {"Arguments": {"--source-bucket.$": "$.s3.bucket"}}
        {"Item": {"pk": {"S.$": "$.JobRunId"}}}

    would keep JSONPath ``.$`` keys under QueryLanguage=JSONata and
    fail purity validation / AWS deploy.
    """
    if isinstance(value, list):
        return [_convert_value_to_jsonata(v) for v in value]
    if isinstance(value, dict):
        # Same key rewrite as _convert_parameters_to_arguments — nested
        # objects must not leak JSONPath '.$' suffixes.
        return _convert_parameters_to_arguments(value)
    if isinstance(value, str):
        s = value.strip()
        if _TODO_TOKEN.match(s):
            return _as_jsonata_string(f"$states.input.{s}")
        if s.startswith("$.") or s.startswith("$$") or s == "$" or s == "$$":
            return _as_jsonata_string(_jsonpath_to_jsonata_expr(s))
        if s.startswith("{%") and s.endswith("%}"):
            return s
        if _VAR_REF.match(s):
            return _as_jsonata_string(s)
        return value
    return value


def _convert_parameters_to_arguments(params: dict) -> dict:
    """JSONPath Parameters object → JSONata Arguments object (AWS rules).

    - Keys ending in '.$' become bare keys with {% path-expr %} values.
    - Nested objects/arrays are walked recursively (same key rewrite
      at every depth, so ``Arguments.--source-bucket.$`` and
      ``Item.pk.S.$`` are both cleaned).
    - TODO_ tokens and $. paths become {% $states.input... %} expressions.
    - Plain constants stay as plain JSON literals.
    - Values that are already ``{% ... %}`` JSONata strings are kept.
    """
    if not isinstance(params, dict):
        return params
    out: dict[str, Any] = {}
    for key, val in params.items():
        key_s = str(key)
        if key_s.endswith(".$"):
            new_key = key_s[:-2]
            if isinstance(val, str):
                s = val.strip()
                # Already a JSONata expression (e.g. after a prior pass
                # or enrichment wrote {% %} directly under a '.$' key).
                if s.startswith("{%") and s.endswith("%}"):
                    out[new_key] = s
                else:
                    out[new_key] = _as_jsonata_string(_jsonpath_to_jsonata_expr(s))
            else:
                out[new_key] = _convert_value_to_jsonata(val)
        else:
            out[key_s] = _convert_value_to_jsonata(val)
    return out


def _items_to_jsonata(items_path: str) -> str:
    """Map ItemsPath → JSONata ``Items`` expression."""
    return _as_jsonata_string(_jsonpath_to_jsonata_expr(items_path))


def _choice_rule_from_condition(condition: str, next_step: str) -> dict:
    """Convert a condition string into a pure-JSONata ASL Choice rule.

    Prefer strict parseable forms (enrichment is instructed to emit
    ``$.path == value``, ``$.path > N``, ``$.path is present``, etc.)
    and converts them to JSONata ``{% $states.input.status = 'approved' %}``.
    """
    text = (condition or "").strip()

    def rule(expr: str, comment: str | None = None) -> dict:
        out: dict[str, Any] = {
            "Condition": _as_jsonata_string(expr),
            "Next": next_step,
        }
        if comment:
            out["Comment"] = comment
        return out

    if not text:
        return rule(
            "$states.input.status = true",
            "empty condition from workflow_spec",
        )

    # Already a JSONata Condition string
    if text.startswith("{%") and text.endswith("%}"):
        return {"Condition": text, "Next": next_step}

    # Likely already JSONata body without wrappers
    if "$states" in text or text.startswith("$exists") or text.startswith("$not"):
        return rule(text)

    m = _COND_BOOL.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        bool_val = m.group(2).lower()
        return rule(f"{expr_path} = {bool_val}")

    m = _COND_NUM.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        op, num_s = m.group(2), m.group(3)
        return rule(f"{expr_path} {op} {num_s}")

    m = _COND_NEQ.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        raw_val = m.group(2).strip()
        if re.match(r"^-?\d+(?:\.\d+)?$", raw_val):
            return rule(
                f"{expr_path} != {raw_val}",
                f"NOTE: original condition was inequality (!=); review: {text}",
            )
        if raw_val.lower() in ("true", "false"):
            return rule(
                f"{expr_path} != {raw_val.lower()}",
                f"NOTE: original condition was inequality (!=); review: {text}",
            )
        val = raw_val.replace("'", "\\'")
        return rule(
            f"{expr_path} != '{val}'",
            f"NOTE: original condition was inequality (!=); review: {text}",
        )

    m = _COND_EQ.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        raw_val = m.group(2).strip()
        # Unquoted numeric/boolean literals must stay unquoted -- JSONata's
        # type system treats '5' != 5.
        if re.match(r"^-?\d+(?:\.\d+)?$", raw_val):
            return rule(f"{expr_path} = {raw_val}")
        if raw_val.lower() in ("true", "false"):
            return rule(f"{expr_path} = {raw_val.lower()}")
        val = raw_val.replace("'", "\\'")
        return rule(f"{expr_path} = '{val}'")

    m = _COND_PRESENT.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        return rule(f"$exists({expr_path})")

    m = _COND_NOT_PRESENT.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        return rule(f"$not($exists({expr_path}))")

    m = _COND_CONTAINS.match(text)
    if m:
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        val = m.group(2).strip().replace("'", "\\'")
        return rule(f"$contains({expr_path}, '{val}')")

    m = _COND_SOFT_STATUS.match(text)
    if m:
        label = m.group(1).strip().lower().replace("'", "\\'")
        return rule(
            f"$states.input.status = '{label}'",
            f"soft-parsed condition label: {text}",
        )

    m = _COND_SOFT_PATH_STATUS.match(text)
    if m and (text.startswith("$.") or text.startswith("$$")):
        expr_path = _jsonpath_to_jsonata_expr(m.group(1))
        return rule(
            f"$exists({expr_path})",
            f"soft-parsed path-only condition: {text}",
        )

    # No known pattern matched. Silently guessing (e.g. treating the whole
    # string as a status label) risks generating a Choice rule that looks
    # valid but never matches what the author meant -- that's worse than
    # failing loudly, since it deploys without complaint and just silently
    # never takes this branch. Surface it as a generation error instead.
    raise AslGenerationError(
        f"Could not translate condition into a JSONata Choice rule: {text!r}. "
        "Rephrase as e.g. \"$.field == value\", \"$.field > 5\", "
        "\"$.field is present\", or a raw JSONata body."
    )


# ---------------------------------------------------------------------------
# ASL generation
# ---------------------------------------------------------------------------

def _retry_to_asl(retriers: list[dict] | None) -> list[dict] | None:
    if not retriers:
        return None
    out = []
    for r in retriers:
        item: dict[str, Any] = {"ErrorEquals": list(r.get("error_equals") or ["States.ALL"])}
        if "interval_seconds" in r:
            item["IntervalSeconds"] = int(r["interval_seconds"])
        if "max_attempts" in r:
            item["MaxAttempts"] = int(r["max_attempts"])
        if "backoff_rate" in r:
            item["BackoffRate"] = float(r["backoff_rate"])
        out.append(item)
    return out


def _catch_to_asl(catchers: list[dict] | None) -> list[dict] | None:
    if not catchers:
        return None
    out = []
    for c in catchers:
        if not c.get("next_step"):
            continue
        item: dict[str, Any] = {
            "ErrorEquals": list(c.get("error_equals") or ["States.ALL"]),
            "Next": c["next_step"],
        }
        # JSONata mode has no ResultPath -- the equivalent is Assign, which
        # binds the error onto a workflow variable so the target state (and
        # anything after it) can still read it via {% $<key> %}.
        # Always Assign (default errorInfo) so failure notifiers can rely
        # on $errorInfo even when enrichment omitted result_path.
        rp = c.get("result_path")
        if isinstance(rp, str) and rp.strip():
            key = rp.strip()
            key = key[2:] if key.startswith("$.") else key.lstrip("$")
            key = re.sub(r"[^A-Za-z0-9_]", "_", key) or "errorInfo"
        else:
            key = "errorInfo"
        item["Assign"] = {key: "{% $states.errorOutput %}"}
        out.append(item)
    return out


def _apply_common_transition(state: dict, step: dict) -> None:
    """Attach Next/End from workflow_spec step to an ASL state."""
    if step.get("end") is True:
        state["End"] = True
    elif step.get("next_step"):
        state["Next"] = step["next_step"]


def _build_branch_asl(branch: dict, notes: list[str]) -> dict:
    """Build a nested state machine fragment for Parallel/Map."""
    steps = branch.get("steps") or []
    if not steps:
        raise AslGenerationError("Parallel/Map branch has no steps")
    start = branch.get("start_at") or steps[0]["id"]
    states: dict[str, dict] = {}
    for step in steps:
        states[step["id"]] = _step_to_state(step, notes)
    return {"StartAt": start, "States": states}


def _apply_output_reshape(
    state: dict, step: dict, sid: str, notes: list[str]
) -> None:
    """Apply workflow_spec's output/result_selector/result_path to Output.

    The schema allows result_path/result_selector on Task, Parallel, Map,
    Pass, and Wait -- this used to only run for Task steps, silently
    dropping the field on the other four types.
    """
    if step.get("output") is not None:
        state["Output"] = _convert_value_to_jsonata(step["output"])
    elif step.get("result_selector") is not None:
        state["Output"] = _convert_parameters_to_arguments(step["result_selector"])
    elif isinstance(step.get("result_path"), str) and step["result_path"].strip():
        # JSONPath ResultPath "$.foo" MERGES the task result under "foo"
        # while preserving the rest of the input. A bare Output object
        # {foo: result} would REPLACE the state output and drop prior
        # fields (s3.bucket, etc.), breaking downstream Arguments that
        # still read $states.input.s3.bucket. Use JSONata $merge so
        # behaviour matches ResultPath.
        rp = step["result_path"].strip()
        key = rp[2:] if rp.startswith("$.") else rp.lstrip("$")
        # Only simple top-level keys (no dots) are safe as object keys in
        # the merge expression; nested paths fall back to Assign.
        if key and re.match(r"^[A-Za-z0-9_]+$", key):
            state["Output"] = (
                "{% $merge([$states.input, {'" + key + "': $states.result}]) %}"
            )
            notes.append(
                f"Step '{sid}': mapped legacy result_path '{rp}' to "
                f"JSONata Output $merge with key '{key}' (preserves input)."
            )
        elif key and re.match(r"^[A-Za-z0-9_.]+$", key):
            # Nested path — keep result accessible via Assign variable
            # derived from the last segment.
            leaf = key.split(".")[-1] or "result"
            leaf = re.sub(r"[^A-Za-z0-9_]", "_", leaf)
            assign = state.get("Assign") if isinstance(state.get("Assign"), dict) else {}
            assign = dict(assign)
            assign[leaf] = "{% $states.result %}"
            state["Assign"] = assign
            # Still pass input through so later states keep context.
            state["Output"] = "{% $states.input %}"
            notes.append(
                f"Step '{sid}': mapped nested result_path '{rp}' to "
                f"Assign.{leaf} + Output=$states.input."
            )


def _step_to_state(step: dict, notes: list[str]) -> dict:
    """Translate one workflow_spec step into an ASL state object."""
    sid = step["id"]
    stype = step.get("type")
    if stype not in SUPPORTED_STATE_TYPES:
        raise AslGenerationError(f"Unsupported step type '{stype}' on step '{sid}'")

    state: dict[str, Any] = {"Type": stype}
    if step.get("description"):
        state["Comment"] = step["description"]

    if stype == "Task":
        resource = step.get("resource")
        if not resource:
            raise AslGenerationError(f"Task step '{sid}' missing resource")
        state["Resource"] = resource
        if step.get("parameters"):
            # Pure JSONata: Arguments replaces Parameters (no .$ path keys).
            state["Arguments"] = _convert_parameters_to_arguments(step["parameters"])
        _apply_output_reshape(state, step, sid, notes)
        if step.get("timeout_seconds") is not None:
            state["TimeoutSeconds"] = int(step["timeout_seconds"])
        retry = _retry_to_asl(step.get("retry"))
        if retry:
            state["Retry"] = retry
        catch = _catch_to_asl(step.get("catch"))
        if catch:
            state["Catch"] = catch
        _apply_common_transition(state, step)

    elif stype == "Wait":
        if step.get("seconds") is not None:
            state["Seconds"] = int(step["seconds"])
        elif step.get("timestamp"):
            state["Timestamp"] = step["timestamp"]
        elif step.get("seconds_path"):
            # JSONPath SecondsPath → JSONata Seconds expression
            state["Seconds"] = _as_jsonata_string(
                _jsonpath_to_jsonata_expr(step["seconds_path"])
            )
        else:
            raise AslGenerationError(f"Wait step '{sid}' needs seconds or timestamp")
        _apply_output_reshape(state, step, sid, notes)
        _apply_common_transition(state, step)

    elif stype == "Choice":
        choices = step.get("choices") or []
        if not choices:
            raise AslGenerationError(f"Choice step '{sid}' has no choices")
        state["Choices"] = [
            _choice_rule_from_condition(c.get("condition", ""), c["next_step"])
            for c in choices
            if c.get("next_step")
        ]
        if step.get("default_next"):
            state["Default"] = step["default_next"]
        # Choice never uses Next/End at the top level.

    elif stype == "Parallel":
        branches = step.get("branches") or []
        if not branches:
            raise AslGenerationError(f"Parallel step '{sid}' has no branches")
        state["Branches"] = [_build_branch_asl(b, notes) for b in branches]
        retry = _retry_to_asl(step.get("retry"))
        if retry:
            state["Retry"] = retry
        catch = _catch_to_asl(step.get("catch"))
        if catch:
            state["Catch"] = catch
        _apply_output_reshape(state, step, sid, notes)
        _apply_common_transition(state, step)

    elif stype == "Map":
        iterator = step.get("iterator")
        items_path = step.get("items_path") or step.get("items")
        if not iterator or not items_path:
            raise AslGenerationError(
                f"Map step '{sid}' requires iterator and items_path"
            )
        # Pure JSONata: Items replaces ItemsPath
        state["Items"] = (
            items_path
            if isinstance(items_path, str)
            and items_path.strip().startswith("{%")
            else _items_to_jsonata(str(items_path))
        )
        # ItemProcessor is the JSONata-era name; Iterator still accepted by
        # AWS for compatibility — emit ItemProcessor for pure JSONata ASL.
        state["ItemProcessor"] = _build_branch_asl(iterator, notes)
        max_concurrency = step.get("max_concurrency")
        if max_concurrency is not None:
            state["MaxConcurrency"] = max_concurrency
        items_batcher = step.get("items_batcher")
        if items_batcher and items_batcher.get("max_items_per_batch"):
            state["ItemBatcher"] = {
                "MaxItemsPerBatch": items_batcher["max_items_per_batch"]
            }
        retry = _retry_to_asl(step.get("retry"))
        if retry:
            state["Retry"] = retry
        catch = _catch_to_asl(step.get("catch"))
        if catch:
            state["Catch"] = catch
        _apply_output_reshape(state, step, sid, notes)
        _apply_common_transition(state, step)

    elif stype == "Succeed":
        pass  # terminal; no Next/End

    elif stype == "Fail":
        if step.get("error"):
            state["Error"] = step["error"]
        if step.get("cause"):
            state["Cause"] = step["cause"]

    elif stype == "Pass":
        if step.get("parameters"):
            # Pass with Assign-like reshape via Output in JSONata
            state["Output"] = _convert_parameters_to_arguments(step["parameters"])
        elif step.get("result") is not None:
            state["Output"] = step["result"]
        else:
            _apply_output_reshape(state, step, sid, notes)
        _apply_common_transition(state, step)

    return state


def _collect_assign_var_names(spec: dict) -> set[str]:
    """Names Catch (and similar) will bind via Assign from result_path.

    Walks the whole steps tree so nested Map/Parallel catchers count too.
    Every Catch gets an Assign (default ``errorInfo``) even when the
    spec omitted result_path -- keep this set in sync with _catch_to_asl.
    """
    names: set[str] = set()
    for step in _iter_spec_steps(spec.get("steps")):
        for c in step.get("catch") or []:
            if not isinstance(c, dict):
                continue
            if not c.get("next_step"):
                continue
            rp = c.get("result_path")
            if isinstance(rp, str) and rp.strip():
                key = rp.strip()
                key = key[2:] if key.startswith("$.") else key.lstrip("$")
                key = re.sub(r"[^A-Za-z0-9_]", "_", key) or "errorInfo"
            else:
                key = "errorInfo"
            names.add(key)
    return names


def _collect_catch_target_ids(spec: dict) -> set[str]:
    """Step ids that are Catch / on_failure destinations."""
    targets: set[str] = set()
    for step in _iter_spec_steps(spec.get("steps")):
        for c in step.get("catch") or []:
            if isinstance(c, dict) and c.get("next_step"):
                targets.add(str(c["next_step"]))
        for fid in step.get("on_failure") or []:
            if fid:
                targets.add(str(fid))
    return targets


def _message_references_error(msg: Any, assign_vars: set[str]) -> bool:
    if not isinstance(msg, str):
        return False
    if "errorOutput" in msg:
        return True
    for v in assign_vars:
        if v and v in msg:
            return True
    return False


def _ensure_failure_messages_include_error(
    asl: dict,
    spec: dict,
    notes: list[str],
) -> None:
    """Guarantee Catch-target alert steps include the assigned error in Message.

    Enrichment sometimes emits a static SNS/email body on a failure
    notifier. In JSONata, Catch binds the error via Assign (e.g.
    $errorInfo); static text alone hides the failure cause from ops.
    This is a mechanical, service-agnostic pass: any state that is a
    Catch target and exposes a string ``Message`` (or ``HtmlMessage`` /
    ``Text``) gets the error variable appended when missing.
    """
    targets = _collect_catch_target_ids(spec)
    if not targets:
        return
    assign_vars = set(_ASSIGN_VARS) or {"errorInfo"}
    err_var = "errorInfo" if "errorInfo" in assign_vars else next(iter(assign_vars))
    err_ref = f"${err_var}"

    body_keys = ("Message", "HtmlMessage", "Text", "Body")

    states = asl.get("States") or {}
    for name, state in states.items():
        if name not in targets or not isinstance(state, dict):
            continue
        args = state.get("Arguments")
        if not isinstance(args, dict):
            continue
        # Only touch notifier-like states that already have a body field
        # or are SNS/SES/Chat publish integrations.
        resource = str(state.get("Resource") or "")
        is_notifier = any(
            s in resource
            for s in (
                "sns:publish",
                "ses:sendEmail",
                "ses:sendRawEmail",
                "chime.",
                "sqs:sendMessage",
            )
        )
        has_body = any(k in args for k in body_keys)
        if not (is_notifier or has_body):
            continue

        for key in body_keys:
            if key not in args and not (is_notifier and key == "Message"):
                continue
            msg = args.get(key)
            if key not in args and is_notifier and key == "Message":
                args["Message"] = _as_jsonata_string(f"$string({err_ref})")
                notes.append(
                    f"State '{name}': set Message to $string({err_ref}) "
                    f"(Catch target had no Message)."
                )
                break
            if _message_references_error(msg, assign_vars):
                break
            if isinstance(msg, str) and msg.strip():
                s = msg.strip()
                if s.startswith("{%") and s.endswith("%}"):
                    inner = s[2:-2].strip()
                    args[key] = _as_jsonata_string(
                        f"($string({inner}) & ' | error: ' & $string({err_ref}))"
                    )
                else:
                    safe = s.replace("\\", "\\\\").replace("'", "\\'")
                    args[key] = _as_jsonata_string(
                        f"'{safe} | error: ' & $string({err_ref})"
                    )
                notes.append(
                    f"State '{name}': appended {err_ref} to {key} "
                    f"(Catch-target failure notifier)."
                )
                break
            else:
                args[key] = _as_jsonata_string(f"$string({err_ref})")
                notes.append(
                    f"State '{name}': set {key} to $string({err_ref}) "
                    f"(Catch target)."
                )
                break


def generate_asl(spec: dict) -> tuple[dict, list[str]]:
    """Translate a validated workflow_spec into a pure-JSONata ASL definition.

    Always sets top-level ``QueryLanguage: "JSONata"``. No JSONPath fields
    (Parameters, ItemsPath, Variable, InputPath, ResultPath, OutputPath,
    ResultSelector) are emitted.

    Returns (asl_dict, generation_notes). Raises AslGenerationError on
    structural problems that make translation impossible.
    """
    global _ASSIGN_VARS
    if not isinstance(spec, dict):
        raise AslGenerationError("workflow_spec must be a dict")
    steps = spec.get("steps")
    if not steps:
        raise AslGenerationError("workflow_spec.steps is empty")

    # Populate before _step_to_state so path conversion rewrites
    # $.errorInfo → $errorInfo for Catch Assign targets.
    _ASSIGN_VARS = _collect_assign_var_names(spec)

    notes: list[str] = []
    name = spec.get("name") or "GeneratedWorkflow"
    summary = spec.get("summary") or ""
    start_at = spec.get("start_at") or steps[0]["id"]

    states: dict[str, dict] = {}
    for step in steps:
        sid = step.get("id")
        if not sid:
            raise AslGenerationError("Every step must have an id")
        if sid in states:
            raise AslGenerationError(f"Duplicate top-level step id '{sid}'")
        states[sid] = _step_to_state(step, notes)

    asl: dict[str, Any] = {
        "Comment": f"{name}: {summary}".strip(": "),
        "QueryLanguage": "JSONata",
        "StartAt": start_at,
        "States": states,
    }
    if spec.get("version"):
        asl["Version"] = str(spec["version"])

    # Mechanical correctness: Catch-target notifiers always include the
    # assigned error payload in their Message/body (enrichment may have
    # left a static string).
    _ensure_failure_messages_include_error(asl, spec, notes)

    notes.append("Generated with QueryLanguage=JSONata (no JSONPath fields).")
    _ASSIGN_VARS = set()
    return asl, notes


# ---------------------------------------------------------------------------
# Validation group 1 — structural / syntactic
# ---------------------------------------------------------------------------

def _validate_json_shape(asl: Any, report: ValidationReport) -> dict | None:
    if not isinstance(asl, dict):
        report.issues.append(
            ValidationIssue(
                "structural",
                "not_object",
                "ASL root must be a JSON object",
            )
        )
        return None
    report.json_ok = True
    return asl


def _validate_structure(asl: dict, report: ValidationReport) -> None:
    ok = True
    if "StartAt" not in asl or not isinstance(asl["StartAt"], str) or not asl["StartAt"]:
        report.issues.append(
            ValidationIssue(
                "structural",
                "missing_start_at",
                "ASL must have a non-empty string StartAt",
            )
        )
        ok = False
    if "States" not in asl or not isinstance(asl["States"], dict) or not asl["States"]:
        report.issues.append(
            ValidationIssue(
                "structural",
                "missing_states",
                "ASL must have a non-empty States object",
            )
        )
        ok = False
        report.structure_ok = False
        return

    states: dict = asl["States"]

    def check_state(name: str, state: dict, path: str) -> None:
        nonlocal ok
        if not isinstance(state, dict):
            report.issues.append(
                ValidationIssue(
                    "structural",
                    "state_not_object",
                    f"State '{name}' is not an object",
                    path,
                )
            )
            ok = False
            return
        stype = state.get("Type")
        if stype not in SUPPORTED_STATE_TYPES:
            report.issues.append(
                ValidationIssue(
                    "structural",
                    "bad_type",
                    f"State '{name}' has unsupported Type '{stype}'",
                    path,
                )
            )
            ok = False
            return

        has_next = "Next" in state
        has_end = state.get("End") is True

        if stype in ("Succeed", "Fail"):
            if has_next or has_end:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "terminal_transition",
                        f"{stype} state '{name}' must not have Next or End",
                        path,
                    )
                )
                ok = False
            if stype == "Fail" and not state.get("Error"):
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "fail_missing_error",
                        f"Fail state '{name}' requires Error",
                        path,
                    )
                )
                ok = False
        elif stype == "Choice":
            choices = state.get("Choices")
            if not isinstance(choices, list) or not choices:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "choice_empty",
                        f"Choice state '{name}' needs a non-empty Choices array",
                        path,
                    )
                )
                ok = False
            else:
                for i, rule in enumerate(choices):
                    if not isinstance(rule, dict) or "Next" not in rule:
                        report.issues.append(
                            ValidationIssue(
                                "structural",
                                "choice_rule_bad",
                                f"Choice '{name}' rule[{i}] must be an object with Next",
                                f"{path}.Choices[{i}]",
                            )
                        )
                        ok = False
            if has_next or has_end:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "choice_has_next_end",
                        f"Choice state '{name}' must not use top-level Next/End",
                        path,
                    )
                )
                ok = False
        else:
            if has_next and has_end:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "next_and_end",
                        f"State '{name}' cannot have both Next and End",
                        path,
                    )
                )
                ok = False
            if not has_next and not has_end:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "missing_transition",
                        f"State '{name}' (Type={stype}) needs Next or End:true",
                        path,
                    )
                )
                ok = False

        if stype == "Task":
            if not state.get("Resource") or not isinstance(state["Resource"], str):
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "task_no_resource",
                        f"Task state '{name}' requires Resource string",
                        path,
                    )
                )
                ok = False
        elif stype == "Wait":
            wait_keys = ("Seconds", "Timestamp", "SecondsPath", "TimestampPath")
            present = [k for k in wait_keys if k in state]
            if len(present) != 1:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "wait_fields",
                        f"Wait state '{name}' must have exactly one of {wait_keys}",
                        path,
                    )
                )
                ok = False
        elif stype == "Parallel":
            branches = state.get("Branches")
            if not isinstance(branches, list) or not branches:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "parallel_no_branches",
                        f"Parallel state '{name}' needs non-empty Branches",
                        path,
                    )
                )
                ok = False
            else:
                for i, br in enumerate(branches):
                    if not isinstance(br, dict) or "StartAt" not in br or "States" not in br:
                        report.issues.append(
                            ValidationIssue(
                                "structural",
                                "branch_shape",
                                f"Parallel '{name}' branch[{i}] needs StartAt + States",
                                f"{path}.Branches[{i}]",
                            )
                        )
                        ok = False
                    else:
                        for sn, st in (br.get("States") or {}).items():
                            check_state(sn, st, f"{path}.Branches[{i}].States.{sn}")
        elif stype == "Map":
            if "Iterator" not in state and "ItemProcessor" not in state:
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "map_no_iterator",
                        f"Map state '{name}' needs Iterator (or ItemProcessor)",
                        path,
                    )
                )
                ok = False
            else:
                it = state.get("Iterator") or state.get("ItemProcessor") or {}
                if not isinstance(it, dict) or "StartAt" not in it or "States" not in it:
                    report.issues.append(
                        ValidationIssue(
                            "structural",
                            "iterator_shape",
                            f"Map '{name}' Iterator needs StartAt + States",
                            f"{path}.Iterator",
                        )
                    )
                    ok = False
                else:
                    for sn, st in (it.get("States") or {}).items():
                        check_state(sn, st, f"{path}.Iterator.States.{sn}")
            if (
                "Items" not in state
                and "ItemsPath" not in state
                and "ItemSelector" not in state
            ):
                # JSONata uses Items; legacy JSONPath used ItemsPath.
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "map_no_items",
                        f"Map state '{name}' should have Items (JSONata) or ItemsPath",
                        path,
                    )
                )
                ok = False

        for field_name, arr in (("Retry", state.get("Retry")), ("Catch", state.get("Catch"))):
            if arr is None:
                continue
            if not isinstance(arr, list):
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        f"{field_name.lower()}_not_array",
                        f"State '{name}'.{field_name} must be an array",
                        path,
                    )
                )
                ok = False
                continue
            for i, item in enumerate(arr):
                if not isinstance(item, dict) or not item.get("ErrorEquals"):
                    report.issues.append(
                        ValidationIssue(
                            "structural",
                            f"{field_name.lower()}_item",
                            f"State '{name}'.{field_name}[{i}] needs ErrorEquals",
                            f"{path}.{field_name}[{i}]",
                        )
                    )
                    ok = False
                if field_name == "Catch" and not item.get("Next"):
                    report.issues.append(
                        ValidationIssue(
                            "structural",
                            "catch_no_next",
                            f"State '{name}'.Catch[{i}] needs Next",
                            f"{path}.Catch[{i}]",
                        )
                    )
                    ok = False

    for name, state in states.items():
        check_state(name, state, f"States.{name}")

    report.structure_ok = ok


# ---------------------------------------------------------------------------
# Validation group 2 — graph / reachability
# ---------------------------------------------------------------------------

def _collect_outgoing(state: dict) -> list[str]:
    targets: list[str] = []
    if "Next" in state and isinstance(state["Next"], str):
        targets.append(state["Next"])
    if state.get("Default"):
        targets.append(state["Default"])
    for rule in state.get("Choices") or []:
        if isinstance(rule, dict) and rule.get("Next"):
            targets.append(rule["Next"])
    for c in state.get("Catch") or []:
        if isinstance(c, dict) and c.get("Next"):
            targets.append(c["Next"])
    return targets


def _walk_reachable(
    start: str,
    states: dict,
    seen: set[str],
    depth: int = 0,
) -> None:
    if start in seen or start not in states or depth > 500:
        return
    seen.add(start)
    state = states[start]
    for t in _collect_outgoing(state):
        _walk_reachable(t, states, seen, depth + 1)
    stype = state.get("Type")
    if stype == "Parallel":
        for br in state.get("Branches") or []:
            if isinstance(br, dict):
                br_states = br.get("States") or {}
                br_start = br.get("StartAt")
                if br_start:
                    _walk_reachable(br_start, br_states, set(), 0)
    elif stype == "Map":
        it = state.get("Iterator") or state.get("ItemProcessor") or {}
        if isinstance(it, dict) and it.get("StartAt"):
            _walk_reachable(it["StartAt"], it.get("States") or {}, set(), 0)


def _validate_reachability(asl: dict, report: ValidationReport) -> None:
    ok = True
    states: dict = asl.get("States") or {}
    start = asl.get("StartAt")

    if start not in states:
        report.issues.append(
            ValidationIssue(
                "reachability",
                "start_missing",
                f"StartAt '{start}' is not present in States",
            )
        )
        report.reachability_ok = False
        return

    # All Next/Default/Catch/Choice targets must resolve (top-level).
    for name, state in states.items():
        for t in _collect_outgoing(state):
            if t not in states:
                report.issues.append(
                    ValidationIssue(
                        "reachability",
                        "dangling_ref",
                        f"State '{name}' references missing state '{t}'",
                        f"States.{name}",
                    )
                )
                ok = False
        if state.get("Type") == "Parallel":
            for i, br in enumerate(state.get("Branches") or []):
                if not isinstance(br, dict):
                    continue
                br_states = br.get("States") or {}
                br_start = br.get("StartAt")
                if br_start and br_start not in br_states:
                    report.issues.append(
                        ValidationIssue(
                            "reachability",
                            "branch_start_missing",
                            f"Parallel '{name}' branch[{i}] StartAt '{br_start}' missing",
                            f"States.{name}.Branches[{i}]",
                        )
                    )
                    ok = False
                for sn, st in br_states.items():
                    for t in _collect_outgoing(st):
                        if t not in br_states:
                            report.issues.append(
                                ValidationIssue(
                                    "reachability",
                                    "branch_dangling",
                                    f"Parallel '{name}' branch[{i}] state '{sn}' -> missing '{t}'",
                                    f"States.{name}.Branches[{i}].States.{sn}",
                                )
                            )
                            ok = False
        if state.get("Type") == "Map":
            it = state.get("Iterator") or state.get("ItemProcessor") or {}
            it_states = (it or {}).get("States") or {}
            it_start = (it or {}).get("StartAt")
            if it_start and it_start not in it_states:
                report.issues.append(
                    ValidationIssue(
                        "reachability",
                        "iterator_start_missing",
                        f"Map '{name}' Iterator StartAt '{it_start}' missing",
                        f"States.{name}.Iterator",
                    )
                )
                ok = False
            for sn, st in it_states.items():
                for t in _collect_outgoing(st):
                    if t not in it_states:
                        report.issues.append(
                            ValidationIssue(
                                "reachability",
                                "iterator_dangling",
                                f"Map '{name}' Iterator state '{sn}' -> missing '{t}'",
                                f"States.{name}.Iterator.States.{sn}",
                            )
                        )
                        ok = False

    reachable: set[str] = set()
    _walk_reachable(start, states, reachable)
    unreachable = set(states.keys()) - reachable
    # States only reachable via Catch from an unreachable parent are still
    # flagged; pure dead code is almost always a generation bug.
    for name in sorted(unreachable):
        report.issues.append(
            ValidationIssue(
                "reachability",
                "unreachable",
                f"State '{name}' is not reachable from StartAt",
                f"States.{name}",
            )
        )
        ok = False

    # Must be able to terminate.
    def has_terminal(sts: dict, from_name: str, seen: set[str]) -> bool:
        if from_name in seen or from_name not in sts:
            return False
        seen.add(from_name)
        st = sts[from_name]
        t = st.get("Type")
        if t in ("Succeed", "Fail") or st.get("End") is True:
            return True
        for nxt in _collect_outgoing(st):
            if has_terminal(sts, nxt, seen):
                return True
        if t == "Parallel":
            # Parallel completes when all branches complete; each branch
            # must itself be able to terminate.
            for br in st.get("Branches") or []:
                if not isinstance(br, dict):
                    return False
                br_start = br.get("StartAt")
                if not br_start or not has_terminal(br.get("States") or {}, br_start, set()):
                    return False
            # After parallel, follow Next/End of the Parallel state itself.
            if st.get("End") is True:
                return True
            for nxt in _collect_outgoing(st):
                if has_terminal(sts, nxt, seen):
                    return True
            return False
        if t == "Map":
            it = st.get("Iterator") or st.get("ItemProcessor") or {}
            it_start = it.get("StartAt")
            if not it_start or not has_terminal(it.get("States") or {}, it_start, set()):
                return False
            if st.get("End") is True:
                return True
            for nxt in _collect_outgoing(st):
                if has_terminal(sts, nxt, seen):
                    return True
            return False
        return False

    if not has_terminal(states, start, set()):
        report.issues.append(
            ValidationIssue(
                "reachability",
                "no_terminal",
                "No path from StartAt reaches a Succeed/Fail or End:true state",
            )
        )
        ok = False

    report.reachability_ok = ok


# ---------------------------------------------------------------------------
# Validation group 3 — consistency with workflow_spec
# ---------------------------------------------------------------------------

def _validate_spec_consistency(
    asl: dict, spec: dict, report: ValidationReport
) -> None:
    ok = True
    states: dict = asl.get("States") or {}
    start = asl.get("StartAt")
    expected_start = spec.get("start_at") or (spec.get("steps") or [{}])[0].get("id")
    if expected_start and start != expected_start:
        report.issues.append(
            ValidationIssue(
                "spec_consistency",
                "start_mismatch",
                f"ASL StartAt '{start}' != workflow_spec start_at '{expected_start}'",
            )
        )
        ok = False

    # Top-level step ids must become top-level states (merged ones may be absent).
    merged = set(spec.get("merged_step_ids") or [])
    added = set(spec.get("added_step_ids") or [])
    top_spec_ids = {s["id"] for s in (spec.get("steps") or []) if s.get("id")}
    expected_states = top_spec_ids  # top-level only for States keys
    actual_states = set(states.keys())

    missing = expected_states - actual_states
    if missing:
        report.issues.append(
            ValidationIssue(
                "spec_consistency",
                "missing_states",
                f"ASL States missing ids present in workflow_spec.steps: {sorted(missing)}",
            )
        )
        ok = False

    extra = actual_states - expected_states
    if extra:
        report.issues.append(
            ValidationIssue(
                "spec_consistency",
                "extra_states",
                f"ASL States has ids not in top-level workflow_spec.steps: {sorted(extra)}",
            )
        )
        ok = False

    # Type fidelity for every top-level step that exists in both.
    type_map = {s["id"]: s.get("type") for s in (spec.get("steps") or [])}
    for sid, expected_type in type_map.items():
        if sid not in states:
            continue
        actual_type = states[sid].get("Type")
        if expected_type and actual_type != expected_type:
            report.issues.append(
                ValidationIssue(
                    "spec_consistency",
                    "type_mismatch",
                    f"State '{sid}' Type '{actual_type}' != spec type '{expected_type}'",
                    f"States.{sid}",
                )
            )
            ok = False

    # Control-flow: next_step / end / default_next / catch next_step
    for step in spec.get("steps") or []:
        sid = step.get("id")
        if not sid or sid not in states:
            continue
        state = states[sid]
        stype = step.get("type")

        if stype not in ("Choice", "Succeed", "Fail"):
            if step.get("end") is True:
                if state.get("End") is not True:
                    report.issues.append(
                        ValidationIssue(
                            "spec_consistency",
                            "end_missing",
                            f"Spec step '{sid}' has end:true but ASL state does not",
                            f"States.{sid}",
                        )
                    )
                    ok = False
            elif step.get("next_step"):
                if state.get("Next") != step["next_step"]:
                    report.issues.append(
                        ValidationIssue(
                            "spec_consistency",
                            "next_mismatch",
                            f"State '{sid}' Next '{state.get('Next')}' != "
                            f"spec next_step '{step['next_step']}'",
                            f"States.{sid}",
                        )
                    )
                    ok = False

        if stype == "Choice":
            if step.get("default_next") and state.get("Default") != step["default_next"]:
                report.issues.append(
                    ValidationIssue(
                        "spec_consistency",
                        "default_mismatch",
                        f"Choice '{sid}' Default mismatch",
                        f"States.{sid}",
                    )
                )
                ok = False
            spec_nexts = {c.get("next_step") for c in (step.get("choices") or [])}
            asl_nexts = {
                r.get("Next") for r in (state.get("Choices") or []) if isinstance(r, dict)
            }
            if spec_nexts - asl_nexts:
                report.issues.append(
                    ValidationIssue(
                        "spec_consistency",
                        "choice_targets",
                        f"Choice '{sid}' missing ASL targets for {sorted(spec_nexts - asl_nexts)}",
                        f"States.{sid}",
                    )
                )
                ok = False

        if stype == "Task":
            if step.get("resource") and state.get("Resource") != step["resource"]:
                report.issues.append(
                    ValidationIssue(
                        "spec_consistency",
                        "resource_mismatch",
                        f"Task '{sid}' Resource does not match workflow_spec",
                        f"States.{sid}",
                    )
                )
                ok = False

        # Catch targets
        spec_catch_nexts = {
            c.get("next_step") for c in (step.get("catch") or []) if c.get("next_step")
        }
        asl_catch_nexts = {
            c.get("Next") for c in (state.get("Catch") or []) if isinstance(c, dict)
        }
        if spec_catch_nexts - asl_catch_nexts:
            report.issues.append(
                ValidationIssue(
                    "spec_consistency",
                    "catch_targets",
                    f"State '{sid}' Catch missing targets {sorted(spec_catch_nexts - asl_catch_nexts)}",
                    f"States.{sid}",
                )
            )
            ok = False

    # added_step_ids must be Succeed/Fail only and present
    for aid in added:
        found = None
        for s in _iter_spec_steps(spec.get("steps")):
            if s.get("id") == aid:
                found = s
                break
        if not found:
            report.issues.append(
                ValidationIssue(
                    "spec_consistency",
                    "added_not_in_spec",
                    f"added_step_id '{aid}' not found in workflow_spec steps tree",
                )
            )
            ok = False
        elif found.get("type") not in ("Succeed", "Fail"):
            report.issues.append(
                ValidationIssue(
                    "spec_consistency",
                    "added_not_terminal",
                    f"added_step_id '{aid}' has type '{found.get('type')}' "
                    f"(only Succeed/Fail allowed)",
                )
            )
            ok = False

    report.spec_consistency_ok = ok


# ---------------------------------------------------------------------------
# Validation group 4 — AWS pragmatic constraints
# ---------------------------------------------------------------------------

# Fields that only make sense in legacy JSONPath-mode ASL. If any of these
# appear anywhere in a QueryLanguage=JSONata document, the generator (or a
# hand-edit) has leaked JSONPath into an otherwise-pure-JSONata definition,
# and AWS will reject the deploy. This is the invariant generate_asl()
# claims to guarantee end-to-end -- worth checking, not just assuming.
_JSONPATH_ONLY_FIELDS = {
    "InputPath",
    "OutputPath",
    "ResultPath",
    "ResultSelector",
    "Parameters",
    "ItemsPath",
}


def _check_jsonata_purity(asl: dict, report: ValidationReport) -> bool:
    ok = True
    if asl.get("QueryLanguage") != "JSONata":
        report.issues.append(
            ValidationIssue(
                "aws_constraints",
                "not_jsonata",
                "Top-level QueryLanguage must be 'JSONata' for this generator's output",
            )
        )
        return False

    def walk(node: Any, path: str) -> None:
        nonlocal ok
        if isinstance(node, dict):
            for key, val in node.items():
                if key in _JSONPATH_ONLY_FIELDS:
                    report.issues.append(
                        ValidationIssue(
                            "aws_constraints",
                            "jsonpath_field_leaked",
                            f"'{key}' is a JSONPath-only field but document is "
                            f"QueryLanguage=JSONata",
                            f"{path}.{key}",
                        )
                    )
                    ok = False
                if isinstance(key, str) and key.endswith(".$"):
                    report.issues.append(
                        ValidationIssue(
                            "aws_constraints",
                            "jsonpath_key_leaked",
                            f"Key '{key}' uses JSONPath '.$' suffix in a JSONata document",
                            f"{path}.{key}",
                        )
                    )
                    ok = False
                if key == "Condition" and isinstance(val, str):
                    v = val.strip()
                    if not (v.startswith("{%") and v.endswith("%}")):
                        report.issues.append(
                            ValidationIssue(
                                "aws_constraints",
                                "condition_not_wrapped",
                                f"Choice Condition must be wrapped in {{% ... %}}: {v!r}",
                                f"{path}.Condition",
                            )
                        )
                        ok = False
                walk(val, f"{path}.{key}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(asl, "$")
    return ok


def _validate_aws_constraints(asl: dict, report: ValidationReport) -> None:
    ok = _check_jsonata_purity(asl, report)
    states: dict = asl.get("States") or {}

    if len(states) > MAX_STATES_SOFT:
        report.issues.append(
            ValidationIssue(
                "aws_constraints",
                "too_many_states",
                f"State count {len(states)} exceeds soft limit {MAX_STATES_SOFT}",
            )
        )
        ok = False

    for name, state in states.items():
        if not STATE_NAME_RE.match(name):
            report.issues.append(
                ValidationIssue(
                    "aws_constraints",
                    "bad_state_name",
                    f"State name '{name}' must match [A-Za-z0-9_]{{1,80}}",
                    f"States.{name}",
                )
            )
            ok = False

        resource = state.get("Resource")
        if resource is not None:
            if not isinstance(resource, str) or not resource.strip():
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "empty_resource",
                        f"State '{name}' Resource must be a non-empty string",
                        f"States.{name}",
                    )
                )
                ok = False
            elif not (
                resource.startswith("arn:")
                or resource.startswith("TODO_")
            ):
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "resource_format",
                        f"State '{name}' Resource should be an ARN or TODO_ placeholder",
                        f"States.{name}",
                    )
                )
                ok = False

        for field_name, minimum in (
            ("TimeoutSeconds", 1),
            ("HeartbeatSeconds", 1),
            ("Seconds", 0),
        ):
            if field_name in state:
                val = state[field_name]
                if not isinstance(val, (int, float)) or val < minimum:
                    report.issues.append(
                        ValidationIssue(
                            "aws_constraints",
                            "numeric_limit",
                            f"State '{name}'.{field_name}={val} is invalid (min {minimum})",
                            f"States.{name}",
                        )
                    )
                    ok = False

        for i, r in enumerate(state.get("Retry") or []):
            if not isinstance(r, dict):
                continue
            if "IntervalSeconds" in r and (
                not isinstance(r["IntervalSeconds"], (int, float))
                or r["IntervalSeconds"] < 1
            ):
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "retry_interval",
                        f"State '{name}' Retry[{i}].IntervalSeconds must be >= 1",
                        f"States.{name}.Retry[{i}]",
                    )
                )
                ok = False
            if "MaxAttempts" in r and (
                not isinstance(r["MaxAttempts"], (int, float)) or r["MaxAttempts"] < 0
            ):
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "retry_max_attempts",
                        f"State '{name}' Retry[{i}].MaxAttempts must be >= 0",
                        f"States.{name}.Retry[{i}]",
                    )
                )
                ok = False
            if "BackoffRate" in r and (
                not isinstance(r["BackoffRate"], (int, float)) or r["BackoffRate"] < 1
            ):
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "retry_backoff",
                        f"State '{name}' Retry[{i}].BackoffRate must be >= 1",
                        f"States.{name}.Retry[{i}]",
                    )
                )
                ok = False
            errs = r.get("ErrorEquals")
            if not isinstance(errs, list) or not errs:
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "retry_error_equals",
                        f"State '{name}' Retry[{i}].ErrorEquals must be non-empty",
                        f"States.{name}.Retry[{i}]",
                    )
                )
                ok = False

        for i, c in enumerate(state.get("Catch") or []):
            if not isinstance(c, dict):
                continue
            errs = c.get("ErrorEquals")
            if not isinstance(errs, list) or not errs:
                report.issues.append(
                    ValidationIssue(
                        "aws_constraints",
                        "catch_error_equals",
                        f"State '{name}' Catch[{i}].ErrorEquals must be non-empty",
                        f"States.{name}.Catch[{i}]",
                    )
                )
                ok = False

        # Forbidden combinations already partly covered in structural;
        # double-check End + Next.
        if state.get("End") is True and "Next" in state:
            report.issues.append(
                ValidationIssue(
                    "aws_constraints",
                    "end_and_next",
                    f"State '{name}' has both End and Next (forbidden)",
                    f"States.{name}",
                )
            )
            ok = False

    report.aws_constraints_ok = ok


# ---------------------------------------------------------------------------
# Validation group 5 — extras (round-trip, live AWS API, graph sanity)
# ---------------------------------------------------------------------------

def _sfn_client():
    """Build a Step Functions client from API settings / default credential chain."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError(
            "boto3 is not installed — pip install boto3 to enable live AWS validation"
        ) from exc

    from orchestrator.api.config.settings import get_api_settings

    settings = get_api_settings()
    session_kwargs: dict[str, Any] = {}
    if settings.aws_profile:
        session_kwargs["profile_name"] = settings.aws_profile
    session = boto3.Session(**session_kwargs)
    return session.client("stepfunctions", region_name=settings.aws_region)


def call_aws_validate_state_machine(asl: dict) -> dict:
    """Call AWS ValidateStateMachineDefinition and return a details dict.

    Never raises — failures are captured in the returned structure so the
    rest of the validation report can still be produced.
    """
    from orchestrator.api.config.settings import get_api_settings

    settings = get_api_settings()
    if not settings.aws_validate_asl:
        return {
            "attempted": False,
            "skipped": True,
            "reason": "AWS_VALIDATE_ASL=false",
            "region": settings.aws_region,
        }

    try:
        client = _sfn_client()
        definition = json.dumps(asl)
        sm_type = (settings.aws_state_machine_type or "STANDARD").upper()
        if sm_type not in ("STANDARD", "EXPRESS"):
            sm_type = "STANDARD"

        response = client.validate_state_machine_definition(
            definition=definition,
            type=sm_type,
        )
        # AWS returns result = RESULT_PASS | RESULT_FAIL and diagnostics list.
        result = response.get("result") or response.get("Result")
        diagnostics = response.get("diagnostics") or response.get("Diagnostics") or []
        passed = result in ("RESULT_PASS", "PASS", True) or (
            result is None and not diagnostics
        )
        # Some SDK shapes only return diagnostics on failure.
        if result is None and diagnostics:
            severity_fail = any(
                (d.get("severity") or d.get("Severity") or "").upper()
                in ("ERROR", "FATAL")
                for d in diagnostics
                if isinstance(d, dict)
            )
            passed = not severity_fail

        return {
            "attempted": True,
            "passed": bool(passed),
            "result": result,
            "diagnostics": diagnostics,
            "region": settings.aws_region,
            "type": sm_type,
            "raw_keys": sorted(response.keys()),
        }
    except Exception as exc:  # noqa: BLE001 — surface any credential/API error
        error_code = None
        resp = getattr(exc, "response", None)
        if isinstance(resp, dict):
            error_code = (resp.get("Error") or {}).get("Code")
        name = type(exc).__name__
        # Credential / client-setup problems are environmental, not ASL
        # defects — report them but do not fail the overall validation.
        soft = name in {
            "NoCredentialsError",
            "PartialCredentialsError",
            "CredentialRetrievalError",
            "ProfileNotFound",
            "EndpointConnectionError",
            "ConnectTimeoutError",
            "ReadTimeoutError",
        } or error_code in {
            "UnrecognizedClientException",
            "InvalidClientTokenId",
            "ExpiredToken",
            "AccessDeniedException",
        }
        return {
            "attempted": True,
            "passed": False,
            "soft_failure": soft,
            "error": str(exc),
            "error_code": error_code,
            "region": settings.aws_region,
            "exception_type": name,
        }


def _validate_extras(
    asl: dict, spec: dict, report: ValidationReport, graph_dot: str
) -> None:
    ok = True

    # Round-trip: every top-level spec step id appears as a state name.
    top_ids = {s["id"] for s in (spec.get("steps") or []) if s.get("id")}
    state_ids = set((asl.get("States") or {}).keys())
    if top_ids != state_ids:
        report.issues.append(
            ValidationIssue(
                "extras",
                "round_trip_ids",
                f"Round-trip id set mismatch: only_in_spec={sorted(top_ids - state_ids)}, "
                f"only_in_asl={sorted(state_ids - top_ids)}",
            )
        )
        ok = False

    # Deploy readiness: TODO_ is valid ASL syntax (aws_constraints
    # above explicitly allows it) but not a deployable value. Checked
    # here, not there, because this is the group that answers "is this
    # actually ready to hand to CreateStateMachine" rather than "is
    # this syntactically sound ASL".
    report.unresolved_placeholders = find_asl_placeholders(asl)
    if report.unresolved_placeholders:
        report.issues.append(
            ValidationIssue(
                "extras",
                "unresolved_placeholders",
                "Unresolved TODO_ placeholders remain: "
                + ", ".join(report.unresolved_placeholders),
            )
        )
        ok = False

    # Graph DOT must at least mention StartAt and contain digraph.
    if not graph_dot or "digraph" not in graph_dot:
        report.issues.append(
            ValidationIssue(
                "extras",
                "graph_empty",
                "Graph DOT is empty or missing digraph header",
            )
        )
        ok = False
    else:
        start = asl.get("StartAt") or ""
        if start and f'"{start}"' not in graph_dot and start not in graph_dot:
            report.issues.append(
                ValidationIssue(
                    "extras",
                    "graph_missing_start",
                    f"Graph DOT does not mention StartAt state '{start}'",
                )
            )
            ok = False

    # Live AWS ValidateStateMachineDefinition
    details = call_aws_validate_state_machine(asl)
    report.aws_details = details
    if details.get("skipped") or not details.get("attempted"):
        report.aws_validate_ok = None
    elif details.get("passed"):
        report.aws_validate_ok = True
    elif details.get("soft_failure"):
        # Env/credential problem — show in UI but do not fail extras/overall.
        report.aws_validate_ok = None
        report.issues.append(
            ValidationIssue(
                "extras",
                "aws_validate_unavailable",
                details.get("error") or "AWS validation unavailable (credentials/network)",
            )
        )
    else:
        report.aws_validate_ok = False
        msg = details.get("error") or details.get("result") or "AWS validation failed"
        diagnostics = details.get("diagnostics") or []
        if diagnostics:
            diag_txt = "; ".join(
                str(d.get("message") or d.get("Message") or d)
                for d in diagnostics[:5]
            )
            msg = f"{msg}: {diag_txt}"
        report.issues.append(
            ValidationIssue(
                "extras",
                "aws_validate_failed",
                msg,
            )
        )
        ok = False

    report.extras_ok = ok


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def validate_asl(asl: Any, spec: dict | None = None, graph_dot: str = "") -> ValidationReport:
    """Run all five validation groups. `spec` is required for groups 3 & 5
    full fidelity; AWS live validation still runs when asl is valid JSON.
    """
    report = ValidationReport()
    root = _validate_json_shape(asl, report)
    if root is None:
        return report

    _validate_structure(root, report)
    _validate_reachability(root, report)

    if spec is not None:
        _validate_spec_consistency(root, spec, report)
    else:
        report.spec_consistency_ok = True  # skipped, not failed

    _validate_aws_constraints(root, report)

    if spec is not None:
        _validate_extras(root, spec, report, graph_dot)
    else:
        # No spec to round-trip against, but deploy-readiness still
        # applies -- same placeholder scan _validate_extras() would
        # have run, just without the spec-dependent round-trip check.
        ok = True
        report.unresolved_placeholders = find_asl_placeholders(root)
        if report.unresolved_placeholders:
            report.issues.append(
                ValidationIssue(
                    "extras",
                    "unresolved_placeholders",
                    "Unresolved TODO_ placeholders remain: "
                    + ", ".join(report.unresolved_placeholders),
                )
            )
            ok = False
        report.extras_ok = ok
        details = call_aws_validate_state_machine(root)
        report.aws_details = details
        if details.get("skipped") or not details.get("attempted"):
            report.aws_validate_ok = None
        elif details.get("passed"):
            report.aws_validate_ok = True
        elif details.get("soft_failure"):
            report.aws_validate_ok = None
            report.issues.append(
                ValidationIssue(
                    "extras",
                    "aws_validate_unavailable",
                    details.get("error")
                    or "AWS validation unavailable (credentials/network)",
                )
            )
        else:
            report.aws_validate_ok = False
            report.issues.append(
                ValidationIssue(
                    "extras",
                    "aws_validate_failed",
                    details.get("error")
                    or details.get("result")
                    or "AWS validation failed",
                )
            )
            report.extras_ok = False
        if graph_dot and "digraph" not in graph_dot:
            report.extras_ok = False

    return report


# ---------------------------------------------------------------------------
# Graphviz rendering (Step Functions console style)
# ---------------------------------------------------------------------------

_TYPE_STYLE = {
    "Task": ("#E0E7FF", "#4338CA", "box"),
    "Choice": ("#FEF3C7", "#B45309", "diamond"),
    "Wait": ("#E0F2FE", "#0369A1", "hexagon"),
    "Parallel": ("#F3E8FF", "#7C3AED", "box"),
    "Map": ("#FCE7F3", "#BE185D", "box"),
    "Succeed": ("#D4F4DD", "#2F855A", "box"),
    "Fail": ("#FDE2E1", "#C53030", "box"),
    "Pass": ("#F3F4F6", "#4B5563", "box"),
}


def _dot_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


def _state_label(name: str, state: dict) -> str:
    stype = state.get("Type", "?")
    comment = state.get("Comment") or ""
    resource = state.get("Resource") or ""
    lines = [name, f"[{stype}]"]
    if stype == "Task" and resource:
        short = resource.split(":::")[-1] if ":::" in resource else resource
        if len(short) > 40:
            short = short[:37] + "..."
        lines.append(short)
    elif comment:
        c = comment if len(comment) <= 40 else comment[:37] + "..."
        lines.append(c)
    return "\\n".join(_dot_escape(x) for x in lines)


def render_asl_graph(asl: dict) -> str:
    """Produce Graphviz DOT approximating the Step Functions console graph.

    Recurses into Map Iterator/ItemProcessor and Parallel Branches so nested
    states (and their Next / Choice / Catch edges) appear in the diagram —
    not only the iterator entry node.
    """
    root_states: dict = asl.get("States") or {}
    start = asl.get("StartAt") or ""
    lines = [
        "digraph ASL {",
        "  rankdir=TB;",
        '  graph [pad="0.3", nodesep="0.4", ranksep="0.5"];',
        '  node [style="rounded,filled", fontname="Helvetica", fontsize=11];',
        '  edge [fontname="Helvetica", fontsize=9];',
        '  __start [label="Start", shape=circle, fillcolor="#111827", '
        'fontcolor="white", width=0.4, height=0.4];',
    ]
    if start:
        lines.append(f'  __start -> "{_dot_escape(start)}";')

    end_used = False

    def ensure_end_node() -> None:
        nonlocal end_used
        if end_used:
            return
        end_used = True
        lines.append(
            '  __end [label="End", shape=circle, fillcolor="#111827", '
            'fontcolor="white", width=0.4, height=0.4];'
        )

    drawn_nodes: set[str] = set()
    drawn_edges: set[tuple[str, str, str]] = set()

    def emit_node(name: str, state: dict) -> None:
        if name in drawn_nodes:
            return
        drawn_nodes.add(name)
        stype = state.get("Type", "Task")
        fill, stroke, shape = _TYPE_STYLE.get(stype, ("#E5E7EB", "#374151", "box"))
        label = _state_label(name, state)
        lines.append(
            f'  "{_dot_escape(name)}" [label="{label}", fillcolor="{fill}", '
            f'color="{stroke}", shape={shape}];'
        )

    def edge(
        src: str, dst: str, style: str, color: str, label: str | None = None
    ) -> None:
        key = (src, dst, style + (label or ""))
        if key in drawn_edges:
            return
        drawn_edges.add(key)
        lab = f', label="{_dot_escape(label)}"' if label else ""
        lines.append(
            f'  "{_dot_escape(src)}" -> "{_dot_escape(dst)}" '
            f'[style={style}, color="{color}"{lab}];'
        )

    def walk_states(states: dict, *, cluster_label: str | None = None) -> None:
        if not isinstance(states, dict) or not states:
            return

        cluster_id = None
        if cluster_label:
            cluster_id = f"cluster_{abs(hash(cluster_label)) % 10_000_000}"
            lines.append(f"  subgraph {cluster_id} {{")
            lines.append("    style=rounded;")
            lines.append('    color="#CBD5E1";')
            lines.append(f'    label="{_dot_escape(cluster_label)}";')
            lines.append("    fontsize=10;")

        for name, state in states.items():
            if not isinstance(state, dict):
                continue
            emit_node(name, state)
            stype = state.get("Type")

            if "Next" in state and state["Next"]:
                edge(name, state["Next"], "solid", "#4B5563")

            # Terminal states always link to the shared End marker —
            # including inside Map iterators and Parallel branches.
            # Restricting this to top-level only hid every per-item End
            # (Persist, Notify, …) and made graphs look like they had no
            # terminal whenever the only Ends lived under a Map.
            # Succeed/Fail are terminals by type (no End field required).
            if state.get("End") is True or stype in ("Succeed", "Fail"):
                ensure_end_node()
                edge(name, "__end", "solid", "#4B5563")

            if state.get("Default"):
                edge(name, state["Default"], "dashed", "#B45309", "default")

            for rule in state.get("Choices") or []:
                if isinstance(rule, dict) and rule.get("Next"):
                    lab = None
                    cond = rule.get("Condition")
                    if isinstance(cond, str) and cond.strip():
                        # "{% $states.input.status = 'approved' %}" → status = approved
                        t = cond.strip()
                        if t.startswith("{%") and t.endswith("%}"):
                            t = t[2:-2].strip()
                        t = t.replace("$states.input.", "").replace("$states.context.", "")
                        lab = t[:28] + ("..." if len(t) > 28 else "")
                    elif "StringEquals" in rule:
                        lab = str(rule["StringEquals"])
                    elif "NumericGreaterThan" in rule:
                        lab = f"> {rule['NumericGreaterThan']}"
                    elif "NumericGreaterThanEquals" in rule:
                        lab = f">= {rule['NumericGreaterThanEquals']}"
                    elif "Comment" in rule:
                        lab = str(rule["Comment"])[:30]
                    edge(name, rule["Next"], "dashed", "#B45309", lab)

            for c in state.get("Catch") or []:
                if isinstance(c, dict) and c.get("Next"):
                    errs = c.get("ErrorEquals") or []
                    lab = "catch"
                    if errs and errs != ["States.ALL"]:
                        lab = "catch: " + ",".join(str(e) for e in errs[:2])
                    edge(name, c["Next"], "dashed", "#C53030", lab)

            if stype == "Parallel":
                for i, br in enumerate(state.get("Branches") or []):
                    if not isinstance(br, dict):
                        continue
                    br_start = br.get("StartAt")
                    br_states = br.get("States") or {}
                    if br_start:
                        edge(name, br_start, "solid", "#7C3AED", f"branch {i}")
                    if isinstance(br_states, dict) and br_states:
                        walk_states(
                            br_states,
                            cluster_label=f"{name} branch {i}",
                        )

            if stype == "Map":
                it = state.get("Iterator") or state.get("ItemProcessor") or {}
                if not isinstance(it, dict):
                    continue
                it_start = it.get("StartAt")
                it_states = it.get("States") or {}
                if it_start:
                    edge(name, it_start, "solid", "#BE185D", "iterator")
                if isinstance(it_states, dict) and it_states:
                    walk_states(
                        it_states,
                        cluster_label=f"{name} iterator",
                    )

        if cluster_id:
            lines.append("  }")

    walk_states(root_states)
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_asl_from_spec(spec: dict) -> AslResult:
    """Generate ASL, render graph, run all validations. Always returns a
    result object; check ``result.validations.ok`` before deploying.
    """
    asl, notes = generate_asl(spec)
    graph_dot = render_asl_graph(asl)
    report = validate_asl(asl, spec=spec, graph_dot=graph_dot)
    return AslResult(
        asl=asl,
        validations=report,
        graph_dot=graph_dot,
        generation_notes=notes,
    )