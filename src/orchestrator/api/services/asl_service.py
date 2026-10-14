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

# Matches the first path segment of a "$states.input.<key>" reference
# inside a JSONata expression, e.g. "foo" in "$states.input.foo.bar[0]".
# Used by _check_output_consumer_consistency to check that a downstream
# state only reads keys its predecessor's Output actually produces.
_STATES_INPUT_REF_RE = re.compile(r"\$states\.input\.([A-Za-z_][A-Za-z0-9_]*)")

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
    """Fold a step description into a trailing clause for help text.

    - Humanizes any TODO_ tokens so the fill form never echoes raw
      placeholder names.
    - Lowercases the first letter for mid-sentence use.
    - Caller is responsible for the connecting preposition ("to",
      "when", etc.); use _clause_connector() for that.
    """
    d = (desc or "").strip().rstrip(".")
    if not d:
        return d
    d = _PLACEHOLDER_PATTERN.sub(
        lambda m: _humanize_todo_token(m.group(0)), d
    )
    return d[0].lower() + d[1:] if d else d


def _clause_connector(clause: str) -> str:
    """Pick 'to' / 'when' / 'if' / 'after' so help text stays grammatical.

    Step descriptions often already start with a subordinate conjunction
    ("if the job fails…", "when approved…"). Prefixing those with "to"
    produces "to if the job fails…". Detect that and skip the extra "to".
    """
    c = (clause or "").lstrip()
    if not c:
        return ""
    first = c.split(None, 1)[0].lower() if c else ""
    if first in {
        "if", "when", "whenever", "after", "before", "once", "while",
        "unless", "until", "because", "since", "although", "though",
        "where", "wherever", "as",
    }:
        return f" {c}"
    return f" to {c}"


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

        # Precompute every step that references each token, so a value
        # asked for once but reused by several steps (see the "ask once,
        # reuse everywhere" wiring rule) gets a description that says so,
        # instead of only crediting whichever step happened to be seen
        # first while walking the list below.
        token_step_ids: dict[str, list[str]] = {}
        if isinstance(components, list):
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                cid0 = str(comp.get("id") or comp.get("name") or "")
                resource0 = str(comp.get("resource") or comp.get("service") or "")
                service_field0 = str(comp.get("service") or "")
                params_blob0 = json.dumps(comp.get("parameters") or {}, default=str)
                cdesc0 = (comp.get("description") or "").strip()
                for t in token_set:
                    if not t:
                        continue
                    if (
                        t == resource0.strip()
                        or t in params_blob0
                        or t in service_field0
                        or t in cdesc0
                    ):
                        ids = token_step_ids.setdefault(t, [])
                        if cid0 and cid0 not in ids:
                            ids.append(cid0)

        def _also_used_by(t: str, primary_cid: str) -> str:
            """' Step X also uses this same value.' / '' if not shared."""
            extra = [s for s in token_step_ids.get(t, []) if s and s != primary_cid]
            if not extra:
                return ""
            names = [f"`{s}`" for s in extra]
            if len(names) == 1:
                joined = names[0]
                verb = "also uses"
            elif len(names) == 2:
                joined = f"{names[0]} and {names[1]}"
                verb = "also use"
            else:
                joined = ", ".join(names[:-1]) + f", and {names[-1]}"
                verb = "also use"
            return f" Step {joined} {verb} this same answer, so you only need to fill it in once."

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
                        clause = _as_clause(cdesc)
                        if hint:
                            label, arn_example, _name_example = hint
                            details[t] = (
                                f"The {label} that {step_ref} will use"
                                + (_clause_connector(clause) if clause else "")
                                + ". Enter its full AWS ID (called an ARN) —"
                                f" it looks like `{arn_example}`."
                            )
                        else:
                            step_ref_cap = f"Step `{cid}`" if cid else "This step"
                            details[t] = (
                                f"{step_ref_cap} needs the full AWS ID "
                                "(called an ARN) of whatever it calls. We "
                                "couldn't tell exactly what kind of resource "
                                "this is — it might be a Lambda function, an "
                                "SNS topic, a DynamoDB table, or similar."
                            )
                    else:
                        clause = _as_clause(cdesc)
                        leaf = (field_key or "").split(".")[-1] if field_key else ""
                        if as_key_only:
                            # e.g. TODO_ATTRIBUTE_NAME as a DynamoDB Item key
                            details[t] = (
                                f"You're naming a field here, not entering data. "
                                f"Whatever you type becomes the label {step_ref} "
                                "uses" + (f" when it {clause}." if clause else ".")
                            )
                        # Field-aware label: JobName + Glue → "Glue job name";
                        # TopicArn + SNS → "SNS topic ARN". Generic for any
                        # service via resource-derived hint + field leaf.
                        elif hint and kind in ("arn", "name"):
                            label, arn_example, name_example = hint
                            if kind == "arn":
                                field_bit = f" ({leaf})" if leaf and not leaf.lower().endswith("arn") else ""
                                details[t] = (
                                    f"The full AWS ID (ARN) of the {label}"
                                    f"{field_bit} that {step_ref} uses"
                                    + (_clause_connector(clause) if clause else "")
                                    + f". It looks like `{arn_example}`."
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
                                        f"The {label.split()[0]} {role_h} that "
                                        f"{step_ref} uses"
                                        + (_clause_connector(clause) if clause else "")
                                        + f". Just the plain name — not the "
                                        f"full ARN — e.g. `{name_example}`."
                                    )
                                else:
                                    details[t] = (
                                        f"The name of the {label} that "
                                        f"{step_ref} uses"
                                        + (_clause_connector(clause) if clause else "")
                                        + f". Just the plain name — not the "
                                        f"full ARN — e.g. `{name_example}`."
                                    )
                        elif field_key and kind == "arn":
                            details[t] = (
                                f"The full AWS ID (ARN) for `{leaf}`, needed "
                                f"by {step_ref}"
                                + (f" — {clause}." if clause else ".")
                            )
                        elif field_key and kind == "name":
                            details[t] = (
                                f"The plain name (not a full ARN) to use for "
                                f"`{leaf}` in {step_ref}"
                                + (f" — {clause}." if clause else ".")
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
                                f"A value {step_ref} needs — something "
                                f"related to the {label}."
                            )
                        elif cid:
                            details[t] = (
                                f"A value step `{cid}` needs: "
                                f"{_humanize_todo_token(t).lower()}."
                            )

                    # A value asked for once but reused by other steps (the
                    # "ask once, reuse everywhere" wiring rule) should say
                    # so, regardless of which branch above wrote the text.
                    if t in details:
                        details[t] += _also_used_by(t, cid)

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
                            + ((_clause_connector(clause) + ".") if clause else ".")
                            + f" Must be the full ARN, e.g. `{arn_example}`."
                        )
                    elif kind == "name":
                        details[t] = (
                            f"{label} name used by {step_ref}"
                            + ((_clause_connector(clause) + ".") if clause else ".")
                            + f" Just the name, not the ARN — e.g. `{name_example}`."
                        )
                    else:
                        details[t] = (
                            f"{label} name or ARN used by {step_ref}"
                            + ((_clause_connector(clause) + ".") if clause else ".")
                            + f" e.g. `{arn_example}`."
                        )
                elif sdesc:
                    details[t] = (
                        f"AWS service + resource identifier (e.g. a Lambda "
                        f"function name/ARN or SNS topic ARN) used by "
                        f"{step_ref}"
                        + (_clause_connector(clause) if clause else "")
                        + "."
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


# Keys that are exactly a TODO_ token (or contain only such tokens) must
# rewrite to a structural identifier, not a runtime data value.
_STRUCTURAL_ARG_KEY_RE = re.compile(
    r"^(--[A-Za-z0-9][A-Za-z0-9_\-]*|[A-Za-z][A-Za-z0-9_]*)$"
)


def _looks_like_data_value_not_identifier(s: str) -> bool:
    """True when a string looks like a runtime data value, not a structural
    field / argument name.

    Used to refuse rewriting dict keys into values (e.g. replacing a
    ``TODO_GLUE_JOB_ARG_SOURCE`` key with ``s3://bucket/key``), which
    produces invalid ASL. Safe identifiers and conventional argument
    names (``JobName``, ``--source_csv_uri``, ``TableName``) return False.
    """
    if not s or not isinstance(s, str):
        return False
    v = s.strip()
    if not v:
        return False
    lower = v.lower()
    if lower.startswith(("s3://", "http://", "https://", "arn:")):
        return True
    # Spaces or JSON/path punctuation → almost certainly a value
    if any(ch in v for ch in (" ", "{", "}", "[", "]", '"', "'")):
        return True
    # JSONata / path expressions are values, not keys
    if v.startswith("{%") or v.startswith("$"):
        return True
    return False


def _is_safe_structural_key(original_key: str, candidate: str) -> bool:
    """Whether rewriting ``original_key`` → ``candidate`` is safe.

    - Always reject candidates that look like runtime data values.
    - When the original key is itself a TODO_ placeholder (the dangerous
      enrichment pattern), require the candidate to look like a real
      argument/field name (``--foo_bar`` or ``CamelCase`` / ``snake_case``).
    """
    if _looks_like_data_value_not_identifier(candidate):
        return False
    # Original key is exactly one TODO_ token → only accept identifier-shaped
    # replacements so "DynamoDB" or a URI cannot become the key by accident.
    if _PLACEHOLDER_PATTERN.fullmatch(original_key.strip()):
        return bool(_STRUCTURAL_ARG_KEY_RE.match(candidate.strip()))
    return True


# Whole-string shapes that are a REFERENCE to a single TODO_ token, not
# text that happens to contain one -- used by apply_placeholder_overrides
# to replace the entire expression (not splice a substring) when that
# token gets filled in. Covers the token at every stage it can appear:
#   pre-translation JSONPath  (workflow_spec parameters): "$.TODO_X"
#   post-translation JSONata, wrapped   (generated ASL):  "{% $states.input.TODO_X %}"
#   post-translation JSONata, unwrapped (e.g. a Condition
#     string assembled without the {% %} delimiters yet): "$states.input.TODO_X"
_FULL_TODO_REFERENCE_PATTERNS = [
    re.compile(r"^\{%\s*\$states\.input\.(TODO_[A-Z0-9_]+)\s*%\}$"),
    re.compile(r"^\$states\.input\.(TODO_[A-Z0-9_]+)$"),
    re.compile(r"^\$\.(TODO_[A-Z0-9_]+)$"),
]


def apply_placeholder_overrides(node: Any, overrides: dict[str, str]) -> Any:
    """Deep-replace TODO_ tokens in a JSON tree (string values AND dict keys).

    Used after Stage 4 so user-supplied values from /approve overrides
    land in the locked ASL / workflow_spec — not only in the thin Plan
    service/description fields. Enrichment may still emit TODOs in
    Parameters, Resource, and (rarely) as DynamoDB Item attribute *keys*;
    without key rewriting, Apply placeholders would leave those keys
    unresolved.

    Safety: a key is rewritten only when the replacement still looks like
    a structural identifier (argument/field name). If the fill value looks
    like runtime data (URI, ARN, JSONata, free text), or when a pure
    TODO_ key would become a non-identifier, the key is left unchanged so
    we never produce ASL such as ``{"s3://bucket/key": "..."}`` under
    Glue Arguments.

    Whole-expression TODO references get whole-expression replacement,
    not a substring splice: a workflow_spec parameter written as
    ``"--source-bucket.$": "$.TODO_SOURCE_S3_BUCKET"`` (pre-translation
    JSONPath) or an already-generated ``"{% $states.input.TODO_SOURCE_S3_BUCKET %}"``
    (post-translation JSONata) is a *reference* to the token, not text
    containing it -- filling the token means "this whole expression is
    now the literal value", not "splice the literal into the middle of
    the path". A blind ``str.replace`` turns
    ``{% $states.input.TODO_SOURCE_S3_BUCKET %}`` into
    ``{% $states.input.my-input-bucket %}`` -- still-broken JSONata (an
    unquoted hyphenated identifier), just with the corruption spliced
    one level deeper instead of left bare. Matching the whole reference
    first and substituting the entire node with the plain literal
    produces the correct ``"my-input-bucket"`` / ``"s3://bucket/key"``
    instead. Free text, ARNs, and plain non-path literal values (e.g.
    ``"JobName": "TODO_JOB_NAME"``) don't match these patterns and keep
    using the existing substring behavior below.
    """
    if not overrides:
        return node
    if isinstance(node, str):
        stripped = node.strip()
        for pattern in _FULL_TODO_REFERENCE_PATTERNS:
            m = pattern.match(stripped)
            if m:
                token = m.group(1)
                if token in overrides and overrides[token] is not None:
                    return overrides[token]
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
                        candidate = k.replace(token, value)
                        if not _is_safe_structural_key(k, candidate):
                            continue
                        new_k = candidate
            new_v = apply_placeholder_overrides(v, overrides)
            # Path-style TODO under a '.$' key filled with a constant:
            # strip the '.$' so generation does not re-wrap the constant
            # as invalid JSONata (see _convert_parameters_to_arguments).
            if (
                isinstance(new_k, str)
                and new_k.endswith(".$")
                and isinstance(new_v, str)
                and not _is_path_or_jsonata_expression(new_v)
            ):
                new_k = new_k[:-2]
            new_dict[new_k] = new_v
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
    # Which generator actually produced `asl`: "mechanical" (the default,
    # deterministic translator), "llm" (asl_generation/llm_service.py), or
    # "mechanical_fallback" (LLM path failed under asl_generation_mode
    # "llm_with_fallback" and the mechanical translator ran instead). Purely
    # informational -- nothing downstream requires a particular value.
    source: str = "mechanical"


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

# A "{% ... %}" wrapper whose contents look like a bare, unquoted literal
# rather than an actual JSONata expression: no leading "$" (no state
# input / context / workflow-variable reference), no operators, no
# function calls, no string quoting -- just letters/digits/./_/-//:.
# This is the shape a bucket name, table name, job name, etc. takes
# when something upstream mistakenly wrapped a plain constant instead
# of leaving it as a JSON literal. Real JSONata expressions always
# involve at least one of "$", "(", quotes, or an operator, so this
# pattern should never false-positive on legitimate JSONata.
_BARE_LITERAL_INSIDE_WRAPPER = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_./:-]*$"
)


def _is_path_or_jsonata_expression(s: str) -> bool:
    """True when ``s`` is a runtime path / JSONata expression, not a constant.

    Used by Parameters→Arguments conversion so that a filled-in literal
    under a leftover ``.$`` key (e.g. ``"--bucket.$": "my-input-bucket"``
    after placeholder fill replaced ``$.TODO_BUCKET``) is emitted as a
    plain JSON string, never wrapped in ``{% ... %}``.

    Real expressions always start with ``$`` (JSONPath / JSONata / $var)
    or ``{%``, or are an unresolved ``TODO_*`` token that must bind from
    execution input. Everything else is a constant (bucket name, S3 URI,
    table name, file key, etc.).
    """
    text = (s or "").strip()
    if not text:
        return False
    if text.startswith("$") or text.startswith("{%"):
        return True
    if _TODO_TOKEN.match(text):
        return True
    return False


def _unwrap_if_mistakenly_wrapped_literal(s: str) -> str:
    """If ``s`` is a "{% ... %}" string whose inner text is a bare literal
    (not a real JSONata expression), return the unwrapped plain string.
    Otherwise return ``s`` unchanged.

    This guards the "already wrapped, trust it" fast path in
    ``_convert_value_to_jsonata`` / ``_convert_parameters_to_arguments``:
    without it, a literal that some upstream stage incorrectly wrapped
    (e.g. ``"{% my-input-bucket %}"``) sails through unchanged and
    reaches AWS as invalid JSONata (the un-quoted hyphens parse as
    subtraction). Genuine JSONata -- ``$states.input.x``, ``$errorInfo``,
    ``'literal'``, function calls, comparisons -- never matches this
    pattern and is left untouched.
    """
    text = s.strip()
    if not (text.startswith("{%") and text.endswith("%}")):
        return s
    inner = text[2:-2].strip()
    if inner and _BARE_LITERAL_INSIDE_WRAPPER.match(inner):
        return inner
    return s


def sanitize_asl_literals(node: Any) -> Any:
    """Recursively apply ``_unwrap_if_mistakenly_wrapped_literal`` to every
    string in an already-built ASL document (any depth, any field).

    ``_convert_value_to_jsonata`` / ``_convert_parameters_to_arguments``
    only ever see this guard while translating a workflow_spec's
    ``parameters`` into ``Arguments`` during ``generate_asl()``. A raw
    ASL document handed to us directly -- e.g. a client-supplied payload
    to ``/asl/validate``, or any other path that bypasses spec-to-ASL
    translation -- never passes through that translator and so never
    gets the fix, even though it's exactly the kind of mistakenly
    wrapped literal (``"{% s3://bucket/key %}"``) the guard exists for.

    Safe to call on any ASL tree, generated or hand-edited, generated-
    then-re-validated, or freshly uploaded: genuine JSONata expressions
    (anything with ``$``, quotes, operators, or function calls) are left
    completely untouched by the underlying regex, so this is a no-op on
    already-correct documents.

    Also rewrites Catch[].ResultPath (JSONPath-only, illegal under
    QueryLanguage=JSONata) into Catch[].Assign matching the mechanical
    translator -- so an LLM that still copies result_path → ResultPath
    does not need a full repair-loop retry for that alone.
    """
    if isinstance(node, dict):
        out = {k: sanitize_asl_literals(v) for k, v in node.items()}
        catchers = out.get("Catch")
        if isinstance(catchers, list):
            out["Catch"] = [_sanitize_catch_entry(c) for c in catchers]
        return out
    if isinstance(node, list):
        return [sanitize_asl_literals(v) for v in node]
    if isinstance(node, str):
        return _unwrap_if_mistakenly_wrapped_literal(node)
    return node


def _sanitize_catch_entry(catcher: Any) -> Any:
    """Convert Catch ResultPath → Assign for JSONata purity.

    Mirrors ``_catch_to_asl``: key from ResultPath (strip ``$.``), default
    ``errorInfo``, value ``{% $states.errorOutput %}``. Drops ResultPath
    (and ResultSelector if present). Leaves an existing Assign alone when
    ResultPath is absent.
    """
    if not isinstance(catcher, dict):
        return catcher
    item = dict(catcher)
    rp = item.pop("ResultPath", None)
    item.pop("ResultSelector", None)
    item.pop("OutputPath", None)
    item.pop("InputPath", None)
    if "Assign" not in item:
        if isinstance(rp, str) and rp.strip():
            key = rp.strip()
            key = key[2:] if key.startswith("$.") else key.lstrip("$")
            key = re.sub(r"[^A-Za-z0-9_]", "_", key) or "errorInfo"
        else:
            key = "errorInfo"
        item["Assign"] = {key: "{% $states.errorOutput %}"}
    return item

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


# JSONPath intrinsic helpers that enrichment still emits (workflow_spec stage
# is allowed to use familiar States.* forms). Under QueryLanguage=JSONata
# these must become real JSONata — leaving them as plain strings produces
# invalid mixed ASL (the user's generated Output
# ``{"output": "States.StringToJson($.Payload.body)"}`` is the canonical
# failure mode).
# Accept both JSONPath ($.Payload.body) and already-rooted JSONata
# ($states.result.Payload.body) inside States.StringToJson(...).
_STATES_STRING_TO_JSON = re.compile(
    r"^States\.StringToJson\(\s*(\$[.\w\[\]'\"$]*)\s*\)$",
    re.IGNORECASE,
)
# Same intrinsic embedded inside a larger JSONata body / {% %} string.
_STATES_STRING_TO_JSON_EMBEDDED = re.compile(
    r"States\.StringToJson\(\s*(\$[.\w\[\]'\"$]*)\s*\)",
    re.IGNORECASE,
)
_STATES_FORMAT = re.compile(
    r"^States\.Format\(\s*'((?:\\'|[^'])*)'\s*,\s*(.+)\s*\)$",
    re.IGNORECASE,
)


def _path_with_root(path: str, root: str) -> str:
    """Map a JSONPath ``$`` / ``$.foo`` reference onto a JSONata root.

    ``root`` is typically ``$states.input`` (Arguments) or
    ``$states.result`` (ResultSelector / Output reshape of a Task).
    """
    text = (path or "").strip()
    if not text:
        return root
    if text == "$":
        return root
    if text.startswith("$."):
        return root + "." + text[2:]
    if text.startswith("$$."):
        return "$states.context." + text[3:]
    if text.startswith("$states"):
        return text
    return _jsonpath_to_jsonata_expr(text)


def _format_args_to_jsonata(args_blob: str, root: str) -> list[str]:
    """Split States.Format argument list into JSONata expression fragments."""
    parts: list[str] = []
    buf = ""
    in_str = False
    str_ch = ""
    depth = 0
    for ch in args_blob:
        if in_str:
            buf += ch
            if ch == str_ch and not buf.endswith("\\" + str_ch):
                in_str = False
            continue
        if ch in ("'", '"'):
            in_str = True
            str_ch = ch
            buf += ch
            continue
        if ch == "(":
            depth += 1
            buf += ch
            continue
        if ch == ")":
            depth -= 1
            buf += ch
            continue
        if ch == "," and depth == 0:
            parts.append(buf.strip())
            buf = ""
            continue
        buf += ch
    if buf.strip():
        parts.append(buf.strip())
    out: list[str] = []
    for p in parts:
        if (p.startswith("'") and p.endswith("'")) or (
            p.startswith('"') and p.endswith('"')
        ):
            # String literal — keep single-quoted for JSONata
            inner = p[1:-1].replace("\\'", "'")
            out.append("'" + inner.replace("'", "\\'") + "'")
        elif p.startswith("$"):
            out.append(_path_with_root(p, root))
        else:
            # Bare number / identifier — quote as string if non-numeric
            if re.match(r"^-?\d+(\.\d+)?$", p):
                out.append(p)
            else:
                out.append("'" + p.replace("'", "\\'") + "'")
    return out


def _states_intrinsic_to_jsonata(expr: str, root: str = "$states.input") -> str | None:
    """Translate a States.* intrinsic into a JSONata expression body.

    Returns the bare JSONata body (no ``{% %}``) or None if ``expr`` is
    not a recognized intrinsic. ``root`` selects whether ``$`` paths bind
    to state input (Arguments) or task result (ResultSelector/Output).

    Also rewrites *embedded* ``States.StringToJson(...)`` inside a larger
    expression (e.g. a ``$merge([...])`` body) so mixed forms like
    ``States.StringToJson($states.result.Payload.body)`` never reach AWS.
    """
    text = (expr or "").strip()
    # Strip a single outer {% %} wrapper if present so we can match the body.
    if text.startswith("{%") and text.endswith("%}"):
        text = text[2:-2].strip()

    m = _STATES_STRING_TO_JSON.match(text)
    if m:
        return "$parse(" + _path_with_root(m.group(1), root) + ")"
    m = _STATES_FORMAT.match(text)
    if m:
        # States.Format('s3://{}/{}', 'bucket', $.doc)
        # → JSONata string concat. For simple single-{} templates we
        # substitute positionally; multi-{} uses successive & joins of
        # the literal segments and args (good enough for S3 URI builds).
        template = m.group(1).replace("\\'", "'")
        args = _format_args_to_jsonata(m.group(2), root)
        # Split template on '{}' placeholders
        segments = template.split("{}")
        pieces: list[str] = []
        for i, seg in enumerate(segments):
            if seg:
                pieces.append("'" + seg.replace("'", "\\'") + "'")
            if i < len(args):
                pieces.append(args[i])
        if not pieces:
            return "''"
        return " & ".join(pieces)

    # Embedded States.StringToJson anywhere in the expression body.
    if _STATES_STRING_TO_JSON_EMBEDDED.search(text):
        def _repl(match: re.Match[str]) -> str:
            return "$parse(" + _path_with_root(match.group(1), root) + ")"

        return _STATES_STRING_TO_JSON_EMBEDDED.sub(_repl, text)
    return None


def _scrub_states_intrinsics_in_tree(
    node: Any, *, path_root: str = "$states.result"
) -> Any:
    """Final pass: replace any residual States.StringToJson / States.Format
    strings in an ASL tree with pure JSONata (including inside ``{% %}``).
    """
    if isinstance(node, dict):
        return {
            k: _scrub_states_intrinsics_in_tree(v, path_root=path_root)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [
            _scrub_states_intrinsics_in_tree(v, path_root=path_root) for v in node
        ]
    if isinstance(node, str):
        s = node.strip()
        # Whole-string intrinsic (with or without {% %})
        converted = _states_intrinsic_to_jsonata(s, root=path_root)
        if converted is not None:
            # If original was {% %}-wrapped or contains operators, keep wrap
            if (s.startswith("{%") and s.endswith("%}")) or converted.startswith(
                "$"
            ) or " & " in converted or converted.startswith("$parse"):
                # Prefer {% %} when original had it or result is an expression
                if s.startswith("{%"):
                    return _as_jsonata_string(converted)
                # Output field that was a bare States.* string → wrap
                if converted.startswith("$parse") or " & " in converted:
                    return _as_jsonata_string(converted)
                return _as_jsonata_string(converted)
            return converted
        # Embedded in a {% %} body that didn't fully match
        if "States.StringToJson" in s or "States.Format" in s:
            inner = s
            wrapped = False
            if inner.startswith("{%") and inner.endswith("%}"):
                inner = inner[2:-2].strip()
                wrapped = True
            fixed = _states_intrinsic_to_jsonata(inner, root=path_root)
            if fixed is not None:
                return _as_jsonata_string(fixed) if wrapped or True else fixed
        return node
    return node


def _convert_value_to_jsonata(
    value: Any, *, path_root: str = "$states.input"
) -> Any:
    """Recursively apply AWS JSONata value rules to Arguments / Output trees.

    Nested dicts are converted with the same key rules as top-level
    Parameters → Arguments (strip ``.$`` suffixes, path-ify string
    values). Without that, enrichment shapes like::

        {"Arguments": {"--source-bucket.$": "$.s3.bucket"}}
        {"Item": {"pk": {"S.$": "$.JobRunId"}}}

    would keep JSONPath ``.$`` keys under QueryLanguage=JSONata and
    fail purity validation / AWS deploy.

    ``path_root`` selects where bare ``$`` / ``$.foo`` paths bind:
    ``$states.input`` for Arguments (default), ``$states.result`` for
    ResultSelector / task Output reshape.
    """
    if isinstance(value, list):
        return [_convert_value_to_jsonata(v, path_root=path_root) for v in value]
    if isinstance(value, dict):
        # Same key rewrite as _convert_parameters_to_arguments — nested
        # objects must not leak JSONPath '.$' suffixes.
        return _convert_parameters_to_arguments(value, path_root=path_root)
    if isinstance(value, str):
        s = value.strip()
        intrinsic = _states_intrinsic_to_jsonata(s, root=path_root)
        if intrinsic is not None:
            return _as_jsonata_string(intrinsic)
        if _TODO_TOKEN.match(s):
            return _as_jsonata_string(f"$states.input.{s}")
        if s.startswith("$.") or s.startswith("$$") or s == "$" or s == "$$":
            return _as_jsonata_string(_path_with_root(s, path_root))
        if s.startswith("{%") and s.endswith("%}"):
            # {% States.StringToJson(...) %} or {% $merge([..., States.StringToJson(...)]) %}
            if "States.StringToJson" in s or "States.Format" in s:
                fixed = _states_intrinsic_to_jsonata(s, root=path_root)
                if fixed is not None:
                    return _as_jsonata_string(fixed)
            return _unwrap_if_mistakenly_wrapped_literal(s)
        if _VAR_REF.match(s):
            return _as_jsonata_string(s)
        return value
    return value


def _convert_parameters_to_arguments(
    params: dict, *, path_root: str = "$states.input"
) -> dict:
    """JSONPath Parameters object → JSONata Arguments object (AWS rules).

    - Keys ending in '.$' become bare keys with {% path-expr %} values
      **only when the value is actually a path / JSONata expression /
      unresolved TODO_ token / States.* intrinsic**.
    - If a ``.$`` key holds a plain constant (typical after placeholder
      fill replaced ``$.TODO_X`` with a literal bucket name, S3 URI,
      table name, etc.), the ``.$`` suffix is stripped and the value is
      emitted as a plain JSON literal — never wrapped in ``{% ... %}``.
      Wrapping constants produces invalid JSONata (``{% s3://... %}``,
      ``{% my-input-bucket %}``) that AWS ValidateStateMachineDefinition
      rejects.
    - Nested objects/arrays are walked recursively (same key rewrite
      at every depth, so ``Arguments.--source-bucket.$`` and
      ``Item.pk.S.$`` are both cleaned).
    - TODO_ tokens and $. paths become {% $states.input... %} expressions
      (or ``path_root`` when converting ResultSelector → Output).
    - ``States.StringToJson`` / ``States.Format`` are rewritten to
      JSONata ``$parse`` / string concat so they never leak into pure
      JSONata ASL as opaque strings.
    - Plain constants stay as plain JSON literals.
    - Values that are already ``{% ... %}`` JSONata strings are kept
      (mistakenly-wrapped bare literals are unwrapped).
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
                    out[new_key] = _unwrap_if_mistakenly_wrapped_literal(s)
                else:
                    intrinsic = _states_intrinsic_to_jsonata(s, root=path_root)
                    if intrinsic is not None:
                        out[new_key] = _as_jsonata_string(intrinsic)
                    elif _is_path_or_jsonata_expression(s):
                        # Real path / $var / unresolved TODO_ → JSONata wrap.
                        if _TODO_TOKEN.match(s):
                            out[new_key] = _as_jsonata_string(
                                f"$states.input.{s}"
                            )
                        else:
                            out[new_key] = _as_jsonata_string(
                                _path_with_root(s, path_root)
                            )
                    else:
                        # Plain constant under a leftover '.$' key (filled
                        # placeholder). Emit as JSON literal; do not wrap.
                        out[new_key] = s
            else:
                out[new_key] = _convert_value_to_jsonata(
                    val, path_root=path_root
                )
        else:
            out[key_s] = _convert_value_to_jsonata(val, path_root=path_root)
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
        state["Output"] = _convert_value_to_jsonata(
            step["output"], path_root="$states.result"
        )
    elif step.get("result_selector") is not None:
        # ResultSelector paths bind to the *task result* ($), not state
        # input — convert with path_root=$states.result so
        # States.StringToJson($.Payload.body) becomes
        # {% $parse($states.result.Payload.body) %}.
        selected = _convert_parameters_to_arguments(
            step["result_selector"], path_root="$states.result"
        )
        rp = step.get("result_path")
        if isinstance(rp, str) and rp.strip():
            # ResultPath + ResultSelector together: reshape task result,
            # then merge under the ResultPath key (JSONPath semantics).
            key = rp.strip()
            key = key[2:] if key.startswith("$.") else key.lstrip("$")
            if key and re.match(r"^[A-Za-z0-9_]+$", key):
                # Build {% $merge([$states.input, {"key": <selected>}]) %}
                # where selected is an object of JSONata field expressions.
                if isinstance(selected, dict):
                    pairs = []
                    for sk, sv in selected.items():
                        if not re.match(r"^[A-Za-z0-9_]+$", str(sk)):
                            continue
                        if (
                            isinstance(sv, str)
                            and sv.startswith("{%")
                            and sv.endswith("%}")
                        ):
                            body = sv[2:-2].strip()
                        elif isinstance(sv, str):
                            body = "'" + sv.replace("'", "\\'") + "'"
                        else:
                            continue
                        pairs.append('"' + str(sk) + '": ' + body)
                    inner = "{" + ", ".join(pairs) + "}"
                    state["Output"] = (
                        '{% $merge([$states.input, {"'
                        + key
                        + '": '
                        + inner
                        + "}]) %}"
                    )
                    notes.append(
                        f"Step '{sid}': ResultPath '{rp}' + ResultSelector "
                        f"→ JSONata Output $merge under '{key}'."
                    )
                    return
        state["Output"] = selected
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
            echo = step.get("echo")
            if isinstance(echo, dict) and echo:
                # Fields the underlying AWS API never returns (e.g. a
                # caller-supplied output URI) get merged alongside the
                # real $states.result, so a value defined once upstream
                # (execution input / TODO_ token) can still be read
                # downstream via $states.input.<key>.<field> -- the same
                # shape as a genuine API-returned field like JobRunId.
                echo_pairs = []
                for ek, ev in echo.items():
                    ek_s = str(ek)
                    if not re.match(r"^[A-Za-z0-9_]+$", ek_s):
                        continue
                    expr = _jsonpath_to_jsonata_expr(str(ev))
                    # Double-quoted string keys only. AWS JSONata rejects
                    # unquoted identifier keys in object constructors
                    # (UNSUPPORTED_JSONATA_EXPRESSION: "Reference to 'X'
                    # at the top level is not supported").
                    echo_pairs.append('"' + ek_s + '": ' + expr)
                if echo_pairs:
                    result_expr = (
                        "$merge([$states.result, {"
                        + ", ".join(echo_pairs)
                        + "}])"
                    )
                    notes.append(
                        f"Step '{sid}': echoed {list(echo.keys())} into "
                        f"result_path '{rp}' alongside $states.result "
                        "(value defined once, threaded to a later step)."
                    )
                else:
                    result_expr = "$states.result"
            else:
                result_expr = "$states.result"
            # ResultPath semantics via $merge: task result under <key>,
            # prior input preserved. Object keys MUST be double-quoted
            # strings — unquoted identifiers are rejected by AWS with
            # UNSUPPORTED_JSONATA_EXPRESSION ("Reference to 'X' at the
            # top level is not supported").
            state["Output"] = (
                '{% $merge([$states.input, {"'
                + key
                + '": '
                + result_expr
                + "}]) %}"
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


# Parent-scoped fields referenced from inside a Map iterator that must be
# injected via ItemSelector for Distributed Map (child executions do not
# inherit the parent $states.input).
_EXECUTION_INPUT_FIELD_RE = re.compile(
    r"\$states\.context\.Execution\.Input\.([A-Za-z_][A-Za-z0-9_]*)"
)
# $states.input.foo inside iterator may be parent-scoped when ItemSelector
# has not yet been applied; we only auto-inject common external keys when
# Distributed and selector is missing — see _ensure_map_item_selector.
_STATES_INPUT_FIELD_RE = re.compile(
    r"\$states\.input\.([A-Za-z_][A-Za-z0-9_]*)"
)


def _walk_strings(node: Any):
    """Yield every string in a JSON-like tree."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


def _collect_map_parent_field_refs(processor: dict) -> set[str]:
    """Fields that the iterator reads from parent execution scope.

    Distributed Map children only see ItemSelector output as
    ``$states.input``. Collect:
    - ``$states.context.Execution.Input.<field>``
    - ``$states.input.<field>`` except the reserved item key ``item``
      (those parent-scoped fields must be injected via ItemSelector).
    """
    found: set[str] = set()
    for s in _walk_strings(processor):
        for m in _EXECUTION_INPUT_FIELD_RE.finditer(s):
            found.add(m.group(1))
        for m in _STATES_INPUT_FIELD_RE.finditer(s):
            field = m.group(1)
            if field != "item":
                found.add(field)
    return found


def _rewrite_map_iterator_parent_refs(node: Any) -> Any:
    """Rewrite parent Execution.Input refs to ItemSelector-local input.

    After ItemSelector injects ``run_id`` (etc.) onto the item payload,
    the iterator should read ``$states.input.run_id`` rather than
    ``$states.context.Execution.Input.run_id`` (unreliable in Distributed
    child executions).
    """
    if isinstance(node, str):
        return _EXECUTION_INPUT_FIELD_RE.sub(
            r"$states.input.\1", node
        )
    if isinstance(node, dict):
        return {k: _rewrite_map_iterator_parent_refs(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_rewrite_map_iterator_parent_refs(v) for v in node]
    return node


def _rewrite_bare_states_input_to_item(node: Any, item_key: str = "item") -> Any:
    """When ItemSelector wraps the raw item under ``item``, fix path builds
    that concatenated the whole ``$states.input`` (object/string item).

    Matches JSONata fragments like ``& $states.input %}`` or
    ``& $states.input)`` so ``'s3://bucket/' & $states.input`` becomes
    ``'s3://bucket/' & $states.input.item``.
    """
    if isinstance(node, str):
        # Only rewrite $states.input when it is not already a field access
        # ($states.input.foo stays). Word-boundary after "input".
        return re.sub(
            r"\$states\.input(?!\s*\.|[A-Za-z0-9_])",
            f"$states.input.{item_key}",
            node,
        )
    if isinstance(node, dict):
        return {
            k: _rewrite_bare_states_input_to_item(v, item_key)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_rewrite_bare_states_input_to_item(v, item_key) for v in node]
    return node


def _ensure_map_item_selector(
    state: dict,
    processor: dict,
    *,
    distributed: bool,
    explicit_selector: dict | None,
    notes: list[str],
    sid: str,
) -> dict:
    """Ensure Distributed (and parent-ref) Maps get a safe ItemSelector.

    Generic rules (any domain):
    - Always place the raw Map item under a stable key ``item`` from
      ``$states.context.Map.Item.Value`` when we synthesize a selector.
    - Copy any parent fields the iterator needs
      (``$states.context.Execution.Input.<field>``) into the selector from
      parent ``$states.input.<field>``.
    - Explicit enrichment ItemSelector wins for keys it defines; we only
      fill gaps.
    """
    parent_fields = _collect_map_parent_field_refs(processor)
    need_synthetic = distributed or bool(parent_fields)

    if not need_synthetic and not explicit_selector:
        return processor

    # Start from explicit selector (converted) or empty
    selector: dict[str, Any] = {}
    if isinstance(explicit_selector, dict) and explicit_selector:
        selector = dict(
            _convert_parameters_to_arguments(
                explicit_selector, path_root="$states.input"
            )
        )

    # Ensure raw item is addressable under "item" when missing and we are
    # synthesizing or Distributed (child input must be an object if we
    # also inject parent fields).
    has_item_value = any(
        isinstance(v, str) and "Map.Item.Value" in v for v in selector.values()
    )
    if (distributed or parent_fields) and not has_item_value and "item" not in selector:
        selector["item"] = "{% $states.context.Map.Item.Value %}"

    for field in sorted(parent_fields):
        if field not in selector:
            selector[field] = _as_jsonata_string(f"$states.input.{field}")

    if not selector:
        return processor

    state["ItemSelector"] = selector
    # Iterator should read injected fields from $states.input
    processor = _rewrite_map_iterator_parent_refs(processor)
    if "item" in selector and has_item_value is False:
        # Only rewrite bare $states.input → $states.input.item when we
        # introduced the item wrapper (otherwise enrichment already shaped
        # the item payload).
        if not (isinstance(explicit_selector, dict) and explicit_selector):
            processor = _rewrite_bare_states_input_to_item(processor, "item")
        elif "item" in selector and not any(
            isinstance(explicit_selector.get(k), str)
            and "Map.Item.Value" in str(explicit_selector.get(k))
            for k in explicit_selector
            if k != "item"
        ):
            # Explicit selector lacked Map.Item.Value; we added item=
            processor = _rewrite_bare_states_input_to_item(processor, "item")

    notes.append(
        f"Step '{sid}': ItemSelector ensures Map item + parent fields "
        f"({', '.join(sorted(selector.keys()))}) reach the iterator."
    )
    return processor


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
        processor = _build_branch_asl(iterator, notes)

        # --- Distributed Map + fail-whole-map intent ---
        # Prefer explicit workflow_spec fields; fall back to description /
        # comment heuristics so plans that say "Distributed Map" or
        # "fail the entire map if any item fails" still get the right ASL
        # when enrichment omitted processor_config / tolerated_failure_*.
        proc_cfg = step.get("processor_config") or step.get("ProcessorConfig")
        text_blob = " ".join(
            str(x or "")
            for x in (
                step.get("description"),
                step.get("comment"),
                step.get("Comment"),
                sid,
            )
        ).lower()
        distributed = (
            step.get("distributed") is True
            or str(step.get("distributed") or "").lower() in ("true", "1", "yes")
            or str(step.get("map_mode") or "").upper() == "DISTRIBUTED"
            or (
                isinstance(proc_cfg, dict)
                and str(
                    proc_cfg.get("Mode") or proc_cfg.get("mode") or ""
                ).upper()
                == "DISTRIBUTED"
            )
            or bool(
                re.search(
                    r"\bdistributed\s+map\b|\bdistributed\b|"
                    r"\bat\s+once\b|\bhigh\s+fan-?out\b|"
                    r"\bfor\s+every\b.*\bat\s+once\b",
                    text_blob,
                )
            )
        )
        fail_whole = (
            step.get("tolerated_failure_count") == 0
            or step.get("ToleratedFailureCount") == 0
            or step.get("tolerated_failure_percentage") == 0
            or step.get("ToleratedFailurePercentage") == 0
            or bool(
                re.search(
                    r"fail(s|ing)?\s+(the\s+)?(entire\s+|whole\s+)?map|"
                    r"fail(s|ing)?\s+if\s+any|"
                    r"failure-?intolerant|"
                    r"tolerated\s+failure\s*(count|percentage)?\s*[:=]?\s*0|"
                    r"no\s+tolerated\s+failure|"
                    r"stop\s+on\s+(first\s+)?(item\s+)?failure",
                    text_blob,
                )
            )
        )

        if distributed or proc_cfg:
            cfg: dict[str, Any] = {}
            if isinstance(proc_cfg, dict):
                mode = proc_cfg.get("Mode") or proc_cfg.get("mode")
                exec_type = (
                    proc_cfg.get("ExecutionType")
                    or proc_cfg.get("execution_type")
                )
                if mode:
                    cfg["Mode"] = str(mode).upper()
                if exec_type:
                    cfg["ExecutionType"] = str(exec_type).upper()
            if distributed and "Mode" not in cfg:
                cfg["Mode"] = "DISTRIBUTED"
            if cfg.get("Mode") == "DISTRIBUTED" and "ExecutionType" not in cfg:
                cfg["ExecutionType"] = "STANDARD"
            if cfg:
                processor["ProcessorConfig"] = cfg
                if distributed:
                    notes.append(
                        f"Step '{sid}': Distributed Map "
                        f"(ProcessorConfig.Mode=DISTRIBUTED)."
                    )
        state["ItemProcessor"] = processor

        max_concurrency = step.get("max_concurrency")
        if max_concurrency is not None:
            state["MaxConcurrency"] = max_concurrency

        # Explicit tolerated-failure fields from the spec win; otherwise
        # apply 0/0 when the plan text requires fail-whole-map.
        for src, dest in (
            ("tolerated_failure_count", "ToleratedFailureCount"),
            ("tolerated_failure_percentage", "ToleratedFailurePercentage"),
            ("ToleratedFailureCount", "ToleratedFailureCount"),
            ("ToleratedFailurePercentage", "ToleratedFailurePercentage"),
        ):
            if step.get(src) is not None and dest not in state:
                state[dest] = step[src]
        if fail_whole:
            state.setdefault("ToleratedFailureCount", 0)
            state.setdefault("ToleratedFailurePercentage", 0)
            notes.append(
                f"Step '{sid}': fail-whole-map "
                f"(ToleratedFailureCount/Percentage=0)."
            )

        # ItemSelector (JSONata) — reshape each item before the iterator.
        # Explicit enrichment item_selector / parameters win for defined
        # keys; Distributed Maps and iterators that read parent fields
        # always get a safe selector (item + injected parent fields).
        explicit_selector = (
            step.get("item_selector")
            or step.get("ItemSelector")
            or step.get("parameters")
        )
        if not isinstance(explicit_selector, dict):
            explicit_selector = None
        processor = _ensure_map_item_selector(
            state,
            processor,
            distributed=distributed,
            explicit_selector=explicit_selector,
            notes=notes,
            sid=sid,
        )
        state["ItemProcessor"] = processor

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


def _check_bare_result_indexing(asl: dict, report: ValidationReport) -> bool:
    """Flag "$[" -- a bare, unqualified JSONata root context indexed
    directly, e.g. "{% $[0].foo %}" -- inside a Parallel/Map state's own
    Output, Assign, or Catch[].Assign fields (including nested branches
    and Iterator/ItemProcessor bodies).

    Only $states.result and $states.errorOutput are documented, reserved
    references to a state's raw result. Bare "$" is never specified
    anywhere to be pre-bound to that array, so indexing it directly
    (rather than writing $states.result[0], say) relies on undefined
    behavior instead of the one guaranteed reference AWS actually
    provides for a Parallel/Map state's per-branch/per-item result array.
    """
    ok = True

    def scan_value(val: Any, path: str) -> None:
        nonlocal ok
        if isinstance(val, str) and "$[" in val:
            report.issues.append(
                ValidationIssue(
                    "aws_constraints",
                    "bare_result_index",
                    f"{path} uses bare '$[...]' indexing; use "
                    "'$states.result[...]' (or '$states.errorOutput[...]' "
                    "in a Catch) instead -- bare '$' is not documented to "
                    "be bound to the raw array result",
                    path,
                )
            )
            ok = False
        elif isinstance(val, dict):
            for k, v in val.items():
                scan_value(v, f"{path}.{k}")
        elif isinstance(val, list):
            for i, v in enumerate(val):
                scan_value(v, f"{path}[{i}]")

    def walk_states(states_map: dict | None, path: str) -> None:
        for name, state in (states_map or {}).items():
            state_path = f"{path}.{name}"
            if state.get("Type") in ("Parallel", "Map"):
                if "Output" in state:
                    scan_value(state["Output"], f"{state_path}.Output")
                if "Assign" in state:
                    scan_value(state["Assign"], f"{state_path}.Assign")
                for i, c in enumerate(state.get("Catch") or []):
                    if isinstance(c, dict) and "Assign" in c:
                        scan_value(
                            c["Assign"], f"{state_path}.Catch[{i}].Assign"
                        )
                for br in state.get("Branches") or []:
                    if isinstance(br, dict):
                        walk_states(br.get("States"), f"{state_path}.Branches")
                iterator = state.get("ItemProcessor") or state.get("Iterator")
                if isinstance(iterator, dict):
                    walk_states(
                        iterator.get("States"), f"{state_path}.ItemProcessor"
                    )
            else:
                for br in state.get("Branches") or []:
                    if isinstance(br, dict):
                        walk_states(br.get("States"), f"{state_path}.Branches")

    walk_states(asl.get("States"), "States")
    return ok


def _literal_output_keys(output: Any) -> set[str] | None:
    """Return a state's Output top-level key set IF Output is a literal
    JSON object -- unlike a "{% $merge(...) %}" string or an absent
    Output (defaulting to $states.result), a literal object's key set is
    fully known statically, with no ambiguity to guess at. Returns None
    whenever the shape can't be determined this way; callers must skip
    the check in that case rather than assume.
    """
    if isinstance(output, dict):
        return set(output.keys())
    return None


def _referenced_input_keys(node: Any) -> set[str]:
    """Collect the first path segment of every "$states.input.<key>"
    reference found anywhere (recursively) inside a state's Arguments
    (or any nested dict/list within it, e.g. a Payload sub-object)."""
    found: set[str] = set()

    def walk(val: Any) -> None:
        if isinstance(val, str):
            for m in _STATES_INPUT_REF_RE.finditer(val):
                found.add(m.group(1))
        elif isinstance(val, dict):
            for v in val.values():
                walk(v)
        elif isinstance(val, list):
            for v in val:
                walk(v)

    walk(node)
    return found


def _check_output_consumer_consistency(asl: dict, report: ValidationReport) -> bool:
    """When a state's Output is a literal JSON object, the set of
    top-level keys the next state will see in $states.input is fully
    known statically. If that next state (reached via Next) references
    $states.input.<key> for a key outside that set, the reference is
    guaranteed to resolve to undefined at runtime -- flag it as a hard
    error instead of letting it fail silently during execution (e.g. a
    Parallel state's Output producing {"a": ..., "b": ...} while the next
    state reads $states.input.combined, which was never produced).

    Deliberately conservative: only checks the direct Next edge (not
    Catch targets, not multi-hop chains), and only fires when the
    predecessor's Output is a literal object -- a string/JSONata Output,
    or no Output at all, is left unchecked rather than risk a false
    positive from a shape we can't verify statically.
    """
    ok = True
    states: dict = asl.get("States") or {}

    for name, state in states.items():
        next_name = state.get("Next")
        if not next_name or next_name not in states:
            continue
        known_keys = _literal_output_keys(state.get("Output"))
        if known_keys is None:
            continue

        next_state = states[next_name]
        referenced = _referenced_input_keys(next_state.get("Arguments"))
        missing = referenced - known_keys
        if missing:
            report.issues.append(
                ValidationIssue(
                    "aws_constraints",
                    "dangling_input_reference",
                    f"State '{next_name}' references "
                    f"$states.input.{{{', '.join(sorted(missing))}}}, but "
                    f"preceding state '{name}' Output only produces "
                    f"{{{', '.join(sorted(known_keys))}}} -- this resolves "
                    "to undefined at runtime",
                    f"States.{next_name}.Arguments",
                )
            )
            ok = False
    return ok


def _validate_aws_constraints(asl: dict, report: ValidationReport) -> None:
    ok = _check_jsonata_purity(asl, report)
    ok = _check_bare_result_indexing(asl, report) and ok
    ok = _check_output_consumer_consistency(asl, report) and ok
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

        # Glue / Lambda-style nested Arguments: keys must be structural
        # identifiers (JobName, --source_csv_uri, TableName, …), never
        # runtime data values (s3://…, arn:…, free text). Catches the
        # class of bug where a TODO_ token was used as a key and then
        # filled with a URI or similar.
        args = state.get("Arguments")
        if isinstance(args, dict):
            nested = args.get("Arguments")
            keys_to_check: list[tuple[str, str]] = []
            for k in args:
                if k != "Arguments":
                    keys_to_check.append((k, f"States.{name}.Arguments"))
            if isinstance(nested, dict):
                for k in nested:
                    keys_to_check.append(
                        (k, f"States.{name}.Arguments.Arguments")
                    )
            for key, path in keys_to_check:
                if isinstance(key, str) and _looks_like_data_value_not_identifier(key):
                    report.issues.append(
                        ValidationIssue(
                            "aws_constraints",
                            "arguments_key_is_value",
                            f"State '{name}' has an Arguments key that looks "
                            f"like a data value ('{key[:80]}'), not a field or "
                            f"job argument name. Argument keys must be "
                            f"identifiers (e.g. JobName, --source_csv_uri, "
                            f"TableName); put URIs/ARNs/values on the value side.",
                            path,
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
        # AWS ValidateStateMachineDefinition result field is OK | FAIL
        # (API reference). Some SDK/docs shapes also use RESULT_PASS /
        # RESULT_FAIL — accept both so a real pass is never mis-classified
        # as failure (which was blocking deploy with a green
        # "[extras/aws_validate_failed] OK" message).
        result = response.get("result") or response.get("Result")
        diagnostics = response.get("diagnostics") or response.get("Diagnostics") or []
        result_norm = (result if isinstance(result, str) else str(result or "")).upper()
        passed = result_norm in (
            "OK",
            "PASS",
            "RESULT_PASS",
            "TRUE",
        ) or (result is None and not diagnostics)
        # Some SDK shapes only return diagnostics on failure.
        if result is None and diagnostics:
            severity_fail = any(
                (d.get("severity") or d.get("Severity") or "").upper()
                in ("ERROR", "FATAL")
                for d in diagnostics
                if isinstance(d, dict)
            )
            passed = not severity_fail
        # If AWS said OK/PASS but also returned ERROR diagnostics, trust
        # the diagnostics (defensive).
        if passed and diagnostics:
            severity_fail = any(
                (d.get("severity") or d.get("Severity") or "").upper()
                in ("ERROR", "FATAL")
                for d in diagnostics
                if isinstance(d, dict)
            )
            if severity_fail:
                passed = False

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
    asl: dict,
    spec: dict,
    report: ValidationReport,
    graph_dot: str,
    include_aws_validate: bool = True,
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

    # Live AWS ValidateStateMachineDefinition. Skippable via
    # include_aws_validate=False for callers that need the mechanical
    # checks repeatedly (e.g. asl_generation/llm_service.py's repair
    # loop) without hammering the AWS API once per retry -- see that
    # module's docstring for why. aws_validate_ok stays None (its
    # "not attempted" value) rather than True/False, since skipping is
    # not the same as passing.
    if include_aws_validate:
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
                parts: list[str] = []
                for d in diagnostics[:8]:
                    if not isinstance(d, dict):
                        parts.append(str(d))
                        continue
                    severity = d.get("severity") or d.get("Severity") or ""
                    code = d.get("code") or d.get("Code") or ""
                    message = d.get("message") or d.get("Message") or str(d)
                    loc = d.get("location") or d.get("Location") or ""
                    bit = message
                    if code:
                        bit = f"[{code}] {bit}"
                    if severity:
                        bit = f"{severity}: {bit}"
                    if loc:
                        bit = f"{bit} @ {loc}"
                    parts.append(bit)
                msg = f"{msg}: " + "; ".join(parts)
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

def validate_asl(
    asl: Any,
    spec: dict | None = None,
    graph_dot: str = "",
    include_aws_validate: bool = True,
) -> ValidationReport:
    """Run all five validation groups. `spec` is required for groups 3 & 5
    full fidelity; AWS live validation runs when asl is valid JSON, unless
    include_aws_validate=False (used by repair loops that call this
    repeatedly and don't want to hit the live AWS API on every attempt --
    see asl_generation/llm_service.py).
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
        _validate_extras(root, spec, report, graph_dot, include_aws_validate)
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
        if include_aws_validate:
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

    def walk_states(
        states: dict, *, cluster_label: str | None = None, top_level: bool = True
    ) -> None:
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

            # Only top-level terminals connect to the global End marker.
            # Nested Map/Parallel End:true (or Succeed/Fail) ends an item or
            # branch, not the workflow — linking them to __end is misleading.
            if top_level and (
                state.get("End") is True or stype in ("Succeed", "Fail")
            ):
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
                            top_level=False,
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
                        top_level=False,
                    )

        if cluster_id:
            lines.append("  }")

    walk_states(root_states, top_level=True)
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_asl_from_spec(spec: dict, *, mode: str | None = None) -> AslResult:
    """Generate ASL, render graph, run all validations. Always returns a
    result object; check ``result.validations.ok`` before deploying.

    ``mode`` overrides ``APISettings.asl_generation_mode`` for this one
    call (mainly for tests/tooling); every real call site omits it and
    gets whatever ASL_GENERATION_MODE is configured to. This is the ONLY
    place that decides whether generation is deterministic or
    LLM-driven -- every caller (session_service, the LangGraph nodes)
    is unchanged and unaware of the choice. Default mode is
    "mechanical", so existing behavior is unaffected unless
    ASL_GENERATION_MODE is explicitly set.
    """
    # Local import, matching this module's existing convention for
    # config access (see _sfn_client / call_aws_validate_state_machine
    # above) rather than importing it at module scope.
    from orchestrator.api.config.settings import get_api_settings

    effective_mode = mode or get_api_settings().asl_generation_mode
    source = "mechanical"

    if effective_mode in ("llm", "llm_with_fallback"):
        # Imported here, not at module scope: asl_generation/llm_service.py
        # imports THIS module back (for validate_asl/sanitize_asl_literals/
        # render_asl_graph), so a top-level import here would be circular.
        from orchestrator.api.services.asl_generation.llm_service import (
            AslLlmGenerationError,
            generate_asl_via_llm,
        )

        try:
            asl, notes = generate_asl_via_llm(spec)
            source = "llm"
        except AslLlmGenerationError as exc:
            if effective_mode == "llm_with_fallback":
                asl, notes = generate_asl(spec)
                notes = [
                    f"LLM ASL generation failed, fell back to the mechanical "
                    f"translator ({exc}).",
                    *notes,
                ]
                source = "mechanical_fallback"
            else:
                raise AslGenerationError(str(exc)) from exc
    else:
        asl, notes = generate_asl(spec)
    # Final, unconditional normalization pass over the WHOLE generated
    # document -- not just the values that flowed through
    # _convert_parameters_to_arguments. generate_asl() assembles a state
    # from several independent branches (Task Arguments, Pass Output,
    # Wait Seconds, Choice Condition, Map Items, Catch Assign, ...), and
    # any branch that sets a field from spec data without routing it
    # through _convert_value_to_jsonata would silently skip the
    # mistakenly-wrapped-literal guard. Running sanitize_asl_literals
    # here, once, over the fully assembled tree makes that guarantee
    # structural instead of depending on every branch remembering to
    # call the per-value guard correctly. No-op on a document that's
    # already correct (see _BARE_LITERAL_INSIDE_WRAPPER / the guard's
    # own docstring for why real JSONata is never touched).
    asl = sanitize_asl_literals(asl)
    # Scrub residual States.StringToJson / States.Format (including
    # embedded inside $merge Output expressions).
    asl = _scrub_states_intrinsics_in_tree(asl, path_root="$states.result")
    graph_dot = render_asl_graph(asl)
    report = validate_asl(asl, spec=spec, graph_dot=graph_dot)
    return AslResult(
        asl=asl,
        validations=report,
        graph_dot=graph_dot,
        generation_notes=notes,
        source=source,
    )