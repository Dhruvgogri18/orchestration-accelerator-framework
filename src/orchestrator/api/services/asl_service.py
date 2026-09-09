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
    any TODO_ token in any string value at any depth.

    Deliberately broader than the aws_constraints group's TODO_ check
    below, which only ever looks at a Task's `Resource`: enrichment
    can just as easily leave a TODO_ token in a Parameters value, a
    Comment, or an InputPath, and a workflow with any of those is
    exactly as undeployable as one with a placeholder Resource.
    """
    out = _seen if _seen is not None else []
    if isinstance(node, str):
        for m in _PLACEHOLDER_PATTERN.findall(node):
            if m not in out:
                out.append(m)
    elif isinstance(node, dict):
        for v in node.values():
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


# Known AWS integrations we can recognize from resource/description text,
# used to give placeholders a concrete label + example instead of a
# generic "AWS resource ARN". Checked in order, so this list stays short
# and specific rather than trying to be exhaustive. Each entry carries
# BOTH an ARN-shaped example and a bare-name-shaped example, since the
# same integration (e.g. S3) can need either depending on which field
# the placeholder actually fills -- a Task Resource is always an ARN,
# but an S3 GetObject "Bucket" parameter wants the bare bucket name, and
# handing out the wrong shape as an "e.g." is exactly what leads users
# to paste an ARN (or an s3:// URI) where a plain name belongs, or vice
# versa. See classify_placeholder_kind() / validate_placeholder_value().
_SERVICE_ARN_HINTS: list[tuple[str, str, str, str]] = [
    ("lambda", "Lambda function", "arn:aws:lambda:us-east-1:123456789012:function:MyFunction", "MyFunction"),
    ("dynamodb", "DynamoDB table", "arn:aws:dynamodb:us-east-1:123456789012:table/MyTable", "MyTable"),
    ("sns", "SNS topic", "arn:aws:sns:us-east-1:123456789012:MyTopic", "MyTopic"),
    ("sqs", "SQS queue", "arn:aws:sqs:us-east-1:123456789012:MyQueue", "MyQueue"),
    ("s3", "S3 bucket", "arn:aws:s3:::my-bucket", "my-bucket"),
    ("ecs", "ECS task definition", "arn:aws:ecs:us-east-1:123456789012:task-definition/MyTask", "MyTask"),
    ("ses", "SES identity", "arn:aws:ses:us-east-1:123456789012:identity/example.com", "example.com"),
    ("glue", "Glue job", "arn:aws:glue:us-east-1:123456789012:job/MyJob", "MyJob"),
    ("states", "Step Functions state machine", "arn:aws:states:us-east-1:123456789012:stateMachine:MyStateMachine", "MyStateMachine"),
    ("kms", "KMS key", "arn:aws:kms:us-east-1:123456789012:key/abcd1234-a123-4567-b890-1234567890ab", "abcd1234-a123-4567-b890-1234567890ab"),
    ("rds", "RDS database instance", "arn:aws:rds:us-east-1:123456789012:db:my-database", "my-database"),
]


def _detect_service_hint(*texts: str) -> tuple[str, str, str] | None:
    """Best-effort (label, example ARN, example bare name) guess from
    non-placeholder text.

    Strips TODO_ tokens out of the combined text first -- a bare
    placeholder can't tell us what AWS service it belongs to, and
    without this a token could "match" against its own name.
    """
    blob = _PLACEHOLDER_PATTERN.sub("", " ".join(t for t in texts if t)).lower()
    for keyword, label, arn_example, name_example in _SERVICE_ARN_HINTS:
        if keyword in blob:
            return label, arn_example, name_example
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

    Preference order: the token's own naming convention (TODO_..._ARN
    is unambiguous) beats structural context, since it's what the
    extraction stage itself decided this value represents.
    """
    upper = (token or "").upper()
    if upper.endswith("_ARN"):
        return "arn"
    if upper.endswith(("_URL", "_URI", "_ENDPOINT")):
        return "url"
    # NOTE: deliberately does NOT include "_KEY" -- an object/file "key"
    # (e.g. TODO_CSV_OBJECT_KEY) is commonly a slash-separated path
    # ("incoming/data.csv"), not a bare resource name, so it's left to
    # fall through to structural context (and ultimately "generic")
    # rather than being force-fit into the no-slash "name" rule.
    if upper.endswith(("_NAME", "_BUCKET", "_TABLE", "_QUEUE", "_TOPIC")):
        return "name"

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
                        if _ARN_FIELD_SUFFIX_RE.search(field_key):
                            return "arn"
                        if _URL_FIELD_SUFFIX_RE.search(field_key):
                            return "url"
                        if _NAME_FIELD_SUFFIX_RE.search(field_key):
                            return "name"
                        return "generic"

        for param in workflow_spec.get("parameters") or []:
            if isinstance(param, dict) and str(param.get("name") or "") == token:
                constraints = str(param.get("constraints") or "").lower()
                if "arn" in constraints:
                    return "arn"

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
    thing>" sentence instead of repeating it verbatim.
    """
    d = (desc or "").strip().rstrip(".")
    if not d:
        return d
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

                    hint = _detect_service_hint(
                        "" if is_whole_resource else resource,
                        service_field,
                        cdesc,
                        params_blob,
                    )
                    kind = classify_placeholder_kind(workflow_spec, t, plan=plan)
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
                        # Only borrow the service hint's own wording
                        # ("S3 bucket ARN/name") when this token's kind
                        # says it IS that resource identifier. A token
                        # sitting in an ambiguous field (e.g. an S3 "Key"
                        # parameter -- an object path, not the bucket)
                        # would otherwise get mislabeled "S3 bucket
                        # name" just because the step also touches S3.
                        if hint and kind in ("arn", "name"):
                            label, arn_example, name_example = hint
                            if kind == "arn":
                                details[t] = (
                                    f"{label} ARN used by {step_ref}"
                                    + (f" to {clause}." if clause else ".")
                                    + f" Must be the full ARN, e.g. `{arn_example}`."
                                )
                            else:
                                details[t] = (
                                    f"{label} name used by {step_ref}"
                                    + (f" to {clause}." if clause else ".")
                                    + f" Just the name, not the ARN — e.g. `{name_example}`."
                                )
                        elif cdesc:
                            details[t] = (
                                f"{cdesc} (step `{cid}`)." if cid else cdesc
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
    """Deep-replace TODO_ tokens (or any string keys) in a JSON tree.

    Used after Stage 4 so user-supplied values from /approve overrides
    land in the locked ASL / workflow_spec — not only in the thin Plan
    service/description fields. Enrichment routinely re-emits TODOs in
    Parameters and Resource; without this step, Apply placeholders in
    the UI never changes the deployable definition.
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
        return {k: apply_placeholder_overrides(v, overrides) for k, v in node.items()}
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
# Choice condition → ASL Choice rule
# ---------------------------------------------------------------------------

# JSONPath variable: $.foo, $.Payload.status, $.items[0].id, etc.
_VAR = r"(\$\.[A-Za-z0-9_.\[\]'\"-]+)"

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


def _choice_rule_from_condition(condition: str, next_step: str) -> dict:
    """Convert a condition string into an ASL Choice rule.

    Prefer strict parseable forms (enrichment is instructed to emit
    ``$.path == value``, ``$.path > N``, ``$.path is present``, etc.).
    Fall back to conservative StringEquals on ``$.status`` so the
    definition stays valid and the original label is preserved in Comment.
    """
    text = (condition or "").strip()
    if not text:
        return {
            "Variable": "$.status",
            "StringEquals": "true",
            "Next": next_step,
            "Comment": "empty condition from workflow_spec",
        }

    m = _COND_BOOL.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "BooleanEquals": m.group(2).lower() == "true",
            "Next": next_step,
        }

    m = _COND_NUM.match(text)
    if m:
        path, op, num_s = m.group(1), m.group(2), m.group(3)
        num = float(num_s)
        key = {
            ">": "NumericGreaterThan",
            ">=": "NumericGreaterThanEquals",
            "<": "NumericLessThan",
            "<=": "NumericLessThanEquals",
        }[op]
        value: int | float = int(num) if num == int(num) else num
        return {"Variable": path, key: value, "Next": next_step}

    m = _COND_NEQ.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "StringEquals": m.group(2).strip(),
            "Next": next_step,
            "Comment": f"NOTE: original condition was inequality (!=); "
            f"review and replace with a positive match if needed: {text}",
        }

    m = _COND_EQ.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "StringEquals": m.group(2).strip(),
            "Next": next_step,
        }

    m = _COND_PRESENT.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "IsPresent": True,
            "Next": next_step,
        }

    m = _COND_NOT_PRESENT.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "IsPresent": False,
            "Next": next_step,
        }

    m = _COND_CONTAINS.match(text)
    if m:
        return {
            "Variable": m.group(1),
            "StringMatches": f"*{m.group(2).strip()}*",
            "Next": next_step,
        }

    # Soft: bare status-like label → $.status == label
    m = _COND_SOFT_STATUS.match(text)
    if m:
        return {
            "Variable": "$.status",
            "StringEquals": m.group(1).strip().lower(),
            "Next": next_step,
            "Comment": f"soft-parsed condition label: {text}",
        }

    # Soft: bare JSONPath → treat as boolean / presence of that path
    m = _COND_SOFT_PATH_STATUS.match(text)
    if m and text.startswith("$."):
        return {
            "Variable": m.group(1),
            "IsPresent": True,
            "Next": next_step,
            "Comment": f"soft-parsed path-only condition: {text}",
        }

    # Last resort: keep the human label visible on $.status
    return {
        "Variable": "$.status",
        "StringEquals": text,
        "Next": next_step,
        "Comment": f"unparsed condition from workflow_spec — review: {text}",
    }


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
    return [
        {
            "ErrorEquals": list(c.get("error_equals") or ["States.ALL"]),
            "Next": c["next_step"],
        }
        for c in catchers
        if c.get("next_step")
    ]


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
            state["Parameters"] = step["parameters"]
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
        else:
            raise AslGenerationError(f"Wait step '{sid}' needs seconds or timestamp")
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
        _apply_common_transition(state, step)

    elif stype == "Map":
        iterator = step.get("iterator")
        items_path = step.get("items_path")
        if not iterator or not items_path:
            raise AslGenerationError(
                f"Map step '{sid}' requires iterator and items_path"
            )
        state["ItemsPath"] = items_path
        state["Iterator"] = _build_branch_asl(iterator, notes)
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
        _apply_common_transition(state, step)

    elif stype == "Succeed":
        pass  # terminal; no Next/End

    elif stype == "Fail":
        if step.get("error"):
            state["Error"] = step["error"]
        if step.get("cause"):
            state["Cause"] = step["cause"]

    elif stype == "Pass":
        _apply_common_transition(state, step)

    return state


def generate_asl(spec: dict) -> tuple[dict, list[str]]:
    """Translate a validated workflow_spec into an ASL definition.

    Returns (asl_dict, generation_notes). Raises AslGenerationError on
    structural problems that make translation impossible.
    """
    if not isinstance(spec, dict):
        raise AslGenerationError("workflow_spec must be a dict")
    steps = spec.get("steps")
    if not steps:
        raise AslGenerationError("workflow_spec.steps is empty")

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
        "StartAt": start_at,
        "States": states,
    }
    if spec.get("version"):
        asl["Version"] = str(spec["version"])

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
            if "ItemsPath" not in state and "ItemSelector" not in state:
                # ItemsPath is the classic field; soft-warn via issue if missing
                report.issues.append(
                    ValidationIssue(
                        "structural",
                        "map_no_items_path",
                        f"Map state '{name}' should have ItemsPath",
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

def _validate_aws_constraints(asl: dict, report: ValidationReport) -> None:
    ok = True
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
                    if "StringEquals" in rule:
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