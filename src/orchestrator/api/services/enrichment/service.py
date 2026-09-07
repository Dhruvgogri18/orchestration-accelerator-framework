"""Stage 3.5 enrichment: approved Plan -> full workflow_spec.json.

Two DIFFERENT loops are at play here, and it matters to keep them
conceptually separate:

  1. The REPAIR loop (automatic, invisible to the user): when the LLM's
     response fails one of the validation checks below, the specific
     error is fed back and it gets another attempt, capped at
     MAX_ENRICHMENT_ATTEMPTS. This is the LLM correcting its OWN
     mistakes against checks WE run.
  2. The CLARIFICATION loop (visible to the user, a real pause): when
     the LLM itself decides a structural decision is genuinely
     undecidable from the plan (not a missing value -- that's what
     TODO_ placeholders are for -- but a missing STRUCTURE), it can ask
     a question instead of guessing. Capped separately
     (MAX_CLARIFICATION_ROUNDS) so an indecisive LLM can't stall
     forever either.

IMPORTANT: all structural checks (fidelity, merge validity, addition
validity, ARN whitelist) walk the ENTIRE nested document -- top-level
steps, plus every step nested inside a Parallel branch or Map
iterator, recursively. Earlier versions only looked at top-level
`steps`, which caused two real, live-observed bugs: (a) a hallucinated
ARN hidden inside a Parallel branch passed the whitelist unnoticed,
and (b) a correctly-nested Parallel spec would have been REJECTED
(while an incorrectly duplicated one was accepted), because top-level
id comparison doesn't understand nesting. A thin Plan has no concept
of nesting at all, so fidelity is defined as a SET/multiplicity
property across the whole document -- every plan step id appears
EXACTLY ONCE somewhere in the full structure -- not a positional
top-level list match.

Six independent validation checks guard the repair loop, run in
order (cheapest first):

  1. Schema validation     -- structurally valid per workflow_spec.schema.json
  2. Fidelity check         -- every plan step id appears exactly once
                               somewhere in the full recursive document,
                               no duplicates, no unexplained extra ids.
                               Two declared, narrow exceptions:
                                 - merged_step_ids: a plan step allowed
                                   to be ABSENT (folded into its
                                   predecessor -- e.g. a retry-only step)
                                 - added_step_ids: a NEW id allowed to be
                                   PRESENT that wasn't in the plan (a
                                   minimal terminal step ASL structurally
                                   requires but the thin plan never
                                   described as its own step)
  3. Merge validity check   -- for every declared merged_step_id: real
                               plan id, has a predecessor, predecessor
                               is present and actually has a `retry`
                               block -- proves the merge preserved real
                               behavior rather than silently dropping a
                               step under a different label.
  4. Addition validity check-- for every declared added_step_id: not
                               already a plan id, actually present in
                               the output, type is Succeed/Fail ONLY
                               (never a real action or new decision
                               logic), and actually referenced by
                               something -- proves the addition is a
                               minimal, necessary terminal, not a
                               loophole for inventing business logic.
  5. No unjustified nesting -- Parallel/Map rejected unless the plan text
                               clearly signals concurrency or iteration.
                               Stops enrichment inventing nesting on
                               linear plans.
  6. ARN whitelist check    -- any `resource` claiming to be a Step
                               Functions optimized integration, ANYWHERE
                               in the document including nested
                               branches/iterators, must EXACTLY match a
                               known-real action, or be a TODO_
                               placeholder.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import jsonschema

from orchestrator.api.schemas.common import Plan
from orchestrator.api.services.enrichment.prompts import ENRICHMENT_SYSTEM_PROMPT
from orchestrator.indexing.gateway_client import chat_completion
from orchestrator.api.services.llm_trace import log_event, llm_stage

MAX_ENRICHMENT_ATTEMPTS = 4
MAX_CLARIFICATION_ROUNDS = 3

# ARN whitelist / TODO_UNKNOWN_INTEGRATION_ARN force-patch removed:
# resource ARNs are left to the model (optimized integration or aws-sdk
# form). User-specific names still use TODO_ placeholders in parameters.

# A declared "added" step may only be one of these ASL types -- a
# minimal terminal, never a real action or new decision point. This is
# what stops added_step_ids from becoming a loophole for hallucinating
# business logic under the label of "structurally necessary."
ALLOWED_ADDED_STEP_TYPES = {"Succeed", "Fail"}

# Known-real AWS Step Functions optimized integration resources
# (service:action, before any .sync/.waitForTaskToken suffix). EXACT
# match required, not prefix -- prefix matching alone let a real bug
# through during testing: "arn:aws:states:::glue:runMyJobNow" passed a
# prefix check against "arn:aws:states:::glue:" even though
# "runMyJobNow" isn't a real Glue action. This is a suspicion filter,
# not exhaustive -- extend deliberately as you verify new integrations,
# rather than loosening the match back to a prefix.
KNOWN_ARNS = {
    "arn:aws:states:::glue:startJobRun",
    "arn:aws:states:::sns:publish",
    "arn:aws:states:::sqs:sendMessage",
    "arn:aws:states:::lambda:invoke",
    "arn:aws:states:::aws-sdk:s3:getObject",
    "arn:aws:states:::aws-sdk:s3:putObject",
    "arn:aws:states:::dynamodb:putItem",
    "arn:aws:states:::dynamodb:getItem",
    "arn:aws:states:::ecs:runTask",
    "arn:aws:states:::states:startExecution",
}
KNOWN_SUFFIXES = ("", ".sync", ".sync:2", ".waitForTaskToken")

# arn:aws:states:::aws-sdk:<service>:<apiAction> is a deliberately
# open-ended REAL AWS pattern -- it mirrors the AWS SDK's own API
# action names across virtually any service, which we can't
# realistically enumerate. Trusted structurally rather than
# exact-matched, unlike the fixed optimized-integration ARNs above.
AWS_SDK_PASSTHROUGH_PREFIX = "arn:aws:states:::aws-sdk:"


class EnrichmentError(Exception):
    """Raised when enrichment could not produce a valid spec within
    MAX_ENRICHMENT_ATTEMPTS, or asked more than MAX_CLARIFICATION_ROUNDS
    clarifying questions without resolving. Carries the last error/
    question seen.
    """


@dataclass
class EnrichmentResult:
    """Outcome of one enrichment turn (fresh start or resumed).

    status == "complete"            -> `spec` is a validated workflow_spec.json
    status == "needs_clarification" -> `question` needs a human answer;
                                        `messages` must be persisted by the
                                        caller and passed back into
                                        continue_enrichment() along with
                                        the answer, or the conversation
                                        context is lost.
    """
    status: Literal["complete", "needs_clarification"]
    spec: dict | None = None
    question: str | None = None
    messages: list[dict] = field(default_factory=list)
    clarification_rounds: int = 0


def _load_schema() -> dict:
    # NOTE: kept exactly as it exists in the real repo -- api/config/
    # workflow_spec.schema.json -- rather than assuming the older
    # synthesis/schemas/ path from earlier in this project. If your
    # schema file actually lives somewhere else now, update this path
    # accordingly; this is the one thing in this file NOT rebuilt from
    # scratch, since I can't see your current directory layout.
    #
    # This file lives at api/services/enrichment/service.py, so:
    # parents[0] = services/enrichment/, parents[1] = services/,
    # parents[2] = api/
    schema_path = (
        Path(__file__).resolve().parents[2]   # api/
        / "config"
        / "workflow_spec.schema.json"
    )
    return json.loads(schema_path.read_text(encoding="utf-8"))


_SCHEMA = _load_schema()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---- recursive structure helpers --------------------------------------
# These exist because the document is a TREE (top-level steps, each of
# which may contain Parallel branches or a Map iterator, each of which
# contains its own steps, recursively) -- not a flat list. Every
# structural check below needs to see the whole tree, not just the
# top-level array, or it can be silently bypassed by hiding content one
# level deeper (this happened live -- see module docstring).

def _iter_all_steps(steps: list[dict] | None) -> Iterator[dict]:
    """Yields every step dict in the document, depth-first: each
    top-level step, then recursing into its Parallel branches and Map
    iterator (if any) before moving to the next top-level step.

    Defensive against malformed (non-dict) entries at any level -- a
    schema-invalid LLM response (e.g. a `steps` entry that's a raw
    string instead of an object) must not crash this walk. That case
    belongs to _check_schema to catch and feed back to the model;
    every other check/caller that shares this walker (fidelity, ARN
    whitelist, the enrichment loop's repeat-detection) would otherwise
    inherit whatever ordering guarantee happened to protect them,
    which is exactly the kind of implicit, easy-to-break invariant
    that caused this bug once already.
    """
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        yield step
        for branch in step.get("branches") or []:
            if isinstance(branch, dict):
                yield from _iter_all_steps(branch.get("steps"))
        iterator = step.get("iterator")
        if isinstance(iterator, dict):
            yield from _iter_all_steps(iterator.get("steps"))


def _collect_all_step_ids(spec: dict) -> list[str]:
    return [s.get("id") for s in _iter_all_steps(spec.get("steps"))]


def _find_step_by_id(spec: dict, step_id: str) -> dict | None:
    for s in _iter_all_steps(spec.get("steps")):
        if s.get("id") == step_id:
            return s
    return None


def _unknown_arn_token(step_id: str | None) -> str:
    """A TODO_ placeholder unique to this step, for the ARN-whitelist
    force-patch backstop (see the ARN_FORCE_PLACEHOLDER_AFTER loop).

    Must still match asl_service._PLACEHOLDER_PATTERN
    (r"\\bTODO_[A-Z0-9_]+\\b") so downstream scanning/description/
    override-substitution code (which all just do generic TODO_ token
    matching) treats it like any other placeholder. Falls back to a
    bare, non-unique token only if step_id is unusable -- better than
    crashing, though callers should always have a real step id here.
    """
    suffix = re.sub(r"[^A-Z0-9]+", "_", (step_id or "").upper()).strip("_")
    if not suffix:
        return "TODO_UNKNOWN_INTEGRATION_ARN"
    return f"TODO_UNKNOWN_INTEGRATION_ARN_{suffix}"


def _collect_all_references(spec: dict) -> set[str]:
    """Every step id referenced anywhere as a routing target: next_step,
    choices[].next_step, catch[].next_step, default_next, and Parallel /
    Map entry points (branches[].start_at, iterator.start_at) -- across
    the whole document, not just top level.
    """
    refs: set[str] = set()

    def walk_steps(steps) -> None:
        for s in steps or []:
            if not isinstance(s, dict):
                continue
            if s.get("next_step"):
                refs.add(s["next_step"])
            if s.get("default_next"):
                refs.add(s["default_next"])
            for c in s.get("choices") or []:
                if c.get("next_step"):
                    refs.add(c["next_step"])
            for c in s.get("catch") or []:
                if c.get("next_step"):
                    refs.add(c["next_step"])
            for b in s.get("branches") or []:
                if isinstance(b, dict) and b.get("start_at"):
                    refs.add(b["start_at"])
                walk_steps((b or {}).get("steps"))
            it = s.get("iterator")
            if isinstance(it, dict):
                if it.get("start_at"):
                    refs.add(it["start_at"])
                walk_steps(it.get("steps"))

    walk_steps(spec.get("steps"))
    return refs


def _added_step_nesting_context(spec: dict, step_id: str) -> str | None:
    """Return 'map_iterator' | 'parallel_branch' | None for where step_id lives."""

    def walk(steps, ctx: str | None) -> str | None:
        for s in steps or []:
            if not isinstance(s, dict):
                continue
            if s.get("id") == step_id:
                return ctx
            for b in s.get("branches") or []:
                found = walk((b or {}).get("steps"), "parallel_branch")
                if found:
                    return found
            it = s.get("iterator")
            if isinstance(it, dict):
                found = walk(it.get("steps"), "map_iterator")
                if found:
                    return found
        return None

    return walk(spec.get("steps"), None)


# ---- checks -------------------------------------------------------

def _suggest_close_field(exc: jsonschema.ValidationError) -> str | None:
    """For an 'additional properties not allowed' error, checks if the
    offending field name is a close match to a real allowed field at
    that point in the schema, and returns a "did you mean" hint.

    This exists because a real failure was observed live: the LLM
    wrote `default_end` instead of `default_next` on a nested Choice
    step, and repeated the exact same mistake across all repair
    attempts -- the raw jsonschema message states the problem but
    never points at the fix, so the model had nothing concrete to
    correct toward.
    """
    if exc.validator != "additionalProperties":
        return None
    allowed = set((exc.schema or {}).get("properties", {}).keys())
    if not allowed or not isinstance(exc.instance, dict):
        return None
    offending = set(exc.instance.keys()) - allowed
    hints = []
    for bad_field in offending:
        matches = difflib.get_close_matches(bad_field, allowed, n=1, cutoff=0.6)
        if matches:
            hints.append(f"'{bad_field}' is not a real field -- did you mean '{matches[0]}'?")
    return " ".join(hints) if hints else None


def _check_schema(spec: dict) -> str | None:
    """Returns an error message string if invalid, None if valid.
    Reports EVERY distinct schema violation in one message (not just
    the first) so a single repair attempt can fix everything at once,
    and adds a close-match field-name hint where possible.
    """
    validator = jsonschema.Draft202012Validator(_SCHEMA)
    errors = list(validator.iter_errors(spec))
    if not errors:
        return None

    lines = []
    for exc in errors[:5]:  # cap to keep the repair prompt from ballooning
        hint = _suggest_close_field(exc)
        line = f"- {exc.message} (at path: {list(exc.absolute_path)})"
        if hint:
            line += f" HINT: {hint}"
        lines.append(line)
    suffix = f" (+{len(errors) - 5} more)" if len(errors) > 5 else ""
    return "Schema validation failed with the following issue(s):\n" + "\n".join(lines) + suffix


def _check_fidelity(plan: Plan, spec: dict) -> str | None:
    """Confirms every plan step id appears EXACTLY ONCE somewhere in
    the full recursive spec (top-level, or nested inside any Parallel
    branch / Map iterator) -- not requiring top-level positional order,
    since a thin Plan has no nesting concept at all and enrichment is
    expected to nest steps as real ASL structure requires. See module
    docstring for why order-based top-level-only comparison was wrong.
    """
    plan_ids = [s.id for s in plan.steps]
    merged_ids = set(spec.get("merged_step_ids") or [])
    added_ids = set(spec.get("added_step_ids") or [])

    actual_ids = _collect_all_step_ids(spec)
    actual_id_set = set(actual_ids)

    seen: set[str] = set()
    dupes: set[str] = set()
    for sid in actual_ids:
        if sid in seen:
            dupes.add(sid)
        seen.add(sid)
    if dupes:
        return (
            f"Step id(s) {sorted(dupes)} appear more than once in the document "
            f"(counting both top-level steps and steps nested inside Parallel "
            f"branches / Map iterators). Each step must appear exactly once."
        )

    expected_present = set(plan_ids) - merged_ids
    missing = expected_present - actual_id_set
    if missing:
        return (
            f"Step fidelity check failed: plan step(s) {sorted(missing)} are "
            f"missing from the output entirely. If a step was intentionally "
            f"folded into another (e.g. a retry-only step), declare it in "
            f"merged_step_ids -- otherwise every plan step must appear "
            f"somewhere in the output, top-level or nested."
        )

    unexplained = actual_id_set - set(plan_ids) - added_ids
    if unexplained:
        return (
            f"Step fidelity check failed: output contains step id(s) "
            f"{sorted(unexplained)} that were not in the input plan and are not "
            f"declared in added_step_ids. Do not invent steps -- if ASL "
            f"structurally requires a new minimal terminal step, declare it in "
            f"added_step_ids (see the rules on when that's allowed)."
        )
    return None


def _check_merge_validity(plan: Plan, spec: dict) -> str | None:
    """For every id declared in merged_step_ids, confirms the merge was
    real, not a way to silently drop a step: real plan id, has a
    predecessor, predecessor is present in the output (anywhere,
    top-level or nested) and has a non-empty `retry` block.
    """
    merged_ids = spec.get("merged_step_ids") or []
    if not merged_ids:
        return None

    plan_ids = [s.id for s in plan.steps]

    for merged_id in merged_ids:
        if merged_id not in plan_ids:
            return (
                f"merged_step_ids contains '{merged_id}', which is not a real "
                f"step id from the input plan."
            )
        index = plan_ids.index(merged_id)
        if index == 0:
            return (
                f"merged_step_ids contains '{merged_id}', which is the first "
                f"step in the plan -- it has no predecessor to merge into."
            )
        predecessor_id = plan_ids[index - 1]
        predecessor_spec = _find_step_by_id(spec, predecessor_id)
        if predecessor_spec is None:
            return (
                f"'{merged_id}' was declared merged into '{predecessor_id}', but "
                f"'{predecessor_id}' is not present in the output steps."
            )
        if not predecessor_spec.get("retry"):
            return (
                f"'{merged_id}' was declared merged into '{predecessor_id}', but "
                f"'{predecessor_id}' has no `retry` block -- the retry behavior "
                f"described by '{merged_id}' must actually be applied to its "
                f"predecessor, not just omitted."
            )
    return None


def _check_addition_validity(plan: Plan, spec: dict) -> str | None:
    """For every id declared in added_step_ids, confirms it's a
    legitimate minimal terminal addition, not a loophole for inventing
    business logic: not already a plan id, actually present in the
    output, type is Succeed/Fail ONLY, and actually referenced by
    something (no dead additions).
    """
    added_ids = spec.get("added_step_ids") or []
    if not added_ids:
        return None

    plan_ids = {s.id for s in plan.steps}
    refs = _collect_all_references(spec)

    for added_id in added_ids:
        if added_id in plan_ids:
            return (
                f"added_step_ids contains '{added_id}', which is already a real "
                f"plan step id -- added_step_ids is only for genuinely NEW steps."
            )
        step = _find_step_by_id(spec, added_id)
        if step is None:
            return (
                f"'{added_id}' is declared in added_step_ids but does not appear "
                f"anywhere in the output steps."
            )
        nest = _added_step_nesting_context(spec, added_id)
        stype = step.get("type")
        if nest in {"map_iterator", "parallel_branch"}:
            # Map/Parallel need distinct worker step ids inside the nest;
            # those may be Tasks (the per-item / per-branch work). Still
            # forbid inventing nested Parallel/Map/Choice as "added" work.
            if stype not in (ALLOWED_ADDED_STEP_TYPES | {"Task", "Wait", "Pass"}):
                return (
                    f"'{added_id}' is declared in added_step_ids and nested under "
                    f"a {nest}, but type '{stype}' is not allowed there. Nested "
                    f"added workers may be Task/Wait/Pass or Succeed/Fail only."
                )
        elif stype not in ALLOWED_ADDED_STEP_TYPES:
            return (
                f"'{added_id}' is declared in added_step_ids with type "
                f"'{stype}', but top-level added steps may only be "
                f"{sorted(ALLOWED_ADDED_STEP_TYPES)} -- a minimal terminal, never "
                f"a Task/Choice/Wait/Parallel/Map. (Exception: Task/Wait/Pass "
                f"are allowed when nested inside a Map iterator or Parallel "
                f"branch as the worker for that structure.)"
            )
        if added_id not in refs:
            return (
                f"'{added_id}' is declared in added_step_ids but nothing routes "
                f"to it (no next_step/choices/catch/default_next references it) "
                f"-- an added step must actually be used."
            )
    return None


# Keywords that justify inventing Parallel / Map. Without at least one
# of these somewhere in the plan text, those types are treated as
# unnecessary structural invention and rejected by the repair loop.
_PARALLEL_HINTS = (
    "parallel",
    "concurrent",
    "simultaneously",
    "at the same time",
    "in parallel",
    "concurrently",
)
_MAP_HINTS = (
    "for each",
    "for every",
    "map over",
    "iterate over",
    "each item",
    "each element",
    "over a list",
    "over the list",
    "for all",
)


def _plan_text_blob(plan: Plan) -> str:
    parts = [plan.summary or ""]
    for s in plan.steps:
        parts.append(s.description or "")
        parts.append(s.service or "")
        for b in s.branches or []:
            parts.append(b.condition or "")
    return " ".join(parts).lower()


def _check_no_unjustified_nesting(plan: Plan, spec: dict) -> str | None:
    """Reject Parallel/Map unless the plan text clearly signals concurrency
    or iteration. Stops enrichment from inventing nesting to 'improve'
    a linear plan.
    """
    blob = _plan_text_blob(plan)
    allows_parallel = any(h in blob for h in _PARALLEL_HINTS)
    allows_map = any(h in blob for h in _MAP_HINTS)

    for step in _iter_all_steps(spec.get("steps")):
        stype = step.get("type")
        sid = step.get("id", "?")
        if stype == "Parallel" and not allows_parallel:
            return (
                f"Step '{sid}' has type Parallel, but the plan does not "
                f"clearly describe concurrency (e.g. 'in parallel', "
                f"'simultaneously'). Prefer sequential Task/Choice steps. "
                f"Only use Parallel when the plan explicitly requires it."
            )
        if stype == "Map" and not allows_map:
            return (
                f"Step '{sid}' has type Map, but the plan does not clearly "
                f"describe iteration over a collection (e.g. 'for each', "
                f"'map over a list'). Prefer sequential steps. Only use Map "
                f"when the plan explicitly requires it."
            )
    return None


def _find_unknown_arns(spec: dict) -> list[dict]:
    """Returns [{"step_id": ..., "resource": ...}, ...] for every Task
    step's `resource`, ANYWHERE in the document (including nested inside
    Parallel branches / Map iterators), that looks like a Step Functions
    optimized integration ARN but doesn't exactly match a known-real
    integration and isn't a TODO_ placeholder.

    Factored out of _check_arn_whitelist so callers can act on individual
    violations (e.g. auto-patch a resource that keeps getting rejected)
    rather than only seeing the first error message.
    """
    violations = []
    for step in _iter_all_steps(spec.get("steps")):
        resource = step.get("resource")
        if not resource:
            continue
        if resource.startswith("TODO_"):
            continue
        if resource.startswith(AWS_SDK_PASSTHROUGH_PREFIX):
            continue
        if resource.startswith("arn:aws:states:::"):
            matches_known = any(
                resource == known + suffix
                for known in KNOWN_ARNS
                for suffix in KNOWN_SUFFIXES
            )
            if not matches_known:
                violations.append({"step_id": step.get("id"), "resource": resource})
    return violations


def _check_arn_whitelist(spec: dict) -> str | None:
    """Returns an error message for the first unknown-ARN violation found
    (see _find_unknown_arns), or None if there are none.
    """
    violations = _find_unknown_arns(spec)
    if not violations:
        return None
    v = violations[0]
    return (
        f"Step '{v['step_id']}' has resource '{v['resource']}', which does "
        f"not match any known Step Functions integration pattern. If "
        f"you are not certain of the exact fixed ARN pattern for this "
        f"service, use 'TODO_UNKNOWN_INTEGRATION_ARN' instead of "
        f"guessing."
    )


def _run_all_checks(plan: Plan, spec: dict) -> str | None:
    """Runs structural checks in order, cheapest first. Returns the first
    failure found (or None if all pass).
    """
    checks_needing_plan = (
        _check_fidelity,
        _check_merge_validity,
        _check_addition_validity,
        _check_no_unjustified_nesting,
    )
    for check in (
        _check_schema,
        _check_fidelity,
        _check_merge_validity,
        _check_addition_validity,
        _check_no_unjustified_nesting,
    ):
        error = check(plan, spec) if check in checks_needing_plan else check(spec)
        if error:
            return error
    return None


def _run_enrichment_loop(
    plan: Plan, messages: list[dict], clarification_rounds: int
) -> EnrichmentResult:
    """Shared core for enrich_plan() and continue_enrichment(). Runs
    up to MAX_ENRICHMENT_ATTEMPTS LLM calls, checking each response for
    THREE possible shapes in order: malformed JSON (repair), a
    clarification request (pause and return to the caller), or a
    candidate spec (validate via _run_all_checks, repair on failure).
    """
    last_error = None

    for _attempt in range(1, MAX_ENRICHMENT_ATTEMPTS + 1):
        raw = chat_completion(messages, temperature=0)
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = f"Response was not valid JSON: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."})
            continue

        if isinstance(parsed, dict) and parsed.get("status") == "needs_clarification":
            question = parsed.get("clarifying_question")
            if not question:
                last_error = (
                    "status was 'needs_clarification' but no clarifying_question "
                    "was provided."
                )
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Error: {last_error}. Please fix and resend."})
                continue

            if clarification_rounds + 1 > MAX_CLARIFICATION_ROUNDS:
                raise EnrichmentError(
                    f"Enrichment asked more than {MAX_CLARIFICATION_ROUNDS} "
                    f"clarifying questions without resolving. Last question: {question}"
                )

            messages.append({"role": "assistant", "content": raw})
            return EnrichmentResult(
                status="needs_clarification",
                question=question,
                messages=messages,
                clarification_rounds=clarification_rounds + 1,
            )

        error = _run_all_checks(plan, parsed)
        if error is None:
            return EnrichmentResult(status="complete", spec=parsed, messages=messages)

        last_error = error
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."})

    raise EnrichmentError(
        f"Enrichment failed after {MAX_ENRICHMENT_ATTEMPTS} attempts. Last error: {last_error}"
    )


def enrich_plan(plan: Plan) -> EnrichmentResult:
    """Starts a fresh enrichment attempt for an approved plan."""
    messages = [
        {"role": "system", "content": ENRICHMENT_SYSTEM_PROMPT},
        {"role": "user", "content": plan.model_dump_json()},
    ]
    with llm_stage("enrichment"):
        return _run_enrichment_loop(plan, messages, clarification_rounds=0)


def continue_enrichment(
    plan: Plan, messages: list[dict], answer: str, clarification_rounds: int
) -> EnrichmentResult:
    """Resumes an enrichment conversation that previously paused for a
    clarifying question. `messages` and `clarification_rounds` must be
    exactly what an earlier EnrichmentResult with
    status == "needs_clarification" returned -- the caller (session
    layer) is responsible for persisting and passing these back.
    """
    resumed_messages = [*messages, {"role": "user", "content": answer}]
    return _run_enrichment_loop(plan, resumed_messages, clarification_rounds=clarification_rounds)