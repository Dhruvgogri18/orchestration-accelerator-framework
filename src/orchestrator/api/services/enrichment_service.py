"""Stage 3.5 enrichment: approved Plan -> full workflow_spec.json.

Two DIFFERENT loops are at play here, and it matters to keep them
conceptually separate:

  1. The REPAIR loop (automatic, invisible to the user): when the LLM's
     response fails one of the three validation checks below, the
     specific error is fed back and it gets another attempt, capped at
     MAX_ENRICHMENT_ATTEMPTS. This is the LLM correcting its OWN
     mistakes against checks WE run.
  2. The CLARIFICATION loop (visible to the user, a real pause): when
     the LLM itself decides a structural decision is genuinely
     undecidable from the plan (not a missing value -- that's what
     TODO_ placeholders are for -- but a missing STRUCTURE, e.g. "how
     many outcomes does this branch have"), it can ask a question
     instead of guessing. This is deliberately rare -- see
     enrichment_prompts.py rule 6 -- and is capped separately
     (MAX_CLARIFICATION_ROUNDS) so an indecisive LLM can't stall
     forever either.

Three independent validation checks guard the repair loop, run in
order (cheapest first):

  1. Schema validation   -- structurally valid per workflow_spec.schema.json
  2. Fidelity check       -- same step ids/order/count as the input plan
                             (catches silently dropped/invented/reordered
                             steps -- schema validation alone can't see
                             this)
  3. ARN whitelist check  -- any `resource` claiming to be a Step
                             Functions optimized integration must
                             EXACTLY match a known-real action, or be a
                             TODO_ placeholder. Anything else is
                             treated as a hallucinated ARN.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import jsonschema

from orchestrator.api.schemas.common import Plan
from orchestrator.api.services.enrichment_prompts import ENRICHMENT_SYSTEM_PROMPT
from orchestrator.indexing.gateway_client import chat_completion

MAX_ENRICHMENT_ATTEMPTS = 3
MAX_CLARIFICATION_ROUNDS = 3

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
KNOWN_SUFFIXES = ("", ".sync", ".sync2", ".waitForTaskToken")

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
    # src/orchestrator/api/services/enrichment_service.py -> src/orchestrator/api/config/workflow_spec.schema.json
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "config" / "workflow_spec.schema.json"
    )
    return json.loads(schema_path.read_text())


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


def _check_schema(spec: dict) -> str | None:
    """Returns an error message string if invalid, None if valid."""
    try:
        jsonschema.Draft202012Validator(_SCHEMA).validate(spec)
    except jsonschema.ValidationError as exc:
        return f"Schema validation failed: {exc.message} (at path: {list(exc.absolute_path)})"
    return None


def _check_fidelity(plan: Plan, spec: dict) -> str | None:
    """Returns an error message if the spec's steps don't match the
    plan's steps 1:1 in id and order -- catches silently dropped,
    invented, duplicated, or reordered steps that schema validation
    alone cannot detect (a structurally valid spec can still have the
    wrong steps).
    """
    plan_ids = [s.id for s in plan.steps]
    spec_steps = spec.get("steps", [])
    spec_ids = [s.get("id") for s in spec_steps]

    if spec_ids != plan_ids:
        return (
            f"Step fidelity check failed: expected steps {plan_ids} in that exact "
            f"order, got {spec_ids}. Do not add, remove, reorder, or rename steps."
        )
    return None


def _check_arn_whitelist(spec: dict) -> str | None:
    """Returns an error message if any Task step's `resource` looks
    like a Step Functions optimized integration ARN but doesn't
    exactly match a known-real integration (allowing known .sync
    suffixes) and isn't a TODO_ placeholder -- a likely sign of a
    hallucinated integration pattern. The aws-sdk: passthrough form is
    trusted structurally rather than exact-matched (see
    AWS_SDK_PASSTHROUGH_PREFIX above).
    """
    for step in spec.get("steps", []):
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
                return (
                    f"Step '{step.get('id')}' has resource '{resource}', which does "
                    f"not match any known Step Functions integration pattern. If "
                    f"you are not certain of the exact fixed ARN pattern for this "
                    f"service, use 'TODO_UNKNOWN_INTEGRATION_ARN' instead of "
                    f"guessing."
                )
    return None


def _run_all_checks(plan: Plan, spec: dict) -> str | None:
    """Runs the three checks in order, cheapest first, returns the
    first failure found (or None if all pass).
    """
    for check in (_check_schema, _check_fidelity, _check_arn_whitelist):
        error = check(plan, spec) if check is _check_fidelity else check(spec)
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