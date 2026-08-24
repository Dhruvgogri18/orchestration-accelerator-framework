"""Stage 1: Pipeline Analysis -- Prompt2DAG-aligned five-sub-stage chain.

Mirrors Section 5.1 of the Prompt2DAG paper (Alidu, Ciavotta, De Paoli,
2025 -- arXiv:2509.13487): environment inference -> component
identification -> flow structure analysis -> parameter extraction ->
integration analysis, each stage building on the previous stage's
structured output, composed into one versioned `PipelineAnalysis` JSON
artifact (schemas/pipeline_analysis.py).

This module owns its own determinism infrastructure (#2 repair loop via
`_call_llm_json_with_repair`, #4 structured output via
`_chat_completion_structured`, #6 scope prefilter via `classify_scope`)
-- formerly imported from extraction/service.py, now inlined below
since this is the only module that calls them (the old
extraction/service.py split pipeline -- classify_scope /
decompose_steps / wire_control_flow -- had no other live caller and
was deleted). What's new here relative to that old split pipeline is:
the component taxonomy on sub-stage 2, entry_points/parallel_sections
on sub-stage 3, and two entirely new sub-stages (4: parameters, 5:
integrations) the paper describes but the old pipeline never
implemented.

`analyze_pipeline()` is the entry point and is what session_service.py
(api/services/pipeline_analysis/session_service.py) calls. It returns
an `ExtractionResult` -- the exact shape session_service.py already
understands (ready / needs_clarification / out_of_scope) -- with
`.analysis` populated on the ready path (the full PipelineAnalysis
artifact, as a dict) and `.plan` derived from `analysis.flow` so every
existing downstream consumer (plan.render, plan.canonicalize,
plan.matching, enrichment.service, asl_service) keeps working
unchanged -- they only ever depended on the Plan shape, never on how
it was produced.

Self-consistency (#7/#8) is intentionally NOT applied stage-by-stage
here -- five sequential stages already means five LLM calls per turn;
voting each one independently would multiply that further for
uncertain marginal benefit on the later stages (3-5), which are
constrained by the fixed component list from stage 2 and have much
less freedom to disagree. Stage 2 (component identification) is the
one stage where free-form disagreement is most likely -- if that
proves noisy in practice, self-consistency can be layered onto
identify_components() without touching stages 3-5.
"""

from __future__ import annotations

import contextvars
import json
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, ValidationError

from orchestrator.api.schemas.common import ExtractionResult, Plan, PlanBranch, PlanStep, ScopeResult
from orchestrator.api.schemas.pipeline_analysis import (
    ActionClauseResult,
    ComponentDefinition,
    ComponentIdentificationResult,
    EnvironmentInference,
    FlowStructure,
    IntegrationAnalysisResult,
    ParameterSpec,
    PipelineAnalysis,
)
from orchestrator.indexing.gateway_client import chat_completion
from orchestrator.api.services.llm_trace import llm_stage
from orchestrator.api.services.pipeline_analysis.prompts import (
    ACTION_CLAUSE_SYSTEM_PROMPT,
    COMPONENT_IDENTIFICATION_SYSTEM_PROMPT,
    ENVIRONMENT_INFERENCE_SYSTEM_PROMPT,
    FLOW_STRUCTURE_SYSTEM_PROMPT,
    INTEGRATION_ANALYSIS_SYSTEM_PROMPT,
    PARAMETER_EXTRACTION_SYSTEM_PROMPT,
    SCOPE_SYSTEM_PROMPT,
)

MAX_STAGE_ATTEMPTS = 3


class ExtractionError(Exception):
    """Raised when the LLM's response can't be trusted -- bad JSON, or
    JSON that doesn't match the expected shape -- after exhausting the
    repair loop.
    """


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# #4 structured output -- schema-constrained calls with graceful fallback.
# ---------------------------------------------------------------------------

_structured_output_supported = True


def _chat_completion_structured(messages: list[dict], model_cls: type[BaseModel]) -> str:
    global _structured_output_supported
    if _structured_output_supported:
        try:
            return chat_completion(
                messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": model_cls.__name__,
                        "schema": model_cls.model_json_schema(),
                    },
                },
            )
        except TypeError:
            _structured_output_supported = False
    return chat_completion(messages, temperature=0)


# ---------------------------------------------------------------------------
# #2 repair loop -- shared call+parse+validate given an already-built
# messages list: structured-output request (#4) with a repair loop on top.
# ---------------------------------------------------------------------------


def _call_llm_json_with_repair(
    messages: list[dict],
    model_cls: type[BaseModel],
    max_attempts: int = MAX_STAGE_ATTEMPTS,
):
    last_error: str | None = None

    for _attempt in range(1, max_attempts + 1):
        raw = _chat_completion_structured(messages, model_cls)
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = f"Response was not valid JSON: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

        try:
            return model_cls.model_validate(parsed)
        except ValidationError as exc:
            last_error = f"JSON did not match expected shape: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

    raise ExtractionError(
        f"Stage failed after {max_attempts} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# #6 rule-based prefilter, then sub-stage 0 (scope classification).
# ---------------------------------------------------------------------------

_WORKFLOW_HINTS = re.compile(
    r"\b(workflow|step function|stepfunctions|orchestrat|lambda|s3|sns|sqs|"
    r"dynamodb|glue|eventbridge|pipeline|state machine|on failure|retry|"
    r"aws|arn:)\b",
    re.IGNORECASE,
)
_OFF_TOPIC_HINTS = re.compile(
    r"\b(weather|recipe|who won|capital of|joke|distance between|"
    r"translate|poem|lyrics)\b",
    re.IGNORECASE,
)


def _prefilter_scope(history: list[dict]) -> ScopeResult | None:
    if not history or history[-1].get("role") != "user":
        return None
    text = history[-1].get("content") or ""
    if _WORKFLOW_HINTS.search(text):
        return None
    if _OFF_TOPIC_HINTS.search(text):
        return ScopeResult(
            status="out_of_scope",
            message=(
                "This assistant is only designed to help you build AWS Step "
                "Functions workflows from a natural-language description. "
                "It cannot answer general questions (for example travel "
                "distances or unrelated topics). Please describe a workflow "
                "you want to orchestrate — steps, AWS services, and what "
                "should happen on success or failure."
            ),
        )
    return None


def classify_scope(history: list[dict]) -> ScopeResult:
    prefiltered = _prefilter_scope(history)
    if prefiltered is not None:
        return prefiltered
    messages = [{"role": "system", "content": SCOPE_SYSTEM_PROMPT}, *history]
    return _call_llm_json_with_repair(messages, ScopeResult)

# ---------------------------------------------------------------------------
# Sub-stage 1: Environment inference
# ---------------------------------------------------------------------------


def infer_environment(history: list[dict]) -> EnvironmentInference:
    messages = [{"role": "system", "content": ENVIRONMENT_INFERENCE_SYSTEM_PROMPT}, *history]
    return _call_llm_json_with_repair(messages, EnvironmentInference)


# ---------------------------------------------------------------------------
# Sub-stage 2: Component identification
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sub-stage 1.1: Action clause enumeration (see ActionClauseResult's
# docstring in schemas/pipeline_analysis.py for why this has to be its
# own isolated call rather than something component identification
# self-reports).
# ---------------------------------------------------------------------------


def enumerate_action_clauses(history: list[dict]) -> ActionClauseResult:
    messages = [{"role": "system", "content": ACTION_CLAUSE_SYSTEM_PROMPT}, *history]
    return _call_llm_json_with_repair(messages, ActionClauseResult)


# ---------------------------------------------------------------------------
# Sub-stage 2: Component identification
# ---------------------------------------------------------------------------


def _check_component_completeness(
    clauses: list[str], components: list[ComponentDefinition]
) -> str | None:
    """Catches gross under-extraction on compound requests (several of
    PARALLEL / MAP / CHOICE combined in one description) before it ever
    reaches flow structure -- that stage can only wire components that
    exist, it can't invent ones component identification silently
    dropped or merged.

    `clauses` comes from the INDEPENDENT enumerate_action_clauses() pass,
    not from component identification's own output -- checking against a
    self-report would let an under-extracting model under-report the
    checklist too, since it's the same generation grading its own
    homework. Deliberately count-based, not semantic; a generous ratio
    (0.6) leaves room for legitimate folding (retry/timeout wording
    collapses into the component it modifies, per the RETRY RULE)
    without false-positiving on ordinary requests.
    """
    if not clauses:
        return None
    min_expected = max(1, int(len(clauses) * 0.6))
    if len(components) >= min_expected:
        return None
    return (
        f"The independent action-clause pass identified {len(clauses)} "
        f"distinct action clauses in the request, but components only "
        f"has {len(components)} entries -- that's far fewer than "
        f"expected, which means clauses were dropped or silently merged "
        f"(see the COMPLETENESS RULE). The clauses were: "
        f"{json.dumps(clauses)}. Go through them entry by entry and make "
        f"sure each one either has its own component, or is clearly just "
        f"a retry/timeout attribute folded into another component's "
        f"description (RETRY RULE) -- then resend the full corrected "
        f"components JSON."
    )


def identify_components(
    history: list[dict],
    action_clauses: list[str] | None = None,
    max_attempts: int = MAX_STAGE_ATTEMPTS,
) -> ComponentIdentificationResult:
    clause_context = (
        [
            {
                "role": "user",
                "content": (
                    "Action clauses identified from this request (an "
                    "earlier, independent enumeration pass -- your "
                    "components list must account for every one of "
                    f"them; see the COMPLETENESS RULE):\n{json.dumps(action_clauses)}"
                ),
            }
        ]
        if action_clauses
        else []
    )
    messages = [
        {"role": "system", "content": COMPONENT_IDENTIFICATION_SYSTEM_PROMPT},
        *history,
        *clause_context,
    ]
    last_error: str | None = None

    for _attempt in range(1, max_attempts + 1):
        raw = _chat_completion_structured(messages, ComponentIdentificationResult)
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = f"Response was not valid JSON: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

        try:
            result = ComponentIdentificationResult.model_validate(parsed)
        except ValidationError as exc:
            last_error = f"JSON did not match expected shape: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

        if result.status == "ready":
            completeness_error = _check_component_completeness(
                action_clauses or [], result.components or []
            )
            if completeness_error is not None:
                last_error = completeness_error
                messages.append({"role": "assistant", "content": raw})
                messages.append({"role": "user", "content": f"Error: {last_error}"})
                continue

        return result

    raise ExtractionError(
        f"Component identification failed after {max_attempts} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Sub-stage 3: Flow structure analysis
# ---------------------------------------------------------------------------


def _check_components_preserved(
    components: list[ComponentDefinition], flow: FlowStructure
) -> str | None:
    """Same "no invention" contract wire_control_flow already enforces
    for Plan, extended with component_type since this stage carries it
    through rather than re-deriving it.
    """
    expected = [c.id for c in components]
    actual = [s.id for s in flow.steps]
    if expected != actual:
        return (
            "Flow structure stage changed the component list -- expected "
            f"ids {expected} in order, got {actual}. This stage must only "
            "add control flow (next/on_failure/on_timeout/branches/"
            "entry_points/parallel_sections), never add/remove/rename/"
            "reorder components. Resend the full JSON with exactly the "
            "given components, in order."
        )
    by_id = {c.id: c for c in components}
    for s in flow.steps:
        expected_type = by_id[s.id].component_type
        if s.component_type != expected_type:
            return (
                f"Component '{s.id}' component_type changed from "
                f"'{expected_type}' to '{s.component_type}' -- carry the "
                "value through unchanged, it was fixed in the previous stage."
            )
    return None


def _check_decision_gates_wired(flow: FlowStructure) -> str | None:
    """A DecisionGate with no branches (or only one) is a component the
    LLM identified but never actually wired -- exactly the bug pattern
    that produced a disconnected CreateOrder/PublishRejectionEvent pair
    in practice: EvaluateCombinedDecision was typed DecisionGate but
    left with empty branches, so nothing pointed at either outcome step.
    """
    problems = []
    for s in flow.steps:
        if s.component_type != "DecisionGate":
            continue
        if len(s.branches) < 2:
            problems.append(
                f"'{s.id}' is a DecisionGate but has {len(s.branches)} branch(es) -- "
                "a decision needs at least 2 exclusive arms (e.g. pass/fail, "
                "approved/rejected), each {\"condition\": \"...\", \"next\": \"ComponentId\"}."
            )
        if s.next is not None:
            problems.append(
                f"'{s.id}' is a DecisionGate and has both 'next' and 'branches' set -- "
                "omit 'next' on a component that has branches; the branches ARE its "
                "successors."
            )
    if not problems:
        return None
    return "DecisionGate wiring problem(s): " + " ".join(problems)


def _check_flow_connected(flow: FlowStructure) -> str | None:
    """Every component except a declared entry point must be reachable
    by at least one edge (next / branches[].next / on_failure /
    on_timeout) from some other component. A component with no
    incoming edge and not listed in entry_points is an orphan --
    identified during component identification but never actually
    wired into the graph, which produces the exact bug pattern seen in
    practice: terminal steps (e.g. CreateOrder, PublishRejectionEvent)
    left floating with no path leading to them at all.
    """
    incoming: set[str] = set()
    for s in flow.steps:
        if s.next:
            incoming.add(s.next)
        for target in s.on_failure:
            incoming.add(target)
        for target in s.on_timeout:
            incoming.add(target)
        for b in s.branches:
            incoming.add(b.next)

    all_ids = {s.id for s in flow.steps}
    entry_points = set(flow.entry_points)
    orphans = sorted((all_ids - entry_points) - incoming)
    if not orphans:
        return None
    return (
        f"These components have no incoming edge and are not listed in "
        f"entry_points, so they are unreachable: {orphans}. Every "
        "non-entry-point component must be the target of some other "
        "component's next / branches[].next / on_failure / on_timeout. "
        "If a component is meant to run after a decision, add it as a "
        "branch target on the relevant DecisionGate; if after a failure, "
        "add it to the relevant on_failure list."
    )


def analyze_flow_structure(
    history: list[dict],
    components: list[ComponentDefinition],
    max_attempts: int = MAX_STAGE_ATTEMPTS,
) -> FlowStructure:
    components_json = json.dumps([c.model_dump() for c in components])
    messages = [
        {"role": "system", "content": FLOW_STRUCTURE_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Components to wire:\n{components_json}"},
    ]
    last_error: str | None = None

    for _attempt in range(1, max_attempts + 1):
        raw = _chat_completion_structured(messages, FlowStructure)
        cleaned = _strip_code_fences(raw)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            last_error = f"Response was not valid JSON: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

        try:
            flow = FlowStructure.model_validate(parsed)
        except ValidationError as exc:
            last_error = f"JSON did not match expected shape: {exc}"
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": f"Error: {last_error}. Please fix and resend the full JSON."}
            )
            continue

        structure_error = (
            _check_components_preserved(components, flow)
            or _check_decision_gates_wired(flow)
            or _check_flow_connected(flow)
        )
        if structure_error is None:
            return flow

        last_error = structure_error
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Error: {last_error}"})

    raise ExtractionError(
        f"Flow structure analysis failed after {max_attempts} attempts. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Sub-stage 4: Parameter extraction
# ---------------------------------------------------------------------------


def _analysis_so_far(components: list[ComponentDefinition], flow: FlowStructure) -> str:
    return json.dumps(
        {"components": [c.model_dump() for c in components], "flow": flow.model_dump()}
    )


def extract_parameters(
    history: list[dict], components: list[ComponentDefinition], flow: FlowStructure
) -> list[ParameterSpec]:
    from orchestrator.api.schemas.pipeline_analysis import ParameterExtractionResult

    messages = [
        {"role": "system", "content": PARAMETER_EXTRACTION_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Analyzed pipeline so far:\n{_analysis_so_far(components, flow)}"},
    ]
    result = _call_llm_json_with_repair(messages, ParameterExtractionResult)
    return result.parameters


# ---------------------------------------------------------------------------
# Sub-stage 5: Integration analysis
# ---------------------------------------------------------------------------


def analyze_integrations(
    history: list[dict], components: list[ComponentDefinition], flow: FlowStructure
) -> IntegrationAnalysisResult:
    messages = [
        {"role": "system", "content": INTEGRATION_ANALYSIS_SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": f"Analyzed pipeline so far:\n{_analysis_so_far(components, flow)}"},
    ]
    return _call_llm_json_with_repair(messages, IntegrationAnalysisResult)


# ---------------------------------------------------------------------------
# Small helper: run independent sub-stages concurrently
# ---------------------------------------------------------------------------


def _run_concurrently(**calls):
    """Run keyword-named zero-arg callables in parallel threads and return
    {name: result} once all finish. Re-raises the first exception (after
    all threads have finished) so error behavior matches the old
    sequential code -- callers don't need to know this became concurrent.

    Only ever used for sub-stages that are genuinely independent (neither
    reads the other's output) -- see call sites below for why each grouping
    is safe. The LLM calls themselves are network-bound, not CPU-bound, so
    plain threads (no asyncio rewrite needed) are enough to overlap them.

    contextvars.copy_context() is used per-call so llm_trace's
    session_id/request_id/stage ContextVars (and therefore live /status
    polling + the trace log) still show the right session/stage instead of
    leaking None across threads.
    """
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        futures = {
            name: pool.submit(contextvars.copy_context().run, fn)
            for name, fn in calls.items()
        }
        results: dict = {}
        first_exc: Exception | None = None
        for name, fut in futures.items():
            try:
                results[name] = fut.result()
            except Exception as exc:  # noqa: BLE001 -- re-raised below, not swallowed
                if first_exc is None:
                    first_exc = exc
        if first_exc is not None:
            raise first_exc
        return results


# ---------------------------------------------------------------------------
# Composition + entry point
# ---------------------------------------------------------------------------


def _flow_to_plan(flow: FlowStructure) -> Plan:
    """Derive the legacy Plan shape (the contract plan.render,
    plan.canonicalize, plan.matching, and enrichment.service all rely
    on) from the richer FlowStructure. component_type/entry_points/
    parallel_sections are analysis-only extras those consumers were
    never written to expect.
    """
    steps = [
        PlanStep(
            id=s.id,
            service=s.service,
            description=s.description,
            next=s.next,
            on_failure=list(s.on_failure),
            on_timeout=list(s.on_timeout),
            branches=[PlanBranch(condition=b.condition, next=b.next) for b in s.branches],
        )
        for s in flow.steps
    ]
    return Plan(name=flow.pipeline_name, summary=flow.pipeline_summary, steps=steps)


def analyze_pipeline(history: list[dict]) -> ExtractionResult:
    """Entry point: Stage 1 Pipeline Analysis, five sub-stages, gated by
    the existing scope classifier (sub-stage 0). See module docstring.

    Raises ExtractionError (same type extract_turn() raises) if any
    sub-stage exhausts its repair-loop attempts -- session_service.py's
    existing 502 mapping applies unchanged.
    """
    scope = classify_scope(history)
    if scope.status == "out_of_scope":
        return ExtractionResult(status="out_of_scope", message=scope.message)

    # Sub-stages 1 (environment) and the action-clause pass only ever read
    # `history` -- neither depends on the other's output -- so they run
    # concurrently instead of back-to-back. Saves ~1 LLM round trip of
    # wall-clock latency on every turn at zero extra token cost (still
    # exactly the same two calls, just overlapped).
    with llm_stage("pipeline_analysis_env_and_clauses"):
        parallel = _run_concurrently(
            environment=lambda: infer_environment(history),
            action_clauses=lambda: enumerate_action_clauses(history).clauses,
        )
    environment = parallel["environment"]
    action_clauses = parallel["action_clauses"]

    with llm_stage("component_identification"):
        components_result = identify_components(history, action_clauses=action_clauses)
    if components_result.status == "needs_clarification":
        return ExtractionResult(
            status="needs_clarification",
            clarifying_question=components_result.clarifying_question,
        )
    components = components_result.components or []

    with llm_stage("flow_structure_analysis"):
        flow = analyze_flow_structure(history, components)

    # Sub-stages 4 (parameters) and 5 (integrations) both only read
    # (components, flow) from stage 3 -- neither reads the other's
    # output -- so they run concurrently too. Same saving as above.
    with llm_stage("pipeline_analysis_params_and_integrations"):
        parallel = _run_concurrently(
            parameters=lambda: extract_parameters(history, components, flow),
            integrations=lambda: analyze_integrations(history, components, flow),
        )
    parameters = parallel["parameters"]
    integrations = parallel["integrations"]

    analysis = PipelineAnalysis(
        pipeline_name=flow.pipeline_name,
        pipeline_summary=flow.pipeline_summary,
        environment=environment,
        components=components,
        flow=flow,
        parameters=parameters,
        integrations=integrations,
    )
    plan = _flow_to_plan(flow)
    return ExtractionResult(status="ready", plan=plan, analysis=analysis.model_dump())