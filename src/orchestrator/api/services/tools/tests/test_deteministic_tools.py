"""Unit tests for zero-LLM deterministic tools — no AWS, no network."""

from __future__ import annotations

from orchestrator.api.services.tools.registry import execute_tool, tool_names


def test_new_tools_registered():
    names = set(tool_names())
    for n in (
        "normalize_prompt",
        "extract_entities",
        "classify_workflow_pattern",
        "assign_canonical_step_ids",
        "suggest_placeholder_names",
        "get_service_integration_contract",
        "ground_resources",
        "validate_plan_against_aws",
    ):
        assert n in names


def test_normalize_prompt_stable_and_paraphrase_friendly():
    a = execute_tool(
        "normalize_prompt",
        {"text": "Please can you create a workflow that validates then runs two Glue jobs in parallel and loads to DynamoDB"},
    )
    b = execute_tool(
        "normalize_prompt",
        {"text": "I want to build a pipeline: validate input, run both ETL jobs concurrently, then write to Dynamo"},
    )
    assert a.status.value == "ok"
    assert b.status.value == "ok"
    # Same input twice → identical
    a2 = execute_tool(
        "normalize_prompt",
        {"text": "Please can you create a workflow that validates then runs two Glue jobs in parallel and loads to DynamoDB"},
    )
    assert a.data == a2.data
    # Paraphrases should both surface glue + dynamodb (+ baselines)
    for result in (a, b):
        services = set(result.data.get("services") or [])
        assert "glue" in services or "dynamodb" in services
        assert "lambda" in services  # baseline when compute verbs present


def test_classify_pattern_parallel_etl():
    r = execute_tool(
        "classify_workflow_pattern",
        {
            "text": "validate the file, then process two jobs in parallel, run quality checks, load to dynamodb"
        },
    )
    assert r.status.value == "ok"
    assert r.data["pattern"] == "validate_parallel_quality_load"


def test_classify_pattern_map():
    r = execute_tool(
        "classify_workflow_pattern",
        {"text": "for each file in the bucket, invoke a lambda"},
    )
    assert r.data["pattern"] == "map_over_items"


def test_extract_entities_arn_and_name():
    r = execute_tool(
        "extract_entities",
        {
            "text": "Use OrderProcessorLambda and arn:aws:sqs:us-east-1:123456789012:orders-queue"
        },
    )
    assert r.status.value == "ok"
    values = {e["value"] for e in r.data["entities"]}
    assert "OrderProcessorLambda" in values
    assert any(v.startswith("arn:aws:sqs:") for v in values)


def test_integration_contract_lambda():
    r = execute_tool("get_service_integration_contract", {"service": "lambda"})
    assert r.status.value == "ok"
    assert r.data.get("found") is True
    assert "FunctionName" in (r.data.get("required_params") or [])


def test_suggest_placeholders_deterministic():
    r1 = execute_tool(
        "suggest_placeholder_names",
        {
            "entities": [{"kind": "name", "value": "OrderProcessor", "service_hint": "lambda"}],
            "services": ["sns"],
        },
    )
    r2 = execute_tool(
        "suggest_placeholder_names",
        {
            "entities": [{"kind": "name", "value": "OrderProcessor", "service_hint": "lambda"}],
            "services": ["sns"],
        },
    )
    assert r1.data == r2.data
    names = [p["placeholder"] for p in r1.data["placeholders"]]
    assert any(n.startswith("TODO_") for n in names)


def test_assign_canonical_step_ids():
    plan = {
        "steps": [
            {
                "id": "CheckFile",
                "service": "lambda",
                "description": "validate the input file",
                "next": "DoWork",
            },
            {
                "id": "DoWork",
                "service": "glue",
                "description": "transform data",
                "next": None,
            },
        ]
    }
    r = execute_tool("assign_canonical_step_ids", {"plan": plan})
    assert r.status.value == "ok"
    assert "id_map" in r.data
    assert len(r.data["id_map"]) == 2
    # Same plan again → same map
    r2 = execute_tool("assign_canonical_step_ids", {"plan": plan})
    assert r.data["id_map"] == r2.data["id_map"]
