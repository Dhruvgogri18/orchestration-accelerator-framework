"""Context injection and formatting."""

from __future__ import annotations

from orchestrator.api.services.tools.context import (
    format_discovery_for_prompt,
    inject_discovery_into_history,
)


def test_format_empty():
    assert format_discovery_for_prompt(None) is None
    assert format_discovery_for_prompt({}) is None
    assert format_discovery_for_prompt({"resources_by_service": {}}) is None


def test_format_includes_resources():
    discovery = {
        "account_id": "111122223333",
        "services_inferred": ["lambda"],
        "resources_by_service": {
            "lambda": [
                {
                    "service": "lambda",
                    "resource_type": "function",
                    "id": "OrderFn",
                    "name": "OrderFn",
                    "arn": "arn:aws:lambda:us-east-1:111122223333:function:OrderFn",
                    "summary": {"runtime": "python3.12"},
                }
            ]
        },
        "search_hits": [],
    }
    text = format_discovery_for_prompt(discovery)
    assert text is not None
    assert "OrderFn" in text
    assert "111122223333" in text
    assert "TODO_" in text  # guidance to use placeholders when no match


def test_inject_prepends_system():
    history = [{"role": "user", "content": "build a workflow"}]
    discovery = {
        "services_inferred": ["lambda"],
        "resources_by_service": {
            "lambda": [
                {
                    "service": "lambda",
                    "resource_type": "function",
                    "id": "A",
                    "name": "A",
                }
            ]
        },
    }
    out = inject_discovery_into_history(history, discovery)
    assert out[0]["role"] == "system"
    assert "AWS ACCOUNT CONTEXT" in out[0]["content"]
    assert out[-1]["role"] == "user"
    # original history not mutated
    assert len(history) == 1
