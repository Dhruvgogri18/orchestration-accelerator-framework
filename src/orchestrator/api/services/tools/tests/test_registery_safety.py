"""Registry safety: unknown tools, forbidden ops, no cache flag."""

from __future__ import annotations

from unittest.mock import patch

from orchestrator.api.services.tools.registry import (
    clear_tool_cache,
    execute_tool,
    list_tool_schemas,
    tool_names,
)
from orchestrator.api.services.tools.schemas import ToolStatus


def test_tool_names_stable():
    names = tool_names()
    assert "list_resources" in names
    assert "resolve_identifier" in names
    assert "validate_resource_reference" in names
    assert names == sorted(names)


def test_list_tool_schemas_openai_shape():
    schemas = list_tool_schemas()
    assert schemas
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "parameters" in s["function"]


def test_unknown_tool():
    result = execute_tool("not_a_tool", {})
    assert result.status == ToolStatus.ERROR
    assert result.error_code == "unknown_tool"


def test_invoke_denied_operation():
    with patch(
        "orchestrator.api.services.tools.registry._ensure_single_account",
        return_value={"account_id": "123", "region": "us-east-1"},
    ):
        result = execute_tool(
            "invoke_aws_operation",
            {"service": "lambda", "operation": "delete_function", "parameters": {}},
        )
    assert result.status == ToolStatus.FORBIDDEN


def test_clear_tool_cache_is_noop():
    assert clear_tool_cache() is None


def test_meta_never_cache_hit_on_catalogue_tool():
    result = execute_tool("list_aws_services", {})
    assert result.status == ToolStatus.OK
    assert result.meta.get("cache_hit") is False
