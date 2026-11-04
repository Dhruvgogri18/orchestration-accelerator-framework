"""Unit tests for allow-list safety controls."""

from __future__ import annotations

import pytest

from orchestrator.api.services.tools.aws.allowlist import (
    boto3_service_name,
    canonical_service,
    is_operation_allowed,
    is_service_allowed,
)


def test_canonical_service_aliases():
    assert canonical_service("StepFunctions") == "states"
    assert canonical_service("step-functions") == "states"
    assert canonical_service("EventBridge") == "events"
    assert canonical_service("lambda") == "lambda"


def test_service_allowlist():
    assert is_service_allowed("lambda")
    assert is_service_allowed("states")
    assert is_service_allowed("stepfunctions")
    assert not is_service_allowed("not-a-real-service-xyz")


def test_operation_allowlist_read_only():
    assert is_operation_allowed("list_functions")
    assert is_operation_allowed("describe_table")
    assert is_operation_allowed("get_function")
    assert is_operation_allowed("search_resources")


def test_operation_denylist_destructive():
    assert not is_operation_allowed("delete_function")
    assert not is_operation_allowed("create_bucket")
    assert not is_operation_allowed("invoke")
    assert not is_operation_allowed("publish")
    assert not is_operation_allowed("send_message")
    assert not is_operation_allowed("update_state_machine")
    assert not is_operation_allowed("put_object")


def test_boto3_service_name_mapping():
    assert boto3_service_name("states") == "stepfunctions"
    assert boto3_service_name("lambda") == "lambda"
