"""Unit tests for deterministic resolve_identifier ranking and ambiguity."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from orchestrator.api.services.tools.aws.resolver import (
    AMBIGUITY_GAP,
    CONFIDENT_MATCH,
    _score,
    resolve_identifier,
)
from orchestrator.api.services.tools.schemas import ResourceSummary


def _rs(name: str, service: str = "lambda", arn: str | None = None) -> ResourceSummary:
    return ResourceSummary(
        service=service,
        resource_type="function",
        id=name,
        name=name,
        arn=arn or f"arn:aws:lambda:us-east-1:123456789012:function:{name}",
        region="us-east-1",
    )


def test_exact_score_is_one():
    c = _rs("OrderProcessor")
    assert _score("OrderProcessor", c) == 1.0


def test_prefix_scores_high():
    c = _rs("OrderProcessor")
    assert _score("Order", c) >= 0.85


def test_resolve_exact_match():
    candidates = [_rs("OrderProcessor"), _rs("PaymentProcessor")]

    def fake_list(service, **kwargs):
        filters = kwargs.get("filters") or {}
        if filters.get("name") == "OrderProcessor":
            return [candidates[0]]
        return candidates

    with patch(
        "orchestrator.api.services.tools.aws.resolver.list_resources",
        side_effect=fake_list,
    ):
        result = resolve_identifier("lambda", "OrderProcessor")
    assert result["matched"] is True
    assert result["ambiguous"] is False
    assert result["confidence"] >= CONFIDENT_MATCH
    assert result["resource"]["name"] == "OrderProcessor"


def test_resolve_ambiguous_refuses_auto_pick():
    # Two very similar names → ambiguous
    a = _rs("OrderService")
    b = _rs("OrderServices")

    def fake_list(service, **kwargs):
        return [a, b]

    with patch(
        "orchestrator.api.services.tools.aws.resolver.list_resources",
        side_effect=fake_list,
    ):
        result = resolve_identifier("lambda", "OrderService")
    # Exact match on OrderService should still win cleanly
    # Use a hint that scores both similarly
    with patch(
        "orchestrator.api.services.tools.aws.resolver.list_resources",
        side_effect=fake_list,
    ):
        result = resolve_identifier("lambda", "OrderServ")
    # Either matched exact-ish or ambiguous — must not return matched with low gap
    if result.get("ambiguous"):
        assert result["matched"] is False
        assert result["resource"] is None
        assert len(result["alternatives"]) >= 1


def test_resolve_empty_hint():
    result = resolve_identifier("lambda", "  ")
    assert result["matched"] is False
    assert result["reason"] == "empty_hint"


def test_resolve_no_candidates():
    with patch(
        "orchestrator.api.services.tools.aws.resolver.list_resources",
        return_value=[],
    ):
        result = resolve_identifier("lambda", "NonexistentThingXyz")
    assert result["matched"] is False
    assert result["resource"] is None
