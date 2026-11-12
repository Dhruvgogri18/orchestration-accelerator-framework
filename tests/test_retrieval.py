"""Dual-strategy retrieval: dense + structured → merge → re-rank → format."""

from __future__ import annotations

import pytest

from orchestrator.ingestion.models import Chunk
from orchestrator.indexing import vectorstore
from orchestrator.retrieval import retrieve, retrieve_context
from orchestrator.retrieval.format import format_context
from orchestrator.retrieval.merge import merge_hits
from orchestrator.retrieval.models import ScoredHit
from orchestrator.retrieval.rerank import rerank
from orchestrator.retrieval.structured import extract_candidates, retrieve_structured


def _chunk(
    chunk_id: str,
    text: str,
    *,
    repo_id: str = "demo",
    path: str = "config/jobs.yaml",
    kind: str = "job_definition",
    resource_name: str | None = "charge-payment",
    resource_type: str | None = "lambda",
    status: str = "active",
) -> Chunk:
    meta: dict = {
        "repo_id": repo_id,
        "path": path,
        "file_type": "yaml",
        "chunk_kind": kind,
        "unit": chunk_id.split("::")[-1],
        "status": status,
    }
    if resource_name:
        meta["resource_name"] = resource_name
        meta["resource_names"] = [resource_name]
    if resource_type:
        meta["resource_type"] = resource_type
    header = f"[{repo_id}] {path} | {kind}: {resource_name or 'n/a'}"
    return Chunk(id=chunk_id, text=f"{header}\n{text}", metadata=meta)


@pytest.fixture
def seeded_collection(pg_collection, fake_embedder):
    """Small index with three distinct resources (pgvector)."""
    chunks = [
        _chunk(
            "demo::config/jobs.yaml::jobs:charge-payment",
            "- name: charge-payment\n  type: lambda\n  script: src/charge.py\n",
            resource_name="charge-payment",
            resource_type="lambda",
            kind="job_definition",
        ),
        _chunk(
            "demo::config/jobs.yaml::jobs:validate-order",
            "- name: validate-order\n  type: lambda\n  script: src/validate.py\n",
            resource_name="validate-order",
            resource_type="lambda",
            kind="job_definition",
            path="config/jobs.yaml",
        ),
        _chunk(
            "demo::config/env.yaml::sns_topics:ops-alerts",
            "ops-alerts:\n  name: ops-alerts\n  type: sns\n",
            resource_name="ops-alerts",
            resource_type="sns_topic",
            kind="env_resources",
            path="config/env.yaml",
        ),
        _chunk(
            "demo::docs/runbook.md::section:Alerting",
            "When charge-payment fails, publish to ops-alerts.\n",
            resource_name=None,
            resource_type=None,
            kind="doc_section",
            path="docs/runbook.md",
            status="active",
        ),
    ]
    vectors = fake_embedder([c.text for c in chunks])
    vectorstore.upsert_chunks(pg_collection, chunks, vectors)
    return pg_collection


# ---------------------------------------------------------------------------
# unit: structured extract + score path
# ---------------------------------------------------------------------------


def test_extract_candidates_names_and_types():
    names, types = extract_candidates(
        "Wire the charge-payment Lambda to the ops-alerts SNS topic"
    )
    assert "charge-payment" in names
    assert "ops-alerts" in names
    assert "lambda" in types
    assert "sns_topic" in types


def test_structured_finds_named_resource(seeded_collection):
    hits = retrieve_structured(
        "use the charge-payment function",
        collection=seeded_collection,
        repo_id="demo",
    )
    ids = {h.id for h in hits}
    assert any("charge-payment" in i for i in ids)
    assert all(h.exact_score > 0 for h in hits)
    assert all("structured" in h.sources for h in hits)


# ---------------------------------------------------------------------------
# unit: merge + rerank + format
# ---------------------------------------------------------------------------


def test_merge_combines_score_channels():
    dense = [
        ScoredHit(
            id="a",
            text="t",
            metadata={"path": "p"},
            dense_score=0.8,
            sources=["dense"],
        )
    ]
    structured = [
        ScoredHit(
            id="a",
            text="t",
            metadata={"path": "p"},
            exact_score=0.9,
            sources=["structured"],
        ),
        ScoredHit(
            id="b",
            text="u",
            metadata={"path": "q"},
            exact_score=0.5,
            sources=["structured"],
        ),
    ]
    merged = {h.id: h for h in merge_hits(dense, structured)}
    assert set(merged) == {"a", "b"}
    assert merged["a"].dense_score == 0.8
    assert merged["a"].exact_score == 0.9
    assert set(merged["a"].sources) == {"dense", "structured"}


def test_rerank_prefers_multi_source_and_active_jobs():
    hits = [
        ScoredHit(
            id="job",
            text="job body",
            metadata={"chunk_kind": "job_definition", "status": "active", "path": "j"},
            dense_score=0.5,
            exact_score=0.8,
            sources=["dense", "structured"],
        ),
        ScoredHit(
            id="doc",
            text="doc body",
            metadata={"chunk_kind": "doc_section", "status": "active", "path": "d"},
            dense_score=0.7,
            exact_score=0.0,
            sources=["dense"],
        ),
        ScoredHit(
            id="old",
            text="old body",
            metadata={"chunk_kind": "job_definition", "status": "deprecated", "path": "o"},
            dense_score=0.9,
            exact_score=0.9,
            sources=["dense", "structured"],
        ),
    ]
    ranked = rerank(hits, min_score=0.0, top_n=10)
    by_id = {h.id: h.final_score for h in ranked}
    assert by_id["job"] > by_id["old"]
    assert by_id["job"] > by_id["doc"]
    assert ranked[0].id == "job"


def test_format_respects_token_budget():
    hits = [
        ScoredHit(
            id=f"c{i}",
            text="word " * 50,
            metadata={"path": f"p{i}", "chunk_kind": "job_definition"},
            final_score=1.0 - i * 0.1,
            sources=["dense"],
        )
        for i in range(5)
    ]
    # Rough char/4 counter
    ctx, included, truncated = format_context(
        hits, token_budget=80, count_tokens=lambda t: max(1, len(t) // 4)
    )
    assert included
    assert len(included) < len(hits) or not truncated
    assert "Repository context" in ctx


# ---------------------------------------------------------------------------
# integration: full retrieve()
# ---------------------------------------------------------------------------


def test_retrieve_disabled_makes_no_collection_calls(gateway_env):
    result = retrieve("charge-payment workflow", enabled=False)
    assert result.disabled is True
    assert result.hits == []
    assert result.context == ""


def test_retrieve_context_empty_when_disabled(gateway_env):
    assert retrieve_context("anything", enabled=False) == ""


def test_retrieve_end_to_end(seeded_collection, fake_embedder, gateway_env, monkeypatch):
    monkeypatch.setenv("RETRIEVAL_ENABLED", "true")
    from orchestrator.config import get_settings

    get_settings.cache_clear()

    result = retrieve(
        "Build a workflow that runs charge-payment then notifies ops-alerts",
        repo_id="demo",
        collection=seeded_collection,
        embed=fake_embedder,
        enabled=True,
        min_score=0.05,
        dense_top_k=5,
        rerank_top_n=5,
        token_budget=4000,
    )
    assert result.disabled is False
    assert result.dense_count >= 1
    # structured should catch charge-payment / ops-alerts by name
    assert result.structured_count >= 1
    assert result.hits
    assert "Repository context" in result.context
    names = " ".join(
        (h.resource_name or "") + " " + h.text for h in result.hits
    ).lower()
    assert "charge-payment" in names or "ops-alerts" in names


def test_retrieve_context_string(seeded_collection, fake_embedder):
    ctx = retrieve_context(
        "validate-order lambda",
        repo_id="demo",
        collection=seeded_collection,
        embed=fake_embedder,
        enabled=True,
        min_score=0.05,
    )
    assert isinstance(ctx, str)
    assert "validate-order" in ctx.lower() or "Repository context" in ctx
