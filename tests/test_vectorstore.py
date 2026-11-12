"""Part B: the Chroma-backed vector store (uses real Chroma, tiny hand-made vectors)."""

import pytest

from orchestrator.indexing import vectorstore as vs
from orchestrator.ingestion.models import SCHEMA_VERSION, Chunk

NAME = "test_repo_context"


def chunk(cid: str, text: str = "t", **meta) -> Chunk:
    return Chunk(id=cid, text=text, metadata={"repo_id": "r", "path": "p", **meta})


@pytest.fixture
def client(tmp_chroma_dir):
    return vs.get_client(tmp_chroma_dir)


@pytest.fixture
def col(client):
    return vs.get_collection(client=client, name=NAME, embedding_model="fake-embed")


def test_collection_is_cosine_and_records_model_and_schema(col):
    assert col.metadata["embedding_model"] == "fake-embed"
    assert col.metadata["schema_version"] == SCHEMA_VERSION
    vs.upsert_chunks(col, [chunk("a")], [[0.0, 1.0, 0.0]])
    hit = vs.query("q", collection=col, n_results=1, embed=lambda _t: [[1.0, 0.0, 0.0]])[0]
    assert hit.distance == pytest.approx(
        1.0
    )  # orthogonal vectors: cosine distance 1 (L2 would give 2)


def test_reopening_with_the_same_settings_returns_the_same_collection(client, col):
    vs.upsert_chunks(col, [chunk("a")], [[1.0, 0.0]])
    again = vs.get_collection(client=client, name=NAME, embedding_model="fake-embed")
    assert again.count() == 1


def test_a_different_embedding_model_is_refused(client, col):
    with pytest.raises(vs.IndexMismatchError, match="embedding_model"):
        vs.get_collection(client=client, name=NAME, embedding_model="another-model")


def test_a_different_schema_version_is_refused(client):
    client.create_collection(
        "old_schema_idx", metadata={"schema_version": SCHEMA_VERSION + 1, "embedding_model": "m"}
    )
    with pytest.raises(vs.IndexMismatchError, match="schema_version"):
        vs.get_collection(client=client, name="old_schema_idx", embedding_model="m")


def test_create_false_returns_none_when_missing(client):
    assert vs.get_collection(client=client, name="does_not_exist", create=False) is None


def test_creating_without_a_model_name_is_an_error(client):
    with pytest.raises(RuntimeError, match="GATEWAY_EMBEDDING_MODEL"):
        vs.get_collection(client=client, name=NAME)


def test_upsert_replaces_metadata_instead_of_merging_it(col):
    vs.upsert_chunks(col, [chunk("a", downstream=["x", "y"], environment="dev")], [[1.0, 0.0]])
    # The re-indexed chunk no longer has `downstream` (the chunker omits empty lists).
    vs.upsert_chunks(col, [chunk("a", environment="dev")], [[1.0, 0.0]])
    stored = col.get(ids=["a"])["metadatas"][0]
    assert "downstream" not in stored and stored["environment"] == "dev"
    assert stored["content_hash"] and stored["schema_version"] == SCHEMA_VERSION


def test_empty_inputs_are_noops_and_mismatched_lengths_raise(col):
    vs.upsert_chunks(col, [], [])
    vs.delete_ids(col, [])
    assert col.count() == 0
    with pytest.raises(ValueError, match="1 chunks but 0 embeddings"):
        vs.upsert_chunks(col, [chunk("a")], [])


def test_batched_writes_and_deletes(col):
    chunks = [chunk(f"c{i}") for i in range(7)]
    vs.upsert_chunks(col, chunks, [[1.0, float(i)] for i in range(7)], batch_size=3)
    assert col.count() == 7
    vs.delete_ids(col, [f"c{i}" for i in range(5)], batch_size=2)
    assert sorted(col.get()["ids"]) == ["c5", "c6"]


def test_existing_hashes_paginates(col, monkeypatch):
    monkeypatch.setattr(vs, "GET_PAGE_SIZE", 2)
    chunks = [chunk(f"c{i}", text=f"text {i}") for i in range(5)]
    vs.upsert_chunks(col, chunks, [[1.0, float(i)] for i in range(5)])
    other = Chunk(id="x", text="x", metadata={"repo_id": "other", "path": "p"})
    vs.upsert_chunks(col, [other], [[0.0, 1.0]])
    assert vs.existing_hashes(col, "r") == {c.id: c.content_hash for c in chunks}


def test_delete_repo_only_touches_that_repo(col):
    vs.upsert_chunks(
        col,
        [chunk("a"), Chunk(id="b", text="b", metadata={"repo_id": "other"})],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    vs.delete_repo(col, "r")
    assert col.get()["ids"] == ["b"]


@pytest.mark.parametrize(
    "filters, expected",
    [
        (None, None),
        ({}, None),
        ({"environment": "dev"}, {"environment": "dev"}),
        ({"resource_type": ["lambda", "glue"]}, {"resource_type": {"$in": ["lambda", "glue"]}}),
        ({"resource_names__contains": "x"}, {"resource_names": {"$contains": "x"}}),
        (
            {"environment": "dev", "status": "active"},
            {"$and": [{"environment": "dev"}, {"status": "active"}]},
        ),
    ],
)
def test_build_where(filters, expected):
    assert vs.build_where(filters) == expected


def test_query_filters_and_array_membership(col):
    vs.upsert_chunks(
        col,
        [
            chunk("dev-a", environment="dev", resource_names=["alpha", "beta"]),
            chunk("prod-a", environment="prod", resource_names=["alpha"]),
            chunk("dev-b", environment="dev", resource_names=["gamma"]),
        ],
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
    )
    embed = lambda _t: [[1.0, 0.0]]  # noqa: E731
    all_ids = [h.id for h in vs.query("q", collection=col, embed=embed)]
    assert all_ids[:2] == ["dev-a", "prod-a"]  # nearest first
    dev = vs.query("q", collection=col, embed=embed, filters={"environment": "dev"})
    assert [h.id for h in dev] == ["dev-a", "dev-b"]
    contains = vs.query(
        "q", collection=col, embed=embed, filters={"resource_names__contains": "alpha"}
    )
    assert {h.id for h in contains} == {"dev-a", "prod-a"}
    both = vs.query(
        "q",
        collection=col,
        embed=embed,
        filters={"environment": "prod", "resource_names__contains": "alpha"},
    )
    assert [h.id for h in both] == ["prod-a"]


def test_exact_lookup_by_metadata_needs_no_embedding(col):
    vs.upsert_chunks(
        col,
        [chunk("a", text="arn:aws:x", status="deprecated"), chunk("b", status="active")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = vs.get_by_metadata({"status": "deprecated"}, collection=col)
    assert [(h.id, h.text, h.distance) for h in hits] == [("a", "arn:aws:x", None)]
    native = vs.get_by_metadata({"status": {"$eq": "active"}}, collection=col)  # raw Chroma where
    assert [h.id for h in native] == ["b"]


def test_querying_an_empty_collection_returns_nothing(col):
    assert vs.query("q", collection=col, embed=lambda _t: [[1.0, 0.0]]) == []
