"""Part B: the pgvector-backed vector store (needs live Postgres)."""

import pytest

from orchestrator.indexing import vectorstore as vs
from orchestrator.ingestion.models import SCHEMA_VERSION, Chunk
from tests.conftest import FAKE_EMBEDDING_DIM

NAME = "test_repo_context_vs"


def chunk(cid: str, text: str = "t", **meta) -> Chunk:
    return Chunk(id=cid, text=text, metadata={"repo_id": "r", "path": "p", **meta})


def _vec(i: float = 1.0) -> list[float]:
    """Unit-ish vector of FAKE_EMBEDDING_DIM with a distinctive first component."""
    v = [0.0] * FAKE_EMBEDDING_DIM
    v[0] = float(i)
    return v


@pytest.fixture
def client(pg_client):
    return pg_client


@pytest.fixture
def col(client):
    name = f"{NAME}_{id(client)}"
    vs.drop_collection(client, name)
    collection = vs.get_collection(
        client=client,
        name=name,
        embedding_model="fake-embed",
        embedding_dim=FAKE_EMBEDDING_DIM,
    )
    yield collection
    vs.drop_collection(client, name)


def test_collection_is_cosine_and_records_model_and_schema(col):
    assert col.metadata["embedding_model"] == "fake-embed"
    assert col.metadata["schema_version"] == str(SCHEMA_VERSION)
    a = _vec(0.0)
    a[1] = 1.0
    b = _vec(1.0)
    vs.upsert_chunks(col, [chunk("a")], [a])
    hit = vs.query("q", collection=col, n_results=1, embed=lambda _t: [b])[0]
    # orthogonal-ish first two dims: distance near 1 for cosine
    assert hit.distance == pytest.approx(1.0, abs=0.05)


def test_reopening_with_the_same_settings_returns_the_same_collection(client, col):
    vs.upsert_chunks(col, [chunk("a")], [_vec()])
    again = vs.get_collection(
        client=client, name=col.name, embedding_model="fake-embed", embedding_dim=FAKE_EMBEDDING_DIM
    )
    assert again.count() == 1


def test_a_different_embedding_model_is_refused(client, col):
    with pytest.raises(vs.IndexMismatchError, match="embedding_model"):
        vs.get_collection(
            client=client, name=col.name, embedding_model="another-model", embedding_dim=FAKE_EMBEDDING_DIM
        )


def test_a_different_schema_version_is_refused(client, monkeypatch):
    name = f"old_schema_{id(client)}"
    vs.drop_collection(client, name)
    with client._connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO repo_collections (name, schema_version, embedding_model, embedding_dim)
                VALUES (%s, %s, %s, %s)
                """,
                (name, str(SCHEMA_VERSION + 1), "m", FAKE_EMBEDDING_DIM),
            )
        conn.commit()
    try:
        with pytest.raises(vs.IndexMismatchError, match="schema_version"):
            vs.get_collection(client=client, name=name, embedding_model="m", embedding_dim=FAKE_EMBEDDING_DIM)
    finally:
        vs.drop_collection(client, name)


def test_create_false_returns_none_when_missing(client):
    assert vs.get_collection(client=client, name="does_not_exist_xyz", create=False) is None


def test_creating_without_a_model_name_is_an_error(client):
    with pytest.raises(RuntimeError, match="GATEWAY_EMBEDDING_MODEL"):
        vs.get_collection(client=client, name=f"no_model_{id(client)}")


def test_upsert_replaces_metadata_instead_of_merging_it(col):
    vs.upsert_chunks(col, [chunk("a", downstream=["x", "y"], environment="dev")], [_vec()])
    vs.upsert_chunks(col, [chunk("a", environment="dev")], [_vec()])
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
    vs.upsert_chunks(col, chunks, [_vec(float(i)) for i in range(7)], batch_size=3)
    assert col.count() == 7
    vs.delete_ids(col, [f"c{i}" for i in range(5)], batch_size=2)
    assert sorted(col.get()["ids"]) == ["c5", "c6"]


def test_existing_hashes_paginates(col, monkeypatch):
    monkeypatch.setattr(vs, "GET_PAGE_SIZE", 2)
    chunks = [chunk(f"c{i}", text=f"text {i}") for i in range(5)]
    vs.upsert_chunks(col, chunks, [_vec(float(i)) for i in range(5)])
    other = Chunk(id="x", text="x", metadata={"repo_id": "other", "path": "p"})
    vs.upsert_chunks(col, [other], [_vec(9.0)])
    assert vs.existing_hashes(col, "r") == {c.id: c.content_hash for c in chunks}


def test_delete_repo_only_touches_that_repo(col):
    vs.upsert_chunks(
        col,
        [chunk("a"), Chunk(id="b", text="b", metadata={"repo_id": "other"})],
        [_vec(1.0), _vec(0.0)],
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
        [_vec(1.0), _vec(0.9), _vec(0.0)],
    )
    embed = lambda _t: [_vec(1.0)]  # noqa: E731
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
        [_vec(1.0), _vec(0.0)],
    )
    hits = vs.get_by_metadata({"status": "deprecated"}, collection=col)
    assert [(h.id, h.text, h.distance) for h in hits] == [("a", "arn:aws:x", None)]
    native = vs.get_by_metadata({"status": {"$eq": "active"}}, collection=col)
    assert [h.id for h in native] == ["b"]


def test_querying_an_empty_collection_returns_nothing(col):
    assert vs.query("q", collection=col, embed=lambda _t: [_vec()]) == []
