"""Part B: end-to-end indexing of the sample repo (pgvector, fake embedder)."""

import shutil
import uuid
from pathlib import Path

import pytest

from orchestrator.indexing import pipeline, vectorstore
from orchestrator.indexing.pipeline import default_repo_id, index_repository
from tests.conftest import FAKE_EMBEDDING_DIM

SAMPLE = Path(__file__).resolve().parents[1] / "src" / "sample_repos" / "retail-order-platform"
if not SAMPLE.exists():
    SAMPLE = Path(__file__).resolve().parents[1] / "sample_repos" / "retail-order-platform"
INCLUDE = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
EXCLUDE = ["**/.venv/**", "**/node_modules/**", "**/.git/**"]
MODEL = "fake-embed"
TOTAL = 55


def approx(text: str) -> int:
    return -(-len(text) // 4) if text else 0


class CountingEmbedder:
    def __init__(self, inner, fail_on_call: int | None = None):
        self.inner, self.fail_on_call, self.batches = inner, fail_on_call, []

    def __call__(self, texts):
        if self.fail_on_call is not None and len(self.batches) + 1 == self.fail_on_call:
            raise RuntimeError("gateway down")
        self.batches.append(len(texts))
        return self.inner(texts)

    @property
    def total(self) -> int:
        return sum(self.batches)


@pytest.fixture
def repo(tmp_path):
    dest = tmp_path / "retail-order-platform"
    shutil.copytree(SAMPLE, dest)
    return dest


@pytest.fixture
def embedder(fake_embedder):
    return CountingEmbedder(fake_embedder)


@pytest.fixture
def collection_name(pg_client):
    """Unique collection per test; dropped on teardown."""
    name = f"idx_{uuid.uuid4().hex[:12]}"
    yield name
    vectorstore.drop_collection(pg_client, name)


def run(repo, collection_name, embedder, **kw):
    kw.setdefault("embedding_model", MODEL)
    kw.setdefault("embedding_dim", FAKE_EMBEDDING_DIM)
    return index_repository(
        repo,
        collection_name=collection_name,
        include_globs=INCLUDE,
        exclude_globs=EXCLUDE,
        embed=embedder,
        count_tokens=approx,
        max_tokens=800,
        overlap_tokens=120,
        **kw,
    )


def open_collection(collection_name):
    return vectorstore.get_collection(
        client=vectorstore.get_client(),
        name=collection_name,
        embedding_model=MODEL,
        embedding_dim=FAKE_EMBEDDING_DIM,
    )


def cid(path, unit, repo_id="retail-order-platform"):
    return f"{repo_id}::{path}::{unit}"


def test_default_repo_id_is_a_slug(tmp_path):
    assert default_repo_id(tmp_path / "My Repo_v2!") == "my-repo_v2"


def test_first_run_embeds_and_stores_everything_once(repo, collection_name, embedder):
    r = run(repo, collection_name, embedder)
    assert (r.repo_id, r.files_loaded, r.chunks_total) == ("retail-order-platform", 23, TOTAL)
    assert (r.added, r.updated, r.unchanged, r.deleted, r.embedded) == (TOTAL, 0, 0, 0, TOTAL)
    assert embedder.total == TOTAL
    assert open_collection(collection_name).count() == TOTAL
    assert r.redacted_chunks == 0 and r.embedding_model == MODEL


def test_rerunning_an_unchanged_repo_makes_no_embedding_calls(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    before = list(embedder.batches)
    r = run(repo, collection_name, embedder)
    assert (r.added, r.updated, r.deleted, r.embedded, r.unchanged) == (0, 0, 0, 0, TOTAL)
    assert embedder.batches == before


def test_editing_one_file_reembeds_only_its_chunk(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    handler = repo / "lambdas" / "ship_order" / "handler.py"
    handler.write_text(handler.read_text().replace("ACME-FREIGHT", "GLOBEX-FREIGHT"))
    r = run(repo, collection_name, embedder)
    assert (r.added, r.updated, r.unchanged, r.embedded) == (0, 1, TOTAL - 1, 1)
    stored = open_collection(collection_name).get(
        ids=[cid("lambdas/ship_order/handler.py", "file")]
    )
    assert (
        "GLOBEX-FREIGHT" in stored["documents"][0] and "ACME-FREIGHT" not in stored["documents"][0]
    )


def test_a_removed_file_removes_its_chunks(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    (repo / "lambdas" / "dq_check" / "handler.py").unlink()
    r = run(repo, collection_name, embedder)
    assert (r.deleted, r.added, r.updated, r.embedded) == (1, 0, 0, 0)
    col = open_collection(collection_name)
    assert col.count() == TOTAL - 1
    assert col.get(ids=[cid("lambdas/dq_check/handler.py", "file")])["ids"] == []


def test_a_metadata_key_that_disappears_does_not_linger(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    unit = cid("config/jobs.yaml", "jobs:retail-validate-order")
    col = open_collection(collection_name)
    assert col.get(ids=[unit])["metadatas"][0]["downstream"] == [
        "retail-fraud-check",
        "retail-reserve-inventory",
    ]
    jobs = repo / "config" / "jobs.yaml"
    jobs.write_text(
        jobs.read_text().replace(
            "    downstream:\n      - retail-fraud-check\n      - retail-reserve-inventory\n"
            "    description: Validate an order's required fields and total.",
            "    description: Validate an order's required fields and total.",
        )
    )
    r = run(repo, collection_name, embedder)
    assert r.updated >= 1
    assert "downstream" not in col.get(ids=[unit])["metadatas"][0]


def test_rebuild_reembeds_everything(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    r = run(repo, collection_name, embedder, rebuild=True)
    assert (r.added, r.embedded, r.rebuilt) == (TOTAL, TOTAL, True)
    assert open_collection(collection_name).count() == TOTAL


def test_dry_run_reports_but_writes_nothing(repo, collection_name, embedder):
    r = run(repo, collection_name, embedder, dry_run=True)
    assert (r.dry_run, r.added, r.embedded, r.chunks_total) == (True, TOTAL, 0, TOTAL)
    assert embedder.batches == []
    # dry_run with create=False leaves no collection row
    assert (
        vectorstore.get_collection(
            client=vectorstore.get_client(), name=collection_name, create=False
        )
        is None
    )


def test_dry_run_against_an_existing_index_reports_the_diff(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    (repo / "README.md").write_text(
        (repo / "README.md").read_text() + "\n## Extra\n\n" + "more " * 30
    )
    before = list(embedder.batches)
    r = run(repo, collection_name, embedder, dry_run=True)
    assert r.added + r.updated >= 1 and r.embedded == 0 and embedder.batches == before
    assert open_collection(collection_name).count() == TOTAL


def test_an_empty_result_never_wipes_the_existing_index(repo, tmp_path, collection_name, embedder):
    run(repo, collection_name, embedder)
    empty = tmp_path / "retail-order-platform-empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No indexable files"):
        run(empty, collection_name, embedder, repo_id="retail-order-platform")
    assert open_collection(collection_name).count() == TOTAL


def test_two_repos_share_one_collection_without_interfering(repo, collection_name, embedder):
    run(repo, collection_name, embedder, repo_id="repo-a")
    run(repo, collection_name, embedder, repo_id="repo-b")
    col = open_collection(collection_name)
    assert col.count() == 2 * TOTAL
    (repo / "lambdas" / "dq_check" / "handler.py").unlink()
    run(repo, collection_name, embedder, repo_id="repo-a", rebuild=True)
    assert len(vectorstore.existing_hashes(col, "repo-a")) == TOTAL - 1
    assert len(vectorstore.existing_hashes(col, "repo-b")) == TOTAL  # untouched


def test_an_interrupted_run_resumes_without_redoing_finished_work(
    repo, collection_name, fake_embedder, monkeypatch
):
    monkeypatch.setattr(pipeline, "EMBED_GROUP_SIZE", 10)
    flaky = CountingEmbedder(fake_embedder, fail_on_call=3)
    with pytest.raises(RuntimeError, match="gateway down"):
        run(repo, collection_name, flaky)
    assert open_collection(collection_name).count() == 20  # two groups were saved

    good = CountingEmbedder(fake_embedder)
    r = run(repo, collection_name, good)
    assert (r.unchanged, r.added, good.total) == (20, TOTAL - 20, TOTAL - 20)
    assert open_collection(collection_name).count() == TOTAL


def test_stale_chunks_survive_a_run_that_fails_before_the_new_ones_are_saved(
    repo, collection_name, embedder, fake_embedder
):
    run(repo, collection_name, embedder)
    stale_id = cid("lambdas/dq_check/handler.py", "file")
    (repo / "lambdas" / "dq_check" / "handler.py").unlink()  # its chunk is now stale
    (repo / "README.md").write_text(
        (repo / "README.md").read_text() + "\n## More\n\n" + "text " * 40
    )
    with pytest.raises(RuntimeError, match="gateway down"):
        run(repo, collection_name, CountingEmbedder(fake_embedder, fail_on_call=1))
    # Nothing was lost: the old chunk is only removed once the new ones are written.
    assert open_collection(collection_name).get(ids=[stale_id])["ids"] == [stale_id]
    r = run(repo, collection_name, embedder)  # a clean re-run finishes the job
    assert r.deleted == 1 and open_collection(collection_name).get(ids=[stale_id])["ids"] == []


def test_a_different_embedding_model_cannot_be_mixed_in(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    with pytest.raises(vectorstore.IndexMismatchError):
        run(repo, collection_name, embedder, embedding_model="some-other-model")


def test_a_misbehaving_embedder_is_caught(repo, collection_name):
    with pytest.raises(RuntimeError, match="vectors for"):
        run(repo, collection_name, lambda texts: [[1.0] * FAKE_EMBEDDING_DIM] * (len(texts) - 1))


# ---------------------------------------------------------------- what the index lets you ask
@pytest.fixture
def indexed(repo, collection_name, embedder):
    run(repo, collection_name, embedder)
    return open_collection(collection_name)


def test_exact_lookup_gives_the_prod_arn_of_a_function(indexed):
    hits = vectorstore.get_by_metadata(
        {"environment": "prod", "resource_names__contains": "retail-charge-payment"},
        collection=indexed,
    )
    assert len(hits) == 1
    assert "arn:aws:lambda:us-east-1:210987654321:function:retail-charge-payment" in hits[0].text


def test_deprecated_resources_can_be_filtered_out_or_found(indexed):
    found = vectorstore.get_by_metadata({"status": "deprecated"}, collection=indexed)
    assert {h.metadata["path"] for h in found} == {
        "config/jobs.yaml",
        "lambdas/legacy_notifier/handler.py",
    }


@pytest.mark.parametrize(
    "question, filters, resource_name",
    [
        ("Which Glue job loads a CSV from S3 into DynamoDB?", None, "csv-transform-and-load"),
        (
            "Which SNS topic receives operational failure alerts?",
            {"environment": "dev", "resource_type": "sns_topic"},
            "retail-ops-alerts-dev",
        ),
        (
            "Which Glue jobs run in parallel in the daily ETL pipeline?",
            None,
            "daily-batch-etl",
        ),
        (
            "Which lambda reserves inventory in DynamoDB?",
            {"chunk_kind": "lambda_handler"},
            "retail-reserve-inventory",
        ),
    ],
)
def test_retrieval_smoke_with_the_fake_embedder(
    indexed, fake_embedder, question, filters, resource_name
):
    """Wiring check only (the fake embedder is bag-of-words)."""
    hits = vectorstore.query(
        question, collection=indexed, n_results=5, filters=filters, embed=fake_embedder
    )
    assert hits, "nothing retrieved"
    assert any(h.metadata.get("resource_name") == resource_name for h in hits), [h.id for h in hits]
