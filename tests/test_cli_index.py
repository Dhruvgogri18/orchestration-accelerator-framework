"""Part B: `orchestrator index`."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from orchestrator.cli.main import app
from orchestrator.indexing import gateway_client, vectorstore

SAMPLE = Path(__file__).resolve().parents[1] / "sample_repos" / "retail-order-platform"
runner = CliRunner()


@pytest.fixture
def cli_env(gateway_env, use_fake_embeddings, tmp_path, monkeypatch):
    """Dummy gateway settings, fake embedder, and a throwaway Chroma dir."""
    store = tmp_path / "chroma"
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(store))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "cli_test_index")
    from orchestrator.config import get_settings

    get_settings.cache_clear()
    repo = tmp_path / "demo-repo"
    shutil.copytree(SAMPLE, repo)
    calls: list[int] = []
    inner = gateway_client.embed_texts

    def counting(texts):
        calls.append(len(texts))
        return inner(texts)

    monkeypatch.setattr(gateway_client, "embed_texts", counting)
    return store, repo, calls


def stored_count(store: Path) -> int:
    col = vectorstore.get_collection(
        client=vectorstore.get_client(store),
        name="cli_test_index",
        embedding_model="bedrock/amazon.titan-embed-text-v2:0",
    )
    return col.count()


def test_index_builds_the_index_and_a_second_run_embeds_nothing(cli_env):
    store, repo, calls = cli_env
    first = runner.invoke(app, ["index", str(repo)])
    assert first.exit_code == 0, first.stdout
    assert "demo-repo" in first.stdout and sum(calls) > 0
    n = stored_count(store)
    assert n > 40 and sum(calls) == n

    before = list(calls)
    second = runner.invoke(app, ["index", str(repo)])
    assert second.exit_code == 0 and calls == before and stored_count(store) == n


def test_dry_run_changes_nothing(cli_env):
    store, repo, calls = cli_env
    result = runner.invoke(app, ["index", str(repo), "--dry-run"])
    assert result.exit_code == 0 and "Dry run" in result.stdout
    assert calls == [] and not store.exists()


def test_rebuild_and_custom_repo_id(cli_env):
    store, repo, calls = cli_env
    runner.invoke(app, ["index", str(repo), "--repo-id", "team-a"])
    n = sum(calls)
    result = runner.invoke(app, ["index", str(repo), "--repo-id", "team-a", "--rebuild"])
    assert result.exit_code == 0 and sum(calls) == 2 * n and stored_count(store) == n


def test_a_missing_folder_is_a_clean_error_not_a_traceback(cli_env):
    _, repo, _ = cli_env
    result = runner.invoke(app, ["index", str(repo / "nope")])
    assert result.exit_code == 1 and "Indexing failed" in result.stdout


def test_missing_gateway_configuration_is_reported_by_field_name(tmp_path, monkeypatch):
    from orchestrator.config import get_settings
    from tests.conftest import clean_settings_env

    monkeypatch.chdir(tmp_path)  # no .env here
    clean_settings_env(monkeypatch)
    get_settings.cache_clear()
    result = runner.invoke(app, ["index", str(SAMPLE)])
    assert result.exit_code == 1
    assert "Invalid or missing configuration" in result.stdout or "Indexing failed" in result.stdout
