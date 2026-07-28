"""Story 0.1: confirms the CLI actually runs -- doesn't crash, and lists
the commands the README/pyproject.toml promise exist.
"""
from typer.testing import CliRunner

from orchestrator.cli.main import app

runner = CliRunner()


def test_help_runs_without_crashing():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "index" in result.stdout
    assert "generate" in result.stdout
    assert "deploy" in result.stdout


def test_generate_stub_exits_nonzero_not_silently():
    result = runner.invoke(app, ["generate", "do something"])
    assert result.exit_code == 1
    assert "not implemented yet" in result.stdout


def test_deploy_requires_bundle_path():
    result = runner.invoke(app, ["deploy"])
    assert result.exit_code == 2