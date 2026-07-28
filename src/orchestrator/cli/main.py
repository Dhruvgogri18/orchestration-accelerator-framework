"""CLI entrypoint.

Commands:
    orchestrator index    -- build/refresh the repo-context index (deferred, see docs/phase1_architecture.md)
    orchestrator generate -- run the extract -> plan -> spec -> ASL -> validate -> review loop
    orchestrator deploy   -- deploy a previously saved workflow bundle

Story 0.1: this wires up a real, runnable CLI. The command bodies are
intentional stubs -- each one exists, accepts the right arguments, and
tells you clearly it isn't implemented yet, rather than doing nothing
or pretending to work. The actual logic behind each command lands with
Epics 1-8.
"""
from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(
    name="orchestrator",
    help="Turn a plain-English workflow description into a deployed AWS Step Function.",
    no_args_is_help=True,
)
console = Console()


def _not_implemented(command: str, epic_hint: str) -> None:
    """Shared "honest stub" behavior: clear message, non-zero exit, no pretending."""
    console.print(f"[yellow]orchestrator {command}[/yellow] is not implemented yet.")
    console.print(f"[dim]See {epic_hint} in the backlog.[/dim]")
    raise typer.Exit(code=1)


@app.command()
def index() -> None:
    """Build/refresh the repo-context index.

    Deferred: docs/phase1_architecture.md scopes Phase 1 with no
    vector-store/repo-context dependency. This command is kept as a
    placeholder so the CLI shape matches the README until the team
    decides whether repo-context indexing is in or out of scope.
    """
    _not_implemented("index", "deferred pending repo-context scope decision")


@app.command()
def generate(
    prompt: str = typer.Argument(
        None, help="Plain-English workflow description. Omit to start an interactive session."
    ),
) -> None:
    """Run the extract -> plan -> spec -> ASL -> validate -> review loop."""
    _not_implemented("generate", "Epics 1-6")


@app.command()
def deploy(
    bundle_path: str = typer.Argument(
        ..., help="Path to a saved workflow bundle (from a previous 'generate --save-only')."
    ),
) -> None:
    """Deploy a previously saved workflow bundle to AWS Step Functions."""
    _not_implemented("deploy", "Epics 7-8")


@app.callback()
def main(
    version: bool = typer.Option(False, "--version", help="Show the version and exit."),
) -> None:
    """Orchestration Accelerator CLI."""
    if version:
        from importlib.metadata import PackageNotFoundError, version as pkg_version

        try:
            console.print(pkg_version("orchestration-accelerator"))
        except PackageNotFoundError:
            console.print("0.1.0 (not installed as a package)")
        raise typer.Exit()


if __name__ == "__main__":
    app()