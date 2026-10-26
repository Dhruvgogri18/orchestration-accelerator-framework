"""CLI entrypoint.

Commands:
    orchestrator index    -- build/refresh the repo-context index (Part B: chunk -> embed -> Chroma)
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
def index(
    repo_path: str = typer.Argument(
        None, help="Repository folder to index. Defaults to REPO_PATH from .env."
    ),
    repo_id: str = typer.Option(
        None, "--repo-id", help="Name to store the repo under (default: the folder name)."
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Delete this repo's chunks first and re-embed everything."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would change; embeds and writes nothing."
    ),
) -> None:
    """Build or refresh the repo-context index (chunk -> embed -> Chroma).

    Incremental: unchanged chunks are skipped, so re-running costs nothing, and
    an interrupted run is finished by running it again.
    """
    from pydantic import ValidationError
    from rich.table import Table

    from orchestrator.indexing.pipeline import index_repository
    from orchestrator.indexing.vectorstore import IndexMismatchError

    try:
        report = index_repository(repo_path, repo_id=repo_id, rebuild=rebuild, dry_run=dry_run)
    except ValidationError as exc:
        names = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        console.print(f"[red]Invalid or missing configuration for:[/red] {', '.join(names)}")
        raise typer.Exit(code=1) from None
    except (FileNotFoundError, ValueError, IndexMismatchError, RuntimeError) as exc:
        console.print(f"[red]Indexing failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    verb = "Would" if dry_run else "Did"
    table = Table(
        title=f"Index {'dry run' if dry_run else 'run'}: {report.repo_id}", show_header=False
    )
    for label, value in (
        ("Repository", report.repo_path),
        ("Files loaded", report.files_loaded),
        ("Files skipped", len(report.files_skipped)),
        ("Chunks", report.chunks_total),
        (f"{verb} add", report.added),
        (f"{verb} update", report.updated),
        ("Unchanged", report.unchanged),
        (f"{verb} delete", report.deleted),
        ("Embedded", report.embedded),
        ("Chunks with redacted secrets", report.redacted_chunks),
        ("Seconds", report.duration_s),
    ):
        table.add_row(label, str(value))
    console.print(table)
    for skipped in report.files_skipped[:10]:
        console.print(f"[dim]skipped {skipped.path}: {skipped.reason}[/dim]")
    if dry_run:
        console.print("[yellow]Dry run: nothing was embedded or written.[/yellow]")
    try:
        from orchestrator.config import get_settings

        if not get_settings().retrieval_enabled:
            console.print(
                "[dim]RETRIEVAL_ENABLED is false: the index is built, but the workflow "
                "pipeline does not use it yet.[/dim]"
            )
    except Exception:  # noqa: BLE001 - the hint is optional
        pass


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
