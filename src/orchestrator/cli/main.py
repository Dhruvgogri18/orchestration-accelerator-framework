"""CLI entrypoint.

Commands:
    orchestrator index    -- build/refresh the repo-context index (Part B: chunk -> embed -> pgvector)
    orchestrator generate -- NL + repo context -> plan -> ASL -> validate (review, no deploy)
    orchestrator deploy   -- deploy a previously saved workflow bundle (requires --approve)
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="orchestrator",
    help="Turn a plain-English workflow description into a deployed AWS Step Function.",
    no_args_is_help=True,
)
console = Console()


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
    """Build or refresh the repo-context index (chunk -> embed -> pgvector).

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
    save: Path = typer.Option(
        None,
        "--save",
        help="Write plan + ASL + validation bundle JSON to this path.",
    ),
    self_consistency: bool = typer.Option(
        False,
        "--self-consistency",
        help="Run a second independent extraction and require structural agreement.",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Disable paraphrase-stable plan cache for this request.",
    ),
    require_grounding: bool = typer.Option(
        False,
        "--require-grounding",
        help="Block when plan tokens are not found in the repo index.",
    ),
) -> None:
    """Repo-grounded plan + ASL generation with validation (no deploy).

    Deploy is intentionally separate: review the saved bundle, then run
    ``orchestrator deploy bundle.json --approve``.
    """
    text = (prompt or "").strip()
    if not text:
        text = typer.prompt("Describe the workflow").strip()
    if not text:
        console.print("[red]Empty request.[/red]")
        raise typer.Exit(code=1)

    # Enterprise path: LangGraph (RAG → tools → synthesize → ground → present)
    # then optional ASL. No prompt/plan cache. Domain-agnostic.
    from orchestrator.runtime import nl_to_workflow

    _ = (no_cache, self_consistency, require_grounding)  # retained for CLI compat

    result = nl_to_workflow(
        text,
        include_asl=True,
        deploy=False,
        approved=False,
    )
    plan_pkg = result.get("plan_package") or {}

    table = Table(title=f"Pipeline: {result.get('status')}", show_header=False)
    stages = result.get("stages") or plan_pkg.get("stages") or []
    table.add_row("Stages", " → ".join(stages) or "-")
    if result.get("message"):
        table.add_row("Message", str(result["message"]))
    if result.get("clarifying_question"):
        table.add_row("Clarify", str(result["clarifying_question"]))
    if result.get("error"):
        table.add_row("Error", str(result["error"]))
    if plan_pkg.get("grounding"):
        g = plan_pkg["grounding"]
        table.add_row(
            "Grounding",
            f"resolved={g.get('resolved_count', 0)} "
            f"unresolved={len(g.get('unresolved') or [])}",
        )
    if result.get("validation"):
        table.add_row(
            "Validation",
            "ok" if result["validation"].get("ok") else "FAILED",
        )
    if plan_pkg.get("repo_enabled") is not None:
        table.add_row("RAG", "on" if plan_pkg.get("repo_enabled") else "off")
    console.print(table)

    for w in (result.get("warnings") or [])[:8]:
        console.print(f"[dim]warning: {w}[/dim]")

    plan = result.get("plan")
    if plan:
        steps = plan.get("steps") or []
        console.print(
            f"[bold]Plan[/bold] ({len(steps)} steps): {plan.get('name') or '-'}"
        )
        for s in steps[:20]:
            if isinstance(s, dict):
                g = (s.get("grounding") or {}).get("resource_name") or ""
                console.print(
                    f"  • {s.get('id')}: {s.get('service') or '-'} "
                    f"{('← ' + g) if g else ''}"
                )

    if result.get("summary_markdown"):
        console.print("\n[bold]Summary[/bold]")
        console.print(str(result["summary_markdown"])[:3000])

    if save and result.get("status") in (
        "ready",
        "validation_failed",
        "needs_clarification",
    ):
        bundle = {
            "prompt": text,
            "status": result.get("status"),
            "plan": result.get("plan"),
            "asl": result.get("asl"),
            "validation": result.get("validation"),
            "grounding": plan_pkg.get("grounding"),
            "summary_markdown": result.get("summary_markdown"),
            "diagram_dot": result.get("diagram_dot"),
            "placeholders": result.get("placeholders"),
            "warnings": result.get("warnings"),
            "message": result.get("message"),
            "stages": stages,
            "tool_trace": plan_pkg.get("tool_trace"),
        }
        save.parent.mkdir(parents=True, exist_ok=True)
        save.write_text(
            json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        console.print(f"[green]Saved bundle → {save}[/green]")

    status = result.get("status")
    if status in ("error", "validation_failed", "blocked"):
        raise typer.Exit(code=1)
    if status == "needs_clarification":
        raise typer.Exit(code=2)


@app.command()
def deploy(
    bundle_path: Path = typer.Argument(
        ...,
        help="Path to a saved workflow bundle JSON (from `generate --save`).",
        exists=True,
        readable=True,
    ),
    approve: bool = typer.Option(
        False,
        "--approve",
        help="Explicit human approval. Required — deploy refuses without it.",
    ),
    name: str = typer.Option(
        None,
        "--name",
        help="Override state machine name (otherwise uses plan name / ASL Comment).",
    ),
    role_arn: str = typer.Option(
        None,
        "--role-arn",
        help="Execution role ARN (defaults to STEP_FUNCTIONS_EXECUTION_ROLE_ARN).",
    ),
) -> None:
    """Deploy a previously saved, validated workflow bundle to AWS Step Functions.

    Human review gate: ``--approve`` is mandatory. Pre-deploy checks re-run
    validation status and reject any TODO_ placeholders.
    """
    try:
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not read bundle:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    asl = bundle.get("asl")
    if not isinstance(asl, dict) or not asl.get("States"):
        console.print("[red]Bundle has no ASL States — re-run generate.[/red]")
        raise typer.Exit(code=1)

    if not approve:
        console.print(
            "[yellow]Refusing deploy without --approve.[/yellow] "
            "Review the ASL in the bundle, then re-run with --approve."
        )
        raise typer.Exit(code=2)

    from orchestrator.deployment.stepfunctions import deploy_state_machine

    plan = bundle.get("plan") if isinstance(bundle.get("plan"), dict) else {}
    outcome = deploy_state_machine(
        asl,
        name=name or (plan.get("name") if plan else None),
        role_arn=role_arn,
        validation_report=bundle.get("validation")
        if isinstance(bundle.get("validation"), dict)
        else {"ok": True},
        approved=True,
    )

    if not outcome.ok:
        console.print("[red]Deploy blocked or failed[/red]")
        for r in outcome.blocked_reasons:
            console.print(f"  • {r}")
        if outcome.error:
            console.print(f"  error: {outcome.error}")
        raise typer.Exit(code=1)

    console.print(
        f"[green]{outcome.action}[/green] {outcome.state_machine_name} "
        f"→ {outcome.state_machine_arn}"
    )
    if outcome.region:
        console.print(f"region={outcome.region} account={outcome.account_id or '-'}")


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
