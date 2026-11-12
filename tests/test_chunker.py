"""Part B: structure-aware chunking, against the sample repo and edge cases."""

import shutil
from collections import Counter
from pathlib import Path

import pytest

from orchestrator.ingestion.chunker import Registry, chunk_file, chunk_files
from orchestrator.ingestion.loader import load_repository
from orchestrator.ingestion.models import SourceFile

SAMPLE = Path(__file__).resolve().parents[1] / "sample_repos" / "retail-order-platform"
INCLUDE = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
EXCLUDE = ["**/.venv/**", "**/node_modules/**", "**/.git/**"]
REPO = "retail-order-platform"


def approx(text: str) -> int:
    """Deterministic stand-in for the tokenizer so counts do not depend on tiktoken."""
    return -(-len(text) // 4) if text else 0


def sample_chunks(root: Path = SAMPLE, **kw):
    sources, _ = load_repository(root, INCLUDE, EXCLUDE)
    return chunk_files(
        sources, repo_id=REPO, max_tokens=800, overlap_tokens=120, count_tokens=approx, **kw
    )


@pytest.fixture(scope="module")
def chunks():
    return sample_chunks()


@pytest.fixture(scope="module")
def by_id(chunks):
    return {c.id: c for c in chunks}


def cid(path: str, unit: str) -> str:
    return f"{REPO}::{path}::{unit}"


def make(path: str, text: str, ftype: str) -> SourceFile:
    return SourceFile(path=path, text=text, file_type=ftype, file_hash="h")


# ---------------------------------------------------------------- the sample repo
def test_chunk_counts_by_kind(chunks):
    assert Counter(c.metadata["chunk_kind"] for c in chunks) == {
        "job_definition": 14,
        "lambda_handler": 11,
        "glue_script": 3,
        "pipeline": 3,
        "env_resources": 8,
        "config_section": 1,
        "doc_section": 14,
        "python_module": 1,
    }
    assert len(chunks) == 55


def test_every_chunk_fits_and_starts_with_its_header(chunks):
    for c in chunks:
        assert c.text.strip() and approx(c.text) <= 800, c.id
        first = c.text.split("\n", 1)[0]
        assert first.startswith(f"[{REPO}] {c.metadata['path']} | "), c.id


def test_ids_are_unique_and_deterministic(chunks):
    assert len({c.id for c in chunks}) == len(chunks)
    again = sample_chunks()
    assert [(c.id, c.content_hash) for c in chunks] == [(c.id, c.content_hash) for c in again]


def test_windows_line_endings_do_not_change_ids_or_hashes(tmp_path, chunks):
    copy = tmp_path / "repo"
    shutil.copytree(SAMPLE, copy)
    for p in copy.rglob("*"):
        if p.is_file():
            p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    crlf = sample_chunks(copy)
    assert [(c.id, c.content_hash) for c in crlf] == [(c.id, c.content_hash) for c in chunks]


def test_metadata_values_are_always_chroma_safe(chunks):
    for c in chunks:
        for key, value in c.metadata.items():
            if isinstance(value, list):
                assert value and all(isinstance(v, str) and v for v in value), (c.id, key)
                assert value == sorted(set(value)), (c.id, key)
            else:
                assert isinstance(value, str | int | float | bool) and value != "", (c.id, key)


def test_job_catalog_entry_is_one_self_contained_chunk(by_id):
    c = by_id[cid("config/jobs.yaml", "jobs:csv-transform-and-load")]
    m = c.metadata
    assert c.text.count("- name:") == 1 and "retail-ops-alerts-dev" in c.text
    assert "csv-transform-and-load" in c.text.split("\n", 1)[0]
    assert (m["chunk_kind"], m["resource_name"], m["resource_type"], m["status"]) == (
        "job_definition",
        "csv-transform-and-load",
        "glue",
        "active",
    )
    assert m["script_path"] == "glue_jobs/csv_transform_and_load.py"
    assert m["glue_args"] == ["dynamodb_table", "source_s3_uri"]
    assert m["reads"] == ["s3://retail-orders-raw-dev/uploads/"]
    assert m["writes"] == ["dynamodb:orders-dev"]
    assert m["downstream"] == ["retail-ops-alerts-dev"]
    assert (m["start_line"], m["end_line"]) == (5, 21)
    assert "csv-transform-and-load" not in m["mentions"]  # never mentions itself
    assert {"orders-dev", "retail-ops-alerts-dev"} <= set(m["mentions"])


def test_every_catalog_job_has_its_own_chunk(chunks):
    jobs = [c for c in chunks if c.metadata["chunk_kind"] == "job_definition"]
    assert len({c.metadata["resource_name"] for c in jobs}) == 14
    assert all(c.text.count("- name:") == 1 for c in jobs)


def test_environment_files_are_split_per_resource_and_tagged_by_environment(by_id, chunks):
    prod_alerts = by_id[cid("config/environments/prod.yaml", "sns_topics:ops_alerts")]
    assert prod_alerts.metadata["environment"] == "prod"
    assert prod_alerts.metadata["resource_name"] == "retail-ops-alerts-prod"
    assert "arn:aws:sns:us-east-1:210987654321:retail-ops-alerts-prod" in prod_alerts.text
    assert "Operational failure alerts" in prod_alerts.text

    env_chunks = {
        e: [c for c in chunks if c.metadata.get("environment") == e] for e in ("dev", "prod")
    }
    assert len(env_chunks["dev"]) == len(env_chunks["prod"]) == 4
    assert {c.metadata["unit"] for c in env_chunks["dev"]} == {
        c.metadata["unit"] for c in env_chunks["prod"]
    }
    assert not {c.id for c in env_chunks["dev"]} & {c.id for c in env_chunks["prod"]}


def test_lambda_arn_section_supports_exact_lookup_by_resource_name(chunks):
    hits = [
        c
        for c in chunks
        if c.metadata.get("environment") == "prod"
        and "retail-charge-payment" in c.metadata.get("resource_names", [])
    ]
    assert len(hits) == 1
    assert "arn:aws:lambda:us-east-1:210987654321:function:retail-charge-payment" in hits[0].text
    assert "lambda" in hits[0].metadata["resource_types"]


def test_pipelines_json_is_split_per_pipeline(chunks):
    pipes = [c for c in chunks if c.metadata["chunk_kind"] == "pipeline"]
    assert {c.metadata["resource_name"] for c in pipes} == {
        "order-fulfillment",
        "daily-batch-etl",
        "csv-ingest",
    }
    daily = next(c for c in pipes if c.metadata["resource_name"] == "daily-batch-etl")
    assert {"orders-etl-job", "inventory-etl-job", "retail-dq-check", "retail-load-curated"} <= set(
        daily.metadata["mentions"]
    )


def test_python_handlers_get_metadata_from_the_ast_and_the_catalog(by_id):
    m = by_id[cid("lambdas/charge_payment/handler.py", "file")].metadata
    assert m["chunk_kind"] == "lambda_handler" and m["resource_name"] == "retail-charge-payment"
    assert m["env_vars"] == ["PAYMENT_GATEWAY_SECRET_NAME"]
    assert m["aws_services"] == ["secretsmanager"]
    assert m["entry_point"] == "handler.lambda_handler"
    assert "lambda_handler" in m["functions"]

    m = by_id[cid("lambdas/release_inventory/handler.py", "file")].metadata
    assert m["env_vars"] == ["OPS_TOPIC_ARN", "RESERVATIONS_TABLE"]
    assert m["aws_services"] == ["dynamodb", "sns"]


def test_glue_scripts_expose_job_arguments_without_job_name(by_id):
    m = by_id[cid("glue_jobs/csv_transform_and_load.py", "file")].metadata
    assert m["chunk_kind"] == "glue_script" and m["resource_name"] == "csv-transform-and-load"
    assert m["glue_args"] == ["dynamodb_table", "source_s3_uri"]  # same naming as the catalog chunk
    assert m["entry_point"] == "main"


def test_deprecated_resources_are_flagged(by_id, chunks):
    assert (
        by_id[cid("lambdas/legacy_notifier/handler.py", "file")].metadata["status"] == "deprecated"
    )
    assert (
        by_id[cid("config/jobs.yaml", "jobs:retail-legacy-notifier")].metadata["status"]
        == "deprecated"
    )
    flagged = {c.metadata["path"] for c in chunks if c.metadata["mentions_deprecated"]}
    assert "docs/runbook.md" in flagged and "README.md" in flagged
    pipelines = next(
        c for c in chunks if c.metadata.get("heading_path", "").endswith("> Pipelines")
    )
    assert pipelines.metadata["mentions_deprecated"] is False


def test_docs_are_split_by_heading_with_a_breadcrumb(chunks):
    paths = {c.metadata["heading_path"] for c in chunks if c.metadata["path"] == "docs/runbook.md"}
    assert {"Runbook > Failed payment", "Runbook > Who gets alerted"} <= paths
    failed = next(c for c in chunks if c.metadata.get("heading_path") == "Runbook > Failed payment")
    assert (
        "retail-release-inventory" in failed.text
        and "Runbook > Failed payment" in failed.text.split("\n")[0]
    )


# ---------------------------------------------------------------- generic behaviour
def test_markdown_never_splits_inside_a_code_fence_and_merges_bare_headings():
    md = (
        "# Title\n\n## Setup\n\n"
        "Install the dependencies first, then start the local server like this:\n\n"
        "```bash\n# not a heading\n\n## still code\nmake run\n```\n\n"
        "## Next\n\n" + "words " * 40 + "\n"
    )
    out = chunk_file(
        make("a.md", md, "md"), repo_id="r", count_tokens=approx, max_tokens=800, overlap_tokens=0
    )
    paths = [c.metadata["heading_path"] for c in out]
    assert paths == ["Title > Setup", "Title > Next"]  # bare "# Title" merged into its first child
    setup = out[0].text
    assert "# not a heading" in setup and "## still code" in setup and "make run" in setup


def test_oversize_markdown_section_splits_on_paragraphs():
    body = "\n\n".join(f"Paragraph {i}: " + "alpha " * 60 for i in range(8))
    out = chunk_file(
        make("big.md", f"# T\n\n## Big\n\n{body}\n", "md"),
        repo_id="r",
        count_tokens=approx,
        max_tokens=200,
        overlap_tokens=20,
    )
    assert len(out) > 2
    assert all(approx(c.text) <= 200 for c in out)
    assert [c.metadata["part"] for c in out] == list(range(1, len(out) + 1))
    assert all("(part " in c.text.split("\n")[0] for c in out)
    assert len({c.id for c in out}) == len(out)


def test_large_python_file_splits_along_top_level_definitions():
    funcs = "\n\n".join(f"def f{i}(x):\n    return x + {i}  # " + "pad " * 25 for i in range(12))
    src = f'"""Module doc."""\nimport os\n\nimport boto3\n\n{funcs}\n'
    out = chunk_file(
        make("big.py", src, "py"),
        repo_id="r",
        count_tokens=approx,
        max_tokens=250,
        overlap_tokens=20,
    )
    assert len(out) > 1 and all(approx(c.text) <= 250 for c in out)
    assert all(
        "import boto3" in c.text and '"""Module doc."""' in c.text for c in out
    )  # preamble repeated
    owners = Counter(name for c in out for name in c.metadata["functions"])
    assert owners == {f"f{i}": 1 for i in range(12)}  # every function lives in exactly one chunk
    assert all(c.metadata["file_type"] == "py" for c in out)


def test_single_huge_function_and_huge_line_are_hard_split_within_the_limit():
    src = (
        "def big():\n    x = '"
        + "z" * 4000
        + "'\n"
        + "\n".join(f"    y{i} = {i}" for i in range(200))
        + "\n"
    )
    out = chunk_file(
        make("h.py", src, "py"), repo_id="r", count_tokens=approx, max_tokens=150, overlap_tokens=10
    )
    assert len(out) > 3 and all(approx(c.text) <= 150 for c in out)


def test_unparseable_files_fall_back_to_text_windows_without_raising():
    bad_py = make("bad.py", "def (:\n" + "x = 1\n" * 10, "py")
    bad_yaml = make("bad.yaml", "a: [1, 2\nb: : :\n", "yaml")
    bad_json = make("bad.json", '{"a": ', "json")
    multi = make("multi.yaml", "a: 1\n---\nb: 2\n", "yaml")
    for src, kind in (
        (bad_py, "python_module"),
        (bad_yaml, "yaml_text"),
        (bad_json, "json_text"),
        (multi, "yaml_text"),
    ):
        out = chunk_file(src, repo_id="r", count_tokens=approx)
        assert (
            out
            and out[0].metadata["chunk_kind"] == kind
            and out[0].metadata["unit"].startswith("window-")
        )


def test_cloudformation_style_tags_do_not_break_yaml_parsing():
    cfn = (
        "Resources:\n"
        "  Topic:\n    Type: AWS::SNS::Topic\n    Properties:\n"
        "      TopicName: !Sub ${Env}-alerts\n"
        "  Queue:\n    Type: AWS::SQS::Queue\n    Properties:\n"
        "      QueueName: !Ref QueueNameParam\n"
    )
    out = chunk_file(make("template.yaml", cfn, "yaml"), repo_id="r", count_tokens=approx)
    assert out and all(c.metadata["chunk_kind"] != "yaml_text" for c in out)
    assert {c.metadata["resource_name"] for c in out} == {"Topic", "Queue"}


def test_json_list_and_duplicate_names_get_unique_ids():
    src = (
        '[{"name": "a", "script": "x.py"}, {"name": "a", "script": "y.py"}, '
        '{"name": "b", "script": "z.py"}]'
    )
    out = chunk_file(make("jobs.json", src, "json"), repo_id="r", count_tokens=approx)
    assert len(out) == 3 and len({c.id for c in out}) == 3
    assert [c.metadata["resource_name"] for c in out] == ["a", "a", "b"]


def test_secrets_are_redacted_in_chunk_text_and_flagged():
    src = (
        "service:\n  name: billing\n  api_key: sk_live_abcdef1234567890abcd\n  region: us-east-1\n"
    )
    out = chunk_file(make("svc.yaml", src, "yaml"), repo_id="r", count_tokens=approx)
    assert out and all("sk_live_abcdef" not in c.text for c in out)
    assert any("api_key: <REDACTED>" in c.text and c.metadata["redacted"] is True for c in out)


def test_a_registry_is_optional_but_adds_catalog_links():
    sources, _ = load_repository(SAMPLE, INCLUDE, EXCLUDE)
    handler = next(s for s in sources if s.path == "lambdas/ship_order/handler.py")
    bare = chunk_file(handler, repo_id=REPO, count_tokens=approx)[0].metadata
    linked = chunk_file(
        handler, repo_id=REPO, registry=Registry.from_sources(sources), count_tokens=approx
    )[0].metadata
    assert "resource_name" not in bare and bare["chunk_kind"] == "python_module"
    assert (
        linked["resource_name"] == "retail-ship-order" and linked["chunk_kind"] == "lambda_handler"
    )
    assert bare["env_vars"] == linked["env_vars"] == ["ORDERS_TABLE"]  # AST facts need no catalog


def test_unknown_file_types_use_text_windows():
    out = chunk_file(make("notes.txt", "hello\n" * 5, "other"), repo_id="r", count_tokens=approx)
    assert len(out) == 1 and out[0].metadata["chunk_kind"] == "text_window"


def test_a_tiny_section_cascades_into_the_next_one_instead_of_standing_alone():
    md = "# Title\n\n## Tiny\n\nShort.\n\n## Real\n\n" + "content " * 30 + "\n"
    out = chunk_file(
        make("a.md", md, "md"), repo_id="r", count_tokens=approx, max_tokens=800, overlap_tokens=0
    )
    assert [c.metadata["heading_path"] for c in out] == ["Title > Real"]
    assert "Short." in out[0].text and "# Title" in out[0].text
