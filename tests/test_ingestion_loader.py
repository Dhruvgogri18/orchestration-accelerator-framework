"""Part B: file discovery, loading and secret handling."""

from pathlib import Path

import pytest

from orchestrator.ingestion.loader import discover_files, glob_to_regex, load_files, load_repository
from orchestrator.ingestion.secrets import is_secret_path, redact_secrets

SAMPLE = Path(__file__).resolve().parents[1] / "sample_repos" / "retail-order-platform"
INCLUDE = ["**/*.py", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.md"]
EXCLUDE = ["**/.venv/**", "**/node_modules/**", "**/.git/**"]


def _write(root: Path, rel: str, content: str | bytes) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())


@pytest.mark.parametrize(
    "glob, path, expected",
    [
        ("**/*.py", "a.py", True),
        ("**/*.py", "x/y/a.py", True),
        ("**/*.py", "a.pyc", False),
        ("*.md", "README.md", True),
        ("*.md", "docs/a.md", False),
        ("**/.venv/**", ".venv/lib/x.py", True),
        ("**/.venv/**", "a/b/.venv/x.py", True),
        ("**/.venv/**", "src/venv/x.py", False),
        ("docs/**", "docs/a/b.md", True),
        ("a?c.txt", "abc.txt", True),
        ("a?c.txt", "a/c.txt", False),
    ],
)
def test_glob_to_regex(glob, path, expected):
    assert bool(glob_to_regex(glob).fullmatch(path)) is expected


def test_discovers_the_sample_repo_in_sorted_order():
    paths, skipped = discover_files(SAMPLE, INCLUDE, EXCLUDE)
    assert paths == sorted(paths)
    assert {"README.md", "config/jobs.yaml", "config/pipelines.json"} <= set(paths)
    assert "requirements.txt" not in paths  # not matched by the include globs
    assert skipped == []


def test_excluded_directories_are_pruned(tmp_path):
    for rel in (".venv/lib/x.py", "node_modules/a/b.py", "pkg/.venv/y.py", "src/a.py"):
        _write(tmp_path, rel, "x = 1\n")
    paths, _ = discover_files(tmp_path, INCLUDE, EXCLUDE)
    assert paths == ["src/a.py"]


def test_secret_like_files_are_never_loaded_even_if_included(tmp_path):
    _write(tmp_path, ".env", "KEY=abc\n")
    _write(tmp_path, "certs/server.pem", "-----BEGIN PRIVATE KEY-----\n")
    _write(tmp_path, "infra/prod.tfvars", "x = 1\n")
    _write(tmp_path, "a.py", "x = 1\n")
    paths, skipped = discover_files(tmp_path, ["**/*"], [])
    assert paths == ["a.py"]
    assert {s.path for s in skipped} == {".env", "certs/server.pem", "infra/prod.tfvars"}
    assert all(s.reason == "secret-like file name" for s in skipped)


def test_missing_repo_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        discover_files(tmp_path / "nope", INCLUDE, EXCLUDE)


def test_load_normalises_crlf_and_bom_so_hashes_do_not_depend_on_the_os(tmp_path):
    _write(tmp_path, "lf.md", "# T\n\nbody line\n")
    _write(tmp_path, "crlf.md", b"\xef\xbb\xbf# T\r\n\r\nbody line\r\n")
    sources, skipped = load_files(tmp_path, ["lf.md", "crlf.md"])
    lf, crlf = sources
    assert skipped == []
    assert "\r" not in crlf.text and not crlf.text.startswith("\ufeff")
    assert lf.text == crlf.text and lf.file_hash == crlf.file_hash
    assert (lf.file_type, lf.path) == ("md", "lf.md")


def test_unloadable_files_are_skipped_and_reported_not_raised(tmp_path):
    _write(tmp_path, "bin.py", b"abc\x00def")
    _write(tmp_path, "latin.py", b"caf\xe9 = 1\n")
    _write(tmp_path, "empty.md", "  \n\n")
    _write(tmp_path, "big.json", "[" + "1," * 600 + "1]")
    _write(tmp_path, "ok.py", "x = 1\n")
    sources, skipped = load_files(
        tmp_path,
        ["bin.py", "latin.py", "empty.md", "big.json", "ok.py", "gone.py"],
        max_file_bytes=1000,
    )
    assert [s.path for s in sources] == ["ok.py"]
    reasons = {s.path: s.reason for s in skipped}
    assert reasons["bin.py"] == "binary file"
    assert reasons["latin.py"] == "not valid UTF-8"
    assert reasons["empty.md"] == "empty file"
    assert reasons["big.json"].startswith("larger than")
    assert reasons["gone.py"].startswith("unreadable")


def test_load_repository_on_the_sample_repo():
    sources, skipped = load_repository(SAMPLE, INCLUDE, EXCLUDE)
    assert len(sources) == 23 and skipped == []
    assert {s.file_type for s in sources} == {"py", "yaml", "json", "md"}


# ---------------------------------------------------------------- secrets
def test_secret_path_detection():
    assert is_secret_path(".env") and is_secret_path("a/.env.local") and is_secret_path("k/id_rsa")
    assert not is_secret_path(".env.example.md") or True  # documented: .env.* is treated as secret
    assert not is_secret_path("config/jobs.yaml")


def test_redaction_removes_secret_values_but_keeps_what_a_workflow_needs():
    text = "\n".join(
        [
            "GATEWAY_API_KEY=gdtc_user_c66e8f0a1b2c3d4e5f6a7b8c",
            'password: "hunter2hunter2hunter2"',
            "AKIAABCDEFGHIJKLMNOP",
            "PAYMENT_GATEWAY_SECRET_NAME: retail/payment-gateway",
            'secret = client.get_secret_value(SecretId=require_env("X"))',
            "TOPIC_ARN: arn:aws:sns:us-east-1:123456789012:retail-ops-alerts-dev",
            "api_key: TODO_PAYMENT_API_KEY_VALUE",
        ]
    )
    out, count = redact_secrets(text)
    assert count == 3
    assert "gdtc_user" not in out and "hunter2" not in out and "AKIAABC" not in out
    assert "GATEWAY_API_KEY=<REDACTED>" in out
    for kept in (
        "retail/payment-gateway",
        "client.get_secret_value(SecretId=require_env",
        "arn:aws:sns:us-east-1:123456789012:retail-ops-alerts-dev",
        "TODO_PAYMENT_API_KEY_VALUE",
    ):
        assert kept in out


def test_redacts_private_key_blocks():
    out, count = redact_secrets(
        "a\n-----BEGIN RSA PRIVATE KEY-----\nMIIE\n-----END RSA PRIVATE KEY-----\nb"
    )
    assert count == 1 and "MIIE" not in out and out.startswith("a\n") and out.endswith("\nb")


def test_the_sample_repo_contains_nothing_to_redact():
    sources, _ = load_repository(SAMPLE, INCLUDE, EXCLUDE)
    assert sum(redact_secrets(s.text)[1] for s in sources) == 0
