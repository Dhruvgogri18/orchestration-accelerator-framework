"""Story B-02: shared Settings must load against a realistic .env and
carry the Part B retrieval settings."""
import pytest

from orchestrator.api.config.settings import APISettings
from orchestrator.config import Settings, get_settings

from .conftest import clean_settings_env


def _write_env(tmp_path, monkeypatch, text: str):
    monkeypatch.chdir(tmp_path)
    clean_settings_env(monkeypatch)
    (tmp_path / ".env").write_text(text, encoding="utf-8")


REALISTIC_ENV = """\
GATEWAY_API_KEY=test-key
GATEWAY_BASE_URL=http://gateway.test/v1
GATEWAY_CHAT_MODEL=test/chat-model
GATEWAY_EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0
GATEWAY_SEED=42
ORCHESTRATION_API_TIMEOUT=120.0
AWS_VALIDATE_ASL=true
AWS_STATE_MACHINE_TYPE=STANDARD
AWS_STATE_MACHINE_ROLE_ARN=arn:aws:iam::123456789012:role/sfn-test
OAF_LLM_LOG=1
USE_LANGGRAPH=true
LANGSMITH_TRACING=false
"""


def test_settings_load_with_realistic_env(tmp_path, monkeypatch):
    _write_env(tmp_path, monkeypatch, REALISTIC_ENV)
    s = Settings()  # would raise 20 validation errors before B-02
    assert s.gateway_embedding_model == "bedrock/amazon.titan-embed-text-v2:0"
    # unknown keys (OAF_*, USE_LANGGRAPH, ...) are ignored, not rejected
    assert not hasattr(s, "oaf_llm_log")


def test_role_arn_accepts_api_style_name_and_is_optional(tmp_path, monkeypatch):
    _write_env(tmp_path, monkeypatch, REALISTIC_ENV)
    assert Settings().step_functions_execution_role_arn.endswith("role/sfn-test")

    _write_env(tmp_path, monkeypatch, REALISTIC_ENV.replace(
        "AWS_STATE_MACHINE_ROLE_ARN=arn:aws:iam::123456789012:role/sfn-test\n", ""))
    assert Settings().step_functions_execution_role_arn is None


def test_blank_optional_values_are_none(gateway_env, monkeypatch):
    monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL", "")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "")
    monkeypatch.setenv("AWS_PROFILE", "  ")
    get_settings.cache_clear()
    s = get_settings()
    assert s.gateway_embedding_model is None
    assert s.embedding_dimensions is None
    assert s.aws_profile is None


def test_retrieval_is_disabled_by_default(gateway_env):
    assert get_settings().retrieval_enabled is False
    assert APISettings().retrieval_enabled is False


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("false", False)])
def test_retrieval_flag_is_read_by_both_settings_classes(gateway_env, monkeypatch, value, expected):
    monkeypatch.setenv("RETRIEVAL_ENABLED", value)
    get_settings.cache_clear()
    assert get_settings().retrieval_enabled is expected
    assert APISettings().retrieval_enabled is expected


def test_embedding_defaults(gateway_env):
    s = get_settings()
    assert s.embedding_batch_size == 16
    assert s.embedding_dimensions is None


def test_missing_required_key_names_the_field_without_leaking_values(tmp_path, monkeypatch):
    _write_env(tmp_path, monkeypatch, "GATEWAY_BASE_URL=http://gateway.test/v1\n")
    get_settings.cache_clear()
    from orchestrator.indexing import gateway_client

    with pytest.raises(RuntimeError) as exc:
        gateway_client.get_client()
    message = str(exc.value)
    assert "GATEWAY_API_KEY" in message or "gateway_api_key" in message
    assert "http://gateway.test" not in message  # values are never echoed


def test_inline_comment_after_blank_value_is_not_a_value(tmp_path, monkeypatch):
    """python-dotenv parses `KEY=   # note` as the text "# note"."""
    _write_env(tmp_path, monkeypatch, REALISTIC_ENV.replace(
        "GATEWAY_EMBEDDING_MODEL=bedrock/amazon.titan-embed-text-v2:0",
        "GATEWAY_EMBEDDING_MODEL=           # TODO: set me"))
    assert Settings().gateway_embedding_model is None


def test_api_key_is_never_shown_in_repr(gateway_env):
    s = get_settings()
    assert "test-key" not in repr(s)
    assert "test-key" not in str(s)
    assert s.gateway_api_key.get_secret_value() == "test-key"