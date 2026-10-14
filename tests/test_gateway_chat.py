"""Regression: moving gateway_client from os.environ to Settings must not
change chat_completion() behaviour."""
from types import SimpleNamespace

import pytest

from orchestrator.indexing import gateway_client


class FakeChat:
    def __init__(self):
        self.calls = []
        self.completions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="hi"))])


@pytest.fixture
def chat(gateway_env, monkeypatch):
    fake = FakeChat()
    monkeypatch.setattr(gateway_client, "get_client", lambda: SimpleNamespace(chat=fake))
    return fake


def test_uses_configured_model_default_seed_and_zero_temperature(chat):
    assert gateway_client.chat_completion([{"role": "user", "content": "x"}]) == "hi"
    call = chat.calls[0]
    assert call["model"] == "test/chat-model"
    assert call["seed"] == 42
    assert call["temperature"] == 0


def test_seed_none_omits_seed_and_explicit_seed_is_used(chat):
    gateway_client.chat_completion([{"role": "user", "content": "x"}], seed=None)
    gateway_client.chat_completion([{"role": "user", "content": "x"}], seed=7)
    assert "seed" not in chat.calls[0]
    assert chat.calls[1]["seed"] == 7


def test_seed_comes_from_settings(chat, monkeypatch):
    from orchestrator.config import get_settings

    monkeypatch.setenv("GATEWAY_SEED", "99")
    get_settings.cache_clear()
    gateway_client.chat_completion([{"role": "user", "content": "x"}])
    assert chat.calls[0]["seed"] == 99