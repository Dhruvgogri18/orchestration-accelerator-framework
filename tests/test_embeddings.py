"""Story B-01/B-02: embed_texts() behaviour (mocked client, no network)
and the fake-embedder / Chroma fixtures used by later Part B tests."""
from types import SimpleNamespace

import httpx
import openai
import pytest

from orchestrator.config import get_settings
from orchestrator.indexing import gateway_client

DIM = 8


class FakeEmbeddingsAPI:
    def __init__(self, *, dim=DIM, shuffle=False, errors=None):
        self.calls = []
        self.dim = dim
        self.shuffle = shuffle
        self.errors = list(errors or [])

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.errors:
            raise self.errors.pop(0)
        inputs = kwargs["input"]
        dim = kwargs.get("dimensions", self.dim)
        # vector[0] encodes the input text's number so order can be checked
        data = [SimpleNamespace(index=i, embedding=[float(t.split("-")[-1])] + [0.0] * (dim - 1))
                for i, t in enumerate(inputs)]
        return SimpleNamespace(data=list(reversed(data)) if self.shuffle else data)


@pytest.fixture
def api(gateway_env, monkeypatch):
    fake = FakeEmbeddingsAPI()
    monkeypatch.setattr(gateway_client, "get_client",
                        lambda: SimpleNamespace(embeddings=fake))
    # no real sleeping between retries
    monkeypatch.setattr(gateway_client._embed_batch.retry, "sleep", lambda _s: None)
    return fake


def _http_error(cls, status):
    request = httpx.Request("POST", "http://gateway.test/v1/embeddings")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


def test_empty_list_makes_no_calls(api):
    assert gateway_client.embed_texts([]) == []
    assert api.calls == []


def test_batches_and_preserves_order(api, monkeypatch):
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "2")
    get_settings.cache_clear()
    out = gateway_client.embed_texts([f"text-{i}" for i in range(5)])
    assert [len(c["input"]) for c in api.calls] == [2, 2, 1]
    assert [v[0] for v in out] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(c["model"] == "bedrock/amazon.titan-embed-text-v2:0" for c in api.calls)


def test_reorders_out_of_order_response(api):
    api.shuffle = True
    out = gateway_client.embed_texts([f"text-{i}" for i in range(4)])
    assert [v[0] for v in out] == [0.0, 1.0, 2.0, 3.0]


def test_dimensions_param_only_sent_when_configured(api, monkeypatch):
    gateway_client.embed_texts(["text-1"])
    assert "dimensions" not in api.calls[-1]

    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    get_settings.cache_clear()
    out = gateway_client.embed_texts(["text-1"])
    assert api.calls[-1]["dimensions"] == 512
    assert len(out[0]) == 512


def test_gateway_ignoring_dimensions_is_detected(api, monkeypatch):
    api.create = lambda **kw: SimpleNamespace(
        data=[SimpleNamespace(index=i, embedding=[0.0] * 1024) for i, _ in enumerate(kw["input"])])
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "512")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="dimensions"):
        gateway_client.embed_texts(["text-1"])


def test_empty_text_is_rejected_before_any_call(api):
    with pytest.raises(ValueError, match="index"):
        gateway_client.embed_texts(["ok-1", "   ", "ok-2"])
    assert api.calls == []


def test_requires_embedding_model(api, monkeypatch):
    monkeypatch.setenv("GATEWAY_EMBEDDING_MODEL", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError, match="GATEWAY_EMBEDDING_MODEL"):
        gateway_client.embed_texts(["text-1"])


def test_transient_errors_are_retried(api):
    api.errors = [_http_error(openai.RateLimitError, 429), _http_error(openai.InternalServerError, 500)]
    out = gateway_client.embed_texts(["text-3"])
    assert out[0][0] == 3.0
    assert len(api.calls) == 3  # two failures + one success


def test_client_errors_are_not_retried(api):
    api.errors = [_http_error(openai.BadRequestError, 400)]
    with pytest.raises(openai.BadRequestError):
        gateway_client.embed_texts(["text-3"])
    assert len(api.calls) == 1


def test_embedding_text_is_never_logged(api, monkeypatch):
    events = []
    monkeypatch.setattr(gateway_client, "log_event", lambda event, **f: events.append((event, f)))
    gateway_client.embed_texts(["SECRET_REPO_CONTENT-1"])
    assert [e for e, _ in events] == ["embedding_call", "embedding_response"]
    assert "SECRET_REPO_CONTENT" not in repr(events)


# ---- fixtures used by later Part B stories -------------------------------

def test_fake_embedder_is_deterministic_unit_length_and_similarity_aware(fake_embedder):
    a, b, c = fake_embedder([
        "start glue job to transform csv", "run glue job transform csv data", "send birthday email",
    ])
    assert len(a) == fake_embedder.dim == 1024
    assert fake_embedder(["start glue job to transform csv"])[0] == a
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9
    dot = lambda x, y: sum(p * q for p, q in zip(x, y))  # noqa: E731
    assert dot(a, b) > dot(a, c)


def test_use_fake_embeddings_patches_gateway_client(use_fake_embeddings):
    assert gateway_client.embed_texts is use_fake_embeddings


def test_chroma_collection_fixture_ranks_with_fake_embedder(chroma_collection, fake_embedder):
    docs = {
        "d1": "glue job transforms csv from s3",
        "d2": "sns topic publishes failure alert",
        "d3": "dynamodb table stores leave requests",
    }
    chroma_collection.add(ids=list(docs), documents=list(docs.values()),
                          embeddings=fake_embedder(list(docs.values())))
    result = chroma_collection.query(query_embeddings=fake_embedder(["glue csv job"]), n_results=1)
    assert result["ids"][0][0] == "d1"