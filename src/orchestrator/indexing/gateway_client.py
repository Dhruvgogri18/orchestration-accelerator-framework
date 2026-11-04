"""Single point of contact with the internal LLM gateway.

Everything else in the app should call chat_completion() / embed_texts()
from here rather than constructing its own OpenAI client, so auth,
retries, and model names live in exactly one place.

Configuration comes from orchestrator.config.get_settings() (which reads
.env). load_dotenv() is still called below because other modules (e.g.
api.services.llm_trace) read OAF_* variables straight from os.environ.
"""
from __future__ import annotations

import time
from functools import lru_cache

import openai
from dotenv import load_dotenv
# from langsmith import traceable
from openai import OpenAI
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from orchestrator.config import Settings, get_settings

load_dotenv()

try:
    from orchestrator.api.services.llm_trace import (
        count_tokens,
        log_event,
        log_llm_call,
        log_llm_response,
    )
except ImportError:  # pragma: no cover
    def log_llm_call(**kwargs):
        return time.perf_counter()

    def log_llm_response(**kwargs):
        return None

    def log_event(event, **fields):
        return None

    def count_tokens(text):
        return 0


# Sentinel so the default seed is resolved from settings at call time
# (not frozen at import). Pass seed=None to omit the parameter entirely.
_UNSET: object = object()


def _settings() -> Settings:
    """Load settings, turning validation errors into a readable RuntimeError.

    Only field names are reported -- never values -- so keys can't leak
    into logs or tracebacks.
    """
    try:
        return get_settings()
    except ValidationError as exc:
        fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
        raise RuntimeError(
            "Invalid or missing gateway configuration for: "
            f"{', '.join(fields)}. Copy .env.example to .env and fill it in."
        ) from None


@lru_cache
def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client pointed at the gateway."""
    s = _settings()
    return OpenAI(api_key=s.gateway_api_key.get_secret_value(), base_url=s.gateway_base_url)


# @traceable(name="chat_completion", run_type="llm")
@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4))
def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0,
    seed: int | None | object = _UNSET,
    **kwargs,
) -> str:
    """Send a chat completion request to the gateway and return the text.

    `messages` follows the standard OpenAI chat format:
        [{"role": "system"|"user"|"assistant", "content": "..."}]

    Determinism knobs:
      - temperature defaults to 0
      - seed defaults to GATEWAY_SEED (settings; 42 if unset)
      - pass seed=None to omit the seed field (if the gateway rejects it)

    Extra kwargs (max_tokens, etc.) are passed through to
    `chat.completions.create`. `model` always comes from GATEWAY_CHAT_MODEL.

    Retries up to 4 times with exponential backoff on transient failures
    (network errors, rate limits, etc.) -- see tenacity docs if you need
    to narrow this to specific exception types.

    LangSmith: when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set,
    each call is logged as a run (prompt in / completion out).
    """
    s = _settings()
    chat_model = s.gateway_chat_model
    client = get_client()
    if seed is _UNSET:
        seed = s.gateway_seed

    create_kwargs: dict = {
        "model": chat_model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if seed is not None:
        create_kwargs["seed"] = int(seed)

    t0 = log_llm_call(
        messages=messages,
        model=chat_model,
        temperature=temperature,
        seed=seed,
        extra={k: v for k, v in kwargs.items() if k != "messages"},
    )
    try:
        from orchestrator.observability.otel import trace_llm_call
    except Exception:  # noqa: BLE001
        from contextlib import nullcontext as trace_llm_call  # type: ignore
    try:
        with trace_llm_call(model=chat_model, stage="chat_completion"):
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content or ""
        log_llm_response(t0=t0, content=content, model=chat_model)
        return content
    except Exception as exc:
        log_llm_response(t0=t0, content=None, model=chat_model, error=str(exc))
        raise


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

# Only transient failures are retried. A 400 (e.g. input too long) is a
# caller error and will not get better by retrying.
_TRANSIENT_ERRORS = (
    openai.APIConnectionError,  # includes APITimeoutError
    openai.RateLimitError,
    openai.InternalServerError,
)


@retry(
    retry=retry_if_exception_type(_TRANSIENT_ERRORS),
    wait=wait_exponential(multiplier=1, min=1, max=20),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _embed_batch(client: OpenAI, model: str, batch: list[str], dimensions: int | None):
    kwargs: dict = {"model": model, "input": batch}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    return client.embeddings.create(**kwargs)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed texts via the gateway's embeddings endpoint.

    Model: GATEWAY_EMBEDDING_MODEL (e.g. bedrock/amazon.titan-embed-text-v2:0).

    - Returns one vector per input, in the same order as `texts`.
    - Sends batches of EMBEDDING_BATCH_SIZE (default 16).
    - Sends `dimensions` only when EMBEDDING_DIMENSIONS is set; if it is set
      and the returned vectors have a different size, raises rather than
      letting mixed-size vectors reach the vector store.
    - Empty / whitespace-only texts raise ValueError (the provider rejects
      them) -- callers should drop empty chunks before embedding.
    - Transient errors (connection, rate limit, 5xx) are retried with
      backoff; client errors (e.g. oversized input) are raised immediately.
    - Logged via llm_trace.log_event as `embedding_call` /
      `embedding_response`: counts, token estimate, duration -- never the
      text itself (chunks may contain repository content).
    """
    if not texts:
        return []

    s = _settings()
    model = s.gateway_embedding_model
    if not model:
        raise RuntimeError(
            "GATEWAY_EMBEDDING_MODEL is not set. Set it in .env "
            "(e.g. bedrock/amazon.titan-embed-text-v2:0)."
        )

    empty = [i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()]
    if empty:
        raise ValueError(
            f"embed_texts() got empty text at index(es) {empty[:10]}"
            f"{'...' if len(empty) > 10 else ''}; drop empty chunks before embedding."
        )

    client = get_client()
    vectors: list[list[float]] = []
    size = s.embedding_batch_size

    for start in range(0, len(texts), size):
        batch = texts[start : start + size]
        t0 = time.perf_counter()
        log_event(
            "embedding_call",
            model=model,
            batch_start=start,
            batch_size=len(batch),
            total_inputs=len(texts),
            approx_tokens=sum(count_tokens(t) for t in batch),
            dimensions=s.embedding_dimensions,
        )
        try:
            response = _embed_batch(client, model, batch, s.embedding_dimensions)
        except Exception as exc:
            log_event(
                "embedding_response",
                model=model,
                batch_start=start,
                duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise

        data = sorted(response.data, key=lambda item: item.index)
        if len(data) != len(batch):
            raise RuntimeError(
                f"Embeddings response returned {len(data)} vectors for "
                f"{len(batch)} inputs (batch starting at {start})."
            )
        batch_vectors = [list(item.embedding) for item in data]
        if s.embedding_dimensions is not None:
            bad = {len(v) for v in batch_vectors if len(v) != s.embedding_dimensions}
            if bad:
                raise RuntimeError(
                    f"Requested EMBEDDING_DIMENSIONS={s.embedding_dimensions} but the "
                    f"gateway returned vectors of size {sorted(bad)}; the gateway may "
                    "be ignoring the `dimensions` parameter."
                )
        vectors.extend(batch_vectors)
        log_event(
            "embedding_response",
            model=model,
            batch_start=start,
            vectors=len(batch_vectors),
            vector_size=len(batch_vectors[0]) if batch_vectors else None,
            duration_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    return vectors