"""Single point of contact with the internal LLM gateway.

Everything else in the app should call chat_completion() / embed_texts()
from here rather than constructing its own OpenAI client, so auth,
retries, and model names live in exactly one place.

NOTE: this reads config directly from environment variables (via
python-dotenv) rather than from `orchestrator.config`, because that
module is still a stub. Once config.py is implemented, get_client()
and friends should switch to pulling from get_settings() instead of
os.environ directly — the public function signatures below won't need
to change.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
# from langsmith import traceable
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

try:
    from orchestrator.api.services.llm_trace import log_llm_call, log_llm_response
except ImportError:  # pragma: no cover
    def log_llm_call(**kwargs):
        return __import__("time").perf_counter()

    def log_llm_response(**kwargs):
        return None

# Default seed for more reproducible completions when the provider supports it.
# Override with GATEWAY_SEED in .env. Pass seed=None to omit the parameter.
DEFAULT_SEED = int(os.environ.get("GATEWAY_SEED", "42"))


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


@lru_cache
def get_client() -> OpenAI:
    """Return a cached OpenAI-compatible client pointed at the gateway."""
    api_key = _require_env("GATEWAY_API_KEY")
    base_url = _require_env("GATEWAY_BASE_URL")
    return OpenAI(api_key=api_key, base_url=base_url)


# @traceable(name="chat_completion", run_type="llm")
@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(4))
def chat_completion(
    messages: list[dict],
    *,
    temperature: float = 0,
    seed: int | None = DEFAULT_SEED,
    **kwargs,
) -> str:
    """Send a chat completion request to the gateway and return the text.

    `messages` follows the standard OpenAI chat format:
        [{"role": "system"|"user"|"assistant", "content": "..."}]

    Determinism knobs:
      - temperature defaults to 0
      - seed defaults to GATEWAY_SEED (env) or 42
      - pass seed=None to omit the seed field (if the gateway rejects it)

    Extra kwargs (max_tokens, etc.) are passed through to
    `chat.completions.create`. `model` always comes from GATEWAY_CHAT_MODEL.

    Confirmed working against GATEWAY_CHAT_MODEL=codex/gpt-5.6-sol.
    Retries up to 4 times with exponential backoff on transient failures
    (network errors, rate limits, etc.) — see tenacity docs if you need
    to narrow this to specific exception types.

    LangSmith: when LANGSMITH_TRACING=true and LANGSMITH_API_KEY are set,
    each call is logged as a run (prompt in / completion out).
    """
    chat_model = _require_env("GATEWAY_CHAT_MODEL")
    client = get_client()

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
        response = client.chat.completions.create(**create_kwargs)
        content = response.choices[0].message.content or ""
        log_llm_response(t0=t0, content=content, model=chat_model)
        return content
    except Exception as exc:
        log_llm_response(t0=t0, content=None, model=chat_model, error=str(exc))
        raise


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the gateway's embeddings endpoint.

    NOT YET CONFIRMED WORKING. `codex/gpt-5.6-sol` is a chat/coding
    model, not an embedding model — OpenAI keeps embeddings as a
    separate model family (e.g. text-embedding-3-large), and it's not
    yet confirmed whether this gateway proxies POST /v1/embeddings at
    all, or exposes a distinct embedding model alongside the Codex
    models.

    TODO: once that's confirmed, either:
      (a) gateway exposes an embedding model — set GATEWAY_EMBEDDING_MODEL
          in .env and swap the `raise` below for the real call:

              embedding_model = _require_env("GATEWAY_EMBEDDING_MODEL")
              client = get_client()
              response = client.embeddings.create(model=embedding_model, input=texts)
              return [item.embedding for item in response.data]

      (b) gateway is chat-only — replace this function's implementation
          with a local model instead, e.g.:

              from sentence_transformers import SentenceTransformer
              _model = SentenceTransformer("all-MiniLM-L6-v2")
              return _model.encode(texts).tolist()

          `indexing/vectorstore.py` only depends on this function's
          signature (list[str] -> list[list[float]]), so callers don't
          need to change either way.
    """
    raise NotImplementedError(
        "embed_texts() is not implemented yet — pending confirmation of "
        "whether the gateway supports embeddings (see TODO in this function)."
    )
