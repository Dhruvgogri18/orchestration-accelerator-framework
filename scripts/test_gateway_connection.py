"""Standalone connectivity smoke test — confirms the OpenAI SDK can reach
your gateway and get a real completion back, BEFORE any Phase 1 logic is
built on top of it.

Deliberately does not import from `orchestrator.*` (those modules are
still stubs) — this talks to the gateway directly so it can be run and
debugged in isolation.

Usage:
    cp .env.example .env      # fill in GATEWAY_API_KEY at minimum
    python scripts/test_gateway_connection.py
"""
from __future__ import annotations

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> int:
    load_dotenv()

    api_key = os.environ.get("GATEWAY_API_KEY")
    base_url = os.environ.get("GATEWAY_BASE_URL")
    chat_model = os.environ.get("GATEWAY_CHAT_MODEL")

    missing = [
        name
        for name, val in [
            ("GATEWAY_API_KEY", api_key),
            ("GATEWAY_BASE_URL", base_url),
            ("GATEWAY_CHAT_MODEL", chat_model),
        ]
        if not val
    ]
    if missing:
        print(f"Missing required .env values: {', '.join(missing)}")
        print("Copy .env.example to .env and fill them in first.")
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"Base URL   : {base_url}")
    print(f"Chat model : {chat_model}")
    print("Sending a test chat completion...\n")

    try:
        response = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Reply with exactly: gateway ok"}],
            max_tokens=20,
        )
    except Exception as e:
        print(f"❌ Chat completion failed: {e!r}")
        return 1

    text = response.choices[0].message.content
    print(f"✅ Chat completion succeeded. Response: {text!r}")

    # --- Embeddings check (optional) ---
    # Not every gateway proxies POST /v1/embeddings — some only proxy
    # /chat/completions. Uncomment once GATEWAY_EMBEDDING_MODEL is set
    # in .env, to confirm before building indexing/vectorstore.py on it.
    #
    # embedding_model = os.environ.get("GATEWAY_EMBEDDING_MODEL")
    # if embedding_model:
    #     try:
    #         emb = client.embeddings.create(model=embedding_model, input=["test"])
    #         print(f"✅ Embeddings succeeded. Vector length: {len(emb.data[0].embedding)}")
    #     except Exception as e:
    #         print(f"❌ Embeddings failed: {e!r}")
    #         print("   Your gateway may not proxy /v1/embeddings — see README's")
    #         print("   'Embeddings fallback' section for a local alternative.")

    return 0


if __name__ == "__main__":
    sys.exit(main())