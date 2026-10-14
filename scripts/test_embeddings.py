"""Story B-01: live check that the gateway serves the embedding model.

Run against the real gateway (needs GATEWAY_API_KEY, GATEWAY_BASE_URL and
GATEWAY_EMBEDDING_MODEL in .env):

    python scripts/test_embeddings.py

Checks, in order:
  1. basic call           - 3 short inputs -> 3 vectors; prints dimension
  2. batching             - 20 inputs through embed_texts() -> 20 vectors, same order
  3. similarity sanity    - related texts score higher than unrelated ones
  4. `dimensions` param   - is dimensions=512 accepted or rejected (informational)
  5. oversized input      - does the gateway return a clear error (informational)

Checks 1-3 must pass (exit code 0/1). Checks 4-5 only report behaviour so
the team can record it. Nothing here prints keys or full responses.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from orchestrator.config import get_settings  # noqa: E402
from orchestrator.indexing.gateway_client import embed_texts, get_client  # noqa: E402


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


def main() -> int:
    s = get_settings()
    if not s.gateway_embedding_model:
        print("FAIL: GATEWAY_EMBEDDING_MODEL is not set in .env")
        return 1
    print(f"model={s.gateway_embedding_model}  batch_size={s.embedding_batch_size}  "
          f"dimensions={s.embedding_dimensions or 'model default'}")
    failed = False

    # 1. basic call
    try:
        v = embed_texts(["glue job", "sns topic", "def run(): pass"])
        dim = len(v[0])
        ok = len(v) == 3 and all(len(x) == dim for x in v)
        print(f"[{'PASS' if ok else 'FAIL'}] basic call: {len(v)} vectors, dimension {dim}")
        failed |= not ok
    except Exception as exc:
        print(f"[FAIL] basic call: {type(exc).__name__}: {exc}")
        return 1

    # 2. batching (20 inputs -> several requests at the configured batch size)
    texts = [f"workflow step number {i}: run glue job {i}" for i in range(20)]
    try:
        vs = embed_texts(texts)
        single = embed_texts([texts[7]])[0]
        same_order = cosine(vs[7], single) > 0.999
        ok = len(vs) == 20 and same_order
        print(f"[{'PASS' if ok else 'FAIL'}] batching: {len(vs)} vectors, order preserved={same_order}")
        failed |= not ok
    except Exception as exc:
        print(f"[FAIL] batching: {type(exc).__name__}: {exc}")
        failed = True

    # 3. similarity sanity
    try:
        a, b, c = embed_texts([
            "start an AWS Glue job to transform a CSV file",
            "run a Glue ETL job that converts csv data",
            "send a birthday greeting email to the team",
        ])
        close, far = cosine(a, b), cosine(a, c)
        ok = close > far
        print(f"[{'PASS' if ok else 'FAIL'}] similarity: related={close:.3f} unrelated={far:.3f}")
        failed |= not ok
    except Exception as exc:
        print(f"[FAIL] similarity: {type(exc).__name__}: {exc}")
        failed = True

    client = get_client()

    # 4. dimensions parameter (informational)
    try:
        r = client.embeddings.create(model=s.gateway_embedding_model,
                                     input=["dimension test"], dimensions=512)
        print(f"[INFO] dimensions=512 accepted -> vector size {len(r.data[0].embedding)}")
    except Exception as exc:
        print(f"[INFO] dimensions=512 rejected -> {type(exc).__name__}: {str(exc)[:160]}")

    # 5. oversized input (informational). Titan V2 limit is 8,192 tokens.
    try:
        client.embeddings.create(model=s.gateway_embedding_model, input=["word " * 20000])
        print("[INFO] oversized input was ACCEPTED (gateway/provider truncates?) - chunk size "
              "limits must be enforced by us")
    except Exception as exc:
        print(f"[INFO] oversized input rejected -> {type(exc).__name__}: {str(exc)[:160]}")

    print("RESULT:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())