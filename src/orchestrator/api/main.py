"""FastAPI app entrypoint. Only responsible for creating the app and
registering routers -- all actual route logic lives in routers/,
which delegates to services/ (business logic) and repositories/
(data access). Add new routers here as new resource groups are built
(e.g. a future workflow_spec_router once Stage 3.5 enrichment exists).
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI

# src/ on path so `import orchestrator...` works when running this file
# directly. Prefer `pip install -e .` long-term and drop this hack.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")  # repo root

from orchestrator.api.routers.session_router import router as session_router

app = FastAPI(title="Orchestration Accelerator API")
app.include_router(session_router)