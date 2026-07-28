"""FastAPI app entrypoint. Only responsible for creating the app and
registering routers -- all actual route logic lives in routers/,
which delegates to services/ (business logic) and repositories/
(data access). Add new routers here as new resource groups are built
(e.g. a future workflow_spec_router once Stage 3.5 enrichment exists).
"""
from __future__ import annotations

from fastapi import FastAPI

from orchestrator.api.routers.session_router import router as session_router

app = FastAPI(title="Orchestration Accelerator API")
app.include_router(session_router)