"""FastAPI app entrypoint.

Only responsible for:
  - loading settings early (pydantic-settings / .env)
  - ensuring `src` is importable when run without editable install
  - creating the FastAPI app and registering routers

Route logic lives in routers/ → services/ → repositories/.
"""

from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Make `import orchestrator...` work when uvicorn is started from repo root
# without `pip install -e .`
#
# main.py path: <repo>/src/orchestrator/api/main.py
# parents[0] = api/
# parents[1] = orchestrator/
# parents[2] = src/
# parents[3] = repo root
# ---------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_SRC_DIR = _THIS_FILE.parents[2]
_REPO_ROOT = _THIS_FILE.parents[3]

if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Load settings before routers/services import gateway code that needs env.
# pydantic-settings reads environment vars and optional .env — no load_dotenv().
from orchestrator.api.config.settings import get_api_settings  # noqa: E402

settings = get_api_settings()

from orchestrator.api.routers.session_router import router as session_router  # noqa: E402

app = FastAPI(
    title="Orchestration Accelerator API",
    version="0.1.0",
    description=(
        "Session-based workflow extraction API: "
        "create session, message, reject plan, approve plan."
    ),
)

# Streamlit (browser) may call the API directly in some setups.
# Server-side requests from Streamlit Python don't need CORS, but this is harmless.
app.add_middleware(
    CORSMiddleware,
    allow_origins=getattr(
        settings,
        "cors_allow_origins",
        [
            "http://localhost:8501",
            "http://127.0.0.1:8501",
        ],
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(session_router)


@app.get("/health", tags=["health"])
def health() -> dict:
    """Simple liveness check."""
    return {
        "status": "ok",
        "db_path": settings.orchestrator_db_path,
    }


# Optional: `python -m orchestrator.api.main` local run
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "orchestrator.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        app_dir=str(_SRC_DIR),
    )