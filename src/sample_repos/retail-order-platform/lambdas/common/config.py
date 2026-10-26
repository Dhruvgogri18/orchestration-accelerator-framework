"""Shared configuration helpers for Lambda handlers.

Every Lambda reads its resource names from environment variables that are set
from config/jobs.yaml and config/environments/<env>.yaml.
"""
import os


def require_env(name: str) -> str:
    """Return an environment variable or raise a clear error if it is missing."""
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
