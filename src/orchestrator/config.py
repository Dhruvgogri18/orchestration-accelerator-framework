"""Typed application settings, loaded from environment / .env file.
Other modules should import a get_settings() from here rather than
reading os.environ directly.

TODO(Phase 1): implement Settings (pydantic-settings) and get_settings().
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env" , env_file_encoding="utf-8")

    # --- Gateway (OpenAI-compatible) ---
    gateway_api_key: str = Field(..., alias="GATEWAY_API_KEY")
    gateway_base_url: str = Field(..., alias="GATEWAY_BASE_URL")
    gateway_chat_model: str = Field(..., alias="GATEWAY_CHAT_MODEL")
    # Optional: blank in .env until embeddings support is confirmed with
    # whoever manages the gateway. See gateway_client.embed_texts().
    gateway_embedding_model: str | None = Field(None, alias="GATEWAY_EMBEDDING_MODEL")

    # --- AWS ---
    aws_region: str = Field("us-east-1" , alias="AWS_REGION")
    aws_profile: str | None = Field(None, alias="AWS_PROFILE")
    step_functions_execution_role_arn: str = Field(..., alias="STEP_FUNCTIONS_EXECUTION_ROLE_ARN")

    # --- Repository ingestion ---
    repo_path: str = Field(".", alias="REPO_PATH")
    include_globs: str = Field(
        "**/*.py,**/*.yaml,**/*.yml,**/*.json,**/*.md", alias="INCLUDE_GLOBS"
    )
    exclude_globs: str = Field(
        "**/.venv/**,**/node_modules/**,**/.git/**", alias="EXCLUDE_GLOBS"
    )

    # --- Vector store ---
    chroma_persist_dir: str = Field("./data/chroma", alias="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("repo_context", alias="CHROMA_COLLECTION_NAME")

    # --- Chunking ---
    chunk_size_tokens: int = Field(800, alias="CHUNK_SIZE_TOKENS")
    chunk_overlap_tokens: int = Field(120, alias="CHUNK_OVERLAP_TOKENS")

    @property
    def include_glob_list(self) -> list[str]:
        """INCLUDE_GLOBS as a Python list instead of a raw comma-separated string."""
        return [g.strip() for g in self.include_globs.split(",") if g.strip()]

    @property
    def exclude_glob_list(self) -> list[str]:
        """EXCLUDE_GLOBS as a Python list instead of a raw comma-separated string."""
        return [g.strip() for g in self.exclude_globs.split(",") if g.strip()]

@lru_cache
def get_settings() -> Settings:
    """Cached settings instance. Call this everywhere; don't call Settings() directly.

    Cached with lru_cache so .env is only read and validated once per
    process, not on every call.
    """
    return Settings()