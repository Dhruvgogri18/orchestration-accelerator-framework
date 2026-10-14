"""Typed application settings, loaded from environment / .env file.
Other modules should import a get_settings() from here rather than
reading os.environ directly.

Notes (Part B / Story B-02):
  * Unknown keys in .env are ignored (extra="ignore"). .env is shared with
    the API/UI, which define their own keys (see api/config/settings.py).
  * The Step Functions role ARN accepts either STEP_FUNCTIONS_EXECUTION_ROLE_ARN
    or AWS_STATE_MACHINE_ROLE_ARN (the name used by the API and the current
    .env) and is optional -- it is only needed at deploy time.
  * Blank values ("KEY=", or "KEY=   # comment") are treated as "not set" for
    optional fields.
  * RETRIEVAL_ENABLED (default false) gates all Part B behaviour. With it
    off, the pipeline behaves exactly as it did before Part B.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Gateway (OpenAI-compatible) ---
    # SecretStr: never shown in repr()/tracebacks. Use .get_secret_value().
    gateway_api_key: SecretStr = Field(..., alias="GATEWAY_API_KEY")
    gateway_base_url: str = Field(..., alias="GATEWAY_BASE_URL")
    gateway_chat_model: str = Field(..., alias="GATEWAY_CHAT_MODEL")
    gateway_seed: int | None = Field(42, alias="GATEWAY_SEED")

    # Embeddings are served through the same gateway, e.g.
    # bedrock/amazon.titan-embed-text-v2:0 (Titan Text Embeddings V2).
    gateway_embedding_model: str | None = Field(None, alias="GATEWAY_EMBEDDING_MODEL")
    # Texts per embeddings request. Titan embeds one input per underlying
    # call, so keep batches small to stay clear of provider rate limits.
    embedding_batch_size: int = Field(16, ge=1, le=256, alias="EMBEDDING_BATCH_SIZE")
    # Optional output dimension (Titan V2 supports 256 / 512 / 1024). Left
    # unset, the parameter is not sent and the model default (1024) is used.
    embedding_dimensions: int | None = Field(None, ge=1, alias="EMBEDDING_DIMENSIONS")

    # --- Part B feature flag ---
    retrieval_enabled: bool = Field(False, alias="RETRIEVAL_ENABLED")

    # --- AWS ---
    aws_region: str = Field("us-east-1", alias="AWS_REGION")
    aws_profile: str | None = Field(None, alias="AWS_PROFILE")
    step_functions_execution_role_arn: str | None = Field(
        None,
        validation_alias=AliasChoices(
            "STEP_FUNCTIONS_EXECUTION_ROLE_ARN", "AWS_STATE_MACHINE_ROLE_ARN"
        ),
    )

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

    @field_validator(
        "gateway_seed",
        "gateway_embedding_model",
        "embedding_dimensions",
        "aws_profile",
        "step_functions_execution_role_arn",
        mode="before",
    )
    @classmethod
    def _blank_is_none(cls, value):
        """`KEY=` (empty) in .env means "not set", not an empty string.

        Also treats a value that is only a comment ("# ...") as not set:
        python-dotenv parses `KEY=    # note` as the *text* "# note", so
        blank-with-inline-comment lines would otherwise become junk values.
        """
        if isinstance(value, str) and (not value.strip() or value.lstrip().startswith("#")):
            return None
        return value

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