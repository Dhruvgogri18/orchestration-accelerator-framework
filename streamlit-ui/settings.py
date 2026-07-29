"""App settings loaded from environment / .env via pydantic-settings."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UISettings(BaseSettings):
    """
    Values are taken from:
      1) real environment variables
      2) optional .env file (see model_config)
    No load_dotenv() needed.
    """

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # USE_MOCK_API=false in .env
    use_mock_api: bool = Field(default=False, validation_alias="USE_MOCK_API")

    # ORCHESTRATION_API_URL=http://127.0.0.1:8000
    orchestration_api_url: str = Field(
        default="http://127.0.0.1:8000",
        validation_alias="ORCHESTRATION_API_URL",
    )

    orchestration_api_timeout: float = Field(
        default=120.0,
        validation_alias="ORCHESTRATION_API_TIMEOUT",
    )


@lru_cache
def get_settings() -> UISettings:
    """Cached singleton — reads env/.env once per process."""
    return UISettings()