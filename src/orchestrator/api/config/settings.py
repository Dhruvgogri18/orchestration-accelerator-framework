from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    orchestrator_db_path: str = Field(
        default="data/sessions.db",
        validation_alias="ORCHESTRATOR_DB_PATH",
    )


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings()