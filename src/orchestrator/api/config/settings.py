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

    # --- AWS (Stage 4 ValidateStateMachineDefinition) ---
    # Credentials use the standard boto3 chain (env, shared credentials,
    # instance role). These settings control region, optional profile,
    # and whether the live API call is attempted.
    aws_region: str = Field(
        default="us-east-1",
        validation_alias="AWS_REGION",
    )
    aws_profile: str | None = Field(
        default=None,
        validation_alias="AWS_PROFILE",
    )
    # Set AWS_VALIDATE_ASL=false to skip the live API call.
    aws_validate_asl: bool = Field(
        default=True,
        validation_alias="AWS_VALIDATE_ASL",
    )
    # STANDARD | EXPRESS — matches CreateStateMachine type.
    aws_state_machine_type: str = Field(
        default="STANDARD",
        validation_alias="AWS_STATE_MACHINE_TYPE",
    )


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings()
