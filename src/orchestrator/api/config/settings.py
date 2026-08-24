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

    # --- AWS (Stage 5 deploy) ---
    # The IAM role Step Functions assumes to RUN the deployed state
    # machine (states.amazonaws.com trust policy). This is deliberately
    # not something the API creates on your behalf -- minting IAM
    # roles/policies from an LLM-adjacent service is a bad idea. Set it
    # here for a single-role setup, or pass role_arn per deploy request
    # for multi-team/multi-account setups.
    aws_state_machine_role_arn: str | None = Field(
        default=None,
        validation_alias="AWS_STATE_MACHINE_ROLE_ARN",
    )
    # Optional prefix so state machines created by this tool are easy
    # to pick out in an account that has other, unrelated ones — e.g.
    # "orch-" -> "orch-my-workflow". Empty string (default) adds none.
    aws_deploy_name_prefix: str = Field(
        default="",
        validation_alias="AWS_DEPLOY_NAME_PREFIX",
    )


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings()
