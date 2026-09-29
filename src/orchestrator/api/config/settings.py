from functools import lru_cache
from typing import Literal

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

    # --- Orchestration backend ---
    # USE_LANGGRAPH=true  → session routes use api.services.orchestration
    #                       (StateGraph for analysis / approve / enrichment)
    # USE_LANGGRAPH=false → classic pipeline_analysis.session_service path
    # Default false so production stays on the proven path until explicitly enabled.
    use_langgraph: bool = Field(
        default=True,
        validation_alias="USE_LANGGRAPH",
    )

    # Primary chat model (also set via GATEWAY_CHAT_MODEL in gateway_client).
    gateway_chat_model: str | None = Field(
        default=None,
        validation_alias="GATEWAY_CHAT_MODEL",
    )
    # Faster/cheaper model for narrow classification stages (scope,
    # environment, action_clauses). Falls back to gateway_chat_model / gateway default.
    gateway_fast_chat_model: str | None = Field(
        default=None,
        validation_alias="GATEWAY_FAST_CHAT_MODEL",
    )
    # Mark long system prompts for provider prompt-caching when supported.
    use_prompt_cache: bool = Field(
        default=True,
        validation_alias="USE_PROMPT_CACHE",
    )
    # Merge component_identification + flow_structure into one LLM call.
    # Saves one sequential round-trip; disable if quality regresses.
    merge_component_and_flow: bool = Field(
        default=True,
        validation_alias="MERGE_COMPONENT_AND_FLOW",
    )

    # --- Stage 4 ASL generation strategy ---
    # "mechanical"        -> current deterministic Python translator
    #                        (api.services.asl_service.generate_asl). No LLM
    #                        call happens for ASL generation. Default -- the
    #                        existing pipeline is unaffected unless this is
    #                        explicitly changed.
    # "llm"                -> ask the LLM to emit the full ASL document
    #                        (including JSONata expressions) directly from
    #                        workflow_spec.json. The result is still run
    #                        through the same mechanical validate_asl()
    #                        checks as the deterministic path -- checking
    #                        stays code, only generation moves to the LLM.
    #                        Raises AslGenerationError if no attempt passes
    #                        within asl_llm_max_attempts.
    # "llm_with_fallback"  -> same as "llm", but if the LLM can't produce a
    #                        passing definition in time, silently falls back
    #                        to the mechanical translator instead of
    #                        failing the request. A note is added to
    #                        AslResult.generation_notes either way so the
    #                        fallback is visible, not silent, to whoever
    #                        reads generation_notes.
    # Set back to "mechanical" at any time to fully restore prior behavior.
    asl_generation_mode: Literal["mechanical", "llm", "llm_with_fallback"] = Field(
        default="llm",
        validation_alias="ASL_GENERATION_MODE",
    )
    # Repair-loop cap for the LLM ASL path -- mirrors
    # MAX_ENRICHMENT_ATTEMPTS's role in enrichment/service.py. Each retry
    # re-sends the concrete validate_asl() errors from the previous
    # attempt so the model is fixing a specific, named problem, not
    # guessing again from scratch.
    asl_llm_max_attempts: int = Field(
        default=4,
        validation_alias="ASL_LLM_MAX_ATTEMPTS",
    )


@lru_cache
def get_api_settings() -> APISettings:
    return APISettings()