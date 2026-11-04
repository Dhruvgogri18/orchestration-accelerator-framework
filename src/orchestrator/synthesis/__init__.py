"""Synthesis package: grounded plan → ASL → validation → optional deploy.

Public API
----------
  generate_plan(...)     — NL + repo context → structured Plan
  generate_asl(...)      — Plan / workflow_spec → Amazon States Language
  validate_asl(...)      — full validation suite
  pre_deploy_checks(...) — deploy gate reasons
  ground_plan_against_repo(...) — match plan tokens to indexed artifacts
  run_pipeline(...)      — end-to-end enterprise path (preferred entry point)
"""

from orchestrator.synthesis.asl_generator import generate_asl
from orchestrator.synthesis.grounding import (
    apply_grounding_to_plan,
    ground_plan_against_repo,
)
from orchestrator.synthesis.models import (
    AslResult,
    PlanResult,
    SynthesisContext,
    ValidationResult,
)
from orchestrator.synthesis.pipeline import PipelineResult, run_pipeline
from orchestrator.synthesis.planner import generate_plan
from orchestrator.synthesis.validator import pre_deploy_checks, validate_asl

__all__ = [
    "generate_plan",
    "generate_asl",
    "validate_asl",
    "pre_deploy_checks",
    "ground_plan_against_repo",
    "apply_grounding_to_plan",
    "run_pipeline",
    "PlanResult",
    "AslResult",
    "ValidationResult",
    "SynthesisContext",
    "PipelineResult",
]
