"""ASL validation façade wrapping asl_service.validate_asl."""

from __future__ import annotations

import logging
from typing import Any

from orchestrator.synthesis.models import ValidationResult

logger = logging.getLogger(__name__)


def validate_asl(
    asl: dict[str, Any],
    *,
    spec: dict[str, Any] | None = None,
    include_aws_validate: bool = False,
) -> ValidationResult:
    if not asl:
        return ValidationResult(
            ok=False,
            issues=["empty_asl"],
            report={"ok": False, "issues": ["empty_asl"]},
        )

    try:
        from orchestrator.api.services.asl_service import validate_asl as _validate

        try:
            report = _validate(
                asl,
                spec=spec,
                include_aws_validate=include_aws_validate,
            )
        except TypeError:
            report = _validate(asl)
    except Exception as exc:  # noqa: BLE001
        logger.exception("validate_asl failed")
        return ValidationResult(
            ok=False,
            issues=[f"{type(exc).__name__}: {exc}"],
            report={"ok": False, "error": str(exc)},
        )

    return _normalize_report(report)


def _normalize_report(report: Any) -> ValidationResult:
    if report is None:
        return ValidationResult(ok=False, issues=["null_report"])

    if hasattr(report, "model_dump"):
        data = report.model_dump()
    elif isinstance(report, dict):
        data = report
    else:
        data = {
            "ok": bool(getattr(report, "ok", False)),
            "issues": list(getattr(report, "issues", []) or []),
        }

    ok = bool(data.get("ok", False))
    issues: list[str] = []
    groups: dict[str, Any] = {}

    for item in data.get("issues") or data.get("errors") or []:
        if isinstance(item, str):
            issues.append(item)
        elif isinstance(item, dict):
            issues.append(str(item.get("message") or item.get("msg") or item))
        else:
            issues.append(str(item))

    for key in (
        "structural",
        "reachability",
        "spec_consistency",
        "aws_constraints",
        "extras",
        "groups",
    ):
        if key in data and data[key] is not None:
            groups[key] = data[key]
    if isinstance(data.get("groups"), dict):
        groups.update(data["groups"])

    if not issues and groups:
        for gname, gval in groups.items():
            if isinstance(gval, dict) and gval.get("issues"):
                for i in gval["issues"]:
                    issues.append(f"[{gname}] {i}")
            elif isinstance(gval, list):
                for i in gval:
                    issues.append(f"[{gname}] {i}")

    if not ok and not issues:
        issues = ["validation_failed"]

    return ValidationResult(ok=ok, report=data, issues=issues, groups=groups)


def pre_deploy_checks(
    asl: dict[str, Any],
    *,
    validation_ok: bool | None = None,
    execution_role_arn: str | None = None,
) -> list[str]:
    """Return human-readable reasons that block deploy (empty = clear)."""
    reasons: list[str] = []
    if not asl:
        reasons.append("No ASL definition present.")
        return reasons

    if validation_ok is False:
        reasons.append("ASL validation has not passed.")

    import json
    import re

    blob = json.dumps(asl, ensure_ascii=False)
    todos = sorted(set(re.findall(r"\bTODO_[A-Z0-9_]+\b", blob)))
    if todos:
        reasons.append(
            "Unresolved placeholders in ASL: "
            + ", ".join(todos[:12])
            + ("…" if len(todos) > 12 else "")
        )

    if not execution_role_arn:
        try:
            from orchestrator.config import get_settings

            execution_role_arn = get_settings().step_functions_execution_role_arn
        except Exception:  # noqa: BLE001
            pass
    if not execution_role_arn:
        try:
            from orchestrator.api.config.settings import get_api_settings

            s = get_api_settings()
            execution_role_arn = getattr(s, "aws_state_machine_role_arn", None) or getattr(
                s, "step_functions_execution_role_arn", None
            )
        except Exception:  # noqa: BLE001
            pass
    if not execution_role_arn:
        reasons.append(
            "No Step Functions execution role configured "
            "(STEP_FUNCTIONS_EXECUTION_ROLE_ARN / AWS_STATE_MACHINE_ROLE_ARN)."
        )

    return reasons
