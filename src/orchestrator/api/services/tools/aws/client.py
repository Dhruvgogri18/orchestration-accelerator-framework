"""Shared boto3 session / client factory.

Single-account scope
--------------------
Credentials resolve via the standard boto3 chain (env, shared credentials,
instance role).  The account id is resolved once via STS and pinned for the
process lifetime.  Tools refuse to proceed if STS identity cannot be
established when enforcement is on — this prevents silent multi-account
confusion and makes logs attributable to exactly one account.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_session_cache: dict[tuple[str | None, str], Any] = {}
_identity_lock = threading.Lock()
_pinned_identity: "AwsIdentity | None" = None


@dataclass(frozen=True)
class AwsIdentity:
    account_id: str
    arn: str
    user_id: str
    region: str


def _load_settings():
    try:
        from orchestrator.api.config.settings import get_api_settings

        return get_api_settings()
    except Exception:  # pragma: no cover
        return None


def get_default_region() -> str:
    settings = _load_settings()
    if settings is not None:
        return getattr(settings, "aws_region", None) or "us-east-1"
    return "us-east-1"


def get_default_profile() -> str | None:
    settings = _load_settings()
    if settings is not None:
        return getattr(settings, "aws_profile", None)
    return None


def get_boto3_session(region: str | None = None, profile: str | None = None):
    """Return a boto3 Session for (profile, region).

    Sessions are reused only as connection factories — tool *results* are
    never cached (product requirement: always on-the-fly).
    """
    import boto3

    region = region or get_default_region()
    profile = profile if profile is not None else get_default_profile()
    key = (profile, region)

    with _lock:
        if key in _session_cache:
            return _session_cache[key]
        if profile:
            session = boto3.Session(profile_name=profile, region_name=region)
        else:
            session = boto3.Session(region_name=region)
        _session_cache[key] = session
        return session


def get_client(service: str, region: str | None = None, profile: str | None = None):
    """Return a boto3 client for the given service.

    Raises ValueError if the service is not on the allow-list.
    """
    from .allowlist import boto3_service_name, is_service_allowed

    if not is_service_allowed(service):
        raise ValueError(
            f"Service '{service}' is not on the tools allow-list. "
            "Add it to ALLOWED_SERVICES if intentional."
        )
    session = get_boto3_session(region=region, profile=profile)
    return session.client(boto3_service_name(service))


def resolve_identity(region: str | None = None, *, force_refresh: bool = False) -> AwsIdentity:
    """Pin and return the single AWS account identity for this process.

    First successful STS call wins; subsequent calls return the same
    account_id.  If a later call would resolve a *different* account,
    raises RuntimeError (guards against credential rotation into another
    account mid-process).
    """
    global _pinned_identity
    region = region or get_default_region()

    with _identity_lock:
        if _pinned_identity is not None and not force_refresh:
            return _pinned_identity

        client = get_client("sts", region=region)
        resp = client.get_caller_identity()
        identity = AwsIdentity(
            account_id=str(resp["Account"]),
            arn=str(resp.get("Arn") or ""),
            user_id=str(resp.get("UserId") or ""),
            region=region,
        )

        if _pinned_identity is not None and _pinned_identity.account_id != identity.account_id:
            raise RuntimeError(
                f"AWS account changed mid-process: was {_pinned_identity.account_id}, "
                f"now {identity.account_id}. Single-account mode forbids this. "
                "Restart the process with consistent credentials."
            )

        _pinned_identity = identity
        logger.info(
            "Pinned AWS identity account_id=%s arn=%s region=%s",
            identity.account_id,
            identity.arn,
            identity.region,
        )
        return identity


def get_account_id(region: str | None = None) -> str | None:
    """Best-effort account id. Returns None only if STS fails and enforcement is off."""
    try:
        return resolve_identity(region).account_id
    except Exception as exc:  # pragma: no cover
        logger.debug("Could not resolve account id: %s", exc)
        return None


def require_identity(region: str | None = None) -> AwsIdentity:
    """Like resolve_identity but always raises on failure (used by tools entry)."""
    return resolve_identity(region)


def clear_client_cache() -> None:
    """Test helper / forced refresh of sessions and pinned identity."""
    global _pinned_identity
    with _lock:
        _session_cache.clear()
    with _identity_lock:
        _pinned_identity = None
