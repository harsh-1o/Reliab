"""Core utilities: configuration, cryptographic hashing, ID generation, and exceptions."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Sequence


# --- Exceptions ---
class PlatformError(Exception):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ProvenanceError(PlatformError):
    pass


class ImmutabilityError(PlatformError):
    pass


class AdapterTimeoutError(PlatformError):
    pass


class AdapterExecutionError(PlatformError):
    pass


class PolicyViolationError(PlatformError):
    pass


class AuthenticationError(PlatformError):
    pass


class AuthorizationError(PlatformError):
    pass

# --- IDs ---
def generate_id(prefix: str = "id") -> str:
    """Generate a collision-resistant pseudorandom entity identifier using UUID4.

    Note: UUID4 identifiers are NOT time-sortable. Use `generate_ordered_id` when
    lexicographic ordering by creation time is required.
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def generate_ordered_id(prefix: str = "id") -> str:
    """Generate a lexicographically time-sortable entity identifier using microsecond timestamp + random hex."""
    ts_hex = hex(int(time.time() * 1_000_000))[2:]
    rnd_hex = uuid.uuid4().hex[:6]
    return f"{prefix}_{ts_hex}_{rnd_hex}"

# --- Crypto & Manifest ---
def canonical_json(data: Any) -> str:
    """Serialize data into deterministic, canonically formatted JSON (RFC 8785 subset)."""
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o),
    )

def sha256_hash(content: str | bytes) -> str:
    """Compute standard SHA-256 hex digest."""
    if isinstance(content, str):
        content = content.encode("utf-8")
    return hashlib.sha256(content).hexdigest()

def compute_manifest_hash(
    *,
    dataset_checksum: str,
    rag_version: str = "rag_v1",
    model_name: str | None = None,
    model_version: str | None = None,
    model_parameters: dict[str, Any] | None = None,
    model_config_hash: str | None = None,
    prompt_hash: str | None = None,
    evaluator_version: str = "2.0.0",
    experiment_hash: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> str:
    """Compute reproducible run manifest hash from canonical RunProvenance manifest."""
    from rag_platform.models import RunProvenance

    prov = RunProvenance(
        dataset_checksum=dataset_checksum,
        rag_version=rag_version,
        model_name=model_name,
        model_version=model_version,
        model_parameters=model_parameters or {},
        model_config_hash=model_config_hash or "",
        prompt_hash=prompt_hash or "",
        evaluator_version=evaluator_version,
        experiment_hash=experiment_hash or "",
        **kwargs,
    )
    return prov.compute_hash()

# --- Config ---
def _is_production_environment() -> bool:
    env = os.getenv("ENVIRONMENT", os.getenv("APP_ENV", os.getenv("DEPLOYMENT_ENV", ""))).lower()
    return env in ("production", "prod", "staging")


def _resolve_database_url() -> str:
    """Resolve the database URL and fail closed for deployment environments.

    SQLite remains a convenient local-development/test default, but a deployment
    environment must explicitly configure a database and cannot silently fall back
    to a local SQLite file that is unsuitable for distributed workers.
    """
    configured = os.getenv("DATABASE_URL", "").strip()
    if not configured:
        if _is_production_environment():
            raise RuntimeError(
                "STARTUP FAILURE: DATABASE_URL must be explicitly configured in a deployment environment. "
                "Refusing to fall back to SQLite."
            )
        return "sqlite:///./rag_platform.db"

    if _is_production_environment() and configured.lower().startswith("sqlite:"):
        raise RuntimeError(
            "STARTUP FAILURE: SQLite is not supported as the deployment database. "
            "Configure DATABASE_URL with PostgreSQL (or another supported server database)."
        )
    return configured


def _resolve_api_key() -> str:
    """Resolve API key from environment.

    Fails startup with a clear error if running in production mode (DEV_MODE=false / AUTH_ENABLED=true)
    and no key is configured, preventing a silent insecure default.
    """
    is_prod = _is_production_environment()
    default_auth = "true" if is_prod else "false"
    default_dev = "false" if is_prod else "true"

    auth_enabled = os.getenv("AUTH_ENABLED", default_auth).lower() in ("true", "1")
    dev_mode = os.getenv("DEV_MODE", default_dev).lower() in ("true", "1")

    if is_prod and dev_mode:
        raise RuntimeError(
            "STARTUP FAILURE: Application refusing to start with DEV_MODE=true in a deployment environment. "
            "Set DEV_MODE=false and configure AUTH_ENABLED=true."
        )

    key = os.getenv("RAG_PLATFORM_API_KEY", "")
    if auth_enabled and not dev_mode:
        if not key or len(key) < 32:
            raise RuntimeError(
                "STARTUP FAILURE: AUTH_ENABLED=true but RAG_PLATFORM_API_KEY is not set or is too short (<32 chars). "
                "Set a strong secret via RAG_PLATFORM_API_KEY env variable, or set DEV_MODE=true for local development."
            )
    return key or "dev-secret-key-REPLACE-IN-PRODUCTION-32+"


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default_factory=_resolve_database_url)
    auth_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "AUTH_ENABLED",
            "true" if _is_production_environment() else "false",
        ).lower() in ("true", "1")
    )
    api_key: str = field(default_factory=_resolve_api_key)
    dev_mode: bool = field(
        default_factory=lambda: os.getenv(
            "DEV_MODE",
            "false" if _is_production_environment() else "true",
        ).lower() in ("true", "1")
    )
    trusted_proxies: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip()
            for p in os.getenv("TRUSTED_PROXIES", "127.0.0.1,::1,testclient").split(",")
            if p.strip()
        )
    )
    default_max_cases: int = 500
    default_timeout_seconds: int = 60

settings = Settings()


def calculate_percentile(values: Sequence[float | int], percentile: float) -> float:
    """Calculate percentile using standard linear interpolation (NIST Method 7 / NumPy default).

    Computes the p-th percentile (where percentile is between 0.0 and 1.0, e.g. 0.95 for p95)
    over a sequence of numerical observations.

    Method:
      1. Sorts input values in ascending order.
      2. Computes fractional rank: rank = percentile * (N - 1).
      3. Interpolates linearly between adjacent ranks:
         result = sorted_vals[lo] + (rank - lo) * (sorted_vals[hi] - sorted_vals[lo])

    Properties:
      - n=0: Returns 0.0
      - n=1: Returns the single observation
      - p=0.0: Returns min value
      - p=0.5: Returns median (p50)
      - p=1.0: Returns max value
      - Properly handles unsorted input and repeated values without bias.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(float(v) for v in values)
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]

    p = float(percentile)
    if p > 1.0:
        p = p / 100.0
    clamped_pct = max(0.0, min(1.0, p))
    rank = clamped_pct * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def get_settings() -> Settings:
    return settings

