"""Core utilities: configuration, cryptographic hashing, ID generation, and exceptions."""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

# --- Exceptions ---
class PlatformError(Exception):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

class ProvenanceError(PlatformError): pass
class ImmutabilityError(PlatformError): pass
class AdapterTimeoutError(PlatformError): pass
class AdapterExecutionError(PlatformError): pass
class PolicyViolationError(PlatformError): pass
class AuthenticationError(PlatformError): pass
class AuthorizationError(PlatformError): pass

# --- IDs ---
def generate_id(prefix: str = "id") -> str:
    """Generate a collision-resistant pseudorandom entity identifier using UUID4."""
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
    rag_version: str,
    model_config_hash: str,
    prompt_hash: str,
    evaluator_version: str,
    experiment_hash: str,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Compute reproducible run manifest hash from all runtime inputs and configs."""
    payload = {
        "dataset_checksum": dataset_checksum,
        "rag_version": rag_version,
        "model_config_hash": model_config_hash,
        "prompt_hash": prompt_hash,
        "evaluator_version": evaluator_version,
        "experiment_hash": experiment_hash,
    }
    if extra_metadata:
        payload["extra_metadata"] = extra_metadata
    return sha256_hash(canonical_json(payload))

# --- Config ---
@dataclass(frozen=True)
class Settings:
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "sqlite:///./rag_platform.db")
    )
    auth_enabled: bool = field(
        default_factory=lambda: os.getenv("AUTH_ENABLED", "false").lower() in ("true", "1")
    )
    api_key: str = field(
        default_factory=lambda: os.getenv("RAG_PLATFORM_API_KEY", "dev-secret-key-32chars-min-ok")
    )
    dev_mode: bool = field(
        default_factory=lambda: os.getenv("DEV_MODE", "true").lower() in ("true", "1")
    )
    default_max_cases: int = 500
    default_timeout_seconds: int = 60

settings = Settings()


def get_settings() -> Settings:
    return settings

