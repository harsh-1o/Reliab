"""Core utilities: config, crypto, IDs, and exceptions.

# ponytail: everything in one file instead of a 6-folder package maze.
"""

from __future__ import annotations

import hashlib
import json
import os
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

# --- IDs ---
# ponytail: uuid4 hex covers unique sortable IDs in one line.
def generate_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

# --- Crypto & Manifest ---
def canonical_json(data: Any) -> str:
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=lambda o: o.model_dump(mode="json") if hasattr(o, "model_dump") else str(o),
    )

def sha256_hash(content: str | bytes) -> str:
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
    default_max_cases: int = 500
    default_timeout_seconds: int = 60

settings = Settings()
