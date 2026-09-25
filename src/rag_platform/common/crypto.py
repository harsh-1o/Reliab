"""Canonical serialization and deterministic SHA256 hashing.

Ensures bitwise reproducibility of dataset versions, prompt templates,
and the six-dimension execution manifest.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> str:
    """Serialize data into a deterministic, compact JSON string.
    
    Guarantees:
    - Sorted dictionary keys at every level
    - No whitespace between delimiters
    - UTF-8 representation
    - Strict handling of primitives
    """
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_serializer_fallback,
    )


def _json_serializer_fallback(obj: Any) -> Any:
    """Fallback serializer for domain models and dataclasses."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def sha256_hash(content: str | bytes) -> str:
    """Compute standard hexadecimal SHA256 digest."""
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
) -> str:
    """Compute the single cryptographic hash over the 6 reproducibility dimensions.
    
    If any single parameter alters by even 1 byte, the run_manifest_hash changes.
    """
    manifest_payload = {
        "dataset_checksum": dataset_checksum,
        "rag_version": rag_version,
        "model_config_hash": model_config_hash,
        "prompt_hash": prompt_hash,
        "evaluator_version": evaluator_version,
        "experiment_hash": experiment_hash,
    }
    return sha256_hash(canonical_json(manifest_payload))
