"""Platform common utilities."""

from rag_platform.common.config import Settings, settings
from rag_platform.common.crypto import (
    canonical_json,
    compute_manifest_hash,
    sha256_hash,
)
from rag_platform.common.exceptions import (
    AdapterExecutionError,
    AdapterTimeoutError,
    ImmutabilityError,
    PlatformError,
    PolicyViolationError,
    ProvenanceError,
)
from rag_platform.common.ids import generate_id
from rag_platform.common.logging import get_logger, setup_logging

__all__ = [
    "Settings",
    "settings",
    "canonical_json",
    "sha256_hash",
    "compute_manifest_hash",
    "generate_id",
    "setup_logging",
    "get_logger",
    "PlatformError",
    "ProvenanceError",
    "ImmutabilityError",
    "AdapterTimeoutError",
    "AdapterExecutionError",
    "PolicyViolationError",
]
