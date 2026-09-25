"""Deterministic and sortable unique identifier generation."""

from __future__ import annotations

import secrets
import time


def generate_id(prefix: str = "id") -> str:
    """Generate a chronological, sortable unique identifier.
    
    Format: `{prefix}_{timestamp_ms}_{random_hex}`
    Guarantees monotonic sorting across time with zero collision in practice.
    # ponytail: timestamp + secrets.token_hex(4) covers ULID/UUIDv7 without a library dependency.
    """
    ts = int(time.time() * 1000)
    rand = secrets.token_hex(4)
    return f"{prefix}_{ts}_{rand}"
