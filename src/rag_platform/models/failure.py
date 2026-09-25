"""Failure attribution and human review schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from rag_platform.models.enums import FailureCode, Severity


class FailureAttribution(BaseModel):
    """Root-cause diagnostic diagnosis for a failing RAG execution."""

    trace_id: str
    failure_type: FailureCode
    severity: Severity = Severity.MEDIUM
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


class HumanOverride(BaseModel):
    """Human reviewer correction to an automated failure diagnosis."""

    trace_id: str
    original_failure_type: FailureCode
    reviewed_failure_type: FailureCode
    reviewer_id: str
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
