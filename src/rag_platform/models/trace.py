"""Canonical RAG trace schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Provenance for a single passage retrieved by the SUT."""

    document_id: str
    chunk_id: str
    rank: int
    score: float = 0.0
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Claim-centric citation reference."""

    claim_id: str
    claim_text: str
    document_id: str
    chunk_id: str
    span: list[int] | None = None  # Character span [start, end] in answer string


class RagTrace(BaseModel):
    """Canonical trace of an executed RAG interaction under test.
    
    Acts as the single source of truth consumed by all evaluators and attribution logic.
    """

    trace_id: str
    run_id: str
    test_case_id: str
    question: str
    answer: str | None = None
    abstained: bool = False
    abstention_reason: str | None = None
    retrieval_available: bool = True
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    model: str | None = None
    error_code: str | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
