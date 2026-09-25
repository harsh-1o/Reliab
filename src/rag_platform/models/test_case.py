"""Benchmark test case and dataset schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from rag_platform.common.crypto import canonical_json, sha256_hash
from rag_platform.models.enums import Answerability, DatasetStatus


class DocumentReference(BaseModel):
    """Reference to expected supporting document / passage."""

    document_id: str
    chunk_id: str | None = None
    page: int | None = None
    span: list[int] | None = None  # [start_char, end_char]


class TestCase(BaseModel):
    """Individual benchmark evaluation test case."""

    __test__ = False  # Prevent pytest from treating this model as a test suite

    id: str
    question: str
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    relevant_documents: list[DocumentReference] = Field(default_factory=list)
    answerability: Answerability = Answerability.ANSWERABLE
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Convert test case to normalized dictionary for checksum hashing."""
        return {
            "id": self.id,
            "question": self.question.strip(),
            "expected_answer": self.expected_answer.strip() if self.expected_answer else None,
            "expected_facts": sorted([f.strip() for f in self.expected_facts]),
            "relevant_documents": [
                {
                    "document_id": d.document_id,
                    "chunk_id": d.chunk_id,
                    "page": d.page,
                    "span": d.span,
                }
                for d in self.relevant_documents
            ],
            "answerability": self.answerability.value,
            "tags": sorted(self.tags),
        }


def compute_dataset_checksum(cases: list[TestCase]) -> str:
    """Compute deterministic SHA256 checksum over sorted list of test cases."""
    canonical_cases = sorted(
        [c.to_canonical_dict() for c in cases],
        key=lambda x: str(x["id"]),
    )
    return sha256_hash(canonical_json(canonical_cases))


class BenchmarkDataset(BaseModel):
    """Versioned benchmark dataset."""

    id: str
    project_id: str
    name: str
    version: str
    status: DatasetStatus = DatasetStatus.DRAFT
    checksum_sha256: str = ""
    cases: list[TestCase] = Field(default_factory=list)
    description: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def publish(self) -> BenchmarkDataset:
        """Lock and seal the dataset into an immutable PUBLISHED version."""
        if not self.cases:
            raise ValueError("Cannot publish an empty dataset.")
        self.checksum_sha256 = compute_dataset_checksum(self.cases)
        self.status = DatasetStatus.PUBLISHED
        return self
