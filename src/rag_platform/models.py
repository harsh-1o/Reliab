"""Domain models, enums, traces, and policies.

# ponytail: one file covers the entire data model. No imports-of-imports.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rag_platform.core import canonical_json, compute_manifest_hash, sha256_hash


# --- Enums ---
class RunStatus(str, Enum):
    CREATED = "CREATED"
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    SCORING = "SCORING"
    ATTRIBUTING = "ATTRIBUTING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class DatasetStatus(str, Enum):
    DRAFT = "DRAFT"
    PUBLISHED = "PUBLISHED"
    ARCHIVED = "ARCHIVED"


class Answerability(str, Enum):
    ANSWERABLE = "ANSWERABLE"
    UNANSWERABLE = "UNANSWERABLE"


class MetricFamily(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    GENERATION = "GENERATION"
    CITATION = "CITATION"
    ABSTENTION = "ABSTENTION"
    SYSTEM = "SYSTEM"


class FailureCode(str, Enum):
    RET_01 = "RET-01"  # Retrieval miss
    RET_02 = "RET-02"  # Bad ranking
    GEN_01 = "GEN-01"  # Unsupported claim / Hallucination
    GEN_02 = "GEN-02"  # Contradiction
    CIT_01 = "CIT-01"  # Mis-citation
    CIT_02 = "CIT-02"  # Missing citation
    KNW_01 = "KNW-01"  # Knowledge gap
    ABS_01 = "ABS-01"  # Abstention failure
    NUM_01 = "NUM-01"  # Numerical reasoning
    ENT_01 = "ENT-01"  # Entity confusion
    OPS_01 = "OPS-01"  # Infrastructure error (timeout/network)


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


# --- Dataset & Cases ---
class DocumentReference(BaseModel):
    document_id: str
    chunk_id: str | None = None
    page: int | None = None
    span: list[int] | None = None


class TestCase(BaseModel):
    __test__ = False  # Prevent pytest from treating as test class

    id: str
    question: str
    expected_answer: str | None = None
    expected_facts: list[str] = Field(default_factory=list)
    relevant_documents: list[DocumentReference] = Field(default_factory=list)
    answerability: Answerability = Answerability.ANSWERABLE
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_canonical_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question.strip(),
            "expected_answer": self.expected_answer.strip() if self.expected_answer else None,
            "expected_facts": sorted([f.strip() for f in self.expected_facts]),
            "relevant_documents": [
                {"document_id": d.document_id, "chunk_id": d.chunk_id, "page": d.page, "span": d.span}
                for d in self.relevant_documents
            ],
            "answerability": self.answerability.value,
            "tags": sorted(self.tags),
        }


def compute_dataset_checksum(cases: list[TestCase]) -> str:
    canonical_cases = sorted([c.to_canonical_dict() for c in cases], key=lambda x: str(x["id"]))
    return sha256_hash(canonical_json(canonical_cases))


class BenchmarkDataset(BaseModel):
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
        if not self.cases:
            raise ValueError("Cannot publish an empty dataset.")
        self.checksum_sha256 = compute_dataset_checksum(self.cases)
        self.status = DatasetStatus.PUBLISHED
        return self


# --- Traces ---
class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    rank: int
    score: float = 0.0
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    claim_id: str
    claim_text: str
    document_id: str
    chunk_id: str
    span: list[int] | None = None


class RagTrace(BaseModel):
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


# --- Metrics ---
class MetricResult(BaseModel):
    metric_name: str
    metric_family: MetricFamily
    score: float
    reason: str | None = None
    evaluator_version: str = "1.0.0"
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    metric_name: str
    metric_family: MetricFamily
    mean: float
    p50: float
    p95: float
    min: float
    max: float
    count: int


class RunMetricsSummary(BaseModel):
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    total_cases: int = 0
    scored_cases: int = 0
    hallucination_rate: float = 0.0
    abstention_accuracy: float = 1.0
    infra_error_count: int = 0
    p95_latency_ms: float = 0.0
    total_cost_usd: float = 0.0


# --- Failure Attribution ---
class FailureAttribution(BaseModel):
    trace_id: str
    failure_type: FailureCode
    severity: Severity = Severity.MEDIUM
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


class HumanOverride(BaseModel):
    trace_id: str
    original_failure_type: FailureCode
    reviewed_failure_type: FailureCode
    reviewer_id: str
    notes: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --- Run Provenance & Gate ---
class RunProvenance(BaseModel):
    dataset_checksum: str
    rag_version: str
    model_config_hash: str
    prompt_hash: str
    evaluator_version: str
    experiment_hash: str
    manifest_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.manifest_hash:
            self.manifest_hash = compute_manifest_hash(
                dataset_checksum=self.dataset_checksum,
                rag_version=self.rag_version,
                model_config_hash=self.model_config_hash,
                prompt_hash=self.prompt_hash,
                evaluator_version=self.evaluator_version,
                experiment_hash=self.experiment_hash,
            )


class RunOptions(BaseModel):
    max_cases: int | None = None
    concurrency: int = 5
    timeout_seconds: int = 60
    fail_fast: bool = False
    use_cache: bool = True


class RunConfig(BaseModel):
    project_id: str
    dataset_id: str
    dataset_version: str
    system_version: str
    policy_id: str = "prod-default"
    suite: str = "full"
    options: RunOptions = Field(default_factory=RunOptions)


class ReleasePolicy(BaseModel):
    policy_id: str = "prod-default"
    min_faithfulness: float = 0.90
    min_retrieval_recall: float = 0.92
    min_citation_accuracy: float = 0.95
    max_hallucination_rate: float = 0.05
    min_abstention_accuracy: float = 0.90
    max_latency_regression_pct: float = 20.0
    max_cost_regression_pct: float = 25.0
    max_critical_regressions: int = 0


class GateViolation(BaseModel):
    metric_name: str
    baseline_value: float | None = None
    candidate_value: float
    threshold: float
    violation_type: str
    message: str


class GateResult(BaseModel):
    status: GateStatus
    baseline_run_id: str | None = None
    candidate_run_id: str
    policy_id: str
    violations: list[GateViolation] = Field(default_factory=list)
    critical_regressions: int = 0
    report_url: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
