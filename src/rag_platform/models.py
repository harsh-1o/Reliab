"""Domain models, enums, traces, and policies."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from rag_platform.core import canonical_json, sha256_hash


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


class MetricStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class FailureCode(str, Enum):
    OPS_01 = "OPS-01"  # Infrastructure error (timeout/network)
    ABS_01 = "ABS-01"  # Unanswerable abstention failure (answered unanswerable)
    ABS_02 = "ABS-02"  # Answerable false abstention (refused answerable case)
    RET_01 = "RET-01"  # Retrieval miss (evidence chunk absent from top-K)
    RET_02 = "RET-02"  # Bad ranking / distractor contamination (evidence buried below distractors)
    RET_03 = "RET-03"  # Context window truncation (relevant passage cut off)
    GEN_01 = "GEN-01"  # Extrinsic hallucination (unsupported claims)
    GEN_02 = "GEN-02"  # Intrinsic contradiction (claims contradict retrieved evidence)
    CIT_01 = "CIT-01"  # Missing citation (factual claims without citation)
    CIT_02 = "CIT-02"  # Misattributed citation (cited chunk does not substantiate claim)
    KNW_01 = "KNW-01"  # Knowledge gap
    NUM_01 = "NUM-01"  # Numerical reasoning error
    ENT_01 = "ENT-01"  # Entity confusion


class Severity(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class GateStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"


class ClaimStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    CONTRADICTED = "CONTRADICTED"


class ClaimVerification(BaseModel):
    claim_id: str
    claim_text: str
    status: ClaimStatus
    supporting_chunk_id: str | None = None
    confidence: float | None = None
    confidence_type: str = "not_calibrated"
    reason: str = ""


class DiagnosticFinding(BaseModel):
    code: FailureCode
    severity: Severity = Severity.MEDIUM
    confidence: float | None = None
    confidence_type: str = "not_calibrated"
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)


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
        """Convert TestCase to a canonical dictionary representation for cryptographic checksumming.

        Metadata is semantically relevant: it passes configuration, routing tags,
        and domain parameters to SUT adapters and evaluators. Changes to metadata
        alter evaluation behavior and therefore alter the dataset checksum.
        """
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
            "metadata": {k: self.metadata[k] for k in sorted(self.metadata.keys())},
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
    score: float | None = None
    status: MetricStatus = MetricStatus.PASS
    reason: str | None = None
    evaluator_version: str = "2.0.0"
    evaluator_type: str = "deterministic_heuristic"
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
    applicable_count: int = 0
    std_dev: float = 0.0
    ci_lower: float | None = None
    ci_upper: float | None = None
    sample_warning: str | None = None


class RunMetricsSummary(BaseModel):
    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    total_cases: int = 0
    scored_cases: int = 0
    hallucination_rate: float | None = None
    low_faithfulness_rate: float | None = None
    abstention_accuracy: float | None = None
    infra_error_count: int = 0
    p95_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    sample_warning: str | None = None


# --- Failure Attribution ---
class FailureAttribution(BaseModel):
    trace_id: str
    primary_code: FailureCode = FailureCode.OPS_01
    contributing_codes: list[FailureCode] = Field(default_factory=list)
    failure_type: FailureCode | None = None  # Backward-compatible alias for primary_code
    severity: Severity = Severity.MEDIUM
    confidence: float | None = None
    confidence_type: str = "not_calibrated"
    explanation: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    recommended_actions: list[str] = Field(default_factory=list)
    findings: list[DiagnosticFinding] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if self.failure_type is None:
            self.failure_type = self.primary_code
        elif self.primary_code == FailureCode.OPS_01 and self.failure_type != FailureCode.OPS_01:
            self.primary_code = self.failure_type


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
    dataset_id: str | None = None
    dataset_version: str | None = None
    rag_version: str = "rag_v1"
    model_name: str | None = None
    model_version: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    model_config_hash: str = ""
    temperature: float | None = None
    prompt_template: str | None = None
    prompt_hash: str = ""
    system_prompt: str | None = None
    system_prompt_hash: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None
    retriever_config: dict[str, Any] = Field(default_factory=dict)
    reranker_config: dict[str, Any] = Field(default_factory=dict)
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    adapter_type: str = "synthetic"
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    evaluator_version: str = "2.0.0"
    evaluation_config: dict[str, Any] = Field(default_factory=dict)
    experiment_config: dict[str, Any] = Field(default_factory=dict)
    experiment_hash: str = ""
    python_version: str | None = None
    dependency_lock_hash: str | None = None
    environment_info: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    manifest_hash: str = ""

    def canonical_manifest(self) -> dict[str, Any]:
        """Produce the canonical, deterministic dictionary representation of the provenance manifest.

        Only non-empty, non-null fields that genuinely correspond to actual execution are included.
        """
        prompt_h = self.prompt_hash or (sha256_hash(self.prompt_template) if self.prompt_template else None)
        sys_prompt_h = self.system_prompt_hash or (sha256_hash(self.system_prompt) if self.system_prompt else None)
        payload = {
            "dataset_checksum": self.dataset_checksum,
            "dataset_id": self.dataset_id,
            "dataset_version": self.dataset_version,
            "rag_version": self.rag_version,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "model_parameters": self.model_parameters if self.model_parameters else None,
            "temperature": self.temperature,
            "prompt_hash": prompt_h,
            "system_prompt_hash": sys_prompt_h,
            "embedding_model": self.embedding_model,
            "embedding_version": self.embedding_version,
            "retriever_config": self.retriever_config if self.retriever_config else None,
            "reranker_config": self.reranker_config if self.reranker_config else None,
            "chunking_config": self.chunking_config if self.chunking_config else None,
            "adapter_type": self.adapter_type,
            "adapter_config": self.adapter_config if self.adapter_config else None,
            "evaluator_version": self.evaluator_version,
            "evaluation_config": self.evaluation_config if self.evaluation_config else None,
            "experiment_config": self.experiment_config if self.experiment_config else None,
            "python_version": self.python_version,
            "dependency_lock_hash": self.dependency_lock_hash,
            "environment_info": self.environment_info if self.environment_info else None,
            "random_seed": self.random_seed,
        }
        return {k: v for k, v in payload.items() if v not in (None, {}, "", [])}

    def compute_hash(self) -> str:
        """Compute standard cryptographic hash of the canonical manifest."""
        return sha256_hash(canonical_json(self.canonical_manifest()))

    def model_post_init(self, __context: Any) -> None:
        from rag_platform.security import SecretRedactor
        if self.adapter_config:
            self.adapter_config = SecretRedactor.redact_dict(self.adapter_config)
        if self.model_parameters and not self.model_config_hash:
            self.model_config_hash = sha256_hash(canonical_json(self.model_parameters))
        elif self.model_config_hash == "default_model_config_hash":
            self.model_config_hash = sha256_hash(canonical_json(self.model_parameters)) if self.model_parameters else ""
        if self.experiment_config and not self.experiment_hash:
            self.experiment_hash = sha256_hash(canonical_json(self.experiment_config))
        elif self.experiment_hash == "default_experiment_hash":
            self.experiment_hash = sha256_hash(canonical_json(self.experiment_config)) if self.experiment_config else ""
        if self.prompt_template and not self.prompt_hash:
            self.prompt_hash = sha256_hash(self.prompt_template)
        if self.system_prompt and not self.system_prompt_hash:
            self.system_prompt_hash = sha256_hash(self.system_prompt)
        if not self.dependency_lock_hash:
            try:
                from pathlib import Path
                repo_root = Path(__file__).resolve().parent.parent.parent
                candidate_paths = [
                    repo_root / "requirements.lock",
                    Path("requirements.lock"),
                ]
                for lock_file in candidate_paths:
                    if lock_file.is_file():
                        self.dependency_lock_hash = sha256_hash(lock_file.read_text(encoding="utf-8"))
                        break
            except Exception:
                pass
        if not self.manifest_hash:
            self.manifest_hash = self.compute_hash()


class RunOptions(BaseModel):
    max_cases: int | None = Field(default=500, ge=1, le=10000)
    concurrency: int = Field(default=5, ge=1, le=100)
    timeout_seconds: float = 60.0
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


class MetricRegressionPolicy(BaseModel):
    metric_name: str
    # Absolute candidate value bounds
    min_candidate_value: float | None = None
    max_candidate_value: float | None = None
    # Regression limits relative to baseline
    max_absolute_drop: float | None = None
    max_relative_drop_pct: float | None = None
    # Missing data handling (default is fail-closed)
    allow_missing: bool = False
    # Legacy aliases kept for backward compatibility
    min_absolute_score: float | None = None       # alias → min_candidate_value
    max_absolute_score: float | None = None       # alias → max_candidate_value
    max_degradation_pct: float | None = None      # alias → max_relative_drop_pct
    max_absolute_degradation: float | None = None # alias → max_absolute_drop

    def model_post_init(self, __context: Any) -> None:
        # Resolve aliases so callers can use either spelling
        if self.min_candidate_value is None and self.min_absolute_score is not None:
            self.min_candidate_value = self.min_absolute_score
        if self.max_candidate_value is None and self.max_absolute_score is not None:
            self.max_candidate_value = self.max_absolute_score
        if self.max_absolute_drop is None and self.max_absolute_degradation is not None:
            self.max_absolute_drop = self.max_absolute_degradation
        if self.max_relative_drop_pct is None and self.max_degradation_pct is not None:
            self.max_relative_drop_pct = self.max_degradation_pct


class ReleasePolicy(BaseModel):
    policy_id: str = "prod-default"
    min_faithfulness: float = 0.90
    min_retrieval_recall: float = 0.92
    min_citation_accuracy: float = 0.95
    max_hallucination_rate: float = 0.05
    min_abstention_accuracy: float = 0.90
    max_latency_regression_pct: float = 20.0
    max_cost_regression_pct: float = 25.0
    min_cost_budget_usd: float = 0.05
    max_critical_regressions: int = 0
    metric_policies: list[MetricRegressionPolicy] = Field(default_factory=list)


class GateViolation(BaseModel):
    metric_name: str
    baseline_value: float | None = None
    candidate_value: float | None = None
    threshold: float
    operator: str | None = None
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
