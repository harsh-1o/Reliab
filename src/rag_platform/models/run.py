"""Evaluation run, configuration, provenance manifest, and release policy schemas."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field

from rag_platform.common.crypto import compute_manifest_hash
from rag_platform.models.enums import GateStatus, RunStatus


class RunProvenance(BaseModel):
    """The six-dimension provenance envelope guaranteeing bitwise reproducibility."""

    dataset_checksum: str
    rag_version: str
    model_config_hash: str
    prompt_hash: str
    evaluator_version: str
    experiment_hash: str
    manifest_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute the deterministic manifest hash if not already set."""
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
    """Execution options for an evaluation run."""

    max_cases: int | None = None
    concurrency: int = 5
    timeout_seconds: int = 60
    fail_fast: bool = False
    use_cache: bool = True


class RunConfig(BaseModel):
    """Parameters submitted to initiate an evaluation run."""

    project_id: str
    dataset_id: str
    dataset_version: str
    system_version: str
    policy_id: str = "prod-default"
    suite: str = "full"
    options: RunOptions = Field(default_factory=RunOptions)


class ReleasePolicy(BaseModel):
    """Configurable release quality gate thresholds and regression budgets."""

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
    """Details of a single violated threshold or regression budget."""

    metric_name: str
    baseline_value: float | None = None
    candidate_value: float
    threshold: float
    violation_type: str  # THRESHOLD_BREACH | REGRESSION_BUDGET | CRITICAL_FAILURE
    message: str


class GateResult(BaseModel):
    """Machine-readable CI/CD release gate decision."""

    status: GateStatus
    baseline_run_id: str | None = None
    candidate_run_id: str
    policy_id: str
    violations: list[GateViolation] = Field(default_factory=list)
    critical_regressions: int = 0
    report_url: str | None = None
    evaluated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
