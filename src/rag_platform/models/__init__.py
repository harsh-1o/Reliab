"""Canonical domain models and data contracts."""

from rag_platform.models.enums import (
    Answerability,
    DatasetStatus,
    FailureCode,
    GateStatus,
    MetricFamily,
    RunStatus,
    Severity,
)
from rag_platform.models.failure import FailureAttribution, HumanOverride
from rag_platform.models.metric import MetricResult, MetricSummary, RunMetricsSummary
from rag_platform.models.run import (
    GateResult,
    GateViolation,
    ReleasePolicy,
    RunConfig,
    RunOptions,
    RunProvenance,
)
from rag_platform.models.test_case import (
    BenchmarkDataset,
    DocumentReference,
    TestCase,
    compute_dataset_checksum,
)
from rag_platform.models.trace import Citation, RagTrace, RetrievedChunk

__all__ = [
    "RunStatus",
    "DatasetStatus",
    "Answerability",
    "MetricFamily",
    "FailureCode",
    "Severity",
    "GateStatus",
    "DocumentReference",
    "TestCase",
    "BenchmarkDataset",
    "compute_dataset_checksum",
    "RetrievedChunk",
    "Citation",
    "RagTrace",
    "MetricResult",
    "MetricSummary",
    "RunMetricsSummary",
    "FailureAttribution",
    "HumanOverride",
    "RunProvenance",
    "RunOptions",
    "RunConfig",
    "ReleasePolicy",
    "GateViolation",
    "GateResult",
]
