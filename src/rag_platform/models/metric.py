"""Evaluation metric output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_platform.models.enums import MetricFamily


class MetricResult(BaseModel):
    """Atomic score emitted by a metric evaluator for a single trace."""

    metric_name: str
    metric_family: MetricFamily
    score: float
    reason: str | None = None
    evaluator_version: str = "1.0.0"
    cached: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class MetricSummary(BaseModel):
    """Aggregated distribution for a single metric across an evaluation run."""

    metric_name: str
    metric_family: MetricFamily
    mean: float
    p50: float
    p95: float
    min: float
    max: float
    count: int


class RunMetricsSummary(BaseModel):
    """Full aggregate metrics summary for a completed evaluation run."""

    metrics: dict[str, MetricSummary] = Field(default_factory=dict)
    total_cases: int = 0
    scored_cases: int = 0
    hallucination_rate: float = 0.0
    abstention_accuracy: float = 1.0
    infra_error_count: int = 0
    p95_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
