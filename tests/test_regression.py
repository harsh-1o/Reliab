"""Unit tests for Phase 7: regression engine, delta calculations, and release policy gates."""

from __future__ import annotations

import pytest

from rag_platform.models import (
    GateStatus,
    MetricFamily,
    MetricSummary,
    ReleasePolicy,
    RunMetricsSummary,
)
from rag_platform.regression import RegressionEngine


@pytest.fixture
def regression_engine() -> RegressionEngine:
    return RegressionEngine()


@pytest.fixture
def baseline_summary() -> RunMetricsSummary:
    return RunMetricsSummary(
        metrics={
            "faithfulness": MetricSummary(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, mean=0.94, p50=0.95, p95=0.98, min=0.8, max=1.0, count=100),
            "recall_at_5": MetricSummary(metric_name="recall_at_5", metric_family=MetricFamily.RETRIEVAL, mean=0.95, p50=1.0, p95=1.0, min=0.8, max=1.0, count=100),
            "citation_accuracy": MetricSummary(metric_name="citation_accuracy", metric_family=MetricFamily.CITATION, mean=0.97, p50=1.0, p95=1.0, min=0.8, max=1.0, count=100),
        },
        total_cases=100,
        scored_cases=100,
        hallucination_rate=0.02,
        abstention_accuracy=0.95,
        p95_latency_ms=200.0,
        total_cost_usd=0.50,
    )


@pytest.fixture
def compliant_candidate() -> RunMetricsSummary:
    return RunMetricsSummary(
        metrics={
            "faithfulness": MetricSummary(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, mean=0.93, p50=0.94, p95=0.98, min=0.8, max=1.0, count=100),
            "recall_at_5": MetricSummary(metric_name="recall_at_5", metric_family=MetricFamily.RETRIEVAL, mean=0.94, p50=1.0, p95=1.0, min=0.8, max=1.0, count=100),
            "citation_accuracy": MetricSummary(metric_name="citation_accuracy", metric_family=MetricFamily.CITATION, mean=0.96, p50=1.0, p95=1.0, min=0.8, max=1.0, count=100),
        },
        total_cases=100,
        scored_cases=100,
        hallucination_rate=0.03,
        abstention_accuracy=0.92,
        p95_latency_ms=210.0,  # +5% (allowed under 20% budget)
        total_cost_usd=0.55,   # +10% (allowed under 25% budget)
    )


def test_compare_runs_delta_calculation(regression_engine: RegressionEngine, baseline_summary, compliant_candidate):
    b_scores = {"c1": 0.90, "c2": 0.85, "c3": 0.95}
    c_scores = {"c1": 0.92, "c2": 0.40, "c3": 0.90}  # c2 regressed!

    comparison = regression_engine.compare(
        baseline_summary=baseline_summary,
        candidate_summary=compliant_candidate,
        baseline_run_id="run_b",
        candidate_run_id="run_c",
        baseline_case_scores=b_scores,
        candidate_case_scores=c_scores,
    )

    assert "faithfulness" in comparison.metric_deltas
    assert comparison.metric_deltas["faithfulness"].delta_abs == -0.01
    assert comparison.regressed_cases == ["c2"]


def test_evaluate_gate_passes_compliant_candidate(regression_engine: RegressionEngine, baseline_summary, compliant_candidate):
    policy = ReleasePolicy()
    result = regression_engine.evaluate_gate(
        candidate=compliant_candidate,
        baseline=baseline_summary,
        policy=policy,
        candidate_run_id="run_c",
        baseline_run_id="run_b",
    )

    assert result.status == GateStatus.PASS
    assert len(result.violations) == 0


def test_evaluate_gate_fails_on_hallucination_regression(regression_engine: RegressionEngine, baseline_summary, compliant_candidate):
    policy = ReleasePolicy(max_hallucination_rate=0.05)
    # Intentionally corrupt candidate with 15% hallucinations
    bad_candidate = compliant_candidate.model_copy(update={"hallucination_rate": 0.15})

    result = regression_engine.evaluate_gate(
        candidate=bad_candidate,
        baseline=baseline_summary,
        policy=policy,
        candidate_run_id="run_c",
        baseline_run_id="run_b",
    )

    assert result.status == GateStatus.FAIL
    assert any(v.metric_name == "hallucination_rate" for v in result.violations)


def test_evaluate_gate_fails_on_latency_budget_breach(regression_engine: RegressionEngine, baseline_summary, compliant_candidate):
    policy = ReleasePolicy(max_latency_regression_pct=20.0)
    # Baseline p95 is 200ms -> max allowed is 240ms. Jump to 350ms (+75% regression)
    slow_candidate = compliant_candidate.model_copy(update={"p95_latency_ms": 350.0})

    result = regression_engine.evaluate_gate(
        candidate=slow_candidate,
        baseline=baseline_summary,
        policy=policy,
        candidate_run_id="run_c",
        baseline_run_id="run_b",
    )

    assert result.status == GateStatus.FAIL
    assert any(v.metric_name == "p95_latency_ms" for v in result.violations)
