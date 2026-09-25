"""Regression detection, baseline-candidate comparison, and release policy gate evaluation.

# ponytail: single file covers delta calculations, per-case regressions, and gate evaluation.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from rag_platform.models import (
    GateResult,
    GateStatus,
    GateViolation,
    RagTrace,
    ReleasePolicy,
    RunMetricsSummary,
)


class MetricDelta(BaseModel):
    """Calculated difference between baseline and candidate for a single metric."""

    metric_name: str
    baseline: float
    candidate: float
    delta_abs: float
    delta_pct: float


class RunComparison(BaseModel):
    """Full comparative analysis between baseline and candidate evaluation runs."""

    baseline_run_id: str
    candidate_run_id: str
    metric_deltas: dict[str, MetricDelta] = Field(default_factory=dict)
    regressed_cases: list[str] = Field(default_factory=list)
    improved_cases: list[str] = Field(default_factory=list)


class RegressionEngine:
    """Evaluates regressions against baselines and applies composite release policies."""

    def compare(
        self,
        baseline_summary: RunMetricsSummary,
        candidate_summary: RunMetricsSummary,
        baseline_run_id: str,
        candidate_run_id: str,
        baseline_case_scores: dict[str, float] | None = None,
        candidate_case_scores: dict[str, float] | None = None,
    ) -> RunComparison:
        """Calculate metric deltas and detect per-case regressions."""
        deltas: dict[str, MetricDelta] = {}

        # Compare common metrics
        all_metrics = set(baseline_summary.metrics.keys()).union(candidate_summary.metrics.keys())
        for m_name in all_metrics:
            b_val = baseline_summary.metrics[m_name].mean if m_name in baseline_summary.metrics else 0.0
            c_val = candidate_summary.metrics[m_name].mean if m_name in candidate_summary.metrics else 0.0

            delta_abs = round(c_val - b_val, 4)
            delta_pct = round(((c_val - b_val) / b_val * 100), 2) if b_val > 0 else (100.0 if c_val > 0 else 0.0)

            deltas[m_name] = MetricDelta(
                metric_name=m_name,
                baseline=round(b_val, 4),
                candidate=round(c_val, 4),
                delta_abs=delta_abs,
                delta_pct=delta_pct,
            )

        # Per-case regressions (passed in baseline but failed in candidate)
        regressed: list[str] = []
        improved: list[str] = []
        if baseline_case_scores and candidate_case_scores:
            for case_id, b_score in baseline_case_scores.items():
                if case_id in candidate_case_scores:
                    c_score = candidate_case_scores[case_id]
                    if b_score >= 0.80 and c_score < 0.50:
                        regressed.append(case_id)
                    elif b_score < 0.50 and c_score >= 0.80:
                        improved.append(case_id)

        return RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            metric_deltas=deltas,
            regressed_cases=regressed,
            improved_cases=improved,
        )

    def evaluate_gate(
        self,
        candidate: RunMetricsSummary,
        policy: ReleasePolicy,
        candidate_run_id: str,
        baseline: RunMetricsSummary | None = None,
        baseline_run_id: str | None = None,
        comparison: RunComparison | None = None,
    ) -> GateResult:
        """Evaluate composite release policy against candidate run metrics and baseline."""
        violations: list[GateViolation] = []

        # 1. Faithfulness Threshold
        cand_faith = candidate.metrics.get("faithfulness")
        faith_val = cand_faith.mean if cand_faith else 0.0
        if faith_val < policy.min_faithfulness:
            violations.append(
                GateViolation(
                    metric_name="faithfulness",
                    candidate_value=faith_val,
                    threshold=policy.min_faithfulness,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Faithfulness {faith_val} fell below required minimum {policy.min_faithfulness}.",
                )
            )

        # 2. Retrieval Recall Threshold
        cand_recall = candidate.metrics.get("recall_at_5")
        recall_val = cand_recall.mean if cand_recall else 0.0
        if recall_val < policy.min_retrieval_recall:
            violations.append(
                GateViolation(
                    metric_name="recall_at_5",
                    candidate_value=recall_val,
                    threshold=policy.min_retrieval_recall,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Recall@5 {recall_val} fell below required minimum {policy.min_retrieval_recall}.",
                )
            )

        # 3. Hallucination Rate Cap
        if candidate.hallucination_rate > policy.max_hallucination_rate:
            violations.append(
                GateViolation(
                    metric_name="hallucination_rate",
                    candidate_value=candidate.hallucination_rate,
                    threshold=policy.max_hallucination_rate,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Hallucination rate {candidate.hallucination_rate} exceeded maximum ceiling {policy.max_hallucination_rate}.",
                )
            )

        # 4. Abstention Accuracy Threshold
        if candidate.abstention_accuracy < policy.min_abstention_accuracy:
            violations.append(
                GateViolation(
                    metric_name="abstention_accuracy",
                    candidate_value=candidate.abstention_accuracy,
                    threshold=policy.min_abstention_accuracy,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Abstention accuracy {candidate.abstention_accuracy} fell below required {policy.min_abstention_accuracy}.",
                )
            )

        # 5. Baseline Regression Budgets (Latency & Cost)
        if baseline:
            # Latency regression budget
            max_lat = baseline.p95_latency_ms * (1.0 + (policy.max_latency_regression_pct / 100.0))
            if candidate.p95_latency_ms > max_lat:
                violations.append(
                    GateViolation(
                        metric_name="p95_latency_ms",
                        baseline_value=baseline.p95_latency_ms,
                        candidate_value=candidate.p95_latency_ms,
                        threshold=round(max_lat, 2),
                        violation_type="REGRESSION_BUDGET",
                        message=f"P95 latency {candidate.p95_latency_ms}ms regressed more than {policy.max_latency_regression_pct}% over baseline ({baseline.p95_latency_ms}ms).",
                    )
                )

            # Cost regression budget
            max_cost = baseline.total_cost_usd * (1.0 + (policy.max_cost_regression_pct / 100.0))
            if candidate.total_cost_usd > max_cost:
                violations.append(
                    GateViolation(
                        metric_name="total_cost_usd",
                        baseline_value=baseline.total_cost_usd,
                        candidate_value=candidate.total_cost_usd,
                        threshold=round(max_cost, 4),
                        violation_type="REGRESSION_BUDGET",
                        message=f"Cost ${candidate.total_cost_usd} regressed more than {policy.max_cost_regression_pct}% over baseline (${baseline.total_cost_usd}).",
                    )
                )

        # 6. Critical Regressions Cap
        critical_count = len(comparison.regressed_cases) if comparison else 0
        if critical_count > policy.max_critical_regressions:
            violations.append(
                GateViolation(
                    metric_name="critical_regressions",
                    candidate_value=float(critical_count),
                    threshold=float(policy.max_critical_regressions),
                    violation_type="CRITICAL_FAILURE",
                    message=f"{critical_count} critical test case regressions detected (maximum allowed: {policy.max_critical_regressions}).",
                )
            )

        status = GateStatus.PASS if not violations else GateStatus.FAIL

        return GateResult(
            status=status,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            policy_id=policy.policy_id,
            violations=violations,
            critical_regressions=critical_count,
            report_url=f"/reports/{candidate_run_id}.html",
        )
