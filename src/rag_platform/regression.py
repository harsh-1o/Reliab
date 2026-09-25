"""Regression detection, baseline-candidate comparison, statistical deltas, and release policy gate evaluation.
"""

from __future__ import annotations

import math
from typing import Any
from pydantic import BaseModel, Field

from rag_platform.models import (
    GateResult,
    GateStatus,
    GateViolation,
    MetricRegressionPolicy,
    ReleasePolicy,
    RunMetricsSummary,
)


class MetricDelta(BaseModel):
    """Calculated difference between baseline and candidate for a single metric."""

    metric_name: str
    baseline: float
    candidate: float
    delta_abs: float
    delta_pct: float | None = None
    direction: str = "UNCHANGED"  # "IMPROVED", "REGRESSED", "UNCHANGED"
    sample_size_baseline: int = 0
    sample_size_candidate: int = 0


class RunComparison(BaseModel):
    """Full comparative analysis between baseline and candidate evaluation runs."""

    baseline_run_id: str
    candidate_run_id: str
    metric_deltas: dict[str, MetricDelta] = Field(default_factory=dict)
    newly_failed_cases: list[str] = Field(default_factory=list)
    recovered_cases: list[str] = Field(default_factory=list)
    unchanged_cases: list[str] = Field(default_factory=list)
    unchanged_pass_cases: list[str] = Field(default_factory=list)
    unchanged_fail_cases: list[str] = Field(default_factory=list)
    case_transitions: dict[str, dict[str, Any]] = Field(default_factory=dict)
    regressed_cases: list[str] = Field(default_factory=list)  # Backward-compatible alias
    improved_cases: list[str] = Field(default_factory=list)   # Backward-compatible alias
    net_case_drift: int = 0


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
        per_case_threshold: float = 0.15,
    ) -> RunComparison:
        """Calculate metric deltas and detect per-case regressions and recoveries."""
        deltas: dict[str, MetricDelta] = {}

        # Compare common metrics
        all_metrics = set(baseline_summary.metrics.keys()).union(candidate_summary.metrics.keys())
        for m_name in all_metrics:
            b_metric = baseline_summary.metrics.get(m_name)
            c_metric = candidate_summary.metrics.get(m_name)

            b_val = b_metric.mean if b_metric else 0.0
            c_val = c_metric.mean if c_metric else 0.0

            delta_abs = round(c_val - b_val, 4)

            # Mathematically sound delta percentage calculation:
            if abs(b_val) < 1e-6:
                delta_pct = None if abs(c_val) > 1e-6 else 0.0
            else:
                delta_pct = round(((c_val - b_val) / abs(b_val)) * 100, 2)

            # Direction determination based on metric family:
            is_lower_better = any(kw in m_name for kw in ("latency", "cost", "hallucination", "error"))
            if abs(delta_abs) < 0.005:
                direction = "UNCHANGED"
            elif is_lower_better:
                direction = "IMPROVED" if delta_abs < 0 else "REGRESSED"
            else:
                direction = "IMPROVED" if delta_abs > 0 else "REGRESSED"

            deltas[m_name] = MetricDelta(
                metric_name=m_name,
                baseline=round(b_val, 4),
                candidate=round(c_val, 4),
                delta_abs=delta_abs,
                delta_pct=delta_pct,
                direction=direction,
                sample_size_baseline=b_metric.count if b_metric else 0,
                sample_size_candidate=c_metric.count if c_metric else 0,
            )

        # Per-case 4-way classification: NEW_FAILURE, RECOVERED, UNCHANGED_PASS, UNCHANGED_FAIL
        newly_failed: list[str] = []
        recovered: list[str] = []
        unchanged: list[str] = []
        unchanged_pass: list[str] = []
        unchanged_fail: list[str] = []
        transitions: dict[str, dict[str, Any]] = {}

        if baseline_case_scores and candidate_case_scores:
            common_cases = set(baseline_case_scores.keys()).intersection(candidate_case_scores.keys())
            for cid in common_cases:
                b_score = baseline_case_scores[cid]
                c_score = candidate_case_scores[cid]
                score_drop = round(b_score - c_score, 4)

                b_pass = b_score >= 0.70
                c_pass = c_score >= 0.70

                if (b_pass and not c_pass) or score_drop >= per_case_threshold:
                    trans_status = "NEW_FAILURE"
                    newly_failed.append(cid)
                elif (not b_pass and c_pass) or (c_score - b_score) >= per_case_threshold:
                    trans_status = "RECOVERED"
                    recovered.append(cid)
                elif b_pass and c_pass:
                    trans_status = "UNCHANGED_PASS"
                    unchanged_pass.append(cid)
                    unchanged.append(cid)
                else:
                    trans_status = "UNCHANGED_FAIL"
                    unchanged_fail.append(cid)
                    unchanged.append(cid)

                transitions[cid] = {
                    "case_id": cid,
                    "baseline_score": b_score,
                    "candidate_score": c_score,
                    "score_delta": round(c_score - b_score, 4),
                    "transition": trans_status,
                }

        return RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            metric_deltas=deltas,
            newly_failed_cases=newly_failed,
            recovered_cases=recovered,
            unchanged_cases=unchanged,
            unchanged_pass_cases=unchanged_pass,
            unchanged_fail_cases=unchanged_fail,
            case_transitions=transitions,
            regressed_cases=newly_failed,
            improved_cases=recovered,
            net_case_drift=len(recovered) - len(newly_failed),
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
                    message=f"Faithfulness {faith_val:.3f} fell below required minimum {policy.min_faithfulness:.3f}.",
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
                    message=f"Recall@5 {recall_val:.3f} fell below required minimum {policy.min_retrieval_recall:.3f}.",
                )
            )

        # 3. Citation Accuracy Threshold
        cand_cit = candidate.metrics.get("citation_accuracy")
        if cand_cit and cand_cit.mean < policy.min_citation_accuracy:
            violations.append(
                GateViolation(
                    metric_name="citation_accuracy",
                    candidate_value=cand_cit.mean,
                    threshold=policy.min_citation_accuracy,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Citation accuracy {cand_cit.mean:.3f} fell below required {policy.min_citation_accuracy:.3f}.",
                )
            )

        # 4. Hallucination Rate Cap
        if candidate.hallucination_rate > policy.max_hallucination_rate:
            violations.append(
                GateViolation(
                    metric_name="hallucination_rate",
                    candidate_value=candidate.hallucination_rate,
                    threshold=policy.max_hallucination_rate,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Hallucination rate {candidate.hallucination_rate:.3f} exceeded maximum ceiling {policy.max_hallucination_rate:.3f}.",
                )
            )

        # 5. Abstention Accuracy Threshold
        if candidate.abstention_accuracy < policy.min_abstention_accuracy:
            violations.append(
                GateViolation(
                    metric_name="abstention_accuracy",
                    candidate_value=candidate.abstention_accuracy,
                    threshold=policy.min_abstention_accuracy,
                    violation_type="THRESHOLD_BREACH",
                    message=f"Abstention accuracy {candidate.abstention_accuracy:.3f} fell below required {policy.min_abstention_accuracy:.3f}.",
                )
            )

        # 6. Baseline Regression Budgets (Latency & Cost with configurable absolute floor)
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
                        message=f"P95 latency {candidate.p95_latency_ms:.1f}ms regressed more than {policy.max_latency_regression_pct}% over baseline ({baseline.p95_latency_ms:.1f}ms).",
                    )
                )

            # Cost regression budget: max(absolute_minimum, baseline * multiplier)
            cost_ceiling = max(
                policy.min_cost_budget_usd,
                baseline.total_cost_usd * (1.0 + (policy.max_cost_regression_pct / 100.0)),
            )
            if candidate.total_cost_usd > cost_ceiling:
                violations.append(
                    GateViolation(
                        metric_name="total_cost_usd",
                        baseline_value=baseline.total_cost_usd,
                        candidate_value=candidate.total_cost_usd,
                        threshold=round(cost_ceiling, 4),
                        violation_type="REGRESSION_BUDGET",
                        message=f"Cost ${candidate.total_cost_usd:.4f} exceeded allowed budget ceiling of ${cost_ceiling:.4f}.",
                    )
                )

        # 7. Configurable Metric-Specific Regression Policies
        for mp in policy.metric_policies:
            c_entry = candidate.metrics.get(mp.metric_name)
            c_val = c_entry.mean if c_entry else 0.0

            if mp.min_candidate_value is not None and c_val < mp.min_candidate_value:
                violations.append(
                    GateViolation(
                        metric_name=mp.metric_name,
                        candidate_value=c_val,
                        threshold=mp.min_candidate_value,
                        violation_type="METRIC_POLICY_BREACH",
                        message=f"{mp.metric_name} value {c_val:.4f} fell below required minimum {mp.min_candidate_value:.4f}.",
                    )
                )

            if mp.max_candidate_value is not None and c_val > mp.max_candidate_value:
                violations.append(
                    GateViolation(
                        metric_name=mp.metric_name,
                        candidate_value=c_val,
                        threshold=mp.max_candidate_value,
                        violation_type="METRIC_POLICY_BREACH",
                        message=f"{mp.metric_name} value {c_val:.4f} exceeded allowed maximum {mp.max_candidate_value:.4f}.",
                    )
                )

            if baseline:
                b_entry = baseline.metrics.get(mp.metric_name)
                if b_entry:
                    b_val = b_entry.mean
                    drop_abs = b_val - c_val
                    if mp.max_absolute_drop is not None and drop_abs > mp.max_absolute_drop:
                        violations.append(
                            GateViolation(
                                metric_name=mp.metric_name,
                                baseline_value=b_val,
                                candidate_value=c_val,
                                threshold=mp.max_absolute_drop,
                                violation_type="METRIC_POLICY_BREACH",
                                message=f"{mp.metric_name} dropped by {drop_abs:.4f}, exceeding allowed absolute delta {mp.max_absolute_drop:.4f}.",
                            )
                        )
                    if mp.max_relative_drop_pct is not None and b_val > 1e-6:
                        rel_drop = (drop_abs / b_val) * 100.0
                        if rel_drop > mp.max_relative_drop_pct:
                            violations.append(
                                GateViolation(
                                    metric_name=mp.metric_name,
                                    baseline_value=b_val,
                                    candidate_value=c_val,
                                    threshold=mp.max_relative_drop_pct,
                                    violation_type="METRIC_POLICY_BREACH",
                                    message=f"{mp.metric_name} dropped by {rel_drop:.2f}%, exceeding allowed relative delta {mp.max_relative_drop_pct:.2f}%.",
                                )
                            )

        # 8. Critical Regressions Cap
        critical_count = len(comparison.newly_failed_cases) if comparison else 0
        if critical_count > policy.max_critical_regressions:
            violations.append(
                GateViolation(
                    metric_name="critical_regressions",
                    candidate_value=float(critical_count),
                    threshold=float(policy.max_critical_regressions),
                    violation_type="CRITICAL_FAILURE",
                    message=f"{critical_count} critical test case regression(s) detected (maximum allowed: {policy.max_critical_regressions}).",
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
