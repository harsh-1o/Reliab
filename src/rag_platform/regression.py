"""Regression detection, baseline-candidate comparison, statistical deltas, and release policy gate evaluation.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rag_platform.models import (
    GateResult,
    GateStatus,
    GateViolation,
    ReleasePolicy,
    RunMetricsSummary,
)


class MetricDelta(BaseModel):
    """Calculated difference between baseline and candidate for a single metric."""

    metric_name: str
    baseline: float | None = None
    candidate: float | None = None
    delta_abs: float | None = None
    delta_pct: float | None = None
    direction: str = "UNCHANGED"  # "IMPROVED", "REGRESSED", "UNCHANGED", "NOT_APPLICABLE", "BASELINE_MISSING", "CANDIDATE_MISSING"
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
    candidate_missing_cases: list[str] = Field(default_factory=list)
    baseline_missing_cases: list[str] = Field(default_factory=list)
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
        baseline_case_scores: dict[str, float | None] | None = None,
        candidate_case_scores: dict[str, float | None] | None = None,
        per_case_threshold: float = 0.15,
    ) -> RunComparison:
        """Calculate metric deltas and detect per-case regressions and recoveries."""
        deltas: dict[str, MetricDelta] = {}

        # Compare common metrics
        all_metrics = set(baseline_summary.metrics.keys()).union(candidate_summary.metrics.keys())
        for m_name in all_metrics:
            b_metric = baseline_summary.metrics.get(m_name)
            c_metric = candidate_summary.metrics.get(m_name)

            b_has = b_metric is not None and b_metric.count > 0
            c_has = c_metric is not None and c_metric.count > 0

            if not b_has and not c_has:
                deltas[m_name] = MetricDelta(
                    metric_name=m_name,
                    baseline=None,
                    candidate=None,
                    delta_abs=None,
                    delta_pct=None,
                    direction="NOT_APPLICABLE",
                    sample_size_baseline=0,
                    sample_size_candidate=0,
                )
                continue

            if not b_has and c_has:
                assert c_metric is not None
                deltas[m_name] = MetricDelta(
                    metric_name=m_name,
                    baseline=None,
                    candidate=round(c_metric.mean, 4),
                    delta_abs=None,
                    delta_pct=None,
                    direction="BASELINE_MISSING",
                    sample_size_baseline=0,
                    sample_size_candidate=c_metric.count,
                )
                continue

            if b_has and not c_has:
                assert b_metric is not None
                deltas[m_name] = MetricDelta(
                    metric_name=m_name,
                    baseline=round(b_metric.mean, 4),
                    candidate=None,
                    delta_abs=None,
                    delta_pct=None,
                    direction="CANDIDATE_MISSING",
                    sample_size_baseline=b_metric.count,
                    sample_size_candidate=0,
                )
                continue

            # Both have measurements
            assert b_metric is not None and c_metric is not None
            b_val = b_metric.mean
            c_val = c_metric.mean

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
        candidate_missing: list[str] = []
        baseline_missing: list[str] = []
        transitions: dict[str, dict[str, Any]] = {}

        if baseline_case_scores is not None and candidate_case_scores is not None:
            common_cases = sorted(set(baseline_case_scores.keys()).union(candidate_case_scores.keys()))
            for cid in common_cases:
                b_score = baseline_case_scores.get(cid)
                c_score = candidate_case_scores.get(cid)

                if b_score is None and c_score is None:
                    trans_status = "NOT_APPLICABLE"
                    score_delta = None
                elif b_score is None:
                    trans_status = "BASELINE_MISSING"
                    score_delta = None
                    baseline_missing.append(cid)
                elif c_score is None:
                    trans_status = "CANDIDATE_MISSING"
                    score_delta = None
                    candidate_missing.append(cid)
                else:
                    score_drop = round(b_score - c_score, 4)
                    score_delta = round(c_score - b_score, 4)

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
                    "score_delta": score_delta,
                    "transition": trans_status,
                }

        return RunComparison(
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            metric_deltas=deltas,
            newly_failed_cases=sorted(newly_failed),
            recovered_cases=sorted(recovered),
            unchanged_cases=sorted(unchanged),
            unchanged_pass_cases=sorted(unchanged_pass),
            unchanged_fail_cases=sorted(unchanged_fail),
            candidate_missing_cases=sorted(candidate_missing),
            baseline_missing_cases=sorted(baseline_missing),
            case_transitions=transitions,
            regressed_cases=sorted(newly_failed),
            improved_cases=sorted(recovered),
            net_case_drift=len(recovered) - len(newly_failed),
        )

    def evaluate_gate(
        self,
        candidate: RunMetricsSummary | None = None,
        policy: ReleasePolicy | None = None,
        candidate_run_id: str = "candidate",
        baseline: RunMetricsSummary | None = None,
        baseline_run_id: str | None = None,
        comparison: RunComparison | None = None,
        candidate_summary: RunMetricsSummary | None = None,
        baseline_summary: RunMetricsSummary | None = None,
    ) -> GateResult:
        """Evaluate composite release policy against candidate run metrics and baseline."""
        cand = candidate or candidate_summary
        if cand is None:
            raise ValueError("Candidate RunMetricsSummary is required for gate evaluation.")
        candidate = cand
        if baseline is None and baseline_summary is not None:
            baseline = baseline_summary
        if policy is None:
            policy = ReleasePolicy()
        violations: list[GateViolation] = []

        # 0. Case Coverage Evaluation (Requirement 1, 2)
        min_cov = getattr(policy, "min_case_coverage", 1.0)
        if candidate.coverage_ratio < min_cov:
            violations.append(
                GateViolation(
                    metric_name="case_coverage",
                    candidate_value=candidate.coverage_ratio,
                    threshold=min_cov,
                    operator=">=",
                    violation_type="INSUFFICIENT_COVERAGE",
                    message=(
                        f"Evaluation coverage {candidate.coverage_ratio * 100:.1f}% "
                        f"({candidate.evaluated_case_count}/{candidate.required_case_count} cases) fell below "
                        f"required release threshold of {min_cov * 100:.1f}%. Partial/smoke runs cannot pass release gating."
                    ),
                )
            )

        # Baseline Test Case Parity / Candidate Missing Check (Requirement 3)
        if comparison and comparison.candidate_missing_cases:
            if not getattr(policy, "allow_candidate_missing", False):
                missing_sample = sorted(comparison.candidate_missing_cases)[:5]
                violations.append(
                    GateViolation(
                        metric_name="candidate_case_coverage",
                        candidate_value=float(len(comparison.candidate_missing_cases)),
                        threshold=0.0,
                        operator="<=",
                        violation_type="CANDIDATE_MISSING",
                        message=(
                            f"Candidate omitted {len(comparison.candidate_missing_cases)} required test case(s) "
                            f"present in baseline (e.g. {missing_sample}). Baseline cases must not be omitted."
                        ),
                    )
                )

        # 1. Faithfulness Threshold

        cand_faith = candidate.metrics.get("faithfulness")
        if cand_faith is None or cand_faith.count == 0:
            violations.append(
                GateViolation(
                    metric_name="faithfulness",
                    candidate_value=None,
                    threshold=policy.min_faithfulness,
                    operator=">=",
                    violation_type="MISSING_DATA",
                    message=f"Faithfulness metric is missing or unmeasured; cannot satisfy required minimum {policy.min_faithfulness:.3f}.",
                )
            )
        elif cand_faith.mean < policy.min_faithfulness:
            violations.append(
                GateViolation(
                    metric_name="faithfulness",
                    candidate_value=cand_faith.mean,
                    threshold=policy.min_faithfulness,
                    operator=">=",
                    violation_type="THRESHOLD_BREACH",
                    message=f"Faithfulness {cand_faith.mean:.3f} fell below required minimum {policy.min_faithfulness:.3f}.",
                )
            )

        # 2. Retrieval Recall Threshold
        cand_recall = candidate.metrics.get("recall_at_5")
        if cand_recall is None or cand_recall.count == 0:
            violations.append(
                GateViolation(
                    metric_name="recall_at_5",
                    candidate_value=None,
                    threshold=policy.min_retrieval_recall,
                    operator=">=",
                    violation_type="MISSING_DATA",
                    message=f"Recall@5 metric is missing or unmeasured; cannot satisfy required minimum {policy.min_retrieval_recall:.3f}.",
                )
            )
        elif cand_recall.mean < policy.min_retrieval_recall:
            violations.append(
                GateViolation(
                    metric_name="recall_at_5",
                    candidate_value=cand_recall.mean,
                    threshold=policy.min_retrieval_recall,
                    operator=">=",
                    violation_type="THRESHOLD_BREACH",
                    message=f"Recall@5 {cand_recall.mean:.3f} fell below required minimum {policy.min_retrieval_recall:.3f}.",
                )
            )

        # 3. Citation Accuracy Threshold
        cand_cit = candidate.metrics.get("citation_accuracy")
        if cand_cit is None or cand_cit.count == 0:
            if getattr(policy, "min_citation_accuracy", None) is not None:
                violations.append(
                    GateViolation(
                        metric_name="citation_accuracy",
                        candidate_value=None,
                        threshold=policy.min_citation_accuracy,
                        operator=">=",
                        violation_type="MISSING_DATA",
                        message=f"Citation accuracy metric is missing or unmeasured; cannot satisfy required minimum {policy.min_citation_accuracy:.3f}.",
                    )
                )
        elif cand_cit.mean < policy.min_citation_accuracy:
            violations.append(
                GateViolation(
                    metric_name="citation_accuracy",
                    candidate_value=cand_cit.mean,
                    threshold=policy.min_citation_accuracy,
                    operator=">=",
                    violation_type="THRESHOLD_BREACH",
                    message=f"Citation accuracy {cand_cit.mean:.3f} fell below required {policy.min_citation_accuracy:.3f}.",
                )
            )

        # 4. Hallucination Rate Cap
        if candidate.hallucination_rate is not None and candidate.hallucination_rate > policy.max_hallucination_rate:
            violations.append(
                GateViolation(
                    metric_name="hallucination_rate",
                    candidate_value=candidate.hallucination_rate,
                    threshold=policy.max_hallucination_rate,
                    operator="<=",
                    violation_type="THRESHOLD_BREACH",
                    message=f"Hallucination rate {candidate.hallucination_rate:.3f} exceeded maximum ceiling {policy.max_hallucination_rate:.3f}.",
                )
            )

        # 5. Abstention Accuracy Threshold
        if candidate.abstention_accuracy is not None and candidate.abstention_accuracy < policy.min_abstention_accuracy:
            violations.append(
                GateViolation(
                    metric_name="abstention_accuracy",
                    candidate_value=candidate.abstention_accuracy,
                    threshold=policy.min_abstention_accuracy,
                    operator=">=",
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
                        operator="<=",
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
                        operator="<=",
                        violation_type="REGRESSION_BUDGET",
                        message=f"Cost ${candidate.total_cost_usd:.4f} exceeded allowed budget ceiling of ${cost_ceiling:.4f}.",
                    )
                )

        # 7. Configurable Metric-Specific Regression Policies
        for mp in policy.metric_policies:
            c_entry = candidate.metrics.get(mp.metric_name)
            if c_entry is None or c_entry.count == 0:
                if not getattr(mp, "allow_missing", False):
                    violations.append(
                        GateViolation(
                            metric_name=mp.metric_name,
                            candidate_value=None,
                            threshold=mp.min_candidate_value if mp.min_candidate_value is not None else (mp.max_candidate_value or 0.0),
                            operator=">=" if mp.min_candidate_value is not None else "<=",
                            violation_type="MISSING_DATA",
                            message=f"Metric '{mp.metric_name}' is missing or unmeasured; cannot satisfy release policy.",
                        )
                    )
                continue

            c_val = c_entry.mean

            if mp.min_candidate_value is not None and c_val < mp.min_candidate_value:
                violations.append(
                    GateViolation(
                        metric_name=mp.metric_name,
                        candidate_value=c_val,
                        threshold=mp.min_candidate_value,
                        operator=">=",
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
                        operator="<=",
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
