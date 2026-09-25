"""Metric evaluation engine: deterministic retrieval, generation, citation, and abstention metrics.

# ponytail: single file for all metric plugins, hashing cache, and run aggregation.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any

from rag_platform.core import canonical_json, sha256_hash
from rag_platform.models import (
    Answerability,
    MetricFamily,
    MetricResult,
    MetricSummary,
    RagTrace,
    RunMetricsSummary,
    TestCase,
)


class BaseMetric(ABC):
    """Abstract base metric evaluator."""

    name: str
    family: MetricFamily
    version: str = "1.0.0"

    @abstractmethod
    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        """Compute atomic metric score for a given trace and test case."""
        ...


# --- Retrieval Metrics ---
class RecallAtKMetric(BaseMetric):
    """Measures whether expected evidence was retrieved in top-K."""

    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.name = f"recall_at_{k}"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No relevant documents specified in golden set.",
                evaluator_version=self.version,
            )

        gold_doc_ids = {d.document_id for d in case.relevant_documents}
        retrieved_k = trace.retrieved_chunks[: self.k]
        found_doc_ids = {c.document_id for c in retrieved_k}

        intersection = gold_doc_ids.intersection(found_doc_ids)
        score = len(intersection) / len(gold_doc_ids)

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            reason=f"Found {len(intersection)}/{len(gold_doc_ids)} gold documents in top-{self.k}.",
            evaluator_version=self.version,
        )


class MeanReciprocalRankMetric(BaseMetric):
    """Computes Reciprocal Rank (1/rank) for the highest-ranked gold document."""

    def __init__(self) -> None:
        self.name = "mrr"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No golden documents required.",
                evaluator_version=self.version,
            )

        gold_ids = {d.document_id for d in case.relevant_documents}
        for chunk in trace.retrieved_chunks:
            if chunk.document_id in gold_ids:
                rr = 1.0 / chunk.rank
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=round(rr, 4),
                    reason=f"First relevant document '{chunk.document_id}' found at rank {chunk.rank}.",
                    evaluator_version=self.version,
                )

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=0.0,
            reason="No gold documents found in retrieved chunks.",
            evaluator_version=self.version,
        )


class ContextualPrecisionMetric(BaseMetric):
    """Evaluates whether relevant chunks are ranked above distractors."""

    def __init__(self) -> None:
        self.name = "contextual_precision"
        self.family = MetricFamily.RETRIEVAL

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not trace.retrieved_chunks or not case.relevant_documents:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                evaluator_version=self.version,
            )

        gold_ids = {d.document_id for d in case.relevant_documents}
        running_relevant = 0
        precisions = []

        for i, chunk in enumerate(trace.retrieved_chunks, start=1):
            if chunk.document_id in gold_ids:
                running_relevant += 1
                precisions.append(running_relevant / i)

        score = sum(precisions) / len(precisions) if precisions else 0.0
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            reason=f"Calculated precision over {len(precisions)} relevant chunk hits.",
            evaluator_version=self.version,
        )


# --- Generation Metrics ---
class FaithfulnessMetric(BaseMetric):
    """Grounding / Faithfulness: verifies that answer claims are supported by retrieved context."""

    def __init__(self) -> None:
        self.name = "faithfulness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        # Abstaining correctly is 100% faithful
        if trace.abstained or not trace.answer:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="System abstained; no ungrounded claims generated.",
                evaluator_version=self.version,
            )

        context_text = " ".join(c.text.lower() for c in trace.retrieved_chunks)
        if not context_text:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                reason="No retrieved context available to support answer claims.",
                evaluator_version=self.version,
            )

        # ponytail: sentence-level lexical token overlap heuristic until LLM judge plugin is plugged
        sentences = [s.strip() for s in re.split(r"[.!?]", trace.answer) if len(s.strip()) > 3]
        if not sentences:
            return MetricResult(metric_name=self.name, metric_family=self.family, score=1.0, evaluator_version=self.version)

        supported_count = 0
        for sent in sentences:
            sent_words = set(re.findall(r"\w{3,}", sent.lower()))
            if not sent_words:
                supported_count += 1
                continue
            matches = sum(1 for w in sent_words if w in context_text)
            overlap_ratio = matches / len(sent_words)
            if overlap_ratio >= 0.50:
                supported_count += 1

        score = supported_count / len(sentences)
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            reason=f"{supported_count}/{len(sentences)} answer sentences supported by retrieved chunks.",
            evaluator_version=self.version,
        )


class AnswerCorrectnessMetric(BaseMetric):
    """Lexical token F1 / overlap between generated answer and expected golden answer."""

    def __init__(self) -> None:
        self.name = "answer_correctness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.expected_answer:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No expected answer in golden test case.",
                evaluator_version=self.version,
            )

        if not trace.answer:
            score = 1.0 if case.answerability == Answerability.UNANSWERABLE else 0.0
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=score,
                reason="Answer is empty.",
                evaluator_version=self.version,
            )

        ans_words = set(re.findall(r"\w+", trace.answer.lower()))
        exp_words = set(re.findall(r"\w+", case.expected_answer.lower()))

        if not ans_words or not exp_words:
            return MetricResult(metric_name=self.name, metric_family=self.family, score=0.0, evaluator_version=self.version)

        intersection = ans_words.intersection(exp_words)
        precision = len(intersection) / len(ans_words)
        recall = len(intersection) / len(exp_words)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(f1, 4),
            reason=f"Token overlap F1: {round(f1, 3)} (P={round(precision, 2)}, R={round(recall, 2)}).",
            evaluator_version=self.version,
        )


class AbstentionAccuracyMetric(BaseMetric):
    """Verifies that unanswerable cases are refused and answerable cases are attempted."""

    def __init__(self) -> None:
        self.name = "abstention_accuracy"
        self.family = MetricFamily.ABSTENTION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        is_unanswerable = case.answerability == Answerability.UNANSWERABLE
        score = 1.0 if (is_unanswerable == trace.abstained) else 0.0
        reason = (
            "Valid refusal on unanswerable case."
            if (is_unanswerable and trace.abstained)
            else "Valid answer on answerable case."
            if (not is_unanswerable and not trace.abstained)
            else "Failed: answered unanswerable case with hallucination."
            if is_unanswerable
            else "Failed: incorrectly refused an answerable case."
        )

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            reason=reason,
            evaluator_version=self.version,
        )


# --- Evaluation Engine & Cache ---
class EvaluationEngine:
    """Orchestrates metric execution with hashed caching and run-level aggregation."""

    def __init__(self, metrics: list[BaseMetric] | None = None) -> None:
        self.metrics = metrics or [
            RecallAtKMetric(k=5),
            MeanReciprocalRankMetric(),
            ContextualPrecisionMetric(),
            FaithfulnessMetric(),
            AnswerCorrectnessMetric(),
            AbstentionAccuracyMetric(),
        ]
        # ponytail: in-memory cache dict keyed by deterministic sha256 hash
        self._cache: dict[str, MetricResult] = {}

    def _cache_key(self, metric: BaseMetric, trace: RagTrace, case: TestCase) -> str:
        payload = {
            "metric": metric.name,
            "version": metric.version,
            "case_id": case.id,
            "question": case.question,
            "trace_answer": trace.answer,
            "trace_abstained": trace.abstained,
            "chunks": [c.chunk_id for c in trace.retrieved_chunks],
        }
        return sha256_hash(canonical_json(payload))

    async def evaluate_trace(self, trace: RagTrace, case: TestCase, use_cache: bool = True) -> list[MetricResult]:
        results: list[MetricResult] = []
        for metric in self.metrics:
            key = self._cache_key(metric, trace, case)
            if use_cache and key in self._cache:
                cached_res = self._cache[key].model_copy(update={"cached": True})
                results.append(cached_res)
                continue

            result = await metric.compute(trace, case)
            if use_cache:
                self._cache[key] = result
            results.append(result)
        return results

    def aggregate_run(self, traces_with_metrics: list[tuple[RagTrace, list[MetricResult]]]) -> RunMetricsSummary:
        metric_values: dict[str, list[float]] = {}
        metric_families: dict[str, MetricFamily] = {}
        hallucinations = 0
        abstention_scores = []
        latencies = []
        total_cost = 0.0

        for trace, m_list in traces_with_metrics:
            latencies.append(trace.latency_ms)
            if trace.cost_usd:
                total_cost += trace.cost_usd

            for m in m_list:
                metric_values.setdefault(m.metric_name, []).append(m.score)
                metric_families[m.metric_name] = m.metric_family

                if m.metric_name == "faithfulness" and m.score < 0.50:
                    hallucinations += 1
                if m.metric_name == "abstention_accuracy":
                    abstention_scores.append(m.score)

        summaries: dict[str, MetricSummary] = {}
        for name, vals in metric_values.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            p50_idx = int(n * 0.50)
            p95_idx = min(int(n * 0.95), n - 1)

            summaries[name] = MetricSummary(
                metric_name=name,
                metric_family=metric_families[name],
                mean=round(sum(vals) / n, 4),
                p50=round(sorted_vals[p50_idx], 4),
                p95=round(sorted_vals[p95_idx], 4),
                min=round(sorted_vals[0], 4),
                max=round(sorted_vals[-1], 4),
                count=n,
            )

        total_cases = len(traces_with_metrics)
        hallucination_rate = round(hallucinations / total_cases, 4) if total_cases > 0 else 0.0
        abstention_acc = round(sum(abstention_scores) / len(abstention_scores), 4) if abstention_scores else 1.0

        sorted_latencies = sorted(latencies) if latencies else [0]
        p95_lat = sorted_latencies[min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)]

        return RunMetricsSummary(
            metrics=summaries,
            total_cases=total_cases,
            scored_cases=total_cases,
            hallucination_rate=hallucination_rate,
            abstention_accuracy=abstention_acc,
            p95_latency_ms=float(p95_lat),
            total_cost_usd=round(total_cost, 4),
        )
