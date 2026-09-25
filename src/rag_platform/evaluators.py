"""Metric evaluation engine: claim-level faithfulness, fact-anchored correctness,
evidence retrieval, citation validation, and statistical confidence intervals.
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from typing import Any

from rag_platform.core import canonical_json, sha256_hash
from rag_platform.models import (
    Answerability,
    ClaimStatus,
    ClaimVerification,
    DocumentReference,
    MetricFamily,
    MetricResult,
    MetricSummary,
    RagTrace,
    RetrievedChunk,
    RunMetricsSummary,
    TestCase,
)


def extract_claims(text: str) -> list[str]:
    """Extract atomic factual claim units from answer text.
    
    Decomposes paragraphs into declarative sentence/clause propositions,
    filtering conversational padding and meta-discourse.
    """
    if not text or not text.strip():
        return []

    # Clean bracketed citation tags [1], [doc-1] for pure claim analysis
    clean_text = re.sub(r"\[\s*[\w\-]+\s*\]", "", text).strip()

    # Split by sentence boundaries
    raw_sentences = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", clean_text)
        if len(s.strip()) > 5
    ]

    claims: list[str] = []
    # Conversational filler prefixes to strip or ignore
    fillers = (
        "based on the provided context",
        "according to the documents",
        "as stated in the text",
        "in conclusion",
        "to summarize",
        "thank you",
        "here is the answer",
    )

    for s in raw_sentences:
        lower_s = s.lower().strip()
        # Skip pure boilerplate or disclaimers
        if any(lower_s == f for f in fillers):
            continue
        for f in fillers:
            if lower_s.startswith(f + ",") or lower_s.startswith(f + ":"):
                s = s[len(f) + 1 :].strip()
                break

        if len(s) > 8:
            claims.append(s)

    return claims if claims else ([clean_text] if len(clean_text) > 8 else [])


def verify_claim_against_chunks(
    claim: str, chunks: list[RetrievedChunk]
) -> tuple[ClaimStatus, RetrievedChunk | None, str]:
    """Verify an individual claim against all retrieved context chunks.
    
    Returns (ClaimStatus, supporting_chunk, explanation).
    Classifies as:
      - SUPPORTED: Core entities and predicates confirmed in chunk.
      - CONTRADICTED: Directly conflicts with factual values or negation in chunk.
      - UNSUPPORTED: Facts not present in retrieved context (extrinsic hallucination).
    """
    claim_lower = claim.lower()
    claim_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.%]+", claim_lower))
    # Extract numbers, percentages, currency, proper nouns
    claim_entities = set(re.findall(r"\b(?:\d+(?:\.\d+)?%?|\$\d+(?:\.\d+)?|[A-Z][a-z]+)\b", claim))

    best_match_chunk: RetrievedChunk | None = None
    best_overlap = 0.0

    # Common antonym / negation pairs for contradiction detection
    contradiction_pairs = [
        ("increased", "decreased"),
        ("expanded", "contracted"),
        ("grew", "declined"),
        ("approved", "rejected"),
        ("allowed", "prohibited"),
        ("acquired", "divested"),
        ("positive", "negative"),
        ("higher", "lower"),
        ("rose", "fell"),
        ("free", "paid"),
        ("no", "manual"),
        ("none", "all"),
        ("enabled", "disabled"),
        ("success", "failure"),
        ("supported", "unsupported"),
    ]

    for chunk in chunks:
        chunk_text_lower = chunk.text.lower()
        chunk_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.%]+", chunk_text_lower))

        # Check for explicit contradictions
        for w1, w2 in contradiction_pairs:
            if (w1 in claim_lower and w2 in chunk_text_lower) or (w2 in claim_lower and w1 in chunk_text_lower):
                # Verify they share common topic tokens (at least 1 content word)
                shared = {
                    t for t in claim_tokens.intersection(chunk_tokens)
                    if len(t) > 2 and t not in (w1, w2, "the", "and", "for", "with", "this", "that")
                }
                if len(shared) >= 1:
                    return (
                        ClaimStatus.CONTRADICTED,
                        chunk,
                        f"Claim directly contradicts chunk {chunk.chunk_id}: asserts '{w1}' while evidence states '{w2}'.",
                    )

        # Check numeric contradictions (e.g. claim says 55% while chunk says 42.1%)
        chunk_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", chunk_text_lower))
        claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", claim_lower))
        if claim_numbers and chunk_numbers and not claim_numbers.intersection(chunk_numbers):
            shared_words = claim_tokens.intersection(chunk_tokens) - claim_numbers
            if len(shared_words) >= 4:
                return (
                    ClaimStatus.CONTRADICTED,
                    chunk,
                    f"Numerical conflict with chunk {chunk.chunk_id}: claim states {claim_numbers} while evidence records {chunk_numbers}.",
                )

        # Calculate semantic token overlap
        if claim_tokens:
            overlap = len(claim_tokens.intersection(chunk_tokens)) / len(claim_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match_chunk = chunk

    # Supported threshold: key entities and > 50% non-trivial tokens match chunk
    if best_overlap >= 0.50 and best_match_chunk is not None:
        return (
            ClaimStatus.SUPPORTED,
            best_match_chunk,
            f"Supported by chunk {best_match_chunk.chunk_id} ({round(best_overlap * 100, 1)}% token alignment).",
        )

    return (
        ClaimStatus.UNSUPPORTED,
        None,
        f"Unsupported: no retrieved chunk substantiates claim (max alignment {round(best_overlap * 100, 1)}%).",
    )


def is_evidence_match(gold: DocumentReference, chunk: RetrievedChunk) -> bool:
    """Matches evidence reference at chunk-level if chunk_id is specified, else document-level."""
    if gold.document_id != chunk.document_id:
        return False
    if gold.chunk_id is not None and chunk.chunk_id != gold.chunk_id:
        return False
    return True


class BaseMetric(ABC):
    """Abstract base metric evaluator."""

    name: str
    family: MetricFamily
    version: str = "2.0.0"

    @abstractmethod
    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        """Compute atomic metric score for a given trace and test case."""
        ...


# --- Retrieval Metrics ---
class RecallAtKMetric(BaseMetric):
    """Measures whether required evidence (document or chunk level) was retrieved in top-K."""

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
                reason="No relevant documents specified in golden test case.",
                evaluator_version=self.version,
            )

        gold_refs = case.relevant_documents
        retrieved_k = trace.retrieved_chunks[: self.k]

        matched_gold = 0
        chunk_hits = 0
        doc_hits = 0
        for gold in gold_refs:
            if any(is_evidence_match(gold, chunk) for chunk in retrieved_k):
                matched_gold += 1
                if gold.chunk_id:
                    chunk_hits += 1
            if any(gold.document_id == chunk.document_id for chunk in retrieved_k):
                doc_hits += 1

        score = matched_gold / len(gold_refs)
        spec_level = "chunk" if any(g.chunk_id for g in gold_refs) else "document"

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            reason=f"Found {matched_gold}/{len(gold_refs)} gold evidence targets ({spec_level}-level) in top-{self.k}.",
            evaluator_version=self.version,
            metadata={
                "matched_count": matched_gold,
                "total_gold": len(gold_refs),
                "k": self.k,
                "chunk_level_hits": chunk_hits,
                "doc_level_hits": doc_hits,
            },
        )


class MeanReciprocalRankMetric(BaseMetric):
    """Computes Reciprocal Rank (1/rank) for the highest-ranked gold evidence chunk."""

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

        gold_refs = case.relevant_documents
        for chunk in trace.retrieved_chunks:
            if any(is_evidence_match(gold, chunk) for gold in gold_refs):
                rr = 1.0 / max(1, chunk.rank)
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=round(rr, 4),
                    reason=f"First relevant evidence '{chunk.document_id}:{chunk.chunk_id}' found at rank {chunk.rank}.",
                    evaluator_version=self.version,
                )

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=0.0,
            reason="No golden evidence targets found in retrieved chunks.",
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
                reason="Trivially satisfied: no retrieved chunks or no golden references.",
                evaluator_version=self.version,
            )

        gold_refs = case.relevant_documents
        running_relevant = 0
        precisions: list[float] = []

        for i, chunk in enumerate(trace.retrieved_chunks, start=1):
            if any(is_evidence_match(gold, chunk) for gold in gold_refs):
                running_relevant += 1
                precisions.append(running_relevant / i)

        score = sum(precisions) / len(precisions) if precisions else 0.0
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=round(score, 4),
            reason=f"Precision computed across {len(precisions)} relevant evidence chunk hits.",
            evaluator_version=self.version,
        )


# --- Generation Metrics ---
class FaithfulnessMetric(BaseMetric):
    """Claim-level Grounding / Faithfulness: verifies that each generated claim is supported by retrieved evidence."""

    def __init__(self) -> None:
        self.name = "faithfulness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        # Separate abstention dimension from faithfulness:
        if trace.abstained:
            if case.answerability == Answerability.UNANSWERABLE:
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=1.0,
                    reason="Correctly abstained on unanswerable query; zero hallucinated claims produced.",
                    evaluator_version=self.version,
                    metadata={"claims": [], "abstained": True, "is_refusal": True},
                )
            # Falsely abstained: no claims generated
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                reason="System abstained on an answerable query; no grounded claims produced.",
                evaluator_version=self.version,
                metadata={"claims": [], "abstained": True},
            )

        if not trace.answer or not trace.answer.strip():
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                reason="Answer is empty; no claims produced.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        if not trace.retrieved_chunks:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=0.0,
                reason="No retrieved context available to support answer claims.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        # 1. Extract atomic factual claims
        claims = extract_claims(trace.answer)
        if not claims:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No verifiable factual claims detected in response.",
                evaluator_version=self.version,
                metadata={"claims": []},
            )

        # 2. Verify each claim
        verifications: list[ClaimVerification] = []
        supported_count = 0
        contradiction_count = 0

        for idx, claim_text in enumerate(claims):
            status, matched_chunk, reason = verify_claim_against_chunks(claim_text, trace.retrieved_chunks)
            verifications.append(
                ClaimVerification(
                    claim_id=f"clm_{idx+1}",
                    claim_text=claim_text,
                    status=status,
                    supporting_chunk_id=matched_chunk.chunk_id if matched_chunk else None,
                    confidence=0.95 if status == ClaimStatus.SUPPORTED else 0.90,
                    reason=reason,
                )
            )
            if status == ClaimStatus.SUPPORTED:
                supported_count += 1
            elif status == ClaimStatus.CONTRADICTED:
                contradiction_count += 1

        raw_score = supported_count / len(claims)
        # Apply contradiction penalty
        penalty = 0.25 * contradiction_count
        final_score = max(0.0, round(raw_score - penalty, 4))

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=final_score,
            reason=f"Claim-level verdict: {supported_count}/{len(claims)} supported, {contradiction_count} contradicted.",
            evaluator_version=self.version,
            metadata={
                "total_claims": len(claims),
                "supported_claims": supported_count,
                "contradictions": contradiction_count,
                "contradicted_claims": contradiction_count,
                "claims": [c.model_dump() for c in verifications],
            },
        )


class AnswerCorrectnessMetric(BaseMetric):
    """Fact-anchored answer correctness: evaluates golden fact coverage, lexical/entity precision & recall."""

    def __init__(self) -> None:
        self.name = "answer_correctness"
        self.family = MetricFamily.GENERATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not case.expected_answer and not case.expected_facts:
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No expected answer or expected facts specified in golden test case.",
                evaluator_version=self.version,
            )

        if not trace.answer or not trace.answer.strip():
            score = 1.0 if case.answerability == Answerability.UNANSWERABLE else 0.0
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=score,
                reason="Answer is empty.",
                evaluator_version=self.version,
            )

        # 1. Fact Coverage
        fact_coverage = 1.0
        covered_facts: list[str] = []
        missing_facts: list[str] = []
        ans_lower = trace.answer.lower()

        if case.expected_facts:
            for fact in case.expected_facts:
                fact_tokens = [w for w in re.findall(r"\w+", fact.lower()) if len(w) > 2]
                if fact_tokens and all(w in ans_lower for w in fact_tokens[: max(1, int(len(fact_tokens) * 0.7))]):
                    covered_facts.append(fact)
                else:
                    missing_facts.append(fact)
            fact_coverage = len(covered_facts) / len(case.expected_facts)

        # 2. Token / Entity F1
        token_f1 = 1.0
        if case.expected_answer:
            ans_words = set(re.findall(r"\w+", ans_lower))
            exp_words = set(re.findall(r"\w+", case.expected_answer.lower()))

            if ans_words and exp_words:
                intersection = ans_words.intersection(exp_words)
                precision = len(intersection) / len(ans_words)
                recall = len(intersection) / len(exp_words)
                token_f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                token_f1 = 0.0

        # Weighted composite score
        if case.expected_facts and case.expected_answer:
            score = round(0.60 * fact_coverage + 0.40 * token_f1, 4)
            reason = f"Fact coverage: {len(covered_facts)}/{len(case.expected_facts)} ({round(fact_coverage, 2)}), Token F1: {round(token_f1, 3)}."
        elif case.expected_facts:
            score = round(fact_coverage, 4)
            reason = f"Fact coverage: {len(covered_facts)}/{len(case.expected_facts)} facts."
        else:
            score = round(token_f1, 4)
            reason = f"Token overlap F1: {round(token_f1, 3)}."

        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            reason=reason,
            evaluator_version=self.version,
            metadata={
                "fact_coverage": round(fact_coverage, 4),
                "token_f1": round(token_f1, 4),
                "covered_facts": covered_facts,
                "missing_facts": missing_facts,
            },
        )


class CitationSupportMetric(BaseMetric):
    """Citation validation: checks that citations refer to retrieved chunks and actually substantiate claims."""

    def __init__(self) -> None:
        self.name = "citation_accuracy"
        self.family = MetricFamily.CITATION

    async def compute(self, trace: RagTrace, case: TestCase) -> MetricResult:
        if not trace.citations:
            claims = extract_claims(trace.answer or "")
            if claims and len(claims) > 0 and trace.answer and len(trace.answer.strip()) > 30:
                return MetricResult(
                    metric_name=self.name,
                    metric_family=self.family,
                    score=0.50,
                    reason=f"Generated {len(claims)} factual statements without citing evidence.",
                    evaluator_version=self.version,
                    metadata={"missing_citations": True},
                )
            return MetricResult(
                metric_name=self.name,
                metric_family=self.family,
                score=1.0,
                reason="No citations required for query.",
                evaluator_version=self.version,
            )

        chunk_map = {c.chunk_id: c for c in trace.retrieved_chunks}
        doc_map = {c.document_id: c for c in trace.retrieved_chunks}

        valid_count = 0
        total = len(trace.citations)

        for cit in trace.citations:
            matched = chunk_map.get(cit.chunk_id) or doc_map.get(cit.document_id)
            if not matched:
                continue

            claim_words = [w for w in re.findall(r"\w+", cit.claim_text.lower()) if len(w) > 2]
            chunk_words = set(re.findall(r"\w+", matched.text.lower()))
            if claim_words:
                overlap = sum(1 for w in claim_words if w in chunk_words) / len(claim_words)
                if overlap >= 0.40:
                    valid_count += 1
            else:
                valid_count += 1

        score = round(valid_count / total, 4)
        return MetricResult(
            metric_name=self.name,
            metric_family=self.family,
            score=score,
            reason=f"{valid_count}/{total} citations substantiated by retrieved evidence chunks.",
            evaluator_version=self.version,
            metadata={"valid_citations": valid_count, "total_citations": total},
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


# --- Evaluation Engine, Cache, & Statistics ---
class EvaluationEngine:
    """Orchestrates metric execution with hashed caching, statistical confidence intervals, and aggregation."""

    def __init__(self, metrics: list[BaseMetric] | None = None) -> None:
        self.metrics = metrics or [
            RecallAtKMetric(k=5),
            MeanReciprocalRankMetric(),
            ContextualPrecisionMetric(),
            FaithfulnessMetric(),
            AnswerCorrectnessMetric(),
            CitationSupportMetric(),
            AbstentionAccuracyMetric(),
        ]
        self._cache: dict[str, MetricResult] = {}

    def _cache_key(self, metric: BaseMetric, trace: RagTrace, case: TestCase) -> str:
        payload = {
            "metric": metric.name,
            "version": metric.version,
            "case_id": case.id,
            "question": case.question,
            "trace_answer": trace.answer,
            "trace_abstained": trace.abstained,
            "chunks": [(c.document_id, c.chunk_id) for c in trace.retrieved_chunks],
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

                if m.metric_name == "faithfulness" and m.score < 0.60 and not trace.abstained:
                    hallucinations += 1
                if m.metric_name == "abstention_accuracy":
                    abstention_scores.append(m.score)

        summaries: dict[str, MetricSummary] = {}
        for name, vals in metric_values.items():
            if not vals:
                continue
            sorted_vals = sorted(vals)
            n = len(sorted_vals)
            mean_val = sum(vals) / n

            # Calculate sample standard deviation
            variance = sum((x - mean_val) ** 2 for x in vals) / (n - 1) if n > 1 else 0.0
            std_dev = math.sqrt(variance)

            # 95% Confidence Interval (z = 1.96)
            margin_error = 1.96 * (std_dev / math.sqrt(n)) if n > 0 else 0.0
            ci_lower = max(0.0, round(mean_val - margin_error, 4))
            ci_upper = min(1.0, round(mean_val + margin_error, 4))

            p50_idx = int(n * 0.50)
            p95_idx = min(int(n * 0.95), n - 1)

            sample_warning = (
                f"Small sample size (N={n} < 30); statistical variance is elevated."
                if n < 30
                else None
            )

            summaries[name] = MetricSummary(
                metric_name=name,
                metric_family=metric_families[name],
                mean=round(mean_val, 4),
                p50=round(sorted_vals[p50_idx], 4),
                p95=round(sorted_vals[p95_idx], 4),
                min=round(sorted_vals[0], 4),
                max=round(sorted_vals[-1], 4),
                count=n,
                std_dev=round(std_dev, 4),
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                sample_warning=sample_warning,
            )

        total_cases = len(traces_with_metrics)
        hallucination_rate = round(hallucinations / total_cases, 4) if total_cases > 0 else 0.0
        abstention_acc = round(sum(abstention_scores) / len(abstention_scores), 4) if abstention_scores else 1.0

        sorted_latencies = sorted(latencies) if latencies else [0]
        p95_lat = sorted_latencies[min(int(len(sorted_latencies) * 0.95), len(sorted_latencies) - 1)]

        run_warning = (
            f"Run evaluation based on small sample size (N={total_cases} < 30). Regression results may lack statistical power."
            if total_cases < 30
            else None
        )

        return RunMetricsSummary(
            metrics=summaries,
            total_cases=total_cases,
            scored_cases=total_cases,
            hallucination_rate=hallucination_rate,
            abstention_accuracy=abstention_acc,
            p95_latency_ms=float(p95_lat),
            total_cost_usd=round(total_cost, 4),
            sample_warning=run_warning,
        )
