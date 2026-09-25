"""Failure attribution engine: multi-stage root cause diagnosis for RAG systems.

# ponytail: single file for rule-based attribution pipeline, taxonomy mapping, and diagnosis.
"""

from __future__ import annotations

import re
from typing import Any

from rag_platform.models import (
    Answerability,
    FailureAttribution,
    FailureCode,
    MetricResult,
    RagTrace,
    Severity,
    TestCase,
)


class FailureAttributionEngine:
    """Diagnoses root cause of RAG failures following the strict 8-step decision tree."""

    def diagnose(
        self,
        trace: RagTrace,
        case: TestCase,
        metrics: list[MetricResult] | None = None,
    ) -> FailureAttribution | None:
        """Run diagnostic attribution pipeline according to Section 9.2.
        
        Returns FailureAttribution or None if passed.
        """
        metric_map = {m.metric_name: m for m in (metrics or [])}

        # Step 1: Rule out infrastructure errors first (OPS-01)
        # Never label a network timeout, rate limit, or 504 gateway as a model hallucination!
        if trace.error_code == "OPS-01" or (trace.latency_ms >= 5000 and not trace.answer):
            return FailureAttribution(
                trace_id=trace.trace_id,
                failure_type=FailureCode.OPS_01,
                severity=Severity.CRITICAL,
                confidence=1.0,
                explanation="Execution failed due to infrastructure error, timeout, or unreachable endpoint.",
                evidence={"error_code": trace.error_code, "latency_ms": trace.latency_ms, "telemetry": trace.telemetry},
                recommended_actions=[
                    "Check network connectivity and endpoint health",
                    "Increase client timeout threshold",
                    "Inspect worker error logs for stack traces",
                ],
            )

        # Step 2: Check unanswerable cases and abstention (ABS-01)
        if case.answerability == Answerability.UNANSWERABLE:
            if not trace.abstained and trace.answer:
                return FailureAttribution(
                    trace_id=trace.trace_id,
                    failure_type=FailureCode.ABS_01,
                    severity=Severity.CRITICAL,
                    confidence=0.98,
                    explanation="System failed to abstain on unanswerable question, fabricating an answer.",
                    evidence={"question": case.question, "hallucinated_answer": trace.answer},
                    recommended_actions=[
                        "Strengthen system prompt refusal instructions",
                        "Tune abstention confidence threshold",
                        "Ensure knowledge gap prompts explicitly allow refusal",
                    ],
                )
            # Valid refusal
            return None

        # Step 3: Check retrieval evidence top-K (RET-01)
        if case.relevant_documents:
            gold_ids = {d.document_id for d in case.relevant_documents}
            retrieved_ids = [c.document_id for c in trace.retrieved_chunks]

            if not any(g in retrieved_ids for g in gold_ids):
                return FailureAttribution(
                    trace_id=trace.trace_id,
                    failure_type=FailureCode.RET_01,
                    severity=Severity.HIGH,
                    confidence=0.95,
                    explanation=f"Required evidence doc(s) {sorted(gold_ids)} completely absent from top-{len(retrieved_ids)} retrieved chunks.",
                    evidence={"expected_documents": sorted(gold_ids), "retrieved_documents": retrieved_ids},
                    recommended_actions=[
                        "Increase retrieval top-k (e.g. from 5 to 10)",
                        "Add hybrid lexical (BM25) search alongside dense vectors",
                        "Audit document chunking size and boundary overlaps",
                    ],
                )

            # Step 4: Check retrieval ranking (RET-02)
            first_gold_rank = next((c.rank for c in trace.retrieved_chunks if c.document_id in gold_ids), None)
            if first_gold_rank and first_gold_rank > 3:
                return FailureAttribution(
                    trace_id=trace.trace_id,
                    failure_type=FailureCode.RET_02,
                    severity=Severity.MEDIUM,
                    confidence=0.90,
                    explanation=f"Gold evidence was retrieved but buried at rank {first_gold_rank} below distractors.",
                    evidence={"first_gold_rank": first_gold_rank, "retrieved_ranks": [c.document_id for c in trace.retrieved_chunks]},
                    recommended_actions=[
                        "Enable cross-encoder reranker (e.g. Cohere or BGE reranker)",
                        "Tune query rewriting / expansion prompt",
                        "Adjust vector similarity threshold",
                    ],
                )

        # Step 5: Check generation faithfulness / hallucination (GEN-01 / GEN-02)
        faith_metric = metric_map.get("faithfulness")
        if faith_metric and faith_metric.score < 0.60:
            return FailureAttribution(
                trace_id=trace.trace_id,
                failure_type=FailureCode.GEN_01,
                severity=Severity.HIGH,
                confidence=0.92,
                explanation="Answer claims are unsupported by retrieved context (hallucination).",
                evidence={"faithfulness_score": faith_metric.score, "answer": trace.answer},
                recommended_actions=[
                    "Lower temperature to 0.0 for deterministic generation",
                    "Add explicit system instruction: 'Answer ONLY using the provided evidence'",
                    "Implement self-reflection / citation check before response generation",
                ],
            )

        # Step 6: Check citation integrity (CIT-01)
        if trace.citations:
            chunk_map = {c.chunk_id: c for c in trace.retrieved_chunks}
            doc_map = {c.document_id: c for c in trace.retrieved_chunks}

            for cit in trace.citations:
                matched_chunk = chunk_map.get(cit.chunk_id) or doc_map.get(cit.document_id)
                if not matched_chunk:
                    return FailureAttribution(
                        trace_id=trace.trace_id,
                        failure_type=FailureCode.CIT_01,
                        severity=Severity.HIGH,
                        confidence=0.95,
                        explanation=f"Citation refers to doc '{cit.document_id}'/chunk '{cit.chunk_id}' which was never retrieved.",
                        evidence={"cited_doc": cit.document_id, "cited_chunk": cit.chunk_id, "claim": cit.claim_text},
                        recommended_actions=[
                            "Constrain citation generation exclusively to retrieved chunk IDs",
                            "Verify citation parser span regex",
                        ],
                    )

                # Verify cited chunk actually contains tokens from cited claim
                claim_words = set(re.findall(r"[A-Za-z0-9]+", cit.claim_text.lower()))
                chunk_words = set(re.findall(r"[A-Za-z0-9]+", matched_chunk.text.lower()))
                if claim_words and not claim_words.intersection(chunk_words):
                    return FailureAttribution(
                        trace_id=trace.trace_id,
                        failure_type=FailureCode.CIT_01,
                        severity=Severity.HIGH,
                        confidence=0.92,
                        explanation=f"Citation attached to chunk '{matched_chunk.chunk_id}', but chunk text does not support the claim.",
                        evidence={
                            "claim": cit.claim_text,
                            "cited_doc": cit.document_id,
                            "cited_chunk": cit.chunk_id,
                            "chunk_snippet": matched_chunk.text[:200],
                        },
                        recommended_actions=[
                            "Inspect citation prompt alignment",
                            "Use sentence-level citation verification before emitting answer",
                        ],
                    )

        # Step 7: Check answer correctness vs reference (GEN-01 / mismatch)
        corr_metric = metric_map.get("answer_correctness")
        if corr_metric and corr_metric.score < 0.20 and case.expected_answer:
            return FailureAttribution(
                trace_id=trace.trace_id,
                failure_type=FailureCode.GEN_01,
                severity=Severity.MEDIUM,
                confidence=0.85,
                explanation=f"Answer token F1 ({corr_metric.score}) is severely misaligned with expected reference answer.",
                evidence={"expected_answer": case.expected_answer, "actual_answer": trace.answer},
                recommended_actions=[
                    "Inspect generation prompt and reasoning path",
                    "Check if reference answer represents an alternative valid phrasing",
                ],
            )

        # All quality checks passed!
        return None
