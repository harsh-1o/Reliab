"""Failure attribution engine: multi-stage root cause diagnosis supporting primary
and contributing failure classification, canonical taxonomy, and remediation actions.
"""

from __future__ import annotations

import re
from typing import Any

from rag_platform.evaluators import is_evidence_match
from rag_platform.models import (
    Answerability,
    DiagnosticFinding,
    FailureAttribution,
    FailureCode,
    MetricResult,
    RagTrace,
    Severity,
    TestCase,
)


class FailureAttributionEngine:
    """Diagnoses root cause of RAG failures using a causal decision pipeline.
    
    Identifies the primary root-cause failure and all contributing failures across
    infrastructure, abstention, retrieval, generation, and citation stages.
    """

    # Causal priority: earlier stages are considered primary causes of downstream failures
    CAUSAL_PRIORITY: dict[FailureCode, int] = {
        FailureCode.OPS_01: 1,  # System failure blocks entire pipeline
        FailureCode.ABS_01: 2,  # Refusal failure on unanswerable query
        FailureCode.ABS_02: 3,  # False refusal on answerable query
        FailureCode.RET_01: 4,  # Retrieval miss causes generation to lack evidence
        FailureCode.RET_02: 5,  # Bad ranking causes distractors to pollute context
        FailureCode.RET_03: 6,  # Context truncation cuts off evidence
        FailureCode.GEN_02: 7,  # Direct contradiction with retrieved evidence
        FailureCode.GEN_01: 8,  # Extrinsic hallucination
        FailureCode.CIT_01: 9,  # Missing citation
        FailureCode.CIT_02: 10, # Misattributed citation
        FailureCode.KNW_01: 11, # Knowledge gap
        FailureCode.NUM_01: 12, # Numerical error
        FailureCode.ENT_01: 13, # Entity confusion
    }

    def diagnose(
        self,
        trace: RagTrace,
        case: TestCase,
        metrics: list[MetricResult] | None = None,
    ) -> FailureAttribution | None:
        """Run diagnostic attribution pipeline across all evaluation dimensions.
        
        Returns FailureAttribution with primary_code, contributing_codes, and detailed findings,
        or None if all quality thresholds are satisfied.
        """
        metric_map = {m.metric_name: m for m in (metrics or [])}
        findings: list[DiagnosticFinding] = []

        # --- Stage 1: Infrastructure & System Errors (OPS-01) ---
        if trace.error_code == "OPS-01" or (trace.latency_ms >= 5000 and not trace.answer):
            findings.append(
                DiagnosticFinding(
                    code=FailureCode.OPS_01,
                    severity=Severity.CRITICAL,
                    confidence=1.0,
                    explanation="Execution failed due to upstream infrastructure error, gateway timeout, or network drop.",
                    evidence={
                        "error_code": trace.error_code,
                        "latency_ms": trace.latency_ms,
                        "telemetry": trace.telemetry,
                    },
                    recommended_actions=[
                        "Check upstream model endpoint connectivity and health",
                        "Increase client timeout threshold",
                        "Inspect application worker logs for uncaught exceptions",
                    ],
                )
            )

        # --- Stage 2: Abstention & Boundary Handling (ABS-01, ABS-02) ---
        if case.answerability == Answerability.UNANSWERABLE:
            if not trace.abstained and trace.answer and len(trace.answer.strip()) > 5:
                findings.append(
                    DiagnosticFinding(
                        code=FailureCode.ABS_01,
                        severity=Severity.CRITICAL,
                        confidence=0.98,
                        explanation="System failed to abstain on an unanswerable question, producing an ungrounded response.",
                        evidence={"question": case.question, "answer": trace.answer},
                        recommended_actions=[
                            "Strengthen system prompt refusal instructions for unanswerable questions",
                            "Tune refusal confidence threshold in the routing layer",
                            "Provide explicit few-shot examples of appropriate refusal phrasing",
                        ],
                    )
                )
        else:
            # Answerable case falsely refused
            if trace.abstained:
                findings.append(
                    DiagnosticFinding(
                        code=FailureCode.ABS_02,
                        severity=Severity.HIGH,
                        confidence=0.95,
                        explanation="System falsely abstained on an answerable question with valid reference documents.",
                        evidence={
                            "question": case.question,
                            "abstention_reason": trace.abstention_reason,
                        },
                        recommended_actions=[
                            "Reduce over-defensive abstention classifier sensitivity",
                            "Ensure retriever provides sufficient evidence chunks before triggering refusal",
                        ],
                    )
                )

        # --- Stage 3: Retrieval Quality & Ranking (RET-01, RET-02) ---
        if case.relevant_documents:
            gold_refs = case.relevant_documents
            retrieved = trace.retrieved_chunks

            # Check if any gold reference matched
            has_match = any(
                any(is_evidence_match(gold, chunk) for chunk in retrieved)
                for gold in gold_refs
            )

            if not has_match:
                findings.append(
                    DiagnosticFinding(
                        code=FailureCode.RET_01,
                        severity=Severity.HIGH,
                        confidence=0.95,
                        explanation=f"Required evidence doc(s) {[g.document_id for g in gold_refs]} completely absent from top-{len(retrieved)} retrieved chunks.",
                        evidence={
                            "expected_targets": [
                                f"{g.document_id}:{g.chunk_id}" if g.chunk_id else g.document_id
                                for g in gold_refs
                            ],
                            "retrieved_chunks": [f"{c.document_id}:{c.chunk_id}" for c in retrieved],
                        },
                        recommended_actions=[
                            "Increase retrieval top-k (e.g., from 5 to 10)",
                            "Add hybrid lexical (BM25) search alongside dense vector retrieval",
                            "Audit document chunking strategy and boundary overlap sizes",
                        ],
                    )
                )
            else:
                # Evidence present: check rank position
                first_gold_rank = None
                for chunk in retrieved:
                    if any(is_evidence_match(gold, chunk) for gold in gold_refs):
                        first_gold_rank = chunk.rank
                        break

                if first_gold_rank and first_gold_rank > 3:
                    findings.append(
                        DiagnosticFinding(
                            code=FailureCode.RET_02,
                            severity=Severity.MEDIUM,
                            confidence=0.90,
                            explanation=f"Relevant evidence retrieved, but ranked at rank {first_gold_rank} below irrelevant distractors.",
                            evidence={
                                "first_gold_rank": first_gold_rank,
                                "top_chunks": [c.document_id for c in retrieved[:3]],
                            },
                            recommended_actions=[
                                "Enable cross-encoder reranker (e.g. Cohere or BGE reranker)",
                                "Tune query expansion and rewriting prompt",
                                "Increase vector similarity distance threshold",
                            ],
                        )
                    )

        # --- Stage 4: Generation Grounding & Contradictions (GEN-01, GEN-02) ---
        if not trace.abstained and trace.answer:
            faith_metric = metric_map.get("faithfulness")
            if faith_metric:
                meta = faith_metric.metadata or {}
                contradictions = meta.get("contradictions", 0)

                if contradictions > 0:
                    findings.append(
                        DiagnosticFinding(
                            code=FailureCode.GEN_02,
                            severity=Severity.HIGH,
                            confidence=0.94,
                            explanation=f"Generated answer contains {contradictions} assertion(s) directly contradicting retrieved context.",
                            evidence={"answer": trace.answer, "claims": meta.get("claims", [])},
                            recommended_actions=[
                                "Add strict negative constraint: 'Do not state facts opposite to the retrieved context'",
                                "Implement self-correction verification step before emitting output",
                            ],
                        )
                    )

                if faith_metric.score is not None and faith_metric.score < 0.60:
                    findings.append(
                        DiagnosticFinding(
                            code=FailureCode.GEN_01,
                            severity=Severity.HIGH,
                            confidence=0.92,
                            explanation=f"Answer contains ungrounded claims (hallucination) (faithfulness score {faith_metric.score} < 0.60).",
                            evidence={"faithfulness_score": faith_metric.score, "answer": trace.answer},
                            recommended_actions=[
                                "Lower LLM generation temperature to 0.0 for deterministic output",
                                "Enforce prompt instruction: 'Answer exclusively using facts in the provided evidence'",
                                "Enable claim-level citation enforcement in generation prompt",
                            ],
                        )
                    )

        # --- Stage 5: Citation Integrity (CIT-01, CIT-02) ---
        if trace.citations:
            chunk_map = {c.chunk_id: c for c in trace.retrieved_chunks}
            doc_map = {c.document_id: c for c in trace.retrieved_chunks}

            for cit in trace.citations:
                matched_chunk = chunk_map.get(cit.chunk_id) or doc_map.get(cit.document_id)
                if not matched_chunk:
                    findings.append(
                        DiagnosticFinding(
                            code=FailureCode.CIT_01,
                            severity=Severity.HIGH,
                            confidence=0.95,
                            explanation=f"Citation refers to doc '{cit.document_id}'/chunk '{cit.chunk_id}' which was never retrieved.",
                            evidence={"cited_doc": cit.document_id, "cited_chunk": cit.chunk_id, "claim": cit.claim_text},
                            recommended_actions=[
                                "Constrain citation generation strictly to retrieved chunk IDs",
                                "Verify citation parser span regex",
                            ],
                        )
                    )
                    break

                # Verify cited chunk text substantiates claim
                claim_words = [w for w in re.findall(r"[A-Za-z0-9]+", cit.claim_text.lower()) if len(w) >= 1]
                chunk_words = set(re.findall(r"[A-Za-z0-9]+", matched_chunk.text.lower()))
                overlap = sum(1 for w in claim_words if w in chunk_words) / len(claim_words) if claim_words else 0.0
                if overlap < 0.30:
                    findings.append(
                        DiagnosticFinding(
                            code=FailureCode.CIT_01,
                            severity=Severity.HIGH,
                            confidence=0.92,
                            explanation=f"Citation attached to chunk '{matched_chunk.chunk_id}' in doc '{cit.document_id}', but chunk text does not substantiate the claim.",
                            evidence={
                                "claim": cit.claim_text,
                                "cited_doc": cit.document_id,
                                "cited_chunk": cit.chunk_id,
                                "chunk_snippet": matched_chunk.text[:200],
                            },
                            recommended_actions=[
                                "Enforce sentence-level citation verification before emitting answer",
                                "Review citation prompt alignment guidelines",
                            ],
                        )
                    )
                    break

        # --- Stage 6: Semantic Knowledge Alignment (KNW-01) ---
        corr_metric = metric_map.get("answer_correctness")
        if (
            corr_metric
            and corr_metric.score is not None
            and corr_metric.score < 0.25
            and case.expected_answer
            and not trace.abstained
            and not any(f.code in (FailureCode.RET_01, FailureCode.GEN_01, FailureCode.OPS_01) for f in findings)
        ):
            findings.append(
                DiagnosticFinding(
                    code=FailureCode.KNW_01,
                    severity=Severity.MEDIUM,
                    confidence=0.85,
                    explanation=f"Answer correctness score {corr_metric.score} is severely misaligned with expected reference answer.",
                    evidence={"expected_answer": case.expected_answer, "actual_answer": trace.answer},
                    recommended_actions=[
                        "Inspect generation prompt reasoning trajectory",
                        "Check if reference answer represents an alternative valid formulation",
                    ],
                )
            )

        # If zero diagnostic findings, all quality gates passed cleanly!
        if not findings:
            return None

        # Sort findings according to causal priority: earliest stage is PRIMARY
        findings.sort(key=lambda f: self.CAUSAL_PRIORITY.get(f.code, 99))
        primary = findings[0]
        contributing = [f.code for f in findings[1:] if f.code != primary.code]

        # Consolidate recommendations
        all_actions: list[str] = []
        for f in findings:
            for act in f.recommended_actions:
                if act not in all_actions:
                    all_actions.append(act)

        return FailureAttribution(
            trace_id=trace.trace_id,
            primary_code=primary.code,
            contributing_codes=contributing,
            severity=primary.severity,
            confidence=primary.confidence,
            explanation=primary.explanation,
            evidence={
                **primary.evidence,
                "primary_code": primary.code.value,
                "contributing_codes": [c.value for c in contributing],
                "total_diagnostics_detected": len(findings),
            },
            recommended_actions=all_actions[:4],
            findings=findings,
        )
