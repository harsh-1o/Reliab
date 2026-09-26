import pytest

from rag_platform.attribution import FailureAttributionEngine
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    Citation,
    DocumentReference,
    FailureCode,
    RagTrace,
    RetrievedChunk,
    TestCase,
)


@pytest.mark.asyncio
async def test_multi_cause_attribution_primary_and_contributing():
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    case = TestCase(
        id="c_multi",
        question="What were the Q3 operational expenditures?",
        expected_answer="$45M in cloud and facility costs.",
        expected_facts=["$45M expenditure", "cloud and facility"],
        relevant_documents=[DocumentReference(document_id="doc_q3", chunk_id="chk_q3_opex")],
    )

    # Retrieval returned WRONG document (RET-01),
    # Generator hallucinated arbitrary numbers (GEN-01),
    # Citation fabricated a non-existent claim support (CIT-01)
    trace = RagTrace(
        trace_id="t_multi_fail",
        run_id="run_1",
        test_case_id=case.id,
        question="What were the Q3 operational expenditures?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chk_wrong",
                document_id="doc_hr_handbook",
                text="Employees receive 20 days of annual leave.",
                rank=1,
                score=0.4,
            )
        ],
        answer="Operational expenditure was $999M across marketing.",
        citations=[
            Citation(
                claim_id="clm_1",
                claim_text="Operational expenditure was $999M across marketing.",
                document_id="doc_hr_handbook",
                chunk_id="chk_wrong",
            )
        ],
    )

    metrics = await eval_engine.evaluate_trace(trace, case)
    attribution = attr_engine.diagnose(trace, case, metrics)

    # Root cause must be retrieval failure (RET-01 or RET-02) because bad context led to downstream failures
    assert attribution.primary_code in (FailureCode.RET_01, FailureCode.RET_02)
    assert len(attribution.contributing_codes) >= 1

    # Should diagnose hallucination / ungrounded claim as contributing failure
    assert (
        FailureCode.GEN_01 in attribution.contributing_codes
        or FailureCode.CIT_01 in attribution.contributing_codes
    )

    # Validate explanation, confidence, and recommended actions
    assert attribution.confidence > 0.0
    assert len(attribution.explanation) > 10
    assert len(attribution.recommended_actions) >= 1
    assert len(attribution.findings) >= 2


@pytest.mark.asyncio
async def test_attribution_for_clean_trace():
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    case = TestCase(
        id="c_clean",
        question="What is the encryption algorithm?",
        expected_answer="AES-256",
        expected_facts=["AES-256"],
        relevant_documents=[DocumentReference(document_id="doc_enc", chunk_id="chk_aes")],
    )

    trace = RagTrace(
        trace_id="t_clean",
        run_id="run_1",
        test_case_id=case.id,
        question="What is the encryption algorithm?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chk_aes",
                document_id="doc_enc",
                text="The system uses AES-256 encryption for all data at rest.",
                rank=1,
                score=0.98,
            )
        ],
        answer="The system uses AES-256 encryption.",
        citations=[
            Citation(
                claim_id="clm_aes",
                claim_text="The system uses AES-256 encryption.",
                document_id="doc_enc",
                chunk_id="chk_aes",
            )
        ],
    )

    metrics = await eval_engine.evaluate_trace(trace, case)
    attribution = attr_engine.diagnose(trace, case, metrics)

    # Clean trace without any failures correctly returns None
    assert attribution is None
