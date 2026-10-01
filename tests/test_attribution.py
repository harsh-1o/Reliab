"""Unit tests for Phase 5: failure attribution engine verified across all synthetic RAG failure modes."""

from __future__ import annotations

import pytest

from rag_platform.adapters import SyntheticRagAdapter, SyntheticRagMode
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    Answerability,
    DocumentReference,
    FailureCode,
    RunConfig,
    TestCase,
)


@pytest.fixture
def eval_engine() -> EvaluationEngine:
    return EvaluationEngine()


@pytest.fixture
def attr_engine() -> FailureAttributionEngine:
    return FailureAttributionEngine()


@pytest.fixture
def run_config() -> RunConfig:
    return RunConfig(
        project_id="proj_1",
        dataset_id="ds_1",
        dataset_version="v1",
        system_version="mock-v1",
    )


@pytest.fixture
def answerable_case() -> TestCase:
    return TestCase(
        id="c_rev",
        question="What was 2024 revenue?",
        expected_answer="$10.5B",
        relevant_documents=[DocumentReference(document_id="doc_annual", chunk_id="chunk_gold")],
    )


@pytest.fixture
def unanswerable_case() -> TestCase:
    return TestCase(
        id="c_unans",
        question="What is the CEO's favorite constellation?",
        expected_answer=None,
        answerability=Answerability.UNANSWERABLE,
    )


@pytest.mark.asyncio
async def test_diagnose_perfect_rag_returns_none(eval_engine, attr_engine, answerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.PERFECT)
    trace = await adapter.run(answerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, answerable_case)

    diag = attr_engine.diagnose(trace, answerable_case, metrics)
    assert diag is None  # Clean pass!


@pytest.mark.asyncio
async def test_diagnose_distractor_retrieval_miss(eval_engine, attr_engine, answerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.DISTRACTOR)
    trace = await adapter.run(answerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, answerable_case)

    diag = attr_engine.diagnose(trace, answerable_case, metrics)
    assert diag is not None
    assert diag.failure_type == FailureCode.RET_01
    assert diag.confidence >= 0.90
    assert len(diag.recommended_actions) > 0


@pytest.mark.asyncio
async def test_diagnose_hallucination(eval_engine, attr_engine, answerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.HALLUCINATING)
    trace = await adapter.run(answerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, answerable_case)

    diag = attr_engine.diagnose(trace, answerable_case, metrics)
    assert diag is not None
    assert diag.failure_type == FailureCode.GEN_01
    assert "hallucination" in diag.explanation.lower()


@pytest.mark.asyncio
async def test_diagnose_broken_citation(eval_engine, attr_engine, answerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.BROKEN_CITATION)
    trace = await adapter.run(answerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, answerable_case)

    diag = attr_engine.diagnose(trace, answerable_case, metrics)
    assert diag is not None
    # Misattributed chunk text produces CIT_02 (Point 20)
    assert diag.failure_type == FailureCode.CIT_02
    assert "wrong_doc_404" in str(diag.evidence)

    # Completely unretrieved chunk produces CIT_01 (Point 20)
    from rag_platform.models import Citation
    trace_unretrieved = trace.model_copy(update={
        "citations": [Citation(claim_id="cl_1", claim_text="Fact", document_id="never_retrieved_doc", chunk_id="never_retrieved_chunk")]
    })
    diag_unretrieved = attr_engine.diagnose(trace_unretrieved, answerable_case, metrics)
    assert diag_unretrieved.failure_type == FailureCode.CIT_01


@pytest.mark.asyncio
async def test_diagnose_abstention_failure(eval_engine, attr_engine, unanswerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.REFUSAL_BYPASS)
    trace = await adapter.run(unanswerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, unanswerable_case)

    diag = attr_engine.diagnose(trace, unanswerable_case, metrics)
    assert diag is not None
    assert diag.failure_type == FailureCode.ABS_01


@pytest.mark.asyncio
async def test_diagnose_infrastructure_error_never_labeled_hallucination(eval_engine, attr_engine, answerable_case, run_config):
    adapter = SyntheticRagAdapter(SyntheticRagMode.TIMEOUT)
    trace = await adapter.run(answerable_case, run_config)
    metrics = await eval_engine.evaluate_trace(trace, answerable_case)

    diag = attr_engine.diagnose(trace, answerable_case, metrics)
    assert diag is not None
    assert diag.failure_type == FailureCode.OPS_01
    assert diag.failure_type != FailureCode.GEN_01  # Crucial safety check!


@pytest.mark.asyncio
async def test_attribution_strict_citation_resolution_no_fallback(eval_engine, attr_engine, answerable_case, run_config):
    """When retrieved chunks contain doc1/chunk_A and doc1/chunk_B,
    and a citation cites doc1/WRONG_CHUNK, attribution must NOT fall back to doc1.
    It must strictly flag CIT_01 (unretrieved citation).
    """
    from rag_platform.models import Citation, DocumentReference, RagTrace, RetrievedChunk

    case = answerable_case.model_copy(update={
        "relevant_documents": [DocumentReference(document_id="doc1", chunk_id="chunk_A")],
    })

    trace = RagTrace(
        trace_id="tr_strict_cit",
        run_id="run_1",
        test_case_id=case.id,
        question=case.question,
        answer="2024 revenue was 10.5B.",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc1", chunk_id="chunk_A", rank=1, text="2024 revenue was 10.5B."),
            RetrievedChunk(document_id="doc1", chunk_id="chunk_B", rank=2, text="2024 revenue was 10.5B."),
        ],
        citations=[
            Citation(
                claim_id="cl_1",
                claim_text="2024 revenue was 10.5B",
                document_id="doc1",
                chunk_id="WRONG_CHUNK",
            )
        ],
    )
    metrics = await eval_engine.evaluate_trace(trace, case)
    diag = attr_engine.diagnose(trace, case, metrics)
    assert diag is not None
    assert diag.failure_type == FailureCode.CIT_01
    assert "WRONG_CHUNK" in str(diag.evidence)

