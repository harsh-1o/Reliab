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
    assert diag.failure_type == FailureCode.CIT_01
    assert "wrong_doc_404" in str(diag.evidence)


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
