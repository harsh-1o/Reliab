"""Unit tests for Phase 4: metric plugins, evaluation engine, caching, and aggregation."""

from __future__ import annotations

import pytest

from rag_platform.evaluators import (
    AbstentionAccuracyMetric,
    AnswerCorrectnessMetric,
    ContextualPrecisionMetric,
    EvaluationEngine,
    FaithfulnessMetric,
    MeanReciprocalRankMetric,
    RecallAtKMetric,
)
from rag_platform.models import (
    Answerability,
    DocumentReference,
    RagTrace,
    RetrievedChunk,
    TestCase,
)


@pytest.fixture
def sample_case() -> TestCase:
    return TestCase(
        id="c_fin",
        question="What was 2024 gross profit?",
        expected_answer="$5.2 billion",
        relevant_documents=[DocumentReference(document_id="doc_annual")],
    )


@pytest.mark.asyncio
async def test_recall_at_k(sample_case: TestCase):
    metric = RecallAtKMetric(k=3)

    # Found at rank 2 -> Recall@3 = 1.0
    trace_hit = RagTrace(
        trace_id="t1",
        run_id="r1",
        test_case_id="c_fin",
        question="question",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc_noise", chunk_id="p1", rank=1, text="noise"),
            RetrievedChunk(document_id="doc_annual", chunk_id="p2", rank=2, text="gold"),
        ],
    )
    res_hit = await metric.compute(trace_hit, sample_case)
    assert res_hit.score == 1.0

    # Found at rank 5 -> Recall@3 = 0.0
    trace_miss = RagTrace(
        trace_id="t2",
        run_id="r1",
        test_case_id="c_fin",
        question="question",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc_noise", chunk_id="p1", rank=1, text="noise"),
            RetrievedChunk(document_id="doc_noise2", chunk_id="p2", rank=2, text="noise"),
            RetrievedChunk(document_id="doc_noise3", chunk_id="p3", rank=3, text="noise"),
            RetrievedChunk(document_id="doc_annual", chunk_id="p4", rank=4, text="gold"),
        ],
    )
    res_miss = await metric.compute(trace_miss, sample_case)
    assert res_miss.score == 0.0


@pytest.mark.asyncio
async def test_mrr(sample_case: TestCase):
    metric = MeanReciprocalRankMetric()

    trace_rank2 = RagTrace(
        trace_id="t1",
        run_id="r1",
        test_case_id="c_fin",
        question="q",
        retrieved_chunks=[
            RetrievedChunk(document_id="distractor", chunk_id="p1", rank=1, text=""),
            RetrievedChunk(document_id="doc_annual", chunk_id="p2", rank=2, text=""),
        ],
    )
    res = await metric.compute(trace_rank2, sample_case)
    assert res.score == 0.5  # 1/2


@pytest.mark.asyncio
async def test_faithfulness_grounded_vs_hallucinated(sample_case: TestCase):
    metric = FaithfulnessMetric()

    # Grounded answer: vocabulary matches chunk
    trace_grounded = RagTrace(
        trace_id="t1",
        run_id="r1",
        test_case_id="c_fin",
        question="q",
        answer="The 2024 gross profit was $5.2 billion according to annual results.",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc_annual", chunk_id="p1", rank=1, text="In 2024 gross profit was $5.2 billion in annual results.")
        ],
    )
    res_grounded = await metric.compute(trace_grounded, sample_case)
    assert res_grounded.score >= 0.90

    # Hallucinated answer: disjoint vocabulary
    trace_hallucinated = RagTrace(
        trace_id="t2",
        run_id="r1",
        test_case_id="c_fin",
        question="q",
        answer="Company filed liquidation bankruptcy after catastrophic meteor strike obliterated headquarters.",
        retrieved_chunks=[
            RetrievedChunk(document_id="doc_annual", chunk_id="p1", rank=1, text="In 2024 gross profit was $5.2 billion.")
        ],
    )
    res_hallucinated = await metric.compute(trace_hallucinated, sample_case)
    assert res_hallucinated.score <= 0.30


@pytest.mark.asyncio
async def test_abstention_accuracy():
    metric = AbstentionAccuracyMetric()

    unanswerable_case = TestCase(id="u1", question="Where is Atlantis?", answerability=Answerability.UNANSWERABLE)

    # Correct refusal
    trace_abstained = RagTrace(trace_id="t1", run_id="r1", test_case_id="u1", question="q", abstained=True)
    assert (await metric.compute(trace_abstained, unanswerable_case)).score == 1.0

    # Hallucinated answer instead of refusing
    trace_hallucinated = RagTrace(trace_id="t2", run_id="r1", test_case_id="u1", question="q", answer="Atlantis is in Greece.", abstained=False)
    assert (await metric.compute(trace_hallucinated, unanswerable_case)).score == 0.0


@pytest.mark.asyncio
async def test_evaluation_engine_caching_and_aggregation(sample_case: TestCase):
    engine = EvaluationEngine()

    trace = RagTrace(
        trace_id="t_eval",
        run_id="r1",
        test_case_id="c_fin",
        question="What was 2024 gross profit?",
        answer="Gross profit was $5.2 billion.",
        latency_ms=150,
        cost_usd=0.002,
        retrieved_chunks=[
            RetrievedChunk(document_id="doc_annual", chunk_id="p1", rank=1, text="Gross profit was $5.2 billion.")
        ],
    )

    # First run: computed fresh
    results1 = await engine.evaluate_trace(trace, sample_case, use_cache=True)
    assert all(not r.cached for r in results1)

    # Second run: identical input returns cached results
    results2 = await engine.evaluate_trace(trace, sample_case, use_cache=True)
    assert all(r.cached for r in results2)

    # Test aggregate_run calculation
    summary = engine.aggregate_run([(trace, results1)])
    assert summary.total_cases == 1
    assert "recall_at_5" in summary.metrics
    assert "faithfulness" in summary.metrics
    assert summary.metrics["faithfulness"].mean >= 0.90
    assert summary.hallucination_rate == 0.0
    assert summary.p95_latency_ms == 150.0
