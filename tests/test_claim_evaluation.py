import pytest

from rag_platform.evaluators import (
    EvaluationEngine,
    FaithfulnessMetric,
    RecallAtKMetric,
    extract_claims,
    verify_claim_against_chunks,
)
from rag_platform.models import (
    ClaimStatus,
    DocumentReference,
    RagTrace,
    RetrievedChunk,
    TestCase,
)


def test_claim_decomposition_and_verification():
    answer = "The platform delivers 99.99% uptime. It was launched in 2024. It requires no configuration."
    claims = extract_claims(answer)
    assert len(claims) == 3
    assert "99.99% uptime" in claims[0]

    chunks = [
        RetrievedChunk(
            chunk_id="chk_1",
            document_id="doc_infra",
            text="Our platform delivers 99.99% uptime across all regions as recorded in 2024.",
            rank=1,
            score=0.95,
        ),
        RetrievedChunk(
            chunk_id="chk_2",
            document_id="doc_config",
            text="The deployment requires manual configuration of YAML keys.",
            rank=2,
            score=0.88,
        ),
    ]

    # Claim 1: Supported
    status1, chunk1, reason1 = verify_claim_against_chunks(claims[0], chunks)
    assert status1 == ClaimStatus.SUPPORTED
    assert chunk1 is not None and chunk1.chunk_id == "chk_1"

    # Claim 3: Contradicted by chk_2
    status3, chunk3, reason3 = verify_claim_against_chunks(claims[2], chunks)
    assert status3 == ClaimStatus.CONTRADICTED


@pytest.mark.asyncio
async def test_claim_level_faithfulness_penalizes_hallucination_and_contradiction():
    metric = FaithfulnessMetric()
    case = TestCase(
        id="c_faith",
        question="What is the uptime and pricing?",
        expected_answer="99.99% uptime and $50/mo.",
    )

    # Trace with 1 supported claim and 1 contradicted claim
    chunks = [
        RetrievedChunk(
            chunk_id="chk_uptime",
            document_id="doc_sla",
            text="The service guarantees 99.99% uptime SLA.",
            rank=1,
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="chk_pricing",
            document_id="doc_pricing",
            text="The standard plan is strictly paid at $50 per month.",
            rank=2,
            score=0.85,
        ),
    ]

    trace_contradicted = RagTrace(
        trace_id="t_contra",
        run_id="run_1",
        test_case_id=case.id,
        question="What is the uptime and pricing?",
        retrieved_chunks=chunks,
        answer="The service guarantees 99.99% uptime. The standard plan is completely free of charge.",
    )

    res = await metric.compute(trace_contradicted, case)
    # The score must be severely penalized (below 0.5) due to contradiction,
    # despite lexical overlap with the word 'standard plan'.
    assert res.score < 0.5
    assert "claims" in res.metadata
    assert res.metadata["contradicted_claims"] >= 1


@pytest.mark.asyncio
async def test_evidence_retrieval_distinguishes_chunk_vs_document():
    metric = RecallAtKMetric(k=3)
    case = TestCase(
        id="c_ret",
        question="What are the encryption standards?",
        relevant_documents=[
            DocumentReference(document_id="doc_sec", chunk_id="chk_aes256", page=12)
        ],
    )

    # System retrieved the right document, but WRONG chunk
    trace_wrong_chunk = RagTrace(
        trace_id="t_wrong_chunk",
        run_id="run_1",
        test_case_id=case.id,
        question="What are the encryption standards?",
        retrieved_chunks=[
            RetrievedChunk(
                chunk_id="chk_intro",
                document_id="doc_sec",
                text="Security Overview: This document describes general practices.",
                rank=1,
                score=0.9,
            )
        ],
        answer="AES-256",
    )

    res = await metric.compute(trace_wrong_chunk, case)
    # Document was matched, but chunk was missed. Score reflects partial chunk miss!
    assert res.score < 1.0
    assert res.metadata["chunk_level_hits"] == 0
    assert res.metadata["doc_level_hits"] == 1


@pytest.mark.asyncio
async def test_abstention_accuracy_decoupled_from_faithfulness():
    eval_engine = EvaluationEngine()
    case_unanswerable = TestCase(
        id="c_unans",
        question="What is the CEO's personal phone number?",
        expected_answer=None,
        answerability="UNANSWERABLE",
    )

    trace_abstained = RagTrace(
        trace_id="t_abs",
        run_id="run_1",
        test_case_id=case_unanswerable.id,
        question="What is the CEO's personal phone number?",
        retrieved_chunks=[],
        answer="I do not have sufficient information in the provided context to answer this question.",
        abstained=True,
    )

    metrics = await eval_engine.evaluate_trace(trace_abstained, case_unanswerable)
    m_dict = {m.metric_name: m for m in metrics}

    # Abstention should be 1.0 (correctly refused)
    assert m_dict["abstention_accuracy"].score == 1.0
    # Faithfulness for refusal should be neutral 1.0 with refused flag
    assert m_dict["faithfulness"].score == 1.0
    assert m_dict["faithfulness"].metadata.get("is_refusal") is True


@pytest.mark.asyncio
async def test_statistical_aggregation_and_sample_size_warning():
    eval_engine = EvaluationEngine()
    case = TestCase(id="c_small", question="Q", expected_answer="A")
    trace = RagTrace(
        trace_id="t1",
        run_id="run_1",
        test_case_id=case.id,
        question="Q",
        retrieved_chunks=[
            RetrievedChunk(chunk_id="c1", document_id="d1", text="A", rank=1, score=1.0)
        ],
        answer="A",
    )

    metrics = await eval_engine.evaluate_trace(trace, case)

    # 5 samples (N < 30)
    traces = [(trace, metrics)] * 5
    summary = eval_engine.aggregate_run(traces)

    assert summary.total_cases == 5
    # Must flag small sample size warning
    faith_metric = summary.metrics["faithfulness"]
    assert faith_metric.sample_warning is not None
    assert "Small sample size" in faith_metric.sample_warning
    assert faith_metric.count == 5
    assert faith_metric.ci_lower <= faith_metric.mean <= faith_metric.ci_upper
