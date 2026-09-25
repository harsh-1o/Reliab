"""Unit tests for Phase 3: RAG adapter interfaces and synthetic mock RAG fixtures."""

from __future__ import annotations

import pytest

from rag_platform.adapters import (
    HttpRagAdapter,
    PythonRagAdapter,
    SyntheticRagAdapter,
    SyntheticRagMode,
)
from rag_platform.models import (
    Answerability,
    DocumentReference,
    RagTrace,
    RunConfig,
    TestCase,
)


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
        id="case_rev",
        question="What was 2024 revenue?",
        expected_answer="$10.5B",
        expected_facts=["2024 revenue was $10.5B"],
        relevant_documents=[DocumentReference(document_id="annual_2024", chunk_id="p12")],
    )


@pytest.fixture
def unanswerable_case() -> TestCase:
    return TestCase(
        id="case_mars",
        question="What is the CEO's favorite constellation?",
        expected_answer=None,
        answerability=Answerability.UNANSWERABLE,
    )


@pytest.mark.asyncio
async def test_python_rag_adapter(answerable_case: TestCase, run_config: RunConfig):
    def echo_sut(case: TestCase, config: RunConfig):
        return f"Echo: {case.question}"

    adapter = PythonRagAdapter(echo_sut)
    trace = await adapter.run(answerable_case, run_config)

    assert isinstance(trace, RagTrace)
    assert trace.answer == "Echo: What was 2024 revenue?"
    assert trace.test_case_id == answerable_case.id


@pytest.mark.asyncio
async def test_synthetic_rag_perfect_mode(answerable_case: TestCase, run_config: RunConfig):
    adapter = SyntheticRagAdapter(SyntheticRagMode.PERFECT)
    trace = await adapter.run(answerable_case, run_config)

    assert trace.answer == "$10.5B"
    assert len(trace.retrieved_chunks) == 1
    assert trace.retrieved_chunks[0].document_id == "annual_2024"
    assert len(trace.citations) == 1
    assert trace.citations[0].document_id == "annual_2024"


@pytest.mark.asyncio
async def test_synthetic_rag_distractor_mode(answerable_case: TestCase, run_config: RunConfig):
    adapter = SyntheticRagAdapter(SyntheticRagMode.DISTRACTOR)
    trace = await adapter.run(answerable_case, run_config)

    assert len(trace.retrieved_chunks) == 1
    assert trace.retrieved_chunks[0].document_id == "noise_doc_99"


@pytest.mark.asyncio
async def test_synthetic_rag_hallucinating_mode(answerable_case: TestCase, run_config: RunConfig):
    adapter = SyntheticRagAdapter(SyntheticRagMode.HALLUCINATING)
    trace = await adapter.run(answerable_case, run_config)

    assert "liquidated" in trace.answer
    assert trace.retrieved_chunks[0].document_id == "annual_2024"


@pytest.mark.asyncio
async def test_synthetic_rag_broken_citation_mode(answerable_case: TestCase, run_config: RunConfig):
    adapter = SyntheticRagAdapter(SyntheticRagMode.BROKEN_CITATION)
    trace = await adapter.run(answerable_case, run_config)

    assert trace.answer == "$10.5B"
    # Citations point to wrong doc
    assert trace.citations[0].document_id == "wrong_doc_404"


@pytest.mark.asyncio
async def test_synthetic_rag_abstention_modes(unanswerable_case: TestCase, run_config: RunConfig):
    # Perfect abstains
    perfect_adapter = SyntheticRagAdapter(SyntheticRagMode.PERFECT)
    trace_abs = await perfect_adapter.run(unanswerable_case, run_config)
    assert trace_abs.abstained is True
    assert trace_abs.abstention_reason == "INSUFFICIENT_EVIDENCE"

    # Refusal bypass hallucinates an answer
    bypass_adapter = SyntheticRagAdapter(SyntheticRagMode.REFUSAL_BYPASS)
    trace_bypass = await bypass_adapter.run(unanswerable_case, run_config)
    assert trace_bypass.abstained is False
    assert trace_bypass.answer is not None


@pytest.mark.asyncio
async def test_synthetic_rag_timeout_mode(answerable_case: TestCase, run_config: RunConfig):
    adapter = SyntheticRagAdapter(SyntheticRagMode.TIMEOUT)
    trace = await adapter.run(answerable_case, run_config)

    assert trace.error_code == "OPS-01"
    assert trace.latency_ms >= 5000


@pytest.mark.asyncio
async def test_http_adapter_connection_error(answerable_case: TestCase, run_config: RunConfig):
    # Points to non-existent local port -> catches connection failure as OPS-01
    adapter = HttpRagAdapter(endpoint_url="http://127.0.0.1:59999/rag", timeout_seconds=1.0)
    trace = await adapter.run(answerable_case, run_config)

    assert trace.error_code == "OPS-01"
    assert "exception" in trace.telemetry
