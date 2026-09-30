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


def test_validate_http_rag_response_valid():
    from rag_platform.adapters import validate_http_rag_response

    # Answerable valid response
    valid_data = {
        "answer": "Revenue was $10B",
        "abstained": False,
        "abstention_reason": None,
        "retrieved_chunks": [
            {"document_id": "doc_1", "chunk_id": "c_1", "text": "text", "score": 0.9}
        ],
        "citations": [
            {"claim_id": "cl_1", "claim_text": "claim", "document_id": "doc_1"}
        ],
        "telemetry": {"latency_ms": 120},
    }
    assert validate_http_rag_response(valid_data) == valid_data

    # Valid abstention response
    valid_abs = {
        "answer": None,
        "abstained": True,
        "abstention_reason": "INSUFFICIENT_EVIDENCE",
    }
    assert validate_http_rag_response(valid_abs) == valid_abs


def test_validate_http_rag_response_invalid_shapes():
    from rag_platform.adapters import HttpRagResponseError, validate_http_rag_response

    # Not a dict
    with pytest.raises(HttpRagResponseError, match="Expected JSON object"):
        validate_http_rag_response(["an", "array"])

    with pytest.raises(HttpRagResponseError, match="Expected JSON object"):
        validate_http_rag_response("plain string")

    # Missing answer on non-abstained response
    with pytest.raises(HttpRagResponseError, match="Missing required field 'answer'"):
        validate_http_rag_response({"abstained": False})

    # Non-string answer
    with pytest.raises(HttpRagResponseError, match="Field 'answer' must be a string"):
        validate_http_rag_response({"answer": 12345, "abstained": False})

    # Non-boolean abstained
    with pytest.raises(HttpRagResponseError, match="Field 'abstained' must be a boolean"):
        validate_http_rag_response({"answer": "yes", "abstained": "not_a_bool"})

    # Invalid abstention_reason
    with pytest.raises(HttpRagResponseError, match="Field 'abstention_reason' must be a string"):
        validate_http_rag_response({"answer": None, "abstained": True, "abstention_reason": 123})

    # Malformed retrieved_chunks
    with pytest.raises(HttpRagResponseError, match="Field 'retrieved_chunks' must be a list"):
        validate_http_rag_response({"answer": "yes", "retrieved_chunks": "not_a_list"})

    with pytest.raises(HttpRagResponseError, match="must be a dict"):
        validate_http_rag_response({"answer": "yes", "retrieved_chunks": ["not_a_dict"]})

    with pytest.raises(HttpRagResponseError, match="invalid 'document_id'"):
        validate_http_rag_response({"answer": "yes", "retrieved_chunks": [{"document_id": 999}]})

    # Malformed citations
    with pytest.raises(HttpRagResponseError, match="Field 'citations' must be a list"):
        validate_http_rag_response({"answer": "yes", "citations": "not_a_list"})

    with pytest.raises(HttpRagResponseError, match="must be a dict"):
        validate_http_rag_response({"answer": "yes", "citations": ["not_a_dict"]})

    with pytest.raises(HttpRagResponseError, match="invalid 'claim_text'"):
        validate_http_rag_response({"answer": "yes", "citations": [{"claim_text": 1234}]})

    # Malformed telemetry
    with pytest.raises(HttpRagResponseError, match="Field 'telemetry' must be a dict"):
        validate_http_rag_response({"answer": "yes", "telemetry": "not_a_dict"})


@pytest.mark.asyncio
async def test_http_adapter_malformed_response_converts_to_ops01(
    answerable_case: TestCase, run_config: RunConfig
):
    import httpx

    # Test 1: Invalid JSON (e.g. plain text HTML error page)
    def invalid_json_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>500 Server Error</html>", headers={"content-type": "text/html"})

    transport1 = httpx.MockTransport(invalid_json_handler)
    client1 = httpx.AsyncClient(transport=transport1)
    adapter1 = HttpRagAdapter(
        endpoint_url="http://127.0.0.1:8000/rag",
        client=client1,
        allow_private_ip=True,
    )
    trace1 = await adapter1.run(answerable_case, run_config)
    assert trace1.error_code == "OPS-01"
    assert "Failed to parse JSON" in trace1.telemetry.get("error", "")

    # Test 2: Invalid Schema (missing answer)
    def invalid_schema_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"abstained": False, "citations": []})

    transport2 = httpx.MockTransport(invalid_schema_handler)
    client2 = httpx.AsyncClient(transport=transport2)
    adapter2 = HttpRagAdapter(
        endpoint_url="http://127.0.0.1:8000/rag",
        client=client2,
        allow_private_ip=True,
    )
    trace2 = await adapter2.run(answerable_case, run_config)
    assert trace2.error_code == "OPS-01"
    assert "Adapter response schema validation error" in trace2.telemetry.get("error", "")


@pytest.mark.asyncio
async def test_http_adapter_retries_and_idempotency_headers(
    answerable_case: TestCase, run_config: RunConfig
):
    import httpx

    call_count = 0
    captured_headers: dict[str, str] = {}

    def flaky_handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count, captured_headers
        call_count += 1
        captured_headers = dict(request.headers)
        if call_count == 1:
            return httpx.Response(503, text="Service Unavailable")
        return httpx.Response(200, json={"answer": "Recovered successfully", "abstained": False})

    transport = httpx.MockTransport(flaky_handler)
    client = httpx.AsyncClient(transport=transport)
    adapter = HttpRagAdapter(
        endpoint_url="http://127.0.0.1:8000/rag",
        client=client,
        allow_private_ip=True,
    )
    trace = await adapter.run(answerable_case, run_config)

    assert call_count == 2
    assert trace.answer == "Recovered successfully"
    assert "idempotency-key" in captured_headers
    assert f"eval_{run_config.project_id}_{run_config.dataset_id}_{answerable_case.id}" in captured_headers["idempotency-key"]
    assert "x-request-id" in captured_headers


def test_python_adapter_registry():
    from rag_platform.adapters import PythonAdapterRegistry

    PythonAdapterRegistry.clear()

    # Valid registration
    def my_handler(case: TestCase, config: RunConfig):
        return "result"

    PythonAdapterRegistry.register("my_model", my_handler)
    assert PythonAdapterRegistry.get("my_model") is my_handler

    # Re-registration overwrite
    def another_handler(case: TestCase, config: RunConfig):
        return "result2"

    PythonAdapterRegistry.register("my_model", another_handler)
    assert PythonAdapterRegistry.get("my_model") is another_handler

    # Invalid names
    with pytest.raises(ValueError, match="non-empty string"):
        PythonAdapterRegistry.register("", my_handler)
    with pytest.raises(ValueError, match="non-empty string"):
        PythonAdapterRegistry.register("   ", my_handler)

    # Invalid callable
    with pytest.raises(ValueError, match="must be callable"):
        PythonAdapterRegistry.register("bad", "not_callable")  # type: ignore

    # Unknown adapter lookup
    with pytest.raises(ValueError, match="Unknown registered python adapter"):
        PythonAdapterRegistry.get("non_existent")

    PythonAdapterRegistry.clear()

