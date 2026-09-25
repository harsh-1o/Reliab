"""RAG System Under Test (SUT) adapters and offline self-testing fixtures.

# ponytail: single file covers protocol, HTTP/Python adapters, and all 6 mock RAGs.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Callable, Protocol

import httpx

from rag_platform.core import generate_id
from rag_platform.models import (
    Answerability,
    Citation,
    RagTrace,
    RetrievedChunk,
    RunConfig,
    TestCase,
)


class RagAdapter(Protocol):
    """Protocol that every SUT adapter must satisfy."""

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        """Execute question against SUT and return canonical trace."""
        ...


class PythonRagAdapter:
    """Wraps an in-process Python callable into a RagAdapter."""

    def __init__(self, target_fn: Callable[[TestCase, RunConfig], RagTrace | Any]) -> None:
        self.target_fn = target_fn

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        start = time.perf_counter()
        res = self.target_fn(case, config)
        if hasattr(res, "__await__"):
            res = await res
        latency_ms = int((time.perf_counter() - start) * 1000)

        if isinstance(res, RagTrace):
            res.latency_ms = max(res.latency_ms, latency_ms)
            return res
        # Auto-wrap raw dict / string output
        return RagTrace(
            trace_id=generate_id("tr"),
            run_id=generate_id("run_adhoc"),
            test_case_id=case.id,
            question=case.question,
            answer=str(res),
            latency_ms=latency_ms,
        )


class HttpRagAdapter:
    """Invokes an external HTTP RAG endpoint and normalizes to canonical RagTrace."""

    def __init__(
        self,
        endpoint_url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        start = time.perf_counter()
        trace_id = generate_id("tr")
        payload = {
            "question": case.question,
            "test_case_id": case.id,
            "metadata": case.metadata,
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                resp = await client.post(self.endpoint_url, json=payload, headers=self.headers)
                latency_ms = int((time.perf_counter() - start) * 1000)

                if resp.status_code >= 400:
                    return RagTrace(
                        trace_id=trace_id,
                        run_id=generate_id("run"),
                        test_case_id=case.id,
                        question=case.question,
                        error_code="OPS-01",
                        latency_ms=latency_ms,
                        telemetry={"http_status": resp.status_code, "body": resp.text},
                    )

                data = resp.json()
                chunks = [
                    RetrievedChunk(
                        document_id=c.get("document_id", "doc_unknown"),
                        chunk_id=c.get("chunk_id", f"c_{i}"),
                        rank=c.get("rank", i + 1),
                        score=float(c.get("score", 0.0)),
                        text=c.get("text", ""),
                    )
                    for i, c in enumerate(data.get("retrieved_chunks", []))
                ]
                citations = [
                    Citation(
                        claim_id=cit.get("claim_id", f"cl_{i}"),
                        claim_text=cit.get("claim_text", ""),
                        document_id=cit.get("document_id", ""),
                        chunk_id=cit.get("chunk_id", ""),
                        span=cit.get("span"),
                    )
                    for i, cit in enumerate(data.get("citations", []))
                ]

                return RagTrace(
                    trace_id=trace_id,
                    run_id=generate_id("run"),
                    test_case_id=case.id,
                    question=case.question,
                    answer=data.get("answer"),
                    abstained=bool(data.get("abstained", False)),
                    abstention_reason=data.get("abstention_reason"),
                    retrieved_chunks=chunks,
                    citations=citations,
                    latency_ms=latency_ms,
                    input_tokens=data.get("input_tokens"),
                    output_tokens=data.get("output_tokens"),
                    cost_usd=data.get("cost_usd"),
                    model=data.get("model"),
                    telemetry=data.get("telemetry", {}),
                )
        except (httpx.TimeoutException, httpx.RequestError) as ex:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                error_code="OPS-01",
                latency_ms=latency_ms,
                telemetry={"exception": str(ex), "type": type(ex).__name__},
            )


# --- Platform Self-Test: 6 Flawed Synthetic RAGs ---
class SyntheticRagMode(str, Enum):
    PERFECT = "PERFECT"
    DISTRACTOR = "DISTRACTOR"           # RET-01 / RET-02
    HALLUCINATING = "HALLUCINATING"     # GEN-01 / GEN-02
    BROKEN_CITATION = "BROKEN_CITATION" # CIT-01
    REFUSAL_BYPASS = "REFUSAL_BYPASS"   # ABS-01
    TIMEOUT = "TIMEOUT"                 # OPS-01


class SyntheticRagAdapter:
    """Offline synthetic RAG engine producing deterministic behaviors for platform self-tests."""

    def __init__(self, mode: SyntheticRagMode = SyntheticRagMode.PERFECT) -> None:
        self.mode = mode

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        trace_id = generate_id("tr")
        has_gold = bool(case.relevant_documents)
        gold_doc = case.relevant_documents[0].document_id if has_gold else "doc_gold"
        gold_chunk = (case.relevant_documents[0].chunk_id if has_gold and case.relevant_documents[0].chunk_id else "chunk_gold")

        if self.mode == SyntheticRagMode.TIMEOUT:
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                error_code="OPS-01",
                latency_ms=5000,
                telemetry={"simulated_error": "Connection timed out after 5000ms"},
            )

        if case.answerability == Answerability.UNANSWERABLE:
            if self.mode == SyntheticRagMode.REFUSAL_BYPASS:
                return RagTrace(
                    trace_id=trace_id,
                    run_id=generate_id("run"),
                    test_case_id=case.id,
                    question=case.question,
                    answer="Mars office is located at Olympus Mons Crater sector 4.",
                    abstained=False,
                    retrieved_chunks=[],
                    latency_ms=85,
                )
            # Normal / correct abstention
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                answer=None,
                abstained=True,
                abstention_reason="INSUFFICIENT_EVIDENCE",
                latency_ms=45,
            )

        # Answerable cases
        if self.mode == SyntheticRagMode.DISTRACTOR:
            # Irrelevant distractor chunks retrieved
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                answer="Irrelevant noise response.",
                retrieved_chunks=[
                    RetrievedChunk(document_id="noise_doc_99", chunk_id="chunk_noise", rank=1, text="Weather is sunny.")
                ],
                latency_ms=90,
            )

        if self.mode == SyntheticRagMode.HALLUCINATING:
            # Retrieves gold context but hallucinates false facts contradicting evidence
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                answer="Revenue plunged to zero and all assets were liquidated.",
                retrieved_chunks=[
                    RetrievedChunk(document_id=gold_doc, chunk_id=gold_chunk, rank=1, text=f"Revenue grew to {case.expected_answer}.")
                ],
                citations=[
                    Citation(claim_id="cl_1", claim_text="Assets were liquidated", document_id=gold_doc, chunk_id=gold_chunk)
                ],
                latency_ms=110,
            )

        if self.mode == SyntheticRagMode.BROKEN_CITATION:
            # Correct answer, but citations point to completely wrong document
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                answer=case.expected_answer or "Verified fact.",
                retrieved_chunks=[
                    RetrievedChunk(document_id=gold_doc, chunk_id=gold_chunk, rank=1, text=f"Fact: {case.expected_answer}"),
                    RetrievedChunk(document_id="wrong_doc_404", chunk_id="chunk_wrong", rank=2, text="Unrelated text"),
                ],
                citations=[
                    Citation(claim_id="cl_1", claim_text=case.expected_answer or "Fact", document_id="wrong_doc_404", chunk_id="chunk_wrong")
                ],
                latency_ms=95,
            )

        # Default: SyntheticRagMode.PERFECT
        return RagTrace(
            trace_id=trace_id,
            run_id=generate_id("run"),
            test_case_id=case.id,
            question=case.question,
            answer=case.expected_answer or "Correct answer.",
            retrieved_chunks=[
                RetrievedChunk(document_id=gold_doc, chunk_id=gold_chunk, rank=1, score=0.98, text=f"Evidence for {case.expected_answer}")
            ],
            citations=[
                Citation(claim_id="cl_1", claim_text=case.expected_answer or "Fact", document_id=gold_doc, chunk_id=gold_chunk)
            ],
            latency_ms=70,
            model="synthetic-perfect-v1",
        )
