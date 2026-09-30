"""RAG System Under Test (SUT) adapters and adapter registry."""

from __future__ import annotations

import asyncio
import threading
import time
from enum import Enum
from typing import Any, Callable, Protocol
from urllib.parse import urljoin, urlsplit

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
from rag_platform.security import SecretRedactor
from rag_platform.ssrf import SSRFProtectionError, validate_url_ssrf


class RagAdapter(Protocol):
    """Protocol that every SUT adapter must satisfy."""

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        """Execute question against SUT and return canonical trace."""
        ...


class HttpRagResponseError(ValueError):
    """Raised when an external RAG HTTP endpoint returns a schema-violating payload."""
    pass


def validate_http_rag_response(raw_data: Any) -> dict[str, Any]:
    """Validate external RAG system response against expected contract.

    Enforces required fields and structural types while keeping legitimate optional fields optional.
    """
    if not isinstance(raw_data, dict):
        raise HttpRagResponseError(
            f"Expected JSON object (dict) in RAG response, got {type(raw_data).__name__}"
        )

    abstained = raw_data.get("abstained")
    if abstained is not None and not isinstance(abstained, bool):
        raise HttpRagResponseError(
            f"Field 'abstained' must be a boolean, got {type(abstained).__name__}"
        )
    is_abstained = bool(abstained)

    answer = raw_data.get("answer")
    if not is_abstained:
        if answer is None:
            raise HttpRagResponseError("Missing required field 'answer' on non-abstained response")
        if not isinstance(answer, str):
            raise HttpRagResponseError(f"Field 'answer' must be a string, got {type(answer).__name__}")
    else:
        if answer is not None and not isinstance(answer, str):
            raise HttpRagResponseError(f"Field 'answer' must be a string or None, got {type(answer).__name__}")

    abstention_reason = raw_data.get("abstention_reason")
    if abstention_reason is not None and not isinstance(abstention_reason, str):
        raise HttpRagResponseError(
            f"Field 'abstention_reason' must be a string or None, got {type(abstention_reason).__name__}"
        )

    chunks = raw_data.get("retrieved_chunks")
    if chunks is not None:
        if not isinstance(chunks, list):
            raise HttpRagResponseError(f"Field 'retrieved_chunks' must be a list, got {type(chunks).__name__}")
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise HttpRagResponseError(f"Chunk at index {idx} in 'retrieved_chunks' must be a dict")
            chunk_doc = chunk.get("document_id")
            if chunk_doc is not None and not isinstance(chunk_doc, str):
                raise HttpRagResponseError(f"Chunk at index {idx} has invalid 'document_id': must be string")

    citations = raw_data.get("citations")
    if citations is not None:
        if not isinstance(citations, list):
            raise HttpRagResponseError(f"Field 'citations' must be a list, got {type(citations).__name__}")
        for idx, cit in enumerate(citations):
            if not isinstance(cit, dict):
                raise HttpRagResponseError(f"Citation at index {idx} in 'citations' must be a dict")
            claim_text = cit.get("claim_text")
            if claim_text is not None and not isinstance(claim_text, str):
                raise HttpRagResponseError(f"Citation at index {idx} has invalid 'claim_text': must be string")

    telemetry = raw_data.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, dict):
        raise HttpRagResponseError(f"Field 'telemetry' must be a dict, got {type(telemetry).__name__}")

    return raw_data


class PythonAdapterRegistry:
    """Trusted server-side registry mapping adapter names to verified Python callables.

    Prevents untrusted remote code execution from JSON payloads while allowing
    pre-registered Python callables to be referenced safely by name via API.

    Multi-process note: When running distributed workers across separate processes,
    callables must be registered during application initialization in each worker process.
    The REST API strictly disallows dynamic code submission, only referencing registered names.
    """

    _registry: dict[str, Callable[[TestCase, RunConfig], RagTrace | Any]] = {}
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def register(cls, name: str, fn: Callable[[TestCase, RunConfig], RagTrace | Any]) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Adapter name must be a non-empty string.")
        if not callable(fn):
            raise ValueError(f"Adapter handler for '{name}' must be callable.")
        with cls._lock:
            cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Callable[[TestCase, RunConfig], RagTrace | Any]:
        with cls._lock:
            if name not in cls._registry:
                available = sorted(list(cls._registry.keys()))
                raise ValueError(f"Unknown registered python adapter '{name}'. Available: {available}")
            return cls._registry[name]

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._registry.clear()


class PythonRagAdapter:
    """Wraps an in-process Python callable into a RagAdapter."""

    def __init__(self, target_fn: Callable[[TestCase, RunConfig], RagTrace | Any]) -> None:
        if target_fn is None or not callable(target_fn):
            raise ValueError("PythonRagAdapter requires a non-None, callable target_fn.")
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
        return RagTrace(
            trace_id=generate_id("tr"),
            run_id=generate_id("run_adhoc"),
            test_case_id=case.id,
            question=case.question,
            answer=str(res),
            latency_ms=latency_ms,
        )


class HttpRagAdapter:
    """Invokes an external HTTP RAG endpoint and normalizes to canonical RagTrace.

    Supports reusable connection pooling via shared httpx.AsyncClient to minimize
    TCP handshake latency across high-throughput evaluation suites.
    Includes strict SSRF protection: validates destinations before connect and re-validates
    every redirect hop against private, link-local, loopback, and metadata ranges.
    """

    def __init__(
        self,
        endpoint_url: str,
        headers: dict[str, str] | None = None,
        timeout_seconds: float = 30.0,
        client: httpx.AsyncClient | None = None,
        allowed_hosts: list[str] | set[str] | None = None,
        allow_private_ip: bool = False,
        dns_resolver: Any = None,
    ) -> None:
        self.endpoint_url = endpoint_url
        self.headers = headers or {}
        self.timeout_seconds = timeout_seconds
        self.allowed_hosts = allowed_hosts
        self.allow_private_ip = allow_private_ip
        self.dns_resolver = dns_resolver
        self._shared_client = client
        self._owns_client = client is None

    def _pin_host(self, client: httpx.AsyncClient, host: str, ip: str) -> None:
        """Encapsulate IP pinning behind transport interface without exposing internal details."""
        transport = getattr(client, "_transport", None)
        pin_fn = getattr(transport, "pin_host", None)
        if callable(pin_fn):
            pin_fn(host, ip)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._shared_client is not None and not self._shared_client.is_closed:
            return self._shared_client
        from rag_platform.ssrf import SSRFProtectedTransport
        transport = SSRFProtectedTransport(
            allowed_hosts=self.allowed_hosts,
            allow_private_ips=self.allow_private_ip,
            dns_resolver=self.dns_resolver,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=50),
        )
        self._shared_client = httpx.AsyncClient(
            transport=transport,
            timeout=self.timeout_seconds,
            follow_redirects=False,
        )
        self._owns_client = True
        return self._shared_client

    async def close(self) -> None:
        """Close internal HTTP client pool if owned."""
        if self._owns_client and self._shared_client is not None and not self._shared_client.is_closed:
            await self._shared_client.aclose()

    async def __aenter__(self) -> HttpRagAdapter:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def run(self, case: TestCase, config: RunConfig) -> RagTrace:
        start = time.perf_counter()
        trace_id = generate_id("trace")
        payload = {
            "question": case.question,
            "test_case_id": case.id,
            "metadata": case.metadata,
        }

        try:
            current_url = self.endpoint_url
            # Pre-flight SSRF validation immediately before connecting
            validated_ips = validate_url_ssrf(
                current_url,
                allowed_hosts=self.allowed_hosts,
                allow_private_ips=self.allow_private_ip,
                dns_resolver=self.dns_resolver,
            )

            client = await self._get_client()

            # Pin validated host IP on transport if supported
            parsed_host = urlsplit(current_url).hostname
            if parsed_host and validated_ips:
                self._pin_host(client, parsed_host, validated_ips[0])

            # Follow redirects manually with strict SSRF validation at every hop
            max_redirects = 5
            redirect_count = 0
            max_retries = 3
            retry_count = 0
            # Build request headers with trace ID and idempotency key to protect SUT against duplicated side-effects during retries
            req_headers = dict(self.headers)
            req_headers.setdefault("X-Request-ID", trace_id)
            req_headers.setdefault("Idempotency-Key", f"eval_{config.project_id}_{config.dataset_id}_{case.id}_{trace_id}")
            retry_status_codes = {429, 502, 503, 504}

            while True:
                try:
                    resp = await client.post(current_url, json=payload, headers=req_headers)
                except (httpx.ConnectError, httpx.RemoteProtocolError) as net_err:
                    if retry_count < max_retries:
                        retry_count += 1
                        backoff = min(2.0, 0.25 * (2 ** retry_count)) + 0.05
                        await asyncio.sleep(backoff)
                        continue
                    raise net_err

                # Transient server / rate limit retries (429, 502, 503, 504)
                if resp.status_code in retry_status_codes and retry_count < max_retries:
                    retry_count += 1
                    retry_after = resp.headers.get("retry-after")
                    backoff = min(2.0, 0.25 * (2 ** retry_count)) + 0.05
                    if retry_after:
                        try:
                            backoff = min(5.0, float(retry_after))
                        except ValueError:
                            pass
                    await asyncio.sleep(backoff)
                    continue

                if resp.is_redirect and "location" in resp.headers:
                    redirect_count += 1
                    if redirect_count > max_redirects:
                        raise httpx.TooManyRedirects("Exceeded maximum redirect hops")
                    location = resp.headers["location"]
                    current_url = urljoin(current_url, location)
                    redir_host = urlsplit(current_url).hostname
                    # Revalidate every redirect destination before following
                    redirect_ips = validate_url_ssrf(
                        current_url,
                        allowed_hosts=self.allowed_hosts,
                        allow_private_ips=self.allow_private_ip,
                        dns_resolver=self.dns_resolver,
                    )
                    if redir_host and redirect_ips:
                        self._pin_host(client, redir_host, redirect_ips[0])
                    continue
                break

            latency_ms = int((time.perf_counter() - start) * 1000)

            if resp.status_code >= 400:
                # Truncate and redact provider error response to prevent data leakage in telemetry
                safe_body = SecretRedactor.redact_text(resp.text[:500])
                return RagTrace(
                    trace_id=trace_id,
                    run_id=generate_id("run"),
                    test_case_id=case.id,
                    question=case.question,
                    error_code="OPS-01",
                    latency_ms=latency_ms,
                    telemetry={
                        "http_status": resp.status_code,
                        "body": safe_body,
                        "retries": retry_count,
                    },
                )

            try:
                raw_json = resp.json()
            except Exception as json_err:
                return RagTrace(
                    trace_id=trace_id,
                    run_id=generate_id("run"),
                    test_case_id=case.id,
                    question=case.question,
                    error_code="OPS-01",
                    latency_ms=latency_ms,
                    telemetry={
                        "error": f"Failed to parse JSON from adapter response: {json_err}",
                        "raw_body": SecretRedactor.redact_text(resp.text[:500]),
                        "stage": "adapter_response_validation",
                    },
                )

            try:
                data = validate_http_rag_response(raw_json)
            except HttpRagResponseError as val_err:
                return RagTrace(
                    trace_id=trace_id,
                    run_id=generate_id("run"),
                    test_case_id=case.id,
                    question=case.question,
                    error_code="OPS-01",
                    latency_ms=latency_ms,
                    telemetry={
                        "error": f"Adapter response schema validation error: {val_err}",
                        "stage": "adapter_response_validation",
                    },
                )

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
                telemetry=SecretRedactor.redact_dict(data.get("telemetry", {})),
            )
        except SSRFProtectionError as ex:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return RagTrace(
                trace_id=trace_id,
                run_id=generate_id("run"),
                test_case_id=case.id,
                question=case.question,
                error_code="OPS-01",
                latency_ms=latency_ms,
                telemetry={"exception": str(ex), "type": "SSRFProtectionError"},
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
                telemetry={"exception": SecretRedactor.redact_text(str(ex)), "type": type(ex).__name__},
            )


# --- Adapter Registry ---
class AdapterRegistry:
    """Central registry mapping adapter identifiers to adapter factories."""

    _factories: dict[str, Callable[..., RagAdapter]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., RagAdapter]) -> None:
        cls._factories[name.lower()] = factory

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> RagAdapter:
        key = name.lower()
        if key not in cls._factories:
            raise ValueError(f"Unknown adapter '{name}'. Registered: {cls.available()}")
        return cls._factories[key](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        return sorted(list(cls._factories.keys()))


# --- Platform Self-Test: Flawed Synthetic RAGs (Development & Demo Only) ---
class SyntheticRagMode(str, Enum):
    PERFECT = "PERFECT"
    DISTRACTOR = "DISTRACTOR"           # RET-01 / RET-02
    HALLUCINATING = "HALLUCINATING"     # GEN-01 / GEN-02
    BROKEN_CITATION = "BROKEN_CITATION" # CIT-01
    REFUSAL_BYPASS = "REFUSAL_BYPASS"   # ABS-01
    TIMEOUT = "TIMEOUT"                 # OPS-01


class SyntheticRagAdapter:
    """Synthetic SUT adapter for offline development, integration tests, and platform calibration.

    NOT intended for production SUT evaluation.
    """
    is_development_adapter: bool = True
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


def _make_python_adapter(
    target_fn: Callable[[TestCase, RunConfig], RagTrace | Any] | None = None,
    adapter_name: str | None = None,
    **_: Any,
) -> PythonRagAdapter:
    if target_fn is not None:
        return PythonRagAdapter(target_fn)
    if adapter_name is not None:
        fn = PythonAdapterRegistry.get(adapter_name)
        return PythonRagAdapter(fn)
    raise ValueError(
        "Python adapter requires a registered 'adapter_name' in adapter_config or a callable 'target_fn'."
    )


# Register standard built-in adapters
AdapterRegistry.register(
    "synthetic",
    lambda mode=SyntheticRagMode.PERFECT, **_: SyntheticRagAdapter(
        SyntheticRagMode(mode) if isinstance(mode, str) else mode
    ),
)
AdapterRegistry.register("python", _make_python_adapter)
AdapterRegistry.register(
    "http",
    lambda endpoint_url="", headers=None, timeout_seconds=30.0, **_: HttpRagAdapter(
        endpoint_url=endpoint_url, headers=headers, timeout_seconds=timeout_seconds
    ),
)


