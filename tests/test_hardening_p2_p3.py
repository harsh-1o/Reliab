"""Tests for P2 and P3 hardening:
1. init_db() raises RuntimeError on persistent DB without allow_non_memory
2. wilson_score_interval() calculates correct z-quantile for variable confidence levels
3. HttpRagAdapter sends X-Request-ID and Idempotency-Key headers
4. RunProvenance resolves requirements.lock deterministically from repository root
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine

from rag_platform.core import sha256_hash
from rag_platform.db import init_db
from rag_platform.evaluators import wilson_score_interval
from rag_platform.models import RunProvenance


def test_init_db_raises_runtime_error_on_persistent_db(tmp_path: Path):
    """init_db() must strictly forbid create_all() on persistent DBs unless allow_non_memory=True."""
    db_file = tmp_path / "persistent.db"
    persistent_engine = create_engine(f"sqlite:///{db_file}")

    # Calling without allow_non_memory=True must raise RuntimeError
    with pytest.raises(RuntimeError) as exc_info:
        init_db(persistent_engine, allow_non_memory=False)

    assert "strictly prohibited on persistent databases" in str(exc_info.value)

    # Calling with allow_non_memory=True should succeed
    init_db(persistent_engine, allow_non_memory=True)
    persistent_engine.dispose()


def test_wilson_score_interval_confidence_scaling():
    """wilson_score_interval() must scale interval width with requested confidence level."""
    p = 0.8
    n = 100

    lower_90, upper_90 = wilson_score_interval(p, n, confidence=0.90)
    lower_95, upper_95 = wilson_score_interval(p, n, confidence=0.95)
    lower_99, upper_99 = wilson_score_interval(p, n, confidence=0.99)

    width_90 = upper_90 - lower_90
    width_95 = upper_95 - lower_95
    width_99 = upper_99 - lower_99

    # Higher confidence level must result in strictly wider intervals
    assert width_90 < width_95 < width_99

    # Check approximate known theoretical values for p=0.8, n=100
    # 90% (z ~ 1.645): [0.7247, 0.8586]
    # 95% (z ~ 1.960): [0.7111, 0.8666]
    # 99% (z ~ 2.576): [0.6841, 0.8814]
    assert 0.72 <= lower_90 <= 0.73
    assert 0.85 <= upper_90 <= 0.86
    assert 0.70 <= lower_95 <= 0.72
    assert 0.86 <= upper_95 <= 0.87
    assert 0.67 <= lower_99 <= 0.69
    assert 0.88 <= upper_99 <= 0.89

    # Invalid confidence bounds must raise ValueError
    with pytest.raises(ValueError):
        wilson_score_interval(p, n, confidence=0.0)
    with pytest.raises(ValueError):
        wilson_score_interval(p, n, confidence=1.0)


@pytest.mark.asyncio
async def test_http_rag_adapter_sends_idempotency_and_request_id_headers(monkeypatch):
    """HttpRagAdapter must include X-Request-ID and Idempotency-Key headers on POST calls."""
    from rag_platform.adapters import HttpRagAdapter
    from rag_platform.models import RunConfig, TestCase

    sent_headers: list[dict[str, str]] = []

    class MockResponse:
        status_code = 200
        is_redirect = False
        headers: dict[str, str] = {}

        def json(self):
            return {
                "answer": "Idempotent answer",
                "retrieved_documents": [{"document_id": "d1", "chunk_id": "c1", "text": "evidence"}],
                "citations": [],
            }

    class MockAsyncClient:
        is_closed = False

        async def post(self, url, json=None, headers=None):
            sent_headers.append(dict(headers or {}))
            return MockResponse()

        async def aclose(self):
            self.is_closed = True

    adapter = HttpRagAdapter(
        endpoint_url="https://sut.internal.example/api/rag",
        headers={"Authorization": "Bearer sut_token"},
    )
    # Inject mocked client and bypass DNS/network validation for testing
    adapter._shared_client = MockAsyncClient()  # type: ignore[assignment]
    monkeypatch.setattr("rag_platform.adapters.validate_url_ssrf", lambda *args, **kwargs: ["93.184.216.34"])

    case = TestCase(id="tc_idem_1", question="What is idempotency?")
    config = RunConfig(
        project_id="proj_idem",
        dataset_id="ds_idem",
        dataset_version="1.0.0",
        system_version="v1.0.0",
    )

    trace = await adapter.run(case, config)
    assert trace.answer == "Idempotent answer"
    assert len(sent_headers) == 1

    headers = sent_headers[0]
    assert "Authorization" in headers
    assert "X-Request-ID" in headers
    assert "Idempotency-Key" in headers
    assert headers["Idempotency-Key"].startswith("eval_proj_idem_ds_idem_tc_idem_1_")


def test_run_provenance_deterministic_repo_root_lockfile_resolution():
    """RunProvenance must resolve requirements.lock relative to repo root even outside repo cwd."""
    provenance = RunProvenance(dataset_checksum="test_checksum_123")
    lock_file = Path(__file__).resolve().parent.parent / "requirements.lock"
    if lock_file.is_file():
        expected_hash = sha256_hash(lock_file.read_text(encoding="utf-8"))
        assert provenance.dependency_lock_hash == expected_hash
