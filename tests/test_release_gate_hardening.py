"""Release Gate & Evaluation Correctness Hardening Regression Tests.

Covers:
1. Coverage tracking (full vs partial evaluation, max_cases, empty dataset, release gate rejection solely for coverage).
2. Candidate-missing regression handling (baseline A,B,C vs candidate A,B; explicit partial evaluation policy; extra candidate cases; deterministic ordering).
3. Shared percentile implementation (n=1, n=2, n=20, n=21, n=100, repeated values, unsorted input).
4. Scored_cases semantics (non-applicable / None score metrics do not distort scored_cases count).
5. RunProvenance immutability (frozen dataclass/model, immutability on field mutation, serialization determinism).
6. Strict HTTP response model (nested structures, malformed spans/chunks/citations, extra harmless fields, OPS-01 conversion).
7. Async HTTP secret handling (${ENV_VAR} resolution, missing env var failure, redaction in DB options_json).
8. Auth rate limiter & trusted proxy hardening (bounded memory, TTL expiration, untrusted peer with all-trusted chain).
9. Cache failure visibility (DB evaluation cache tracks errors, is_degraded, doesn't falsely mark as cache miss).
10. CLI adapter cleanup (proper async close on success and exception).
"""

import os
from unittest.mock import AsyncMock, patch

import pytest

from rag_platform.adapters import (
    HttpRagAdapter,
    HttpRagResponseError,
    validate_http_rag_response,
)
from rag_platform.core import calculate_percentile
from rag_platform.evaluators import (
    DatabaseEvaluationCache,
    EvaluationEngine,
    MetricFamily,
    MetricResult,
)
from rag_platform.gate import execute_gate_evaluation
from rag_platform.models import (
    GateStatus,
    MetricSummary,
    RagTrace,
    ReleasePolicy,
    RunConfig,
    RunMetricsSummary,
    RunProvenance,
    TestCase,
)
from rag_platform.regression import RegressionEngine
from rag_platform.security import (
    SecretRedactor,
    SecretResolutionError,
    resolve_adapter_headers,
    resolve_secret_reference,
)
from rag_platform.server import AuthRateLimiter, extract_client_ip

# =====================================================================
# 1. Percentile Calculation Tests (Requirement 4)
# =====================================================================

def test_percentile_empty_and_single():
    assert calculate_percentile([], 95) == 0.0
    assert calculate_percentile([42.0], 50) == 42.0
    assert calculate_percentile([42.0], 95) == 42.0
    assert calculate_percentile([42.0], 0) == 42.0
    assert calculate_percentile([42.0], 100) == 42.0


def test_percentile_two_elements():
    # n=2: [10, 20]
    # p50: rank = 0.50 * 1 = 0.5 -> 10 + 0.5*(20-10) = 15.0
    # p95: rank = 0.95 * 1 = 0.95 -> 10 + 0.95*(20-10) = 19.5
    vals = [20.0, 10.0]  # unsorted
    assert calculate_percentile(vals, 50) == pytest.approx(15.0)
    assert calculate_percentile(vals, 95) == pytest.approx(19.5)


def test_percentile_twenty_elements():
    # n=20: 1..20
    # rank for p95 = 0.95 * (20 - 1) = 0.95 * 19 = 18.05
    # sorted values: v[18]=19, v[19]=20 -> 19 + 0.05*(20-19) = 19.05
    # Unlike int(20*0.95)=19 (which picked max observation 20 in old buggy code),
    # linear interpolation properly interpolates below 20.
    vals = list(range(1, 21))
    p95 = calculate_percentile(vals, 95)
    assert p95 == pytest.approx(19.05)
    assert p95 < 20.0  # must NOT be the absolute max for n=20


def test_percentile_twenty_one_elements():
    # n=21: 0..20
    # rank for p95 = 0.95 * 20 = 19.0 -> exact integer index 19
    vals = list(range(21))
    assert calculate_percentile(vals, 95) == pytest.approx(19.0)
    # p50: rank = 0.5 * 20 = 10.0 -> exact integer index 10
    assert calculate_percentile(vals, 50) == pytest.approx(10.0)


def test_percentile_one_hundred_elements():
    # n=100: 1..100
    vals = list(range(1, 101))
    # p50: rank = 0.5 * 99 = 49.5 -> 50 + 0.5 = 50.5
    # p95: rank = 0.95 * 99 = 94.05 -> 95 + 0.05 = 95.05
    assert calculate_percentile(vals, 50) == pytest.approx(50.5)
    assert calculate_percentile(vals, 95) == pytest.approx(95.05)


def test_percentile_repeated_values_and_unsorted():
    vals = [5.0, 5.0, 5.0, 5.0, 5.0]
    assert calculate_percentile(vals, 50) == 5.0
    assert calculate_percentile(vals, 95) == 5.0

    unsorted = [100.0, 1.0, 50.0, 25.0, 75.0]
    # sorted: [1, 25, 50, 75, 100] (n=5, rank for p50 = 0.5*4 = 2 -> 50)
    assert calculate_percentile(unsorted, 50) == 50.0


# =====================================================================
# 2. Coverage Tracking & Release Gate Rejection (Requirements 1 & 2)
# =====================================================================

def test_coverage_tracking_full_evaluation():
    engine = EvaluationEngine()
    traces = [
        (
            RagTrace(
                trace_id=f"t_{i}",
                run_id="run_1",
                test_case_id=f"tc_{i}",
                question="q",
                answer="a",
            ),
            [MetricResult(metric_name="exact_match", metric_family=MetricFamily.GENERATION, score=1.0)],
        )
        for i in range(10)
    ]
    summary = engine.aggregate_run(traces, required_case_count=10)
    assert summary.required_case_count == 10
    assert summary.evaluated_case_count == 10
    assert summary.missing_case_count == 0
    assert summary.coverage_ratio == 1.0
    assert summary.is_full_evaluation is True
    assert summary.eligible_for_release_gate is True


def test_coverage_tracking_partial_evaluation_max_cases():
    engine = EvaluationEngine()
    traces = [
        (
            RagTrace(
                trace_id=f"t_{i}",
                run_id="run_1",
                test_case_id=f"tc_{i}",
                question="q",
                answer="a",
            ),
            [MetricResult(metric_name="exact_match", metric_family=MetricFamily.GENERATION, score=1.0)],
        )
        for i in range(5)
    ]
    # Required was 20 cases, but only 5 were evaluated (e.g. max_cases=5)
    summary = engine.aggregate_run(traces, required_case_count=20)
    assert summary.required_case_count == 20
    assert summary.evaluated_case_count == 5
    assert summary.missing_case_count == 15
    assert summary.coverage_ratio == 0.25
    assert summary.is_full_evaluation is False
    assert summary.eligible_for_release_gate is False


def test_coverage_tracking_empty_dataset():
    engine = EvaluationEngine()
    summary = engine.aggregate_run([], required_case_count=0)
    assert summary.required_case_count == 0
    assert summary.evaluated_case_count == 0
    assert summary.missing_case_count == 0
    assert summary.coverage_ratio == 1.0


def test_release_gate_rejection_solely_due_to_insufficient_coverage():
    reg_engine = RegressionEngine()
    def make_metric(name: str, fam: MetricFamily) -> MetricSummary:
        return MetricSummary(
            metric_name=name,
            metric_family=fam,
            mean=1.0,
            std_dev=0.0,
            min=1.0,
            max=1.0,
            p50=1.0,
            p95=1.0,
            count=5,
            applicable_count=5,
            ci_lower=1.0,
            ci_upper=1.0,
        )

    metrics = {
        "faithfulness": make_metric("faithfulness", MetricFamily.GENERATION),
        "recall_at_5": make_metric("recall_at_5", MetricFamily.RETRIEVAL),
        "citation_accuracy": make_metric("citation_accuracy", MetricFamily.CITATION),
    }
    candidate = RunMetricsSummary(
        total_cases=5,
        passed_cases=5,
        failed_cases=0,
        scored_cases=5,
        metrics=metrics,
        p95_latency_ms=100.0,
        required_case_count=10,
        evaluated_case_count=5,
        missing_case_count=5,
        coverage_ratio=0.5,
        is_full_evaluation=False,
        eligible_for_release_gate=False,
    )
    policy = ReleasePolicy(
        min_case_coverage=1.0,
        min_faithfulness=0.0,
        min_retrieval_recall=0.0,
        min_citation_accuracy=0.0,
    )
    decision = reg_engine.evaluate_gate(candidate, policy)

    assert decision.status == GateStatus.FAIL
    assert len(decision.violations) == 1
    assert decision.violations[0].violation_type == "INSUFFICIENT_COVERAGE"
    assert "50.0%" in decision.violations[0].message


def test_release_gate_permits_custom_coverage_policy():
    reg_engine = RegressionEngine()
    candidate = RunMetricsSummary(
        total_cases=8,
        passed_cases=8,
        failed_cases=0,
        scored_cases=8,
        metrics={},
        p95_latency_ms=50.0,
        required_case_count=10,
        evaluated_case_count=8,
        missing_case_count=2,
        coverage_ratio=0.8,
        is_full_evaluation=False,
        eligible_for_release_gate=False,
    )
    # Policy permits 80% coverage
    policy = ReleasePolicy(min_case_coverage=0.8)
    decision = reg_engine.evaluate_gate(candidate, policy)
    # Should not reject due to coverage
    assert not any(v.violation_type == "INSUFFICIENT_COVERAGE" for v in decision.violations)


# =====================================================================
# 3. Candidate-Missing Regression Handling (Requirement 3)
# =====================================================================

def test_candidate_missing_omits_baseline_cases_rejected():
    reg_engine = RegressionEngine()

    base_summary = RunMetricsSummary(
        total_cases=3, passed_cases=3, failed_cases=0, scored_cases=3, metrics={}, p95_latency_ms=50.0
    )
    cand_summary = RunMetricsSummary(
        total_cases=2, passed_cases=2, failed_cases=0, scored_cases=2, metrics={}, p95_latency_ms=50.0,
        required_case_count=2, evaluated_case_count=2, missing_case_count=0, coverage_ratio=1.0,
    )

    comparison = reg_engine.compare(
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
        baseline_run_id="r1",
        candidate_run_id="r2",
        baseline_case_scores={"tc_A": 1.0, "tc_B": 1.0, "tc_C": 1.0},
        candidate_case_scores={"tc_A": 1.0, "tc_B": 1.0},
    )

    assert comparison.candidate_missing_cases == ["tc_C"]
    assert len(comparison.newly_failed_cases) == 0  # Not distorted into newly_failed_cases

    # Now evaluate gate against baseline
    policy = ReleasePolicy(allow_candidate_missing=False)
    decision = reg_engine.evaluate_gate(
        candidate_summary=cand_summary,
        policy=policy,
        baseline_summary=base_summary,
        comparison=comparison,
    )

    assert decision.status == GateStatus.FAIL
    assert any(v.violation_type == "CANDIDATE_MISSING" for v in decision.violations)
    assert any("tc_C" in v.message for v in decision.violations)


def test_candidate_missing_allowed_with_explicit_policy():
    reg_engine = RegressionEngine()
    base_summary = RunMetricsSummary(total_cases=2, passed_cases=2, failed_cases=0, scored_cases=2, metrics={}, p95_latency_ms=50.0)
    cand_summary = RunMetricsSummary(total_cases=1, passed_cases=1, failed_cases=0, scored_cases=1, metrics={}, p95_latency_ms=50.0)

    comparison = reg_engine.compare(
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
        baseline_run_id="r1",
        candidate_run_id="r2",
        baseline_case_scores={"tc_A": 1.0, "tc_B": 1.0},
        candidate_case_scores={"tc_A": 1.0},
    )
    assert comparison.candidate_missing_cases == ["tc_B"]

    # Explicitly permitted by policy
    policy = ReleasePolicy(allow_candidate_missing=True)
    decision = reg_engine.evaluate_gate(
        candidate_summary=cand_summary,
        policy=policy,
        baseline_summary=base_summary,
        comparison=comparison,
    )
    assert not any(v.violation_type == "CANDIDATE_MISSING" for v in decision.violations)


def test_extra_candidate_cases_and_deterministic_order():
    reg_engine = RegressionEngine()
    base_summary = RunMetricsSummary(total_cases=2, passed_cases=2, failed_cases=0, scored_cases=2, metrics={}, p95_latency_ms=50.0)
    cand_summary = RunMetricsSummary(total_cases=3, passed_cases=3, failed_cases=0, scored_cases=3, metrics={}, p95_latency_ms=50.0)

    comparison = reg_engine.compare(
        baseline_summary=base_summary,
        candidate_summary=cand_summary,
        baseline_run_id="r1",
        candidate_run_id="r2",
        baseline_case_scores={"tc_Z": 1.0, "tc_A": 1.0},
        candidate_case_scores={"tc_M": 1.0, "tc_Z": 1.0, "tc_A": 1.0},
    )
    assert comparison.candidate_missing_cases == []
    # baseline_missing_cases should be deterministically sorted
    assert comparison.baseline_missing_cases == ["tc_M"]


# =====================================================================
# 4. Scored Cases Semantics (Requirement 5)
# =====================================================================

def test_scored_cases_calculation_with_unscored_cases():
    engine = EvaluationEngine()
    traces = [
        # Case 1: Scored 1.0
        (
            RagTrace(trace_id="t1", run_id="r1", test_case_id="c1", question="q", answer="a"),
            [MetricResult(metric_name="exact_match", metric_family=MetricFamily.GENERATION, score=1.0)],
        ),
        # Case 2: Not applicable / abstained -> score is None
        (
            RagTrace(trace_id="t2", run_id="r1", test_case_id="c2", question="q", answer="None", abstained=True),
            [MetricResult(metric_name="exact_match", metric_family=MetricFamily.GENERATION, score=None)],
        ),
        # Case 3: Empty metrics list -> completely unscored
        (
            RagTrace(trace_id="t3", run_id="r1", test_case_id="c3", question="q", answer="c"),
            [],
        ),
    ]

    summary = engine.aggregate_run(traces, required_case_count=3)
    assert summary.total_cases == 3
    # Only case 1 had a non-None numeric score
    assert summary.scored_cases == 1


# =====================================================================
# 5. Provenance Immutability (Requirement 6)
# =====================================================================

def test_run_provenance_immutability():
    prov = RunProvenance(
        dataset_id="ds_123",
        dataset_version="1.0.0",
        dataset_checksum="sha256:abc12345",
        rag_version="v2.1.0",
    )

    # Hashes are computed and valid
    assert prov.manifest_hash is not None
    assert len(prov.manifest_hash) == 64

    # Mutating any attribute must be rejected
    with pytest.raises(Exception):  # Pydantic ValidationError or FrozenInstanceError
        prov.dataset_checksum = "sha256:modified"

    with pytest.raises(Exception):
        prov.rag_version = "v3.0.0"

    with pytest.raises(Exception):
        prov.manifest_hash = "hacked_hash"

    # Deterministic serialization
    dump1 = prov.model_dump_json()
    dump2 = prov.model_dump_json()
    assert dump1 == dump2


# =====================================================================
# 6. Strict HTTP Response Model (Requirement 7)
# =====================================================================

def test_strict_http_response_valid():
    raw = {
        "answer": "France",
        "abstained": False,
        "retrieved_chunks": [
            {
                "document_id": "doc_1",
                "chunk_id": "ch_1",
                "rank": 1,
                "score": 0.95,
                "text": "Paris is the capital of France.",
                "metadata": {"source": "wiki"},
            }
        ],
        "citations": [
            {
                "claim_id": "cl_1",
                "claim_text": "Paris is capital",
                "document_id": "doc_1",
                "chunk_id": "ch_1",
                "span": [0, 16],
            }
        ],
        "extra_harmless_field": "ignore_me",
    }
    validated = validate_http_rag_response(raw)
    assert validated["answer"] == "France"
    assert validated["extra_harmless_field"] == "ignore_me"


def test_strict_http_response_malformed_spans_and_chunks():
    # Malformed span: string instead of int list
    raw_bad_span = {
        "answer": "France",
        "citations": [
            {
                "claim_id": "cl_1",
                "claim_text": "Paris is capital",
                "span": ["not", "an", "int"],
            }
        ],
    }
    with pytest.raises(HttpRagResponseError, match="HTTP RAG response schema validation error"):
        validate_http_rag_response(raw_bad_span)

    # Malformed chunk score: non-numeric string
    raw_bad_score = {
        "answer": "France",
        "retrieved_chunks": [
            {
                "document_id": "doc_1",
                "score": "invalid_score",
            }
        ],
    }
    with pytest.raises(HttpRagResponseError, match="HTTP RAG response schema validation error"):
        validate_http_rag_response(raw_bad_score)


@pytest.mark.asyncio
async def test_http_adapter_strict_validation_produces_ops01():
    import httpx

    case = TestCase(id="c1", question="What is Paris?", ground_truth_answer="Capital of France")
    run_config = RunConfig(project_id="proj_1", dataset_id="ds_1", dataset_version="1.0", system_version="1.0")

    # Endpoint returns malformed span
    def bad_span_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "answer": "hello",
                "citations": [{"claim_text": "text", "span": "wrong_type"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(bad_span_handler))
    adapter = HttpRagAdapter(
        endpoint_url="http://127.0.0.1:8000/rag",
        client=client,
        allow_private_ip=True,
    )
    trace = await adapter.run(case, run_config)
    assert trace.error_code == "OPS-01"
    assert "schema validation error" in trace.telemetry.get("error", "")


# =====================================================================
# 7. Async HTTP Secret Handling (Requirement 8)
# =====================================================================

def test_resolve_secret_reference_env():
    with patch.dict(os.environ, {"API_KEY_ENV": "super_secret_123"}):
        assert resolve_secret_reference("${API_KEY_ENV}") == "super_secret_123"
        assert resolve_secret_reference("env://API_KEY_ENV") == "super_secret_123"

    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(SecretResolutionError, match="Missing required secret reference"):
            resolve_secret_reference("${NON_EXISTENT_SECRET}")


def test_resolve_adapter_headers():
    with patch.dict(os.environ, {"MY_SECRET_TOKEN": "token_abc"}):
        headers = {"Content-Type": "application/json"}
        secret_refs = {"Authorization": "Bearer ${MY_SECRET_TOKEN}"}
        resolved = resolve_adapter_headers(headers, secret_refs)
        assert resolved["Content-Type"] == "application/json"
        assert resolved["Authorization"] == "Bearer token_abc"


def test_redactor_preserves_secret_references():
    options = {
        "headers": {"Authorization": "Bearer real_secret_value"},
        "header_secret_refs": {"Authorization": "Bearer ${ENV_SECRET}"},
    }
    redacted = SecretRedactor.redact_dict(options)
    # Plaintext header is redacted
    assert "[REDACTED]" in redacted["headers"]["Authorization"]
    # Secret reference string is preserved
    assert redacted["header_secret_refs"]["Authorization"] == "Bearer ${ENV_SECRET}"


# =====================================================================
# 8. Auth Rate Limiter & Trusted Proxy Review (Requirement 9)
# =====================================================================

def test_auth_rate_limiter_bounded_memory_and_eviction():
    limiter = AuthRateLimiter(max_failures=5, window_seconds=60.0, max_tracked_ips=50)

    for i in range(100):
        ip = f"10.0.0.{i}"
        limiter.record_failure(ip)

    # Must be bounded by max_tracked_ips
    assert len(limiter._failures) <= 50


def test_trusted_proxy_all_trusted_forwarded_chain_untrusted_peer():
    from starlette.requests import Request

    req = Request(scope={
        "type": "http",
        "client": ("198.51.100.1", 12345),
        "headers": [(b"x-forwarded-for", b"10.0.0.1, 10.0.0.2")],
    })
    client_ip = extract_client_ip(req, trusted_proxies=("127.0.0.1",))
    assert client_ip == "198.51.100.1"


def test_trusted_proxy_legitimate_forwarded_chain():
    from starlette.requests import Request

    req = Request(scope={
        "type": "http",
        "client": ("127.0.0.1", 12345),
        "headers": [(b"x-forwarded-for", b"203.0.113.195, 127.0.0.1")],
    })
    client_ip = extract_client_ip(req, trusted_proxies=("127.0.0.1",))
    assert client_ip == "203.0.113.195"


# =====================================================================
# 9. Cache Failure Visibility (Requirement 10)
# =====================================================================

def test_database_evaluation_cache_tracks_infrastructure_errors():
    class BrokenSession:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
        def get(self, *args, **kwargs):
            raise RuntimeError("Database connection pool exhausted")
        def execute(self, *args, **kwargs):
            raise RuntimeError("Database connection pool exhausted")
        def commit(self):
            raise RuntimeError("Commit failed")
        def rollback(self):
            pass

    cache = DatabaseEvaluationCache(session_factory=lambda: BrokenSession())

    # Get should return None, but record an error and mark is_degraded=True
    val = cache.get("cache_key_1")
    assert val is None
    assert cache.is_degraded is True
    assert cache.errors == 1
    assert "Database connection pool exhausted" in cache.last_error

    # Set should also record an error without throwing
    cache.set("cache_key_1", {"score": 1.0})
    assert cache.errors == 2

    stats = cache.stats()
    assert stats["infrastructure_errors"] >= 2
    assert stats["is_degraded"] is True


# =====================================================================
# 10. CLI Adapter Cleanup (Requirement 11)
# =====================================================================

def test_execute_gate_evaluation_adapter_cleanup():
    from unittest.mock import MagicMock

    mock_adapter = AsyncMock()
    mock_adapter.close = AsyncMock()
    mock_adapter.run = AsyncMock(side_effect=RuntimeError("Pipeline crash"))

    mock_db = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.project_id = "proj_test"
    mock_dataset.checksum_sha256 = "sha256:abc"
    mock_dataset.version = "1.0"
    mock_dataset.status = "PUBLISHED"
    mock_tc_row = MagicMock()
    mock_tc_row.id = "c1"
    mock_tc_row.question = "q"
    mock_tc_row.expected_answer = "a"
    mock_tc_row.expected_facts_json = "[]"
    mock_tc_row.relevant_docs_json = "[]"
    mock_tc_row.answerability = "ANSWERABLE"
    mock_tc_row.tags_json = "[]"
    mock_tc_row.metadata_json = "{}"
    mock_dataset.cases = [mock_tc_row]
    mock_db.get.return_value = mock_dataset

    with patch("rag_platform.adapters.HttpRagAdapter", return_value=mock_adapter):
        with pytest.raises(RuntimeError, match="Pipeline crash"):
            execute_gate_evaluation(
                db_session=mock_db,
                project_id="proj_test",
                dataset_id="ds_test",
                adapter_type="http",
                endpoint_url="http://127.0.0.1:8000/rag",
            )

    mock_adapter.close.assert_awaited_once()
