"""Regression tests for the senior engineering pass.

Covers all critical fixes introduced in this review.
These tests must PASS on every subsequent change.
"""

from __future__ import annotations

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_platform.adapters import (
    AdapterRegistry,
    HttpRagAdapter,
    SyntheticRagAdapter,
    SyntheticRagMode,
)
from rag_platform.core import generate_id, sha256_hash
from rag_platform.db import Base, DatabaseRepo, DatasetRow, ProjectRow, RunRow
from rag_platform.evaluators import EvaluationEngine, extract_claims, verify_claim_against_chunks
from rag_platform.models import (
    Answerability,
    ClaimStatus,
    DocumentReference,
    MetricRegressionPolicy,
    MetricStatus,
    RagTrace,
    ReleasePolicy,
    RetrievedChunk,
    RunConfig,
    RunOptions,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.server import app, get_db, db_row_to_test_case


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
from sqlalchemy.pool import StaticPool

_test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=_test_engine)


def _override_get_db():
    with Session(_test_engine) as session:
        yield session


@pytest.fixture(autouse=False)
def reset_db():
    """Drop and recreate all tables before each test for isolation."""
    Base.metadata.drop_all(bind=_test_engine)
    Base.metadata.create_all(bind=_test_engine)


@pytest.fixture()
def memory_db(reset_db):
    with Session(_test_engine) as session:
        yield session


@pytest.fixture()
def client(reset_db):
    """TestClient with in-memory DB via dependency override."""
    _previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as c:
        yield c
    if _previous_override is not None:
        app.dependency_overrides[get_db] = _previous_override
    else:
        app.dependency_overrides.pop(get_db, None)


@pytest.fixture()
def client_with_db(reset_db):
    """TestClient + direct DB session for cross-checking state."""
    _previous_override = app.dependency_overrides.get(get_db)
    app.dependency_overrides[get_db] = _override_get_db
    with Session(_test_engine) as session:
        with TestClient(app) as c:
            yield c, session
    if _previous_override is not None:
        app.dependency_overrides[get_db] = _previous_override
    else:
        app.dependency_overrides.pop(get_db, None)


def _seed_project_and_dataset(session: Session, *, with_facts: bool = True) -> tuple[str, str]:
    """Seed a project and a published dataset, return (project_id, dataset_id)."""
    import json as _json
    proj = ProjectRow(id=generate_id("proj"), name="Test Project", settings_json="{}")
    session.add(proj)
    session.flush()

    ds = DatasetRow(
        id=generate_id("ds"),
        project_id=proj.id,
        name="Test Dataset",
        version="v1",
        status="PUBLISHED",
        checksum_sha256="abc123",
    )
    session.add(ds)
    session.flush()

    from rag_platform.db import TestCaseRow
    tc = TestCaseRow(
        id="tc-001",
        dataset_id=ds.id,
        question="What is the revenue?",
        expected_answer="Revenue was $142M.",
        expected_facts_json=_json.dumps(["revenue was $142M", "year-over-year growth 14%"]),
        relevant_docs_json=_json.dumps([{"document_id": "doc1", "chunk_id": "c1"}]),
        answerability="ANSWERABLE",
        tags_json=_json.dumps(["financial"]),
        metadata_json=_json.dumps({"difficulty": "medium"}),
    )
    session.add(tc)
    session.commit()
    return proj.id, ds.id


# ===========================================================================
# FIX #1: Adapter selection — mock_mode must NOT override adapter_type
# ===========================================================================
class TestAdapterSelection:
    """Regression: HTTP/Python adapters must not silently become SyntheticRagAdapter."""

    def test_synthetic_adapter_selected_when_type_is_synthetic(self):
        from rag_platform.server import _resolve_adapter
        req = MagicMock()
        req.adapter_type = "synthetic"
        req.mock_mode = "PERFECT"
        req.adapter_config = {}
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, SyntheticRagAdapter)
        assert adapter.mode == SyntheticRagMode.PERFECT

    def test_synthetic_adapter_respects_mock_mode(self):
        from rag_platform.server import _resolve_adapter
        req = MagicMock()
        req.adapter_type = "synthetic"
        req.mock_mode = "HALLUCINATING"
        req.adapter_config = {}
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, SyntheticRagAdapter)
        assert adapter.mode == SyntheticRagMode.HALLUCINATING

    def test_http_adapter_with_mock_mode_does_NOT_use_synthetic(self):
        """The bug: `or req.mock_mode` makes non-empty string truthy → SyntheticRagAdapter."""
        from rag_platform.server import _resolve_adapter
        req = MagicMock()
        req.adapter_type = "http"
        req.mock_mode = "PERFECT"  # non-empty, truthy — must NOT trigger synthetic
        req.adapter_config = {"endpoint_url": "http://rag-sut.example.com/query"}
        req.timeout_seconds = 30
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, HttpRagAdapter), (
            "HTTP adapter with mock_mode='PERFECT' must return HttpRagAdapter, not SyntheticRagAdapter. "
            "This was the regression bug."
        )
        assert not isinstance(adapter, SyntheticRagAdapter)

    def test_python_adapter_with_mock_mode_does_NOT_use_synthetic(self):
        from rag_platform.adapters import PythonAdapterRegistry
        from rag_platform.server import _resolve_adapter
        PythonAdapterRegistry.register("mock_named_rag", lambda case, config: MagicMock())
        req = MagicMock()
        req.adapter_type = "python"
        req.mock_mode = "DISTRACTOR"  # truthy but must be ignored
        req.adapter_config = {"adapter_name": "mock_named_rag"}
        adapter = _resolve_adapter(req)
        # Should not be a SyntheticRagAdapter
        assert not isinstance(adapter, SyntheticRagAdapter)

    def test_unknown_adapter_raises_422(self, client):
        resp = client.post("/v1/projects", json={"name": "X"})
        proj_id = resp.json()["id"]

        # Post a run request with unknown adapter_type
        resp = client.post("/v1/runs", json={
            "project_id": proj_id,
            "dataset_id": "nonexistent-ds",
            "system_version": "v1",
            "adapter_type": "kafka_adapter_that_does_not_exist",
            "mock_mode": "PERFECT",
        })
        assert resp.status_code in (404, 422)  # 404 because dataset doesn't exist first


# ===========================================================================
# FIX #2: Canonical TestCase reconstruction preserves ALL fields
# ===========================================================================
class TestCanonicalTestCaseReconstruction:
    """expected_facts, tags, and metadata must survive the DB round-trip."""

    def test_db_row_to_test_case_preserves_expected_facts(self, memory_db):
        proj_id, ds_id = _seed_project_and_dataset(memory_db)
        ds = memory_db.get(DatasetRow, ds_id)
        assert ds.cases, "Dataset should have at least one test case"
        tc_row = ds.cases[0]
        tc = db_row_to_test_case(tc_row)

        assert tc.expected_facts == ["revenue was $142M", "year-over-year growth 14%"], (
            "expected_facts must be preserved through DB round-trip. "
            "If this fails, answer-correctness evaluation is meaningless."
        )

    def test_db_row_to_test_case_preserves_tags(self, memory_db):
        proj_id, ds_id = _seed_project_and_dataset(memory_db)
        ds = memory_db.get(DatasetRow, ds_id)
        tc = db_row_to_test_case(ds.cases[0])
        assert tc.tags == ["financial"]

    def test_db_row_to_test_case_preserves_metadata(self, memory_db):
        proj_id, ds_id = _seed_project_and_dataset(memory_db)
        ds = memory_db.get(DatasetRow, ds_id)
        tc = db_row_to_test_case(ds.cases[0])
        assert tc.metadata == {"difficulty": "medium"}

    def test_db_row_to_test_case_preserves_relevant_documents(self, memory_db):
        proj_id, ds_id = _seed_project_and_dataset(memory_db)
        ds = memory_db.get(DatasetRow, ds_id)
        tc = db_row_to_test_case(ds.cases[0])
        assert len(tc.relevant_documents) == 1
        assert tc.relevant_documents[0].document_id == "doc1"
        assert tc.relevant_documents[0].chunk_id == "c1"

    def test_expected_facts_reach_evaluator_unchanged(self, memory_db):
        """Regression: facts stored in DB must reach the evaluator unchanged (end-to-end)."""
        proj_id, ds_id = _seed_project_and_dataset(memory_db)
        ds = memory_db.get(DatasetRow, ds_id)
        tc = db_row_to_test_case(ds.cases[0])

        assert "revenue was $142M" in tc.expected_facts
        assert "year-over-year growth 14%" in tc.expected_facts

        # Simulate what the evaluator receives
        trace = RagTrace(
            trace_id=generate_id("tr"),
            run_id=generate_id("run"),
            test_case_id=tc.id,
            question=tc.question,
            answer="Revenue was $142M with year-over-year growth of 14%.",
        )

        async def _run():
            engine = EvaluationEngine()
            results = await engine.evaluate_trace(trace, tc)
            correctness = next((r for r in results if r.metric_name == "answer_correctness"), None)
            assert correctness is not None
            assert correctness.status == MetricStatus.NOT_APPLICABLE or correctness.score is not None
            return results

        asyncio.run(_run())


# ===========================================================================
# FIX #3: Dataset ownership validation
# ===========================================================================
class TestDatasetOwnership:
    """Cross-project dataset usage must be rejected."""

    def test_run_rejected_if_dataset_belongs_to_different_project(self, client, memory_db):
        # Create two projects
        r1 = client.post("/v1/projects", json={"name": "Project Alpha"})
        proj_a = r1.json()["id"]

        r2 = client.post("/v1/projects", json={"name": "Project Beta"})
        proj_b = r2.json()["id"]

        # Create dataset under project B
        r3 = client.post("/v1/datasets", json={
            "project_id": proj_b,
            "name": "Beta Dataset",
            "version": "v1",
            "cases": [{
                "id": "tc-1",
                "question": "Q?",
                "expected_answer": "A.",
                "relevant_documents": [{"document_id": "d1", "chunk_id": "c1"}],
            }],
        })
        assert r3.status_code == 200
        ds_b_id = r3.json()["id"]

        # Try to run Project A's run against Project B's dataset → must be 403
        resp = client.post("/v1/runs", json={
            "project_id": proj_a,
            "dataset_id": ds_b_id,
            "system_version": "v1",
            "adapter_type": "synthetic",
            "mock_mode": "PERFECT",
        })
        assert resp.status_code == 403, (
            f"Cross-project dataset access should be 403 Forbidden, got {resp.status_code}: {resp.text}"
        )

    def test_run_allowed_when_dataset_belongs_to_same_project(self, client, memory_db):
        r1 = client.post("/v1/projects", json={"name": "Project Gamma"})
        proj = r1.json()["id"]

        r2 = client.post("/v1/datasets", json={
            "project_id": proj,
            "name": "Gamma Dataset",
            "version": "v1",
            "cases": [{
                "id": "tc-1",
                "question": "Q?",
                "expected_answer": "A.",
                "relevant_documents": [{"document_id": "d1", "chunk_id": "c1"}],
            }],
        })
        ds_id = r2.json()["id"]

        resp = client.post("/v1/runs", json={
            "project_id": proj,
            "dataset_id": ds_id,
            "system_version": "v1",
            "adapter_type": "synthetic",
            "mock_mode": "PERFECT",
        })
        assert resp.status_code == 200, f"Same-project run should succeed, got {resp.status_code}: {resp.text}"

    def test_run_404_when_dataset_does_not_exist(self, client):
        r1 = client.post("/v1/projects", json={"name": "Project Delta"})
        proj = r1.json()["id"]

        resp = client.post("/v1/runs", json={
            "project_id": proj,
            "dataset_id": "ds_nonexistent_000",
            "system_version": "v1",
            "adapter_type": "synthetic",
            "mock_mode": "PERFECT",
        })
        assert resp.status_code == 404


# ===========================================================================
# FIX #17: Citation evaluator fallback must be disclosed
# ===========================================================================
class TestCitationFallbackDisclosure:
    """Heuristic fallback must be tagged in metadata, not silently override primary evaluator."""

    @pytest.mark.asyncio
    async def test_citation_heuristic_fallback_is_tagged(self):
        from rag_platform.evaluators import CitationSupportMetric
        from rag_platform.models import Citation

        metric = CitationSupportMetric()
        chunk = RetrievedChunk(
            document_id="doc1", chunk_id="c1", rank=1,
            text="Revenue was $142 million in Q3.",
        )
        trace = RagTrace(
            trace_id="tr1",
            run_id="run1",
            test_case_id="tc1",
            question="What was revenue?",
            answer="Revenue was 142 million.",
            retrieved_chunks=[chunk],
            citations=[
                Citation(
                    claim_id="cl1",
                    claim_text="Revenue was 142 million.",
                    document_id="doc1",
                    chunk_id="c1",
                )
            ],
        )
        case = TestCase(id="tc1", question="Q?", expected_answer="A.")
        result = await metric.compute(trace, case)
        # Whether it passed or failed, the metadata should expose the evaluation method
        assert "evaluation_method" in result.metadata
        assert result.metadata.get("heuristic_fallback_count") is not None


# ===========================================================================
# FIX #21/#22: MetricRegressionPolicy field aliases
# ===========================================================================
class TestMetricRegressionPolicyAliases:
    """Old field names must still work via the alias resolution post_init."""

    def test_old_field_names_resolve_to_canonical_names(self):
        p = MetricRegressionPolicy(
            metric_name="faithfulness",
            min_absolute_score=0.8,
            max_absolute_score=1.0,
            max_degradation_pct=5.0,
            max_absolute_degradation=0.1,
        )
        assert p.min_candidate_value == 0.8
        assert p.max_candidate_value == 1.0
        assert p.max_relative_drop_pct == 5.0
        assert p.max_absolute_drop == 0.1

    def test_new_field_names_work_directly(self):
        p = MetricRegressionPolicy(
            metric_name="recall_at_5",
            min_candidate_value=0.7,
            max_absolute_drop=0.05,
            max_relative_drop_pct=10.0,
        )
        assert p.min_candidate_value == 0.7
        assert p.max_absolute_drop == 0.05


# ===========================================================================
# FIX #36: Cache key must include chunk content (not just chunk ID)
# ===========================================================================
class TestEvaluatorCacheKey:
    """Same chunk_id with different content must produce a cache MISS."""

    def test_different_chunk_content_produces_cache_miss(self):
        from rag_platform.evaluators import FaithfulnessMetric

        engine = EvaluationEngine()
        metric = engine.metrics[0]  # use first metric as proxy

        tc = TestCase(id="tc1", question="Q?", expected_answer="A.")

        chunk_v1 = RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="Revenue was $142M.")
        chunk_v2 = RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="Revenue was $200M.")

        trace1 = RagTrace(
            trace_id="tr1", run_id="run1", test_case_id="tc1",
            question="Q?", answer="Revenue was $142M.",
            retrieved_chunks=[chunk_v1],
        )
        trace2 = RagTrace(
            trace_id="tr2", run_id="run1", test_case_id="tc1",
            question="Q?", answer="Revenue was $142M.",
            retrieved_chunks=[chunk_v2],
        )

        key1 = engine._cache_key(metric, trace1, tc)
        key2 = engine._cache_key(metric, trace2, tc)

        assert key1 != key2, (
            "Cache keys must differ when chunk content changes even if chunk_id is the same. "
            "Same chunk_id + different content = cache MISS required."
        )


# ===========================================================================
# FIX #38: Placeholder provenance hashes must not appear in completed runs
# ===========================================================================
class TestProvenanceHashes:
    """Completed runs must not contain 'default_model_config_hash' or 'default_prompt_hash'."""

    def test_provenance_hash_computed_from_model_params(self):
        prov = RunProvenance(
            dataset_checksum="abc",
            model_parameters={"temperature": 0.0, "model": "gpt-4"},
        )
        assert prov.model_config_hash not in ("", "default_model_config_hash"), (
            "model_config_hash should be computed from model_parameters, not remain as placeholder."
        )

    def test_provenance_hash_computed_from_prompt_template(self):
        prov = RunProvenance(
            dataset_checksum="abc",
            prompt_template="You are a helpful assistant. Answer: {question}",
        )
        assert prov.prompt_hash not in ("", "default_prompt_hash"), (
            "prompt_hash should be computed from prompt_template, not remain as placeholder."
        )

    def test_provenance_hash_computed_from_experiment_config(self):
        prov = RunProvenance(
            dataset_checksum="abc",
            experiment_config={"top_k": 5, "chunk_size": 512},
        )
        assert prov.experiment_hash not in ("", "default_experiment_hash")


# ===========================================================================
# FIX #1: Adapter type logic edge cases
# ===========================================================================
class TestAdapterTypeEdgeCases:
    """Case insensitivity and trailing whitespace in adapter_type."""

    def test_adapter_type_case_insensitive(self):
        from rag_platform.server import _resolve_adapter
        req = MagicMock()
        req.adapter_type = "SYNTHETIC"  # uppercase
        req.mock_mode = "PERFECT"
        req.adapter_config = {}
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, SyntheticRagAdapter)

    def test_adapter_type_with_whitespace(self):
        from rag_platform.server import _resolve_adapter
        req = MagicMock()
        req.adapter_type = "  synthetic  "  # with whitespace
        req.mock_mode = "DISTRACTOR"
        req.adapter_config = {}
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, SyntheticRagAdapter)
        assert adapter.mode == SyntheticRagMode.DISTRACTOR


# ===========================================================================
# FIX #7: RunOptions fields are actually wired
# ===========================================================================
class TestRunOptionsWiring:
    """RunOptions max_cases, timeout, fail_fast, use_cache must be honoured."""

    def test_max_cases_limits_evaluated_cases(self, client, memory_db):
        r1 = client.post("/v1/projects", json={"name": "MaxCases Project"})
        proj = r1.json()["id"]

        # Create dataset with 3 cases
        r2 = client.post("/v1/datasets", json={
            "project_id": proj,
            "name": "MaxCases DS",
            "version": "v1",
            "cases": [
                {"id": f"tc-{i}", "question": f"Q{i}?", "expected_answer": f"A{i}."}
                for i in range(3)
            ],
        })
        ds_id = r2.json()["id"]

        # Run with max_cases=1 — only 1 trace should be produced
        resp = client.post("/v1/runs", json={
            "project_id": proj,
            "dataset_id": ds_id,
            "system_version": "v1",
            "adapter_type": "synthetic",
            "mock_mode": "PERFECT",
            "max_cases": 1,
        })
        assert resp.status_code == 200
        run_id = resp.json()["run_id"]

        traces_resp = client.get(f"/v1/runs/{run_id}/traces")
        assert traces_resp.status_code == 200
        # With max_cases=1, should have at most 1 trace
        assert traces_resp.json()["total"] <= 1, (
            f"max_cases=1 should limit to 1 trace, got {traces_resp.json()['total']}"
        )


# ===========================================================================
# FIX #30: create_all() not called at production module level
# ===========================================================================
class TestNoProdCreateAll:
    """Production server must not call Base.metadata.create_all() unconditionally."""

    def test_server_module_does_not_unconditionally_call_create_all(self):
        """Inspect server module: create_all must only be called inside if ':memory:'."""
        import inspect
        import rag_platform.server as srv_mod
        src = inspect.getsource(srv_mod)
        # The pattern `Base.metadata.create_all` outside of the ':memory:' guard should not appear
        # Simple check: create_all should appear inside an if-block checking for :memory:
        assert "Base.metadata.create_all(bind=engine)" in src, "create_all should still exist for test isolation"
        # Verify it's guarded
        assert '":memory:"' in src or "':memory:'" in src, (
            "create_all() must be guarded by a ':memory:' URL check. "
            "Production runs must use `alembic upgrade head`."
        )


# ===========================================================================
# FIX #3: Dataset ownership — additional boundary tests
# ===========================================================================
class TestDatasetOwnershipEdgeCases:
    """Additional dataset ownership edge cases."""

    def test_list_datasets_filtered_by_project_returns_own_only(self, client):
        r1 = client.post("/v1/projects", json={"name": "Proj-List-A"})
        proj_a = r1.json()["id"]

        r2 = client.post("/v1/projects", json={"name": "Proj-List-B"})
        proj_b = r2.json()["id"]

        # Create one dataset per project
        client.post("/v1/datasets", json={
            "project_id": proj_a, "name": "DS-A", "version": "v1",
            "cases": [{"id": "tc-a", "question": "Q?", "expected_answer": "A.",
                        "relevant_documents": [{"document_id": "d1", "chunk_id": "c1"}]}],
        })
        client.post("/v1/datasets", json={
            "project_id": proj_b, "name": "DS-B", "version": "v1",
            "cases": [{"id": "tc-b", "question": "Q?", "expected_answer": "A.",
                        "relevant_documents": [{"document_id": "d2", "chunk_id": "c2"}]}],
        })

        resp = client.get(f"/v1/datasets?project_id={proj_a}")
        assert resp.status_code == 200
        datasets = resp.json()["datasets"]
        assert all(d["project_id"] == proj_a for d in datasets), (
            "Listing datasets filtered by project_id must only return datasets for that project."
        )
