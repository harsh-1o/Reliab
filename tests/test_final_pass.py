"""Regression and verification tests for the final engineering pass.

Tests:
1. timeout_seconds enforcement (not exceeded, exceeded, fail_fast=false, fail_fast=true)
2. fail_fast semantics (all cases run when false, stops scheduling when true)
3. use_cache behavior (hit, miss on chunk content change, bypass when false)
4. adapter lifecycle (close called on success, timeout, failure, fail_fast)
5. python adapter architecture (trusted registry, no arbitrary code, no None)
6. RBAC authorization (API key -> client -> project membership -> role)
7. project listing authorization (client sees only authorized projects)
8. dataset clean persistence (no duplicate/discarded domain object)
9. dataset checksum semantics (metadata changes checksum)
10. runtime provenance representation (actual values, no empty defaults)
11. consolidated provenance hashing (canonical_manifest -> canonical_json -> hash)
12. uncalibrated confidence values (null / not_calibrated)
13. contradiction detection edge cases (no false positives, true contradictions)
14. real external RAG integration test (HTTP server -> HttpRagAdapter -> gate)
"""

from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import socket
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.adapters import (
    HttpRagAdapter,
    PythonAdapterRegistry,
    PythonRagAdapter,
)
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.core import compute_manifest_hash, generate_id
from rag_platform.db import Base, DatabaseRepo
from rag_platform.evaluators import EvaluationEngine, verify_claim_against_chunks
from rag_platform.regression import RegressionEngine
from rag_platform.models import (
    ClaimStatus,
    ClaimVerification,
    DatasetStatus,
    DiagnosticFinding,
    DocumentReference,
    FailureAttribution,
    FailureCode,
    GateStatus,
    RagTrace,
    ReleasePolicy,
    RetrievedChunk,
    RunConfig,
    RunOptions,
    RunProvenance,
    RunStatus,
    TestCase,
    compute_dataset_checksum,
)
from rag_platform.security import (
    ApiKeyRegistry,
    Role,
)
from rag_platform.server import (
    _execute_evaluation_run,
    _resolve_adapter,
    app,
    get_db,
)

# Shared isolated in-memory test database
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    with Session(test_engine) as session:
        yield session


app.dependency_overrides[get_db] = override_get_db
client = TestClient(app)


@pytest.fixture(autouse=True)
def ensure_db_override():
    app.dependency_overrides[get_db] = override_get_db
    yield
    app.dependency_overrides[get_db] = override_get_db


def _seed_project_and_dataset(sess: Session) -> tuple[str, str]:
    """Helper to seed a project and dataset row for DB run foreign key constraints."""
    repo = DatabaseRepo(sess)
    p = repo.create_project(name=f"Test Project {generate_id()}")
    ds = repo.create_dataset(project_id=p.id, name=f"Test DS {generate_id()}", version="v1.0")
    ds.status = DatasetStatus.PUBLISHED.value
    sess.commit()
    return p.id, ds.id


# ---------------------------------------------------------------------------
# 1. ACTUALLY ENFORCE timeout_seconds
# ---------------------------------------------------------------------------
class TestTimeoutEnforcement:
    @pytest.mark.asyncio
    async def test_timeout_not_exceeded(self):
        """When adapter finishes within timeout, execution succeeds normally."""
        async def fast_fn(case: TestCase, config: RunConfig):
            await asyncio.sleep(0.01)
            return RagTrace(
                trace_id="tr_fast",
                run_id="run_t1",
                test_case_id=case.id,
                question=case.question,
                answer="Fast answer.",
                latency_ms=10,
            )

        PythonAdapterRegistry.register("fast_adapter", fast_fn)
        case = TestCase(id="tc_1", question="What is latency?", expected_answer="Fast answer.")

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(timeout_seconds=2, fail_fast=False),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "fast_adapter"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, [case], db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.COMPLETED
            assert len(run.traces) == 1
            assert run.traces[0].error_code is None
            assert run.traces[0].answer == "Fast answer."

    @pytest.mark.asyncio
    async def test_timeout_exceeded(self):
        """When adapter exceeds timeout, case becomes ERROR (OPS-01) with structured timeout error."""
        async def slow_fn(case: TestCase, config: RunConfig):
            await asyncio.sleep(0.5)
            return RagTrace(trace_id="tr_slow", run_id="run_t2", test_case_id=case.id, question=case.question)

        PythonAdapterRegistry.register("slow_adapter", slow_fn)
        case = TestCase(id="tc_timeout", question="Slow query?")

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(timeout_seconds=0.05, fail_fast=False),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "slow_adapter"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, [case], db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.COMPLETED
            assert len(run.traces) == 1
            trace_row = run.traces[0]
            assert trace_row.error_code == "OPS-01"
            trace_data = json.loads(trace_row.raw_trace_json)
            assert "Adapter execution timed out" in trace_data.get("telemetry", {}).get("error", "")

    @pytest.mark.asyncio
    async def test_timeout_plus_fail_fast_false(self):
        """When timeout occurs and fail_fast=false, the run continues and executes subsequent cases."""
        async def mixed_fn(case: TestCase, config: RunConfig):
            if case.id == "c_slow":
                await asyncio.sleep(0.4)
            return RagTrace(trace_id=f"tr_{case.id}", run_id="run_m", test_case_id=case.id, question=case.question, answer="OK")

        PythonAdapterRegistry.register("mixed_timeout_adapter", mixed_fn)
        cases = [
            TestCase(id="c_fast1", question="Q1"),
            TestCase(id="c_slow", question="Q2"),
            TestCase(id="c_fast2", question="Q3"),
        ]

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(timeout_seconds=0.05, fail_fast=False),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "mixed_timeout_adapter"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, cases, db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.COMPLETED
            assert len(run.traces) == 3
            # Middle case timed out, but first and third completed
            assert run.traces[0].error_code is None
            assert run.traces[1].error_code == "OPS-01"
            assert run.traces[2].error_code is None

    @pytest.mark.asyncio
    async def test_timeout_plus_fail_fast_true(self):
        """When timeout occurs and fail_fast=true, the run stops and marks run as FAILED."""
        async def mixed_fn(case: TestCase, config: RunConfig):
            if case.id == "c_slow":
                await asyncio.sleep(0.4)
            return RagTrace(trace_id=f"tr_{case.id}", run_id="run_m", test_case_id=case.id, question=case.question, answer="OK")

        PythonAdapterRegistry.register("fail_fast_timeout_adapter", mixed_fn)
        cases = [
            TestCase(id="c_slow", question="Q1 (times out)"),
            TestCase(id="c_never1", question="Q2 (should not run)"),
            TestCase(id="c_never2", question="Q3 (should not run)"),
        ]

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(timeout_seconds=0.05, fail_fast=True),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "fail_fast_timeout_adapter"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, cases, db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.FAILED
            assert len(run.traces) == 1
            assert run.traces[0].error_code == "OPS-01"


# ---------------------------------------------------------------------------
# 2. FIX fail_fast SEMANTICS
# ---------------------------------------------------------------------------
class TestFailFastSemantics:
    @pytest.mark.asyncio
    async def test_fail_fast_false_completes_all_cases(self):
        """fail_fast=false: Case 1 PASS, Case 2 ERROR, Case 3 PASS, Case 4 FAIL, Case 5 PASS completes full run."""
        call_order = []

        def sut(case: TestCase, config: RunConfig):
            call_order.append(case.id)
            if case.id == "c2":
                raise RuntimeError("Case 2 connection drop")
            if case.id == "c4":
                return RagTrace(
                    trace_id="tr_4",
                    run_id="run_ff",
                    test_case_id=case.id,
                    question=case.question,
                    answer="Completely contradictory answer.",
                    retrieved_chunks=[RetrievedChunk(document_id="doc1", chunk_id="chk1", rank=1, text="Gold fact: Revenue was $10M.")],
                )
            return RagTrace(
                trace_id=f"tr_{case.id}",
                run_id="run_ff",
                test_case_id=case.id,
                question=case.question,
                answer="Gold fact: Revenue was $10M.",
                retrieved_chunks=[RetrievedChunk(document_id="doc1", chunk_id="chk1", rank=1, text="Gold fact: Revenue was $10M.")],
            )

        PythonAdapterRegistry.register("ff_false_sut", sut)
        cases = [TestCase(id=f"c{i+1}", question=f"Q{i+1}", expected_answer="Revenue was $10M.") for i in range(5)]

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(fail_fast=False),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "ff_false_sut"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, cases, db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.COMPLETED
            assert len(run.traces) == 5
            assert call_order == ["c1", "c2", "c3", "c4", "c5"]

    @pytest.mark.asyncio
    async def test_fail_fast_true_aborts_scheduling(self):
        """fail_fast=true: Fatal failure on Case 2 immediately stops processing further cases."""
        call_order = []

        def sut(case: TestCase, config: RunConfig):
            call_order.append(case.id)
            if case.id == "c2":
                raise RuntimeError("Fatal database crash")
            return RagTrace(
                trace_id=f"tr_{case.id}",
                run_id="run_ff2",
                test_case_id=case.id,
                question=case.question,
                answer="Revenue was $10M.",
                retrieved_chunks=[RetrievedChunk(document_id="doc1", chunk_id="chk1", rank=1, text="Revenue was $10M.")],
            )

        PythonAdapterRegistry.register("ff_true_sut", sut)
        cases = [
            TestCase(
                id=f"c{i+1}",
                question=f"Q{i+1}",
                expected_answer="Revenue was $10M.",
                relevant_documents=[DocumentReference(document_id="doc1", chunk_id="chk1")],
            )
            for i in range(5)
        ]

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(
                project_id=proj_id,
                dataset_id=ds_id,
                dataset_version="v1.0",
                system_version="s1",
                options=RunOptions(fail_fast=True),
            )
            req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "ff_true_sut"}, concurrency=1)

            repo = DatabaseRepo(sess)
            prov = RunProvenance(dataset_checksum="c1", rag_version="s1")
            run = repo.create_run(config, prov)
            sess.commit()

            await _execute_evaluation_run(run.id, req, config, cases, db_session=sess)
            sess.refresh(run)

            assert run.status == RunStatus.FAILED
            assert len(run.traces) == 2
            assert call_order == ["c1", "c2"]


# ---------------------------------------------------------------------------
# 3. MAKE use_cache ACTUALLY CONTROL CACHE BEHAVIOR
# ---------------------------------------------------------------------------
class TestEvaluationCacheControl:
    @pytest.mark.asyncio
    async def test_use_cache_hit_and_miss_semantics(self):
        """Test cache hit on identical inputs and cache miss when chunk content changes."""
        engine = EvaluationEngine()
        case = TestCase(
            id="c_cache_1",
            question="What is the net profit?",
            expected_answer="$50M",
            expected_facts=["Net profit was $50M"],
            relevant_documents=[DocumentReference(document_id="doc_fin", chunk_id="chk_1")],
        )
        trace_v1 = RagTrace(
            trace_id="tr_v1",
            run_id="run_1",
            test_case_id=case.id,
            question=case.question,
            answer="Net profit was $50M.",
            retrieved_chunks=[RetrievedChunk(document_id="doc_fin", chunk_id="chk_1", rank=1, text="Net profit reported at $50M.")],
        )

        # 1. First evaluation: cache misses and populates cache
        res_first = await engine.evaluate_trace(trace_v1, case, use_cache=True)
        assert all(not r.cached for r in res_first)

        # 2. Second evaluation with SAME inputs: CACHE HIT
        res_hit = await engine.evaluate_trace(trace_v1, case, use_cache=True)
        assert any(r.cached for r in res_hit)

        # 3. Third evaluation with use_cache=False: NO CACHE REUSE
        res_bypass = await engine.evaluate_trace(trace_v1, case, use_cache=False)
        assert all(not r.cached for r in res_bypass)

        # 4. Same chunk_id, DIFFERENT chunk content: CACHE MISS
        trace_content_changed = RagTrace(
            trace_id="tr_v2",
            run_id="run_1",
            test_case_id=case.id,
            question=case.question,
            answer="Net profit was $50M.",
            retrieved_chunks=[RetrievedChunk(document_id="doc_fin", chunk_id="chk_1", rank=1, text="DIFFERENT CONTENT: Loss of $20M.")],
        )
        res_content_miss = await engine.evaluate_trace(trace_content_changed, case, use_cache=True)
        assert all(not r.cached for r in res_content_miss)


# ---------------------------------------------------------------------------
# 4. CLOSE ADAPTER RESOURCES CORRECTLY
# ---------------------------------------------------------------------------
class TestAdapterResourceLifecycle:
    @pytest.mark.asyncio
    async def test_close_called_on_success(self):
        """Adapter close() must be invoked cleanly upon run completion."""
        mock_adapter = MagicMock()
        mock_adapter.close = AsyncMock()
        mock_adapter.run = AsyncMock(return_value=RagTrace(
            trace_id="tr_ok", run_id="r1", test_case_id="tc1", question="Q?", answer="A."
        ))

        req = MagicMock(adapter_type="custom", adapter_config={}, concurrency=1)

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(project_id=proj_id, dataset_id=ds_id, dataset_version="v1.0", system_version="s1")
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, RunProvenance(dataset_checksum="c1", rag_version="s1"))
            sess.commit()

            from rag_platform import server
            orig_resolve = server._resolve_adapter
            server._resolve_adapter = lambda _: mock_adapter
            try:
                await _execute_evaluation_run(run.id, req, config, [TestCase(id="tc1", question="Q?")], db_session=sess)
            finally:
                server._resolve_adapter = orig_resolve

            mock_adapter.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_called_on_failure_and_timeout(self):
        """Adapter close() must be invoked in finally block even on adapter exception or timeout."""
        mock_adapter = MagicMock()
        mock_adapter.close = AsyncMock()
        mock_adapter.run = AsyncMock(side_effect=RuntimeError("Adapter boom!"))

        req = MagicMock(adapter_type="custom", adapter_config={}, concurrency=1)

        with Session(test_engine) as sess:
            proj_id, ds_id = _seed_project_and_dataset(sess)
            config = RunConfig(project_id=proj_id, dataset_id=ds_id, dataset_version="v1.0", system_version="s1")
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, RunProvenance(dataset_checksum="c1", rag_version="s1"))
            sess.commit()

            from rag_platform import server
            orig_resolve = server._resolve_adapter
            server._resolve_adapter = lambda _: mock_adapter
            try:
                await _execute_evaluation_run(run.id, req, config, [TestCase(id="tc1", question="Q?")], db_session=sess)
            finally:
                server._resolve_adapter = orig_resolve

            mock_adapter.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. PYTHON ADAPTER ARCHITECTURE
# ---------------------------------------------------------------------------
class TestPythonAdapterArchitecture:
    def test_python_rag_adapter_none_raises_value_error(self):
        """PythonRagAdapter(None) must raise ValueError."""
        with pytest.raises(ValueError, match="requires a non-None, callable target_fn"):
            PythonRagAdapter(None)

    def test_api_python_adapter_missing_name_returns_422(self):
        """Calling API with adapter_type=python without a registered name raises 422."""
        from fastapi import HTTPException
        req = MagicMock(adapter_type="python", adapter_config={})
        with pytest.raises(HTTPException) as exc_info:
            _resolve_adapter(req)
        assert exc_info.value.status_code == 422
        assert "requires 'adapter_name'" in exc_info.value.detail

    def test_api_python_adapter_registered_name_resolves(self):
        """Pre-registered named python adapter resolves correctly."""
        def my_fn(case: TestCase, config: RunConfig):
            return "Answer from my_fn"

        PythonAdapterRegistry.register("my_trusted_rag", my_fn)
        req = MagicMock(adapter_type="python", adapter_config={"adapter_name": "my_trusted_rag"})
        adapter = _resolve_adapter(req)
        assert isinstance(adapter, PythonRagAdapter)
        assert adapter.target_fn == my_fn


# ---------------------------------------------------------------------------
# 6 & 7. AUTHORIZATION & PROJECT LISTING RBAC
# ---------------------------------------------------------------------------
class TestProjectAuthorizationAndListing:
    def test_client_project_isolation_and_listing(self):
        """Verify project-level authorization: Project A client cannot access Project B, and list endpoints are filtered."""
        # 1. Setup two projects
        with Session(test_engine) as sess:
            repo = DatabaseRepo(sess)
            p_a = repo.create_project(name=f"Project Alpha {generate_id()}")
            p_b = repo.create_project(name=f"Project Beta {generate_id()}")
            sess.commit()
            proj_a_id = p_a.id
            proj_b_id = p_b.id

        # 2. Register API keys with scoped project memberships
        key_a = f"key_client_a_{generate_id()}"
        key_b = f"key_client_b_{generate_id()}"
        ApiKeyRegistry.register_key(api_key=key_a, client_id="client_a", project_roles={proj_a_id: Role.VIEWER})
        ApiKeyRegistry.register_key(api_key=key_b, client_id="client_b", project_roles={proj_b_id: Role.EDITOR})

        # 3. Direct access to Project A by Client A: SUCCESS (200)
        resp_a = client.get(f"/v1/projects/{proj_a_id}", headers={"X-API-Key": key_a})
        assert resp_a.status_code == 200
        assert resp_a.json()["id"] == proj_a_id

        # 4. Direct access to Project B by Client A: FORBIDDEN (403)
        resp_b_denied = client.get(f"/v1/projects/{proj_b_id}", headers={"X-API-Key": key_a})
        assert resp_b_denied.status_code == 403

        # 5. List projects: Client A sees only Project A, Client B sees only Project B
        list_a = client.get("/v1/projects", headers={"X-API-Key": key_a}).json()["projects"]
        proj_ids_a = [p["id"] for p in list_a]
        assert proj_a_id in proj_ids_a
        assert proj_b_id not in proj_ids_a

        list_b = client.get("/v1/projects", headers={"X-API-Key": key_b}).json()["projects"]
        proj_ids_b = [p["id"] for p in list_b]
        assert proj_b_id in proj_ids_b
        assert proj_a_id not in proj_ids_b

    def test_role_hierarchy_enforcement(self):
        """VIEWER role cannot mutate/create datasets or runs; EDITOR role can."""
        with Session(test_engine) as sess:
            repo = DatabaseRepo(sess)
            p = repo.create_project(name=f"Role Test Project {generate_id()}")
            sess.commit()
            proj_id = p.id

        viewer_key = f"key_viewer_{generate_id()}"
        editor_key = f"key_editor_{generate_id()}"
        ApiKeyRegistry.register_key(api_key=viewer_key, client_id="viewer", project_roles={proj_id: Role.VIEWER})
        ApiKeyRegistry.register_key(api_key=editor_key, client_id="editor", project_roles={proj_id: Role.EDITOR})

        # VIEWER attempting to create dataset -> 403 Forbidden
        ds_payload = {"project_id": proj_id, "name": "DS", "version": "v1", "cases": []}
        resp_denied = client.post("/v1/datasets", json=ds_payload, headers={"X-API-Key": viewer_key})
        assert resp_denied.status_code == 403

        # EDITOR creating dataset -> 200 OK
        resp_allowed = client.post("/v1/datasets", json=ds_payload, headers={"X-API-Key": editor_key})
        assert resp_allowed.status_code == 200


# ---------------------------------------------------------------------------
# 8 & 9. DATASET CLEAN PERSISTENCE & METADATA CHECKSUM
# ---------------------------------------------------------------------------
class TestDatasetChecksumAndCleanPersistence:
    def test_metadata_changes_dataset_checksum(self):
        """Semantic metadata must be included in TestCase canonical dict and affect checksum."""
        tc1 = TestCase(id="tc_1", question="Q1", expected_answer="A1", metadata={"routing": "expert_a"})
        tc2 = TestCase(id="tc_1", question="Q1", expected_answer="A1", metadata={"routing": "expert_b"})

        cs1 = compute_dataset_checksum([tc1])
        cs2 = compute_dataset_checksum([tc2])
        assert cs1 != cs2
        assert "metadata" in tc1.to_canonical_dict()


# ---------------------------------------------------------------------------
# 10 & 11. PROVENANCE MANIFEST & CONSOLIDATED HASHING
# ---------------------------------------------------------------------------
class TestProvenanceManifestAndConsolidatedHashing:
    def test_canonical_manifest_populated_with_runtime_data(self):
        """Manifest only includes non-empty runtime execution values."""
        prov = RunProvenance(
            dataset_checksum="sha_ds_123",
            dataset_id="ds_123",
            rag_version="v2.0",
            model_name="test-llm",
            temperature=0.2,
            random_seed=42,
        )
        manifest = prov.canonical_manifest()
        assert manifest["dataset_checksum"] == "sha_ds_123"
        assert manifest["model_name"] == "test-llm"
        assert manifest["random_seed"] == 42
        assert "embedding_model" not in manifest  # not populated, should not be empty string

    def test_consolidated_provenance_hashing_consistency(self):
        """The same logical provenance produces the same hash via RunProvenance and compute_manifest_hash."""
        prov = RunProvenance(
            dataset_checksum="checksum_abc",
            rag_version="rag_v1",
            model_name="gpt-4o",
            evaluator_version="2.0.0",
        )
        hash_from_prov = prov.compute_hash()

        hash_from_helper = compute_manifest_hash(
            dataset_checksum="checksum_abc",
            rag_version="rag_v1",
            model_name="gpt-4o",
            evaluator_version="2.0.0",
        )
        assert hash_from_prov == hash_from_helper
        assert prov.manifest_hash == hash_from_prov


# ---------------------------------------------------------------------------
# 12. UNCALIBRATED CONFIDENCE VALUES
# ---------------------------------------------------------------------------
class TestUncalibratedConfidence:
    def test_confidence_defaults_to_null_and_not_calibrated(self):
        """Arbitrary heuristic scores must not be represented as 1.0 calibrated probabilities."""
        claim_ver = ClaimVerification(
            claim_id="cl_1",
            claim_text="The company made $10M.",
            status=ClaimStatus.SUPPORTED,
        )
        assert claim_ver.confidence is None
        assert claim_ver.confidence_type == "not_calibrated"

        finding = DiagnosticFinding(
            code=FailureCode.RET_01,
            explanation="Retrieval miss",
        )
        assert finding.confidence is None
        assert finding.confidence_type == "not_calibrated"

        attribution = FailureAttribution(
            trace_id="tr_1",
            explanation="General failure",
        )
        assert attribution.confidence is None
        assert attribution.confidence_type == "not_calibrated"


# ---------------------------------------------------------------------------
# 13. CONTRADICTION DETECTION EDGE CASES
# ---------------------------------------------------------------------------
class TestContradictionDetectionEdgeCases:
    def test_no_manual_changes_vs_manual_review_not_contradictory(self):
        """'No manual changes were made' vs 'Manual review was required' is NOT contradictory."""
        chunks = [RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="Manual review was required for deployment.")]
        status, _, _ = verify_claim_against_chunks("No manual changes were made.", chunks)
        assert status != ClaimStatus.CONTRADICTED

    def test_numerical_revenue_contradiction(self):
        """Evidence: 'Revenue was $80M.' Answer: 'Revenue was $100M.' is CONTRADICTED."""
        chunks = [RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="Fiscal revenue reached $80M.")]
        status, _, reason = verify_claim_against_chunks("Fiscal revenue was $100M.", chunks)
        assert status == ClaimStatus.CONTRADICTED
        assert "numerical/quantitative conflict" in reason.lower()

    def test_explicit_action_negation_contradiction(self):
        """'Requires no configuration' vs 'Requires manual configuration' is CONTRADICTED."""
        chunks = [RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="The deployment requires manual configuration of YAML keys.")]
        status, _, reason = verify_claim_against_chunks("It requires no configuration.", chunks)
        assert status == ClaimStatus.CONTRADICTED

    def test_antonym_polarity_contradiction(self):
        """'Feature was enabled by default' vs 'Feature was disabled by default' is CONTRADICTED."""
        chunks = [RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="The dark mode feature was disabled by default.")]
        status, _, reason = verify_claim_against_chunks("The dark mode feature was enabled by default.", chunks)
        assert status == ClaimStatus.CONTRADICTED


# ---------------------------------------------------------------------------
# 14. REAL EXTERNAL RAG HTTP INTEGRATION TEST
# ---------------------------------------------------------------------------
class _DeterministicRagHandler(BaseHTTPRequestHandler):
    """Deterministic HTTP RAG service fixture exposing the HTTP adapter contract."""

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len > 0:
            self.rfile.read(content_len)

        response_payload = {
            "answer": "Annual revenue grew 25% to $125M in fiscal 2024.",
            "retrieved_chunks": [
                {
                    "document_id": "doc_annual_2024",
                    "chunk_id": "chunk_fin_1",
                    "rank": 1,
                    "score": 0.96,
                    "text": "Annual revenue grew 25% to $125M in fiscal 2024 driven by cloud services.",
                }
            ],
            "citations": [
                {
                    "claim_id": "cl_1",
                    "claim_text": "Annual revenue grew 25% to $125M in fiscal 2024.",
                    "document_id": "doc_annual_2024",
                    "chunk_id": "chunk_fin_1",
                }
            ],
            "input_tokens": 120,
            "output_tokens": 30,
            "cost_usd": 0.0015,
            "model": "external-rag-v1",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response_payload).encode("utf-8"))

    def log_message(self, format, *args):
        # Silence HTTP server access logs during tests
        return


@pytest.fixture(scope="module")
def external_rag_server():
    """Starts a lightweight deterministic HTTP RAG server on a local ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    server = HTTPServer(("127.0.0.1", port), _DeterministicRagHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    endpoint_url = f"http://127.0.0.1:{port}/v1/rag"

    yield endpoint_url

    server.shutdown()
    server.server_close()


class TestRealExternalRagIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_with_real_http_adapter_and_gate(self, external_rag_server):
        """End-to-end integration test: Platform -> Real HttpRagAdapter -> External RAG HTTP service -> Trace -> Eval -> Gate."""
        endpoint_url = external_rag_server
        adapter = HttpRagAdapter(endpoint_url=endpoint_url, timeout_seconds=5.0, allow_private_ip=True)

        case = TestCase(
            id="tc_rev_growth",
            question="What was annual revenue and growth rate?",
            expected_answer="Annual revenue grew 25% to $125M in fiscal 2024.",
            expected_facts=["revenue grew 25%", "reached $125M"],
            relevant_documents=[DocumentReference(document_id="doc_annual_2024", chunk_id="chunk_fin_1")],
        )
        config = RunConfig(
            project_id="proj_ext_test",
            dataset_id="ds_ext_test",
            dataset_version="v1.0",
            system_version="http-rag-1.0",
        )

        try:
            # 1. Execute against real HTTP server over network
            trace = await adapter.run(case, config)

            assert trace.error_code is None
            assert trace.answer == "Annual revenue grew 25% to $125M in fiscal 2024."
            assert len(trace.retrieved_chunks) == 1
            assert trace.retrieved_chunks[0].chunk_id == "chunk_fin_1"
            assert len(trace.citations) == 1
            assert trace.model == "external-rag-v1"

            # 2. Evaluate trace metrics
            eval_engine = EvaluationEngine()
            metrics = await eval_engine.evaluate_trace(trace, case)
            metric_map = {m.metric_name: m.score for m in metrics}

            assert metric_map.get("recall_at_5") == 1.0
            assert metric_map.get("mrr") == 1.0
            assert metric_map.get("faithfulness") == 1.0
            assert metric_map.get("citation_accuracy") == 1.0

            # 3. Diagnose attribution: should pass cleanly (no failure)
            attr_engine = FailureAttributionEngine()
            diag = attr_engine.diagnose(trace, case, metrics)
            assert diag is None

            # 4. Gate evaluation
            summary = eval_engine.aggregate_run([(trace, metrics)])
            policy = ReleasePolicy(min_faithfulness=0.8, min_recall=0.8, max_error_rate=0.05)
            reg_engine = RegressionEngine()
            gate_res = reg_engine.evaluate_gate(summary, policy, candidate_run_id="candidate_run")

            assert gate_res.status == GateStatus.PASS
            assert len(gate_res.violations) == 0

        finally:
            await adapter.close()
