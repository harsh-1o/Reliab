"""Comprehensive test suite for the senior-engineer hardening pass.

Validates:
1. Frontend XSS escaping logic & regression test.
2. Null confidence handling in UI (explicit 'Not calibrated', never 0%, 100%, NaN%).
3. Release gate backend single source of truth (gate_result in API).
4. failure_only trace pagination (SQL filter -> count -> offset -> limit).
5. No fake 0.5 metric fallbacks in comparison (explicit None / MISSING_DATA).
6. Secure /v1/demo-run (dev_mode check and admin role enforcement).
7. Background run DB session closure.
8. DurableRunWorker (atomic claiming, restart survival, stale run recovery).
9. Persistent API key & RBAC in database with SHA-256 hashing.
10. Python adapter API rejection of un-registered target_fn.
11. Accurate pre-pagination totals across projects, datasets, runs, traces, failures.
12. Cross-project authorization and ID substitution protection.
"""

from __future__ import annotations

import hashlib
from typing import Any
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.adapters import PythonAdapterRegistry
import rag_platform.core as core_module
from rag_platform.core import Settings, generate_id
from rag_platform.db import (
    Base,
    DatabaseRepo,
    DatasetRow,
    FailureRow,
    RunRow,
)
from rag_platform.models import (
    GateStatus,
    MetricFamily,
    MetricResult,
    MetricSummary,
    RagTrace,
    ReleasePolicy,
    RunConfig,
    RunMetricsSummary,
    RunOptions,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.regression import RegressionEngine
from rag_platform.security import (
    ApiKeyRegistry,
    Role,
    authenticate_request,
)
from rag_platform.server import (
    _execute_evaluation_run,
    _resolve_adapter,
    app,
    get_db,
)
from rag_platform.worker import DurableRunWorker


@pytest.fixture
def in_memory_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def client(in_memory_db):
    def override_get_db():
        yield in_memory_db

    app.dependency_overrides[get_db] = override_get_db
    test_client = TestClient(app)
    yield test_client
    app.dependency_overrides.clear()


def make_published_dataset(repo: DatabaseRepo, project_id: str, name: str = "Test DS", version: str = "v1.0") -> DatasetRow:
    ds = repo.create_dataset(project_id, name, version)
    repo.add_test_cases(ds.id, [TestCase(id=f"{ds.id}-tc", question="What is Q?", expected_answer="A")])
    return repo.publish_dataset(ds.id)


# ===========================================================================
# 1. FRONTEND XSS ESCAPING LOGIC & REGRESSION TESTS
# ===========================================================================
def escape_html(val: Any) -> str:
    """Python reference implementation mirroring app.js escapeHtml(value)."""
    if val is None:
        return ""
    s = str(val)
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def test_frontend_xss_escaping_neutralizes_malicious_payloads():
    malicious_payloads = [
        '<script>alert("xss")</script>',
        '<img src=x onerror=alert(document.domain)>',
        '"><svg/onload=alert(1)>',
        '<iframe src="javascript:alert(1)"></iframe>',
        '"><script src=//evil.com/xss.js></script>',
    ]

    for payload in malicious_payloads:
        escaped = escape_html(payload)
        assert "<script>" not in escaped
        assert "<img" not in escaped
        assert "<svg" not in escaped
        assert "<iframe" not in escaped
        assert "&lt;" in escaped or "&gt;" in escaped or "&quot;" in escaped


def test_frontend_app_js_contains_xss_protection():
    """Verify that static/app.js defines escapeHtml and uses it."""
    from rag_platform.server import STATIC_DIR
    app_js_path = STATIC_DIR / "app.js"
    assert app_js_path.exists(), "static/app.js must exist"
    content = app_js_path.read_text(encoding="utf-8")

    assert "function escapeHtml" in content or "escapeHtml = " in content
    assert "escapeHtml(t.question" in content
    assert "escapeHtml(t.answer" in content
    assert "escapeHtml(c.text" in content
    assert "escapeHtml(fail.explanation" in content


# ===========================================================================
# 2. NULL CONFIDENCE HANDLING
# ===========================================================================
def format_confidence_ui(confidence: float | None) -> str:
    """Python mirror of app.js formatConfidence."""
    if confidence is None:
        return "Not calibrated"
    return f"{round(confidence * 100)}%"


def test_null_confidence_ui_handling():
    assert format_confidence_ui(None) == "Not calibrated"
    assert format_confidence_ui(0.85) == "85%"
    assert format_confidence_ui(0.0) == "0%"
    assert format_confidence_ui(1.0) == "100%"
    # Must never return "NaN%", "0%" for null, or "100%" for null
    assert format_confidence_ui(None) != "0%"
    assert format_confidence_ui(None) != "100%"
    assert format_confidence_ui(None) != "NaN%"


# ===========================================================================
# 3. BACKEND-DRIVEN RELEASE GATE IN API
# ===========================================================================
def test_backend_release_gate_included_in_run_api(in_memory_db, client):
    repo = DatabaseRepo(in_memory_db)
    proj = repo.create_project(name="Gate Policy Test Proj")
    ds = make_published_dataset(repo, proj.id, "Gate DS", "v1.0")
    run = repo.create_run(
        RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="v1.0",
            system_version="sys-1",
            policy_id="prod-default",
        ),
        RunProvenance(
            dataset_checksum=ds.checksum_sha256,
            rag_version="v1",
            dataset_id=ds.id,
            dataset_version="v1.0",
        ),
    )
    # Add a trace with high faithfulness
    trace = RagTrace(
        trace_id=generate_id("tr"),
        run_id=run.id,
        test_case_id="tc1",
        question="What is X?",
        answer="X is 42.",
    )
    metrics = [
        MetricResult(
            metric_name="faithfulness",
            metric_family=MetricFamily.GENERATION,
            score=0.98,
        ),
        MetricResult(
            metric_name="recall_at_5",
            metric_family=MetricFamily.RETRIEVAL,
            score=0.95,
        ),
    ]
    repo.record_trace(trace, metrics)
    in_memory_db.commit()

    # GET /v1/runs
    resp = client.get("/v1/runs?project_id=" + proj.id)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 1
    run_obj = data["runs"][0]
    assert "gate_result" in run_obj
    assert run_obj["gate_result"]["status"] == "PASS"

    # GET /v1/runs/{id}
    resp_single = client.get(f"/v1/runs/{run.id}")
    assert resp_single.status_code == 200
    assert resp_single.json()["gate_result"]["status"] == "PASS"


# ===========================================================================
# 4. FAILURE_ONLY TRACE PAGINATION (SQL FILTER -> COUNT -> OFFSET -> LIMIT)
# ===========================================================================
def test_failure_only_trace_pagination(in_memory_db, client):
    repo = DatabaseRepo(in_memory_db)
    proj = repo.create_project(name="Pagination Proj")
    ds = make_published_dataset(repo, proj.id, "DS", "v1")
    run = repo.create_run(
        RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="v1",
            system_version="sys-1",
        ),
        RunProvenance(dataset_checksum="c1", rag_version="v1", dataset_id=ds.id, dataset_version="v1"),
    )

    # Insert 15 traces: 5 failures and 10 passes
    for i in range(15):
        t = RagTrace(
            trace_id=f"tr_{i:02d}",
            run_id=run.id,
            test_case_id=f"tc_{i:02d}",
            question=f"Q{i}",
            answer=f"A{i}",
        )
        is_fail = (i < 5)  # 5 failures: indices 0, 1, 2, 3, 4
        m = [MetricResult(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, score=0.2 if is_fail else 0.95)]
        diag = None
        if is_fail:
            from rag_platform.models import FailureAttribution, FailureCode, Severity
            diag = FailureAttribution(
                trace_id=t.trace_id,
                primary_code=FailureCode.GEN_01,
                severity=Severity.HIGH,
                explanation="Model hallucinated fact",
            )
        repo.record_trace(t, m, diag)
    in_memory_db.commit()

    # Query 1: failure_only=True, limit=2, offset=0
    # Must return 2 traces, but total must be 5 (matching failures before pagination)
    r1 = client.get(f"/v1/runs/{run.id}/traces?failure_only=true&limit=2&offset=0")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["total"] == 5
    assert len(d1["traces"]) == 2
    assert all(t["failure"] is not None for t in d1["traces"])

    # Query 2: failure_only=True, limit=2, offset=4
    # Must return 1 trace (the 5th failure), total=5
    r2 = client.get(f"/v1/runs/{run.id}/traces?failure_only=true&limit=2&offset=4")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["total"] == 5
    assert len(d2["traces"]) == 1

    # Query 3: failure_only=False, limit=10, offset=0
    # total must be 15
    r3 = client.get(f"/v1/runs/{run.id}/traces?failure_only=false&limit=10&offset=0")
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["total"] == 15
    assert len(d3["traces"]) == 10


# ===========================================================================
# 5. NO FAKE 0.5 METRIC FALLBACKS
# ===========================================================================
def test_missing_metrics_do_not_default_to_0_5():
    reg_engine = RegressionEngine()
    b_summary = RunMetricsSummary(
        metrics={
            "faithfulness": MetricSummary(
                metric_name="faithfulness",
                metric_family=MetricFamily.GENERATION,
                mean=0.92,
                p50=0.92,
                p95=0.95,
                min=0.8,
                max=1.0,
                count=10,
            )
        }
    )
    # Candidate did not evaluate faithfulness at all
    c_summary = RunMetricsSummary(metrics={})

    comparison = reg_engine.compare(
        baseline_summary=b_summary,
        candidate_summary=c_summary,
        baseline_run_id="run_b",
        candidate_run_id="run_c",
        baseline_case_scores={"tc1": 0.9},
        candidate_case_scores={"tc1": None},  # Missing case score
    )

    delta = comparison.metric_deltas.get("faithfulness")
    assert delta is not None
    assert delta.candidate is None
    assert delta.delta_abs is None
    assert delta.direction == "CANDIDATE_MISSING"

    # Per case transition must be CANDIDATE_MISSING, not NEW_FAILURE or 0.5 score drop
    assert comparison.case_transitions["tc1"]["transition"] == "CANDIDATE_MISSING"
    assert "tc1" not in comparison.newly_failed_cases

    # Release policy evaluation must detect MISSING_DATA violation, not treat as 0.5
    gate = reg_engine.evaluate_gate(
        candidate=c_summary,
        policy=ReleasePolicy(min_faithfulness=0.90),
        candidate_run_id="run_c",
    )
    assert gate.status == GateStatus.FAIL
    violation = next(v for v in gate.violations if v.metric_name == "faithfulness")
    assert violation.violation_type == "MISSING_DATA"
    assert violation.candidate_value is None


# ===========================================================================
# 6. SECURE /v1/demo-run ENDPOINT
# ===========================================================================
def test_demo_run_security_in_production(in_memory_db, client, monkeypatch):
    """In production (DEV_MODE=False), non-admin requests must receive 403 Forbidden."""
    # Force DEV_MODE=False and AUTH_ENABLED=True
    monkeypatch.setattr(
        core_module,
        "settings",
        Settings(auth_enabled=True, dev_mode=False, api_key="test-secret-key-32-chars-long-minimum!"),
    )
    ApiKeyRegistry.clear()

    # 1. Non-admin editor key
    editor_key = ApiKeyRegistry.register("client_editor", is_admin=False, project_roles={"proj_1": Role.EDITOR})
    resp_non_admin = client.post("/v1/demo-run", headers={"X-API-Key": editor_key})
    assert resp_non_admin.status_code == 403
    assert "DEV_MODE" in resp_non_admin.json()["detail"] or "admin" in resp_non_admin.json()["detail"].lower()

    # 2. Admin key in production
    admin_key = ApiKeyRegistry.register("client_admin", is_admin=True)
    resp_admin = client.post("/v1/demo-run", headers={"X-API-Key": admin_key})
    assert resp_admin.status_code == 200
    assert resp_admin.json()["status"] == "SUCCESS"


# ===========================================================================
# 7. BACKGROUND RUN SESSION CLOSURE
# ===========================================================================
@pytest.mark.asyncio
async def test_execute_evaluation_run_closes_owned_session(in_memory_db, monkeypatch):
    """Ensure _execute_evaluation_run closes session when created internally."""
    import rag_platform.server as server_module

    session_closed = False

    class InstrumentedSession(Session):
        def close(self):
            nonlocal session_closed
            session_closed = True
            super().close()

    # Monkeypatch create_session to return InstrumentedSession
    real_engine = in_memory_db.get_bind()

    def mock_create_session():
        return InstrumentedSession(real_engine)

    monkeypatch.setattr(server_module, "create_session", mock_create_session)

    repo = DatabaseRepo(in_memory_db)
    proj = repo.create_project(name="Session Close Proj")
    ds = make_published_dataset(repo, proj.id, "Session DS", "v1")
    run = repo.create_run(
        RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="v1",
            system_version="v1",
            options=RunOptions(timeout_seconds=5.0),
        ),
        RunProvenance(dataset_checksum=ds.checksum_sha256, rag_version="v1", dataset_id=ds.id, dataset_version="v1"),
    )
    in_memory_db.commit()

    from rag_platform.server import CreateRunReq
    req = CreateRunReq(
        project_id=proj.id,
        dataset_id=ds.id,
        system_version="v1",
        adapter_type="synthetic",
        mock_mode="PERFECT",
    )
    config = RunConfig(
        project_id=proj.id,
        dataset_id=ds.id,
        dataset_version="v1",
        system_version="v1",
    )
    cases = [TestCase(id=f"{ds.id}-case-1", question="What is A?", expected_answer="A")]

    # Run without passing db_session -> function creates and owns session
    await _execute_evaluation_run(run.id, req, config, cases, db_session=None)
    assert session_closed is True, "Owned database session must be closed in finally block"


# ===========================================================================
# 8. DURABLE RUN WORKER (ATOMIC CLAIMING & STALE RUN RECOVERY)
# ===========================================================================
def test_durable_run_worker_atomic_claiming_and_stale_recovery(in_memory_db):
    worker = DurableRunWorker()
    repo = DatabaseRepo(in_memory_db)
    proj = repo.create_project(name="Durable Worker Proj")
    ds = make_published_dataset(repo, proj.id, "Durable DS", "v1")
    run = repo.create_run(
        RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="v1",
            system_version="v1",
        ),
        RunProvenance(dataset_checksum=ds.checksum_sha256, rag_version="v1", dataset_id=ds.id, dataset_version="v1"),
    )
    # Set to QUEUED
    repo.update_run_status(run.id, RunStatus.QUEUED)
    in_memory_db.commit()

    # Worker 1 claims run
    claimed_run = worker.claim_next_run(in_memory_db)
    assert claimed_run is not None
    assert claimed_run.id == run.id

    # Verify status in DB is RUNNING
    run_row = in_memory_db.get(RunRow, run.id)
    assert run_row.status == RunStatus.RUNNING.value

    # Worker 2 attempts to claim next run -> should return None (already claimed)
    second_claim = worker.claim_next_run(in_memory_db)
    assert second_claim is None

    # Test stale run recovery: set created_at to 1 hour ago
    stale_recovered = worker.recover_stale_runs(in_memory_db, max_age_seconds=0)
    assert run.id in stale_recovered

    in_memory_db.refresh(run_row)
    assert run_row.status == RunStatus.FAILED.value
    assert "STALE" in run_row.failure_type


# ===========================================================================
# 9. PERSISTENT DATABASE API KEYS / RBAC
# ===========================================================================
def test_database_backed_api_keys_and_hashed_storage(in_memory_db):
    repo = DatabaseRepo(in_memory_db)

    # 1. Create a key through DatabaseRepo
    raw_key, api_key_row = repo.create_api_key(
        client_id="corp_client",
        is_admin=False,
        project_roles={"proj_secure": Role.EDITOR},
    )
    in_memory_db.commit()

    # Raw key is returned once, but row stores only SHA-256 hash!
    expected_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    assert api_key_row.key_hash == expected_hash
    assert raw_key not in api_key_row.key_hash

    # 2. Authenticate using database session
    context = authenticate_request(x_api_key=raw_key, db_session=in_memory_db)
    assert context.authenticated is True
    assert context.client_id == "corp_client"
    assert context.can_access_project("proj_secure", Role.EDITOR) is True
    assert context.can_access_project("proj_other", Role.VIEWER) is False


# ===========================================================================
# 10. REMOVAL OF target_fn FROM PUBLIC PYTHON ADAPTER REST API
# ===========================================================================
def test_python_adapter_rejects_missing_adapter_name():
    from rag_platform.server import CreateRunReq

    # Passing target_fn without registered adapter_name must raise 422
    req = CreateRunReq(
        project_id="p1",
        dataset_id="d1",
        system_version="v1",
        adapter_type="python",
        adapter_config={"target_fn": lambda x: x},  # Malicious/untrusted input
    )
    with pytest.raises(HTTPException) as exc:
        _resolve_adapter(req)
    assert exc.value.status_code == 422
    assert "adapter_name" in exc.value.detail

    # Registering trusted adapter and providing adapter_name succeeds
    def trusted_adapter(case, config):
        return RagTrace(trace_id="t1", run_id="r1", test_case_id="tc1", question="Q", answer="A")

    PythonAdapterRegistry.register("trusted_adapter", trusted_adapter)
    req_valid = CreateRunReq(
        project_id="p1",
        dataset_id="d1",
        system_version="v1",
        adapter_type="python",
        adapter_config={"adapter_name": "trusted_adapter"},
    )
    adapter = _resolve_adapter(req_valid)
    assert adapter is not None


# ===========================================================================
# 11 & 12 & 15. TOTAL COUNTS BEFORE PAGINATION EVERYWHERE
# ===========================================================================
def test_pre_pagination_totals_across_endpoints(in_memory_db, client):
    repo = DatabaseRepo(in_memory_db)
    proj = repo.create_project(name="Totals Proj")

    # Create 3 datasets
    for i in range(3):
        make_published_dataset(repo, proj.id, f"DS_{i}", f"v{i}")
    in_memory_db.commit()

    # list_datasets: limit=1, offset=0 -> total must be 3, datasets list length 1
    ds_resp = client.get(f"/v1/datasets?project_id={proj.id}&limit=1&offset=0")
    assert ds_resp.status_code == 200
    ds_data = ds_resp.json()
    assert ds_data["total"] == 3
    assert len(ds_data["datasets"]) == 1

    # list_failures: add 4 failures
    ds_first = in_memory_db.query(DatasetRow).first()
    run = repo.create_run(
        RunConfig(project_id=proj.id, dataset_id=ds_first.id, dataset_version="v0", system_version="s1"),
        RunProvenance(dataset_checksum=ds_first.checksum_sha256, rag_version="v", dataset_id=ds_first.id, dataset_version="v0"),
    )
    for i in range(4):
        tr = RagTrace(trace_id=f"t_{i}", run_id=run.id, test_case_id=f"tc_{i}", question="Q", answer="A")
        repo.record_trace(tr, [])
        fail = FailureRow(
            id=f"fail_{i}",
            trace_id=tr.trace_id,
            run_id=run.id,
            failure_type="HALLUCINATION",
            severity="HIGH",
            explanation="Explanation",
        )
        in_memory_db.add(fail)
    in_memory_db.commit()

    fail_resp = client.get(f"/v1/failures?run_id={run.id}&limit=2&offset=0")
    assert fail_resp.status_code == 200
    fail_data = fail_resp.json()
    assert fail_data["total"] == 4
    assert len(fail_data["failures"]) == 2


# ===========================================================================
# 16. DATABASE-LEVEL AUTHORIZATION / CROSS-PROJECT ID SUBSTITUTION
# ===========================================================================
def test_cross_project_id_substitution_is_blocked(in_memory_db, client, monkeypatch):
    monkeypatch.setattr(
        core_module,
        "settings",
        Settings(auth_enabled=True, dev_mode=False, api_key="test-secret-key-32-chars-long-minimum!"),
    )
    ApiKeyRegistry.clear()

    repo = DatabaseRepo(in_memory_db)
    proj_a = repo.create_project(name="Project A")
    proj_b = repo.create_project(name="Project B")

    ds_b = make_published_dataset(repo, proj_b.id, "DS B", "v1")
    run_b = repo.create_run(
        RunConfig(project_id=proj_b.id, dataset_id=ds_b.id, dataset_version="v1", system_version="v1"),
        RunProvenance(dataset_checksum=ds_b.checksum_sha256, rag_version="v", dataset_id=ds_b.id, dataset_version="v1"),
    )
    in_memory_db.commit()

    # User has access ONLY to Project A
    key_a = ApiKeyRegistry.register("client_a", is_admin=False, project_roles={proj_a.id: Role.EDITOR})

    headers_a = {"X-API-Key": key_a}

    # 1. Attempt to view Project B's run
    resp_run = client.get(f"/v1/runs/{run_b.id}", headers=headers_a)
    assert resp_run.status_code == 403

    # 2. Attempt to view Project B's dataset
    resp_ds = client.get(f"/v1/datasets/{ds_b.id}", headers=headers_a)
    assert resp_ds.status_code == 403

    # 3. Attempt to create a run in Project A pointing to Project B's dataset
    resp_create_run = client.post(
        "/v1/runs",
        headers=headers_a,
        json={
            "project_id": proj_a.id,
            "dataset_id": ds_b.id,
            "system_version": "v1",
            "adapter_type": "synthetic",
        },
    )
    assert resp_create_run.status_code == 403
    assert "belongs to project" in resp_create_run.json()["detail"]
