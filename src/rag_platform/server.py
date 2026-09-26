"""FastAPI REST API control plane and engineering-grade observability dashboard.
"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, Header, HTTPException, Query, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from rag_platform import __version__
from rag_platform.adapters import (
    AdapterRegistry,
    HttpRagAdapter,
    PythonAdapterRegistry,
    PythonRagAdapter,
    SyntheticRagAdapter,
    SyntheticRagMode,
)
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.core import generate_id, get_settings
from rag_platform.db import (
    Base,
    DatabaseRepo,
    DatasetRow,
    DatasetStatus,
    FailureRow,
    ProjectRow,
    RunRow,
    TraceRow,
    create_db_engine,
)
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    Answerability,
    DocumentReference,
    MetricFamily,
    MetricResult,
    RagTrace,
    ReleasePolicy,
    RunConfig,
    RunOptions,
    RunProvenance,
    RunStatus,
    Severity,
    TestCase,
)
from rag_platform.regression import RegressionEngine
from rag_platform.security import (
    BudgetGuard,
    Role,
    SecretRedactor,
    SecurityContext,
    TokenBucketRateLimiter,
    authenticate_request,
    authorize_project,
)

# --- Database setup ---
# Schema is managed by Alembic migrations (`alembic upgrade head`), NOT create_all().
# Exception: SQLite in-memory URLs are only used in tests — safe to auto-create there.
engine = create_db_engine()
_db_url: str = get_settings().database_url
if ":memory:" in _db_url:
    Base.metadata.create_all(bind=engine)  # test isolation only
# else: rely entirely on Alembic

app = FastAPI(
    title="Reliab",
    description="Automated failure attribution, regression testing, and quality release gates for RAG systems.",
    version=__version__,
)

# Static directory setup
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def get_db():
    with Session(engine) as session:
        yield session


def create_session() -> Session:
    """Create a database session, respecting any test dependency overrides."""
    if get_db in app.dependency_overrides:
        gen = app.dependency_overrides[get_db]()
        return next(gen)
    return Session(engine)


def get_auth(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    cookie_api_key: str | None = Cookie(default=None, alias="api_key"),
    db: Session = Depends(get_db),
) -> SecurityContext:
    """FastAPI dependency for authentication (supports Header, Bearer, and Cookie)."""
    return authenticate_request(
        x_api_key=x_api_key,
        authorization=authorization,
        cookie_token=cookie_api_key,
        db_session=db,
    )


@app.get("/health")
@app.get("/v1/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Canonical TestCase reconstruction — FIX #2
# ---------------------------------------------------------------------------
def db_row_to_test_case(row: Any) -> TestCase:
    """Single canonical function for DB TestCaseRow → TestCase domain model.

    Preserves ALL fields stored in the database, including expected_facts, tags,
    and metadata. Prevents silent field loss when TestCase is reconstructed
    ad-hoc in multiple entry points (API, CLI, gate, tests).
    Loss of expected_facts makes answer-correctness evaluation meaningless.
    """
    return TestCase(
        id=row.id,
        question=row.question,
        expected_answer=row.expected_answer,
        expected_facts=json.loads(row.expected_facts_json) if row.expected_facts_json else [],
        relevant_documents=[
            DocumentReference(**d)
            for d in json.loads(row.relevant_docs_json)
        ] if row.relevant_docs_json else [],
        answerability=row.answerability,
        tags=json.loads(row.tags_json) if row.tags_json else [],
        metadata=json.loads(row.metadata_json) if row.metadata_json else {},
    )


# --- API Request & Response Models ---
class CreateProjectReq(BaseModel):
    name: str
    settings: dict[str, Any] = Field(default_factory=dict)


class CreateDatasetReq(BaseModel):
    project_id: str
    name: str
    version: str
    description: str | None = None
    cases: list[TestCase] = Field(default_factory=list, max_length=10000)


class BulkCasesReq(BaseModel):
    cases: list[TestCase] = Field(..., min_length=1, max_length=10000)
    publish: bool = True


class CreateRunReq(BaseModel):
    project_id: str
    dataset_id: str
    dataset_version: str | None = None
    system_version: str
    policy_id: str = "prod-default"
    mock_mode: str = "PERFECT"
    adapter_type: str = "synthetic"
    adapter_config: dict[str, Any] = Field(default_factory=dict)
    model_name: str | None = None
    model_version: str | None = None
    model_parameters: dict[str, Any] = Field(default_factory=dict)
    temperature: float | None = 0.0
    prompt_template: str | None = None
    system_prompt: str | None = None
    embedding_model: str | None = None
    embedding_version: str | None = None
    retriever_config: dict[str, Any] = Field(default_factory=dict)
    reranker_config: dict[str, Any] = Field(default_factory=dict)
    chunking_config: dict[str, Any] = Field(default_factory=dict)
    random_seed: int | None = None
    async_exec: bool = False
    # RunOptions fields — bounded & enforced (Points 9, 10, 29)
    concurrency: int = Field(default=5, ge=1, le=100)
    max_cases: int | None = Field(default=500, ge=1, le=10000)
    timeout_seconds: float = Field(default=60.0, ge=0.01, le=3600.0)
    fail_fast: bool = False
    use_cache: bool = True


class CompareReq(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    policy: ReleasePolicy = Field(default_factory=ReleasePolicy)
    allow_cross_dataset: bool = False


class AuthSessionReq(BaseModel):
    api_key: str


class CreateApiKeyReq(BaseModel):
    client_id: str = Field(..., min_length=1, max_length=64)
    project_roles: dict[str, str] = Field(default_factory=dict)
    is_admin: bool = False


# --- Session Auth & Rate Limiting ---
class AuthRateLimiter:
    """Sliding-window rate limiter protecting authentication endpoints against brute-force attacks."""

    def __init__(self, max_failures: int = 5, window_seconds: float = 60.0):
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_blocked(self, ip: str) -> tuple[bool, float]:
        now = time.monotonic()
        with self._lock:
            timestamps = self._failures.get(ip, [])
            valid = [t for t in timestamps if (now - t) < self.window_seconds]
            self._failures[ip] = valid
            if len(valid) >= self.max_failures:
                retry_after = max(1.0, self.window_seconds - (now - valid[0]))
                return True, retry_after
            return False, 0.0

    def record_failure(self, ip: str) -> None:
        now = time.monotonic()
        with self._lock:
            timestamps = self._failures.setdefault(ip, [])
            timestamps.append(now)

    def record_success(self, ip: str) -> None:
        with self._lock:
            self._failures.pop(ip, None)

    def reset(self) -> None:
        with self._lock:
            self._failures.clear()


auth_rate_limiter = AuthRateLimiter(max_failures=5, window_seconds=60.0)


def extract_client_ip(request: Request) -> str:
    """Safely extract client IP, respecting proxy forwarding headers if present."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "127.0.0.1"


@app.post("/v1/auth/session")
def create_auth_session(
    request: Request,
    req: AuthSessionReq,
    response: Response,
    db: Session = Depends(get_db),
):
    """Authenticate and issue an HttpOnly, SameSite=Strict session cookie with IP rate limiting."""
    client_ip = extract_client_ip(request)
    blocked, retry_after = auth_rate_limiter.is_blocked(client_ip)
    if blocked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed authentication attempts. Please try again later.",
            headers={"Retry-After": str(int(retry_after))},
        )

    settings = get_settings()
    try:
        ctx = authenticate_request(x_api_key=req.api_key, db_session=db)
    except HTTPException:
        auth_rate_limiter.record_failure(client_ip)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key or credential.",
        )

    auth_rate_limiter.record_success(client_ip)
    response.set_cookie(
        key="api_key",
        value=req.api_key,
        httponly=True,
        samesite="strict",
        secure=not settings.dev_mode,
        max_age=86400,
        path="/",
    )
    return {
        "status": "SUCCESS",
        "client_id": ctx.client_id,
        "is_admin": ctx.is_admin,
        "message": "Session authenticated via HttpOnly cookie.",
    }


@app.post("/v1/auth/logout")
def clear_auth_session(response: Response):
    """Clear authenticated session cookie."""
    response.delete_cookie(key="api_key", path="/")
    return {"status": "SUCCESS", "message": "Session cleared."}


# --- API Key Lifecycle Endpoints (Admin Authorization Required) ---
@app.post("/v1/auth/keys")
def create_api_key_endpoint(
    req: CreateApiKeyReq,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Issue a new API key. Requires administrator privileges."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required to issue API credentials.")
    repo = DatabaseRepo(db)
    raw_key, row = repo.create_api_key(
        client_id=req.client_id,
        project_roles=req.project_roles,
        is_admin=req.is_admin,
    )
    db.commit()
    roles = json.loads(row.project_roles_json) if row.project_roles_json else {}
    return {
        "status": "SUCCESS",
        "api_key": raw_key,
        "key_hash": row.key_hash,
        "client_id": row.client_id,
        "is_admin": row.is_admin,
        "project_roles": roles,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


@app.get("/v1/auth/keys")
def list_api_keys_endpoint(
    client_id: str | None = None,
    include_revoked: bool = False,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """List registered API credentials. Requires administrator privileges."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required to list API credentials.")
    repo = DatabaseRepo(db)
    rows = repo.list_api_keys(client_id=client_id, include_revoked=include_revoked)
    results = []
    for r in rows:
        roles = json.loads(r.project_roles_json) if r.project_roles_json else {}
        results.append({
            "key_hash": r.key_hash,
            "client_id": r.client_id,
            "is_admin": r.is_admin,
            "project_roles": roles,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "revoked_at": r.revoked_at.isoformat() if r.revoked_at else None,
            "is_active": r.revoked_at is None,
        })
    return {"keys": results}


@app.post("/v1/auth/keys/{key_hash}/revoke")
def revoke_api_key_endpoint(
    key_hash: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Revoke an active API credential. Requires administrator privileges."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required to revoke API credentials.")
    repo = DatabaseRepo(db)
    success = repo.revoke_api_key(key_hash)
    db.commit()
    if not success:
        raise HTTPException(status_code=404, detail=f"Active API key with hash '{key_hash}' not found or already revoked.")
    return {"status": "SUCCESS", "message": f"API key '{key_hash}' revoked successfully."}


@app.post("/v1/auth/keys/{key_hash}/rotate")
def rotate_api_key_endpoint(
    key_hash: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Rotate an active API credential. Requires administrator privileges."""
    if not auth.is_admin:
        raise HTTPException(status_code=403, detail="Administrator role required to rotate API credentials.")
    repo = DatabaseRepo(db)
    try:
        raw_new_key, new_row = repo.rotate_api_key(key_hash)
        db.commit()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    roles = json.loads(new_row.project_roles_json) if new_row.project_roles_json else {}
    return {
        "status": "SUCCESS",
        "api_key": raw_new_key,
        "key_hash": new_row.key_hash,
        "old_key_hash": key_hash,
        "client_id": new_row.client_id,
        "is_admin": new_row.is_admin,
        "project_roles": roles,
        "created_at": new_row.created_at.isoformat() if new_row.created_at else None,
        "message": "API key rotated successfully. Previous key revoked.",
    }


# --- REST API Endpoints ---
@app.post("/v1/projects")
def create_project(
    req: CreateProjectReq,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    if not auth.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: project creation requires ADMIN role.",
        )
    repo = DatabaseRepo(db)
    proj = repo.create_project(name=req.name, settings=req.settings)
    db.commit()
    return {"id": proj.id, "name": proj.name}


@app.get("/v1/projects")
def list_projects(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    query = select(ProjectRow)
    count_query = select(func.count(ProjectRow.id))
    allowed = auth.get_authorized_projects()
    if allowed is not None:
        query = query.where(ProjectRow.id.in_(allowed))
        count_query = count_query.where(ProjectRow.id.in_(allowed))
    total = db.scalar(count_query) or 0
    query = query.order_by(ProjectRow.name.asc()).offset(offset).limit(limit)
    projects = db.scalars(query).all()
    return {
        "projects": [{"id": p.id, "name": p.name} for p in projects],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


@app.get("/v1/projects/{project_id}")
def get_project(
    project_id: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    authorize_project(project_id, auth, required_role=Role.VIEWER)
    proj = db.get(ProjectRow, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"id": proj.id, "name": proj.name}


@app.post("/v1/datasets")
def create_dataset(
    req: CreateDatasetReq,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    authorize_project(req.project_id, auth, required_role=Role.EDITOR)
    repo = DatabaseRepo(db)
    row = repo.create_dataset(
        project_id=req.project_id,
        name=req.name,
        version=req.version,
        description=req.description,
    )
    if req.cases:
        repo.add_test_cases(row.id, req.cases)
        repo.publish_dataset(row.id)
    db.commit()
    return {"id": row.id, "checksum": row.checksum_sha256, "status": row.status}


@app.post("/v1/datasets/{dataset_id}/cases/bulk")
def add_cases_bulk(
    dataset_id: str,
    req: BulkCasesReq,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Add test cases to a dataset in bulk chunks (Point 30)."""
    repo = DatabaseRepo(db)
    ds = repo.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    authorize_project(ds.project_id, auth, required_role=Role.EDITOR)
    if ds.status == DatasetStatus.PUBLISHED.value:
        raise HTTPException(status_code=400, detail="Cannot add cases to an already published dataset")
    repo.add_test_cases(dataset_id, req.cases)
    if req.publish:
        repo.publish_dataset(dataset_id)
    db.commit()
    return {
        "dataset_id": dataset_id,
        "cases_added": len(req.cases),
        "status": ds.status,
        "checksum": ds.checksum_sha256,
    }


@app.get("/v1/datasets")
def list_datasets(
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    query = select(DatasetRow)
    count_query = select(func.count(DatasetRow.id))
    if project_id:
        authorize_project(project_id, auth, required_role=Role.VIEWER)
        query = query.where(DatasetRow.project_id == project_id)
        count_query = count_query.where(DatasetRow.project_id == project_id)
    else:
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.where(DatasetRow.project_id.in_(allowed))
            count_query = count_query.where(DatasetRow.project_id.in_(allowed))
    total = db.scalar(count_query) or 0
    query = query.order_by(DatasetRow.created_at.desc()).offset(offset).limit(limit)
    rows = db.scalars(query).all()
    return {
        "datasets": [
            {
                "id": r.id,
                "project_id": r.project_id,
                "name": r.name,
                "version": r.version,
                "status": r.status,
                "checksum": r.checksum_sha256,
            }
            for r in rows
        ],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


@app.get("/v1/datasets/{dataset_id}")
def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    ds = db.get(DatasetRow, dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    authorize_project(ds.project_id, auth, required_role=Role.VIEWER)
    repo = DatabaseRepo(db)
    case_count = repo.get_dataset_case_count(ds.id)
    return {
        "id": ds.id,
        "project_id": ds.project_id,
        "name": ds.name,
        "version": ds.version,
        "status": ds.status,
        "checksum": ds.checksum_sha256,
        "description": ds.description,
        "case_count": case_count,
    }


@app.get("/v1/runs")
def list_runs(
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    query = select(RunRow)
    count_query = select(func.count(RunRow.id))
    if project_id:
        authorize_project(project_id, auth, required_role=Role.VIEWER)
        query = query.where(RunRow.project_id == project_id)
        count_query = count_query.where(RunRow.project_id == project_id)
    else:
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.where(RunRow.project_id.in_(allowed))
            count_query = count_query.where(RunRow.project_id.in_(allowed))
    total = db.scalar(count_query) or 0
    query = query.order_by(RunRow.created_at.desc()).offset(offset).limit(limit)
    runs = db.scalars(query).all()
    eval_engine = EvaluationEngine()
    reg_engine = RegressionEngine()

    result = []
    for r in runs:
        summary_dict = json.loads(r.summary_json) if r.summary_json else None
        gate_dict = json.loads(r.gate_result_json) if r.gate_result_json else None

        # Fallback only for legacy runs missing persisted summaries
        if summary_dict is None and r.traces:
            traces = [
                (
                    RagTrace.model_validate_json(t.raw_trace_json),
                    [
                        MetricResult(
                            metric_name=m.metric_name,
                            metric_family=MetricFamily(m.metric_family),
                            score=m.score,
                        )
                        for m in t.metrics
                    ],
                )
                for t in r.traces
            ]
            summary = eval_engine.aggregate_run(traces) if traces else None
            summary_dict = summary.model_dump() if summary else None
            if summary:
                gate_dict = reg_engine.evaluate_gate(
                    summary,
                    ReleasePolicy(policy_id=r.policy_id),
                    candidate_run_id=r.id,
                ).model_dump()

        trace_count = (
            summary_dict.get("total_cases")
            if (summary_dict and "total_cases" in summary_dict)
            else len(r.traces)
        )
        result.append({
            "id": r.id,
            "project_id": r.project_id,
            "dataset_id": r.dataset_id,
            "system_version": r.system_version,
            "manifest_hash": r.manifest_hash,
            "status": r.status,
            "policy_id": r.policy_id,
            "created_at": r.created_at.isoformat(),
            "trace_count": trace_count,
            "summary": summary_dict,
            "gate_result": gate_dict,
        })
    return {"runs": result, "limit": limit, "offset": offset, "total": total}


def _resolve_adapter(req: "CreateRunReq") -> Any:
    """Resolve the correct adapter based on adapter_type.

    adapter_type is the sole authoritative selector.
    Python adapters are resolved strictly from the trusted server-side PythonAdapterRegistry.
    """
    adapter_type_key = req.adapter_type.lower().strip()

    if adapter_type_key == "synthetic":
        mode_str = req.mock_mode if req.mock_mode else "PERFECT"
        try:
            mode = SyntheticRagMode(mode_str)
        except ValueError:
            mode = SyntheticRagMode.PERFECT
        return SyntheticRagAdapter(mode)

    if adapter_type_key == "http":
        endpoint_url = req.adapter_config.get("endpoint_url", "")
        if not endpoint_url:
            raise HTTPException(status_code=422, detail="HTTP adapter requires adapter_config.endpoint_url")
        case_timeout = float(req.timeout_seconds if hasattr(req, "timeout_seconds") and req.timeout_seconds else 60.0)
        # Cleanly enforce overall case timeout >= HTTP connection/read timeout
        http_timeout = float(req.adapter_config.get("timeout_seconds", min(30.0, case_timeout)))
        return HttpRagAdapter(
            endpoint_url=endpoint_url,
            headers=req.adapter_config.get("headers"),
            timeout_seconds=http_timeout,
        )

    if adapter_type_key == "python":
        adapter_name = req.adapter_config.get("adapter_name")
        if not adapter_name:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Python adapter requires 'adapter_name' in adapter_config referencing "
                    "a pre-registered callable in PythonAdapterRegistry."
                ),
            )
        try:
            fn = PythonAdapterRegistry.get(adapter_name)
            return PythonRagAdapter(fn)
        except ValueError as err:
            raise HTTPException(status_code=422, detail=str(err))

    # Any other custom registered adapters
    return AdapterRegistry.get(adapter_type_key, **req.adapter_config)


# Shared two-tier evaluation cache (L1 in-memory LRU + L2 database persistence across processes)
from rag_platform.evaluators import TwoTierEvaluationCache

_eval_cache = TwoTierEvaluationCache(l1_capacity=10_000, l2_capacity=50_000)


async def _execute_evaluation_run(
    run_id: str,
    req: "CreateRunReq",
    config: RunConfig,
    ds_cases: list[TestCase],
    db_session: Session | None = None,
    expected_lease_id: str | None = None,
) -> None:
    """Execute evaluation run with bounded concurrency, strict timeout, fail-fast, and resource lifecycle."""
    eval_engine = EvaluationEngine(cache=_eval_cache)
    attr_engine = FailureAttributionEngine()

    adapter = _resolve_adapter(req)

    # Acquire session (respecting dependency overrides if running in test)
    owned_session = db_session is None
    sess = db_session if db_session is not None else create_session()
    repo = DatabaseRepo(sess)
    now = datetime.now(timezone.utc)
    sess.expire_all()
    run_row = sess.get(RunRow, run_id)
    if not run_row or run_row.status in (
        RunStatus.FAILED.value,
        RunStatus.CANCELLED.value,
        RunStatus.COMPLETED.value,
    ):
        return
    if expected_lease_id and run_row.lease_id != expected_lease_id:
        return

    if not run_row.started_at:
        run_row.started_at = now
    run_row.heartbeat_at = now
    run_row.status = RunStatus.RUNNING.value
    sess.commit()

    timeout_sec = float(getattr(config.options, "timeout_seconds", 60) or 60)
    fail_fast = bool(getattr(config.options, "fail_fast", False))
    use_cache = bool(getattr(config.options, "use_cache", True))

    async def _process_case_safe(case: TestCase) -> tuple[RagTrace, list[MetricResult], Any]:
        start_time = time.perf_counter()
        try:
            if timeout_sec > 0:
                trace = await asyncio.wait_for(adapter.run(case, config), timeout=timeout_sec)
            else:
                trace = await adapter.run(case, config)
        except asyncio.TimeoutError:
            elapsed = int((time.perf_counter() - start_time) * 1000)
            trace = RagTrace(
                trace_id=generate_id("tr"),
                run_id=run_id,
                test_case_id=case.id,
                question=case.question,
                error_code="OPS-01",
                latency_ms=elapsed,
                telemetry={
                    "error": f"Adapter execution timed out after {timeout_sec}s",
                    "timeout_seconds": timeout_sec,
                    "stage": "adapter_run",
                },
            )
        except Exception as exc:
            elapsed = int((time.perf_counter() - start_time) * 1000)
            from rag_platform.security import SecretRedactor
            trace = RagTrace(
                trace_id=generate_id("tr"),
                run_id=run_id,
                test_case_id=case.id,
                question=case.question,
                error_code="OPS-01",
                latency_ms=elapsed,
                telemetry={
                    "error": SecretRedactor.redact_text(str(exc)),
                    "exception_type": type(exc).__name__,
                    "stage": "adapter_run",
                },
            )

        trace.run_id = run_id
        if not trace.trace_id or not trace.trace_id.startswith(f"tr_{run_id}"):
            trace.trace_id = f"tr_{run_id}_{case.id}"
        metrics = await eval_engine.evaluate_trace(trace, case, use_cache=use_cache)
        diag = attr_engine.diagnose(trace, case, metrics)
        return (trace, metrics, diag)

    try:
        max_cases = getattr(config.options, "max_cases", None)
        cases_to_run = ds_cases[:max_cases] if (max_cases is not None and max_cases > 0) else ds_cases

        case_queue: asyncio.Queue[TestCase] = asyncio.Queue()
        for c in cases_to_run:
            case_queue.put_nowait(c)

        results: list[tuple[RagTrace, list[MetricResult], Any]] = []
        results_lock = asyncio.Lock()
        had_fatal_failure = False
        had_cancellation = False

        rate_limit_rps = req.adapter_config.get("rate_limit_rps")
        rate_limiter = (
            TokenBucketRateLimiter(rate=float(rate_limit_rps), capacity=float(req.concurrency * 2))
            if rate_limit_rps
            else None
        )

        async def worker():
            nonlocal had_fatal_failure, had_cancellation
            while not case_queue.empty():
                if (had_fatal_failure and fail_fast) or had_cancellation:
                    break
                try:
                    case = case_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if (had_fatal_failure and fail_fast) or had_cancellation:
                    case_queue.task_done()
                    break

                # Rate limiting guardrail if configured (Point 27)
                if rate_limiter:
                    await rate_limiter.acquire(1.0)

                trace, metrics, diag = await _process_case_safe(case)

                async with results_lock:
                    # Check cancellation and lease validity under lock before committing trace
                    sess.expire_all()
                    cur_run = sess.get(RunRow, run_id)
                    if (
                        not cur_run
                        or cur_run.status != RunStatus.RUNNING.value
                        or (expected_lease_id and cur_run.lease_id != expected_lease_id)
                    ):
                        had_cancellation = True
                        case_queue.task_done()
                        break

                    results.append((trace, metrics, diag))
                    repo.record_trace(trace, metrics, diag)
                    sess.commit()

                case_queue.task_done()

                # Determine if fatal failure
                is_fatal = bool(
                    trace.error_code is not None or
                    (diag is not None and getattr(diag, "severity", None) in (Severity.CRITICAL, Severity.HIGH))
                )
                if is_fatal and fail_fast:
                    had_fatal_failure = True
                    # Drain remaining queue so workers terminate
                    while not case_queue.empty():
                        try:
                            case_queue.get_nowait()
                            case_queue.task_done()
                        except asyncio.QueueEmpty:
                            break
                    break

        worker_count = max(1, min(req.concurrency, len(cases_to_run))) if cases_to_run else 1
        workers = [asyncio.create_task(worker()) for _ in range(worker_count)]
        if workers:
            await asyncio.gather(*workers)

        now = datetime.now(timezone.utc)
        sess.expire_all()
        run_row = sess.get(RunRow, run_id)
        if not run_row or run_row.status != RunStatus.RUNNING.value:
            # Run was cancelled, marked failed by recovery, or already finalized
            return
        if expected_lease_id and run_row.lease_id != expected_lease_id:
            # Worker lease was lost or claimed by another worker
            return

        final_status = RunStatus.FAILED if (had_fatal_failure and fail_fast) else RunStatus.COMPLETED
        failure_reason = "Run aborted due to fatal case failure (fail_fast=true)" if (had_fatal_failure and fail_fast) else None
        failure_type = "FAIL_FAST" if (had_fatal_failure and fail_fast) else None

        # Pre-compute and persist run summary and gate result to avoid N+1 recalculation on list/get
        summary_obj = eval_engine.aggregate_run([(t, m) for t, m, _ in results]) if results else None
        gate_obj = None
        if summary_obj and final_status == RunStatus.COMPLETED:
            from rag_platform.regression import RegressionEngine, ReleasePolicy
            reg_engine = RegressionEngine()
            gate_obj = reg_engine.evaluate_gate(
                summary_obj,
                ReleasePolicy(policy_id=run_row.policy_id),
                candidate_run_id=run_row.id,
            )

        # Atomic conditional update of final state: only succeeds if run is still RUNNING and lease still matches
        final_stmt = (
            update(RunRow)
            .where(
                RunRow.id == run_id,
                RunRow.status == RunStatus.RUNNING.value,
            )
        )
        if expected_lease_id:
            final_stmt = final_stmt.where(RunRow.lease_id == expected_lease_id)

        update_values: dict[str, Any] = {
            "status": final_status.value,
            "finished_at": now,
            "heartbeat_at": now,
        }
        if failure_reason:
            update_values["failure_reason"] = failure_reason
        if failure_type:
            update_values["failure_type"] = failure_type
        if summary_obj:
            update_values["summary_json"] = summary_obj.model_dump_json()
        if gate_obj:
            update_values["gate_result_json"] = gate_obj.model_dump_json()
            update_values["gate_status"] = gate_obj.status.value

        res = sess.execute(final_stmt.values(**update_values))
        sess.commit()
        if res.rowcount == 0:
            import logging
            logging.getLogger("rag_platform.server").warning(
                "run_id=%s final atomic update affected 0 rows; lost lease or recovered.", run_id
            )
            return

        import logging
        logging.getLogger("rag_platform.server").info(
            "run_id=%s finished with %d/%d cases processed (had_fatal_failure=%s, fail_fast=%s)",
            run_id, len(results), len(ds_cases), had_fatal_failure, fail_fast,
        )
    except Exception as exc:
        exc_type = type(exc).__name__
        import logging
        logging.getLogger("rag_platform.server").exception(
            "run_id=%s run-level failure: %s", run_id, exc_type
        )
        try:
            with create_session() as fail_sess:
                fail_stmt = update(RunRow).where(
                    RunRow.id == run_id,
                    RunRow.status == RunStatus.RUNNING.value,
                )
                if expected_lease_id:
                    fail_stmt = fail_stmt.where(RunRow.lease_id == expected_lease_id)
                from rag_platform.security import SecretRedactor
                safe_reason = SecretRedactor.redact_text(str(exc))[:500]
                fail_sess.execute(
                    fail_stmt.values(
                        status=RunStatus.FAILED.value,
                        finished_at=datetime.now(timezone.utc),
                        failure_reason=safe_reason,
                        failure_type=exc_type[:64],
                    )
                )
                fail_sess.commit()
        except Exception:
            pass
    finally:
        # Guarantee adapter resource closure (HTTP connection pools, sockets)
        if hasattr(adapter, "close"):
            try:
                res = adapter.close()
                if hasattr(res, "__await__"):
                    await res
            except Exception:
                pass
        if owned_session:
            sess.close()


@app.post("/v1/runs")
async def create_run(
    req: CreateRunReq,
    background_tasks: BackgroundTasks,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    authorize_project(req.project_id, auth, required_role=Role.EDITOR)
    repo = DatabaseRepo(db)
    ds = repo.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Dataset ownership validation
    if ds.project_id != req.project_id:
        raise HTTPException(
            status_code=403,
            detail=(
                f"Dataset '{req.dataset_id}' belongs to project '{ds.project_id}', "
                f"not to the requested project '{req.project_id}'."
            ),
        )

    # Validate dataset version (Point 13)
    if req.dataset_version and req.dataset_version != ds.version:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Supplied dataset_version '{req.dataset_version}' does not match "
                f"published dataset version '{ds.version}'."
            ),
        )

    cases = [db_row_to_test_case(r) for r in ds.cases]

    # Validate safety limits via BudgetGuard (Points 10, 27)
    effective_max_cases = req.max_cases if req.max_cases is not None else 500
    config_options = RunOptions(
        concurrency=req.concurrency,
        max_cases=effective_max_cases,
        timeout_seconds=getattr(req, "timeout_seconds", 60.0),
        fail_fast=getattr(req, "fail_fast", False),
        use_cache=getattr(req, "use_cache", True),
    )
    try:
        cases_to_evaluate = min(len(cases), effective_max_cases)
        BudgetGuard.validate_run_bounds(cases_to_evaluate, config_options)
    except Exception as err:
        raise HTTPException(status_code=400, detail=str(err))

    env_info = {
        "os": platform.system(),
        "arch": platform.machine(),
        "python": sys.version.split()[0],
    }

    provenance = RunProvenance(
        dataset_checksum=ds.checksum_sha256,
        dataset_id=ds.id,
        dataset_version=ds.version,
        rag_version=req.system_version,
        model_name=req.model_name,
        model_version=req.model_version,
        model_parameters=req.model_parameters,
        temperature=req.temperature,
        prompt_template=req.prompt_template,
        system_prompt=req.system_prompt,
        embedding_model=req.embedding_model,
        embedding_version=req.embedding_version,
        retriever_config=req.retriever_config,
        reranker_config=req.reranker_config,
        chunking_config=req.chunking_config,
        adapter_type=req.adapter_type,
        adapter_config=req.adapter_config,
        evaluator_version="2.0.0",
        python_version=platform.python_version(),
        environment_info=env_info,
        random_seed=req.random_seed,
    )

    config = RunConfig(
        project_id=req.project_id,
        dataset_id=req.dataset_id,
        dataset_version=ds.version,
        system_version=req.system_version,
        policy_id=req.policy_id,
        options=config_options,
    )

    initial_status = RunStatus.QUEUED if req.async_exec else RunStatus.CREATED
    # Sanitize options before persistence (Point 4)
    safe_options = SecretRedactor.redact_dict(req.model_dump())
    safe_options_json = json.dumps(safe_options)

    run = repo.create_run(
        config,
        provenance,
        initial_status=initial_status,
        options_override_json=safe_options_json,
        idempotency_key=idempotency_key,
    )
    db.commit()

    # If run already existed under this idempotency key
    if idempotency_key and (
        getattr(run, "_is_existing", False)
        or run.status in (
            RunStatus.RUNNING.value,
            RunStatus.COMPLETED.value,
            RunStatus.QUEUED.value,
            RunStatus.FAILED.value,
        )
    ):
        if req.async_exec:
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "run_id": run.id,
                    "manifest_hash": run.manifest_hash,
                    "status": run.status,
                    "message": "Existing run returned for idempotency key.",
                },
            )
        else:
            db.refresh(run)
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "run_id": run.id,
                    "manifest_hash": run.manifest_hash,
                    "status": run.status,
                    "gate_status": run.gate_status,
                    "message": "Existing run returned for idempotency key.",
                },
            )

    if req.async_exec:
        # Asynchronous execution via durable database-backed worker
        from rag_platform.worker import DurableRunWorker, WorkerNotificationBus
        # Wake up any waiting daemon workers immediately via push notification
        WorkerNotificationBus.notify_new_run(run.id, db_session=db)
        background_tasks.add_task(DurableRunWorker.process_next_queued_run)
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "run_id": run.id,
                "manifest_hash": run.manifest_hash,
                "status": "QUEUED",
                "message": "Evaluation run accepted for asynchronous execution.",
            },
        )

    # Synchronous execution
    await _execute_evaluation_run(run.id, req, config, cases, db_session=db)
    db.refresh(run)

    eval_engine = EvaluationEngine()
    traces = [
        (
            RagTrace.model_validate_json(t.raw_trace_json),
            [
                MetricResult(
                    metric_name=m.metric_name,
                    metric_family=MetricFamily(m.metric_family),
                    score=m.score,
                )
                for m in t.metrics
            ],
        )
        for t in run.traces
    ]
    summary = eval_engine.aggregate_run(traces) if traces else None
    return {
        "run_id": run.id,
        "manifest_hash": run.manifest_hash,
        "status": run.status,
        "summary": summary.model_dump() if summary else None,
    }


@app.post("/v1/runs/{run_id}/cancel")
def cancel_run(
    run_id: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Cancel an active or queued evaluation run atomically."""
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    authorize_project(run.project_id, auth, required_role=Role.EDITOR)

    repo = DatabaseRepo(db)
    cancelled, current_run = repo.cancel_run_atomic(run.id, reason="Run cancelled by user request.")
    db.commit()

    if not cancelled:
        current_status = current_run.status if current_run else run.status
        return {
            "run_id": run.id,
            "status": current_status,
            "message": f"Run is already in terminal state '{current_status}'.",
        }

    return {
        "run_id": run.id,
        "status": RunStatus.CANCELLED.value,
        "message": "Run evaluation cancelled successfully.",
    }


@app.get("/v1/runs/{run_id}")
def get_run(
    run_id: str,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    authorize_project(run.project_id, auth, required_role=Role.VIEWER)

    summary_dict = json.loads(run.summary_json) if run.summary_json else None
    gate_dict = json.loads(run.gate_result_json) if run.gate_result_json else None

    # Fallback only for legacy runs
    if summary_dict is None and run.traces:
        eval_engine = EvaluationEngine()
        traces = [
            (
                RagTrace.model_validate_json(t.raw_trace_json),
                [
                    MetricResult(
                        metric_name=m.metric_name,
                        metric_family=MetricFamily(m.metric_family),
                        score=m.score,
                    )
                    for m in t.metrics
                ],
            )
            for t in run.traces
        ]
        summary = eval_engine.aggregate_run(traces) if traces else None
        summary_dict = summary.model_dump() if summary else None
        if summary:
            reg_engine = RegressionEngine()
            gate_dict = reg_engine.evaluate_gate(
                summary,
                ReleasePolicy(policy_id=run.policy_id),
                candidate_run_id=run.id,
            ).model_dump()

    trace_count = (
        summary_dict.get("total_cases")
        if (summary_dict and "total_cases" in summary_dict)
        else len(run.traces)
    )
    return {
        "id": run.id,
        "project_id": run.project_id,
        "dataset_id": run.dataset_id,
        "system_version": run.system_version,
        "manifest_hash": run.manifest_hash,
        "status": run.status,
        "policy_id": run.policy_id,
        "created_at": run.created_at.isoformat(),
        "trace_count": trace_count,
        "summary": summary_dict,
        "gate_result": gate_dict,
    }


@app.get("/v1/runs/{run_id}/traces")
def get_run_traces(
    run_id: str,
    failure_only: bool = False,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    authorize_project(run.project_id, auth, required_role=Role.VIEWER)

    # Database-level filtering, count, offset, and limit
    data_query = select(TraceRow).where(TraceRow.run_id == run_id)
    count_query = select(func.count(TraceRow.id)).where(TraceRow.run_id == run_id)

    if failure_only:
        data_query = data_query.join(FailureRow, TraceRow.id == FailureRow.trace_id)
        count_query = count_query.join(FailureRow, TraceRow.id == FailureRow.trace_id)

    total = db.scalar(count_query) or 0
    traces = db.scalars(data_query.order_by(TraceRow.id.asc()).offset(offset).limit(limit)).all()

    items = []
    for t in traces:
        raw = json.loads(t.raw_trace_json) if t.raw_trace_json else {}
        chunks = raw.get("retrieved_chunks", [])
        citations = raw.get("citations", [])

        failure_obj = None
        if t.failure:
            evidence_data = json.loads(t.failure.evidence_json) if t.failure.evidence_json else {}
            failure_obj = {
                "primary_code": t.failure.failure_type,
                "contributing_codes": evidence_data.get("contributing_codes", []),
                "confidence": t.failure.confidence,
                "severity": t.failure.severity,
                "explanation": t.failure.explanation,
                "evidence": evidence_data,
                "recommended_actions": evidence_data.get("recommended_actions", []),
            }

        items.append({
            "trace_id": t.id,
            "test_case_id": t.test_case_id,
            "question": t.question,
            "answer": t.answer,
            "abstained": t.abstained,
            "abstention_reason": t.abstention_reason,
            "latency_ms": t.latency_ms,
            "chunks": chunks,
            "citations": citations,
            "metrics": [{"name": m.metric_name, "score": m.score, "reason": m.reason} for m in t.metrics],
            "failure": failure_obj,
        })
    return {"run_id": run_id, "traces": items, "limit": limit, "offset": offset, "total": total}


@app.get("/v1/failures")
def list_failures(
    project_id: str | None = None,
    run_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    if project_id:
        authorize_project(project_id, auth, required_role=Role.VIEWER)
    if run_id:
        run = db.get(RunRow, run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        authorize_project(run.project_id, auth, required_role=Role.VIEWER)
        if project_id and run.project_id != project_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Run '{run_id}' does not belong to project '{project_id}'."
            )

    query = select(FailureRow)
    count_query = select(func.count(FailureRow.id))

    if run_id:
        query = query.where(FailureRow.run_id == run_id)
        count_query = count_query.where(FailureRow.run_id == run_id)
    elif project_id:
        query = query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id == project_id)
        count_query = count_query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id == project_id)
    else:
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id.in_(allowed))
            count_query = count_query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id.in_(allowed))

    total = db.scalar(count_query) or 0
    query = query.order_by(FailureRow.id.desc()).offset(offset).limit(limit)
    failures = db.scalars(query).all()
    return {
        "failures": [
            {
                "id": f.id,
                "trace_id": f.trace_id,
                "run_id": f.run_id,
                "failure_type": f.failure_type,
                "severity": f.severity,
                "confidence": f.confidence,
                "explanation": f.explanation,
            }
            for f in failures
        ],
        "limit": limit,
        "offset": offset,
        "total": total,
    }


@app.post("/v1/compare")
def compare_runs(
    req: CompareReq,
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    b_run = db.get(RunRow, req.baseline_run_id)
    c_run = db.get(RunRow, req.candidate_run_id)
    if not b_run or not c_run:
        raise HTTPException(status_code=404, detail="Baseline or candidate run not found")

    authorize_project(b_run.project_id, auth, required_role=Role.VIEWER)
    authorize_project(c_run.project_id, auth, required_role=Role.VIEWER)

    # Validate baseline/candidate dataset compatibility (Point 14)
    if not req.allow_cross_dataset and b_run.dataset_checksum != c_run.dataset_checksum:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Incompatible datasets for comparison: baseline dataset checksum ({b_run.dataset_checksum}) "
                f"!= candidate dataset checksum ({c_run.dataset_checksum}). "
                f"Set allow_cross_dataset=True in request to explicitly permit cross-dataset comparison."
            ),
        )

    eval_engine = EvaluationEngine()
    b_traces = [
        (
            RagTrace.model_validate_json(t.raw_trace_json),
            [
                MetricResult(
                    metric_name=m.metric_name,
                    metric_family=MetricFamily(m.metric_family),
                    score=m.score,
                )
                for m in t.metrics
            ],
        )
        for t in b_run.traces
    ]
    c_traces = [
        (
            RagTrace.model_validate_json(t.raw_trace_json),
            [
                MetricResult(
                    metric_name=m.metric_name,
                    metric_family=MetricFamily(m.metric_family),
                    score=m.score,
                )
                for m in t.metrics
            ],
        )
        for t in c_run.traces
    ]

    b_summary = eval_engine.aggregate_run(b_traces)
    c_summary = eval_engine.aggregate_run(c_traces)

    b_case_scores = {
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness" and m.score is not None), None)
        for t in b_run.traces
    }
    c_case_scores = {
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness" and m.score is not None), None)
        for t in c_run.traces
    }

    reg_engine = RegressionEngine()
    comparison = reg_engine.compare(
        b_summary,
        c_summary,
        req.baseline_run_id,
        req.candidate_run_id,
        b_case_scores,
        c_case_scores,
    )
    gate = reg_engine.evaluate_gate(
        c_summary,
        req.policy,
        req.candidate_run_id,
        b_summary,
        req.baseline_run_id,
        comparison,
    )

    return {
        "comparison": comparison.model_dump(),
        "gate_result": gate.model_dump(),
    }


@app.post("/v1/maintenance/cleanup")
def cleanup_retention(
    trace_retention_days: int = Query(default=90, ge=1, le=3650),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Retention cleanup for evaluation traces (Point 33)."""
    if not auth.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: maintenance operations require ADMIN role.",
        )
    repo = DatabaseRepo(db)
    result = repo.purge_expired_data(trace_retention_days=trace_retention_days)
    db.commit()
    return result


@app.post("/v1/demo-run")
async def seed_demo_run(
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Convenience helper to bootstrap an evaluation run with real data for dashboard demonstration."""
    app_settings = get_settings()
    if not app_settings.dev_mode and not auth.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access forbidden: /v1/demo-run is only available when DEV_MODE is active or with ADMIN role.",
        )

    repo = DatabaseRepo(db)
    proj = db.query(ProjectRow).filter_by(name="Enterprise Knowledge Bot").first()
    if not proj:
        proj = repo.create_project(name="Enterprise Knowledge Bot")
        db.commit()
    ds_ver = f"v{int(datetime.now(timezone.utc).timestamp())}"
    cases = [
        TestCase(
            id=f"{ds_ver}-tc-01",
            question="What was the fiscal year 2024 gross margin?",
            expected_answer="Fiscal year 2024 gross margin expanded to 42.1%.",
            expected_facts=["gross margin was 42.1%", "expanded 120 bps"],
            relevant_documents=[DocumentReference(document_id="annual-2024", chunk_id="c-margin")],
        ),
        TestCase(
            id=f"{ds_ver}-tc-02",
            question="What are the Q4 revenue growth drivers?",
            expected_answer="Cloud subscription revenue grew 24% driven by enterprise AI adoption.",
            expected_facts=["cloud revenue grew 24%", "driven by enterprise AI"],
            relevant_documents=[DocumentReference(document_id="q4-earnings", chunk_id="c-cloud")],
        ),
        TestCase(
            id=f"{ds_ver}-tc-03",
            question="Where is the company's Moon helium processing facility located?",
            answerability=Answerability.UNANSWERABLE,
            expected_answer=None,
        ),
        TestCase(
            id=f"{ds_ver}-tc-04",
            question="What is the refund SLA for enterprise clients?",
            expected_answer="Refunds must be requested within 30 days and are processed in 5 business days.",
            expected_facts=["requested within 30 days", "processed in 5 business days"],
            relevant_documents=[DocumentReference(document_id="sla-policy", chunk_id="c-refund")],
        ),
    ]

    ds_row = repo.create_dataset(proj.id, f"Production Financial Golden Set ({ds_ver})", ds_ver)
    repo.add_test_cases(ds_row.id, cases)
    published_ds = repo.publish_dataset(ds_row.id)
    db.commit()

    # Create baseline run (PERFECT)
    prov_b = RunProvenance(
        dataset_checksum=published_ds.checksum_sha256,
        rag_version="git-v1.0.0",
        dataset_id=published_ds.id,
        dataset_version=published_ds.version,
    )
    cfg_b = RunConfig(
        project_id=proj.id,
        dataset_id=published_ds.id,
        dataset_version=published_ds.version,
        system_version="git-v1.0.0",
    )
    run_b = repo.create_run(cfg_b, prov_b)
    repo.update_run_status(run_b.id, RunStatus.RUNNING)
    db.commit()

    adapter_b = SyntheticRagAdapter(SyntheticRagMode.PERFECT)
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    for c in cases:
        tr = await adapter_b.run(c, cfg_b)
        tr.run_id = run_b.id
        m = await eval_engine.evaluate_trace(tr, c)
        a = attr_engine.diagnose(tr, c, m)
        repo.record_trace(tr, m, a)

    repo.update_run_status(run_b.id, RunStatus.COMPLETED)
    db.commit()

    # Create candidate run with realistic distractor & hallucination
    prov_c = RunProvenance(
        dataset_checksum=published_ds.checksum_sha256,
        rag_version="git-v1.1.0-candidate",
        dataset_id=published_ds.id,
        dataset_version=published_ds.version,
    )
    cfg_c = RunConfig(
        project_id=proj.id,
        dataset_id=published_ds.id,
        dataset_version=published_ds.version,
        system_version="git-v1.1.0-candidate",
    )
    run_c = repo.create_run(cfg_c, prov_c)
    repo.update_run_status(run_c.id, RunStatus.RUNNING)
    db.commit()

    adapter_c = SyntheticRagAdapter(SyntheticRagMode.HALLUCINATING)
    for c in cases:
        tr = await adapter_c.run(c, cfg_c)
        tr.run_id = run_c.id
        m = await eval_engine.evaluate_trace(tr, c)
        a = attr_engine.diagnose(tr, c, m)
        repo.record_trace(tr, m, a)

    repo.update_run_status(run_c.id, RunStatus.COMPLETED)
    db.commit()

    return {"status": "SUCCESS", "baseline_run_id": run_b.id, "candidate_run_id": run_c.id}


# --- Serious Engineering Dashboard UI ---
@app.get("/dashboard")
def get_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file), media_type="text/html")
    return HTMLResponse("<h1>Reliab</h1><p>Dashboard static files missing.</p>")
