"""FastAPI REST API control plane and engineering-grade observability dashboard.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

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
    FailureRow,
    ProjectRow,
    RunRow,
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
    Role,
    SecurityContext,
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
    title="RAG Reliability Platform",
    description="Automated failure attribution, regression testing, and quality release gates for RAG systems.",
    version="2.1.0",
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
) -> SecurityContext:
    """FastAPI dependency for authentication."""
    return authenticate_request(x_api_key=x_api_key, authorization=authorization)


@app.get("/health")
@app.get("/v1/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "2.1.0",
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
    cases: list[TestCase] = Field(default_factory=list)


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
    # RunOptions fields — all honoured in _execute_evaluation_run (FIX #7)
    concurrency: int = 5
    max_cases: int | None = None
    timeout_seconds: float = 60.0
    fail_fast: bool = False
    use_cache: bool = True


class CompareReq(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    policy: ReleasePolicy = Field(default_factory=ReleasePolicy)


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
    allowed = auth.get_authorized_projects()
    if allowed is not None:
        query = query.where(ProjectRow.id.in_(allowed))
    query = query.offset(offset).limit(limit)
    projects = db.scalars(query).all()
    return {"projects": [{"id": p.id, "name": p.name} for p in projects], "limit": limit, "offset": offset}


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


@app.get("/v1/datasets")
def list_datasets(
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    if project_id:
        authorize_project(project_id, auth, required_role=Role.VIEWER)
        query = select(DatasetRow).where(DatasetRow.project_id == project_id)
    else:
        query = select(DatasetRow)
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.where(DatasetRow.project_id.in_(allowed))
    query = query.offset(offset).limit(limit)
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
    return {
        "id": ds.id,
        "project_id": ds.project_id,
        "name": ds.name,
        "version": ds.version,
        "status": ds.status,
        "checksum": ds.checksum_sha256,
        "description": ds.description,
        "case_count": len(ds.cases),
    }


@app.get("/v1/runs")
def list_runs(
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    if project_id:
        authorize_project(project_id, auth, required_role=Role.VIEWER)
        query = select(RunRow).where(RunRow.project_id == project_id).order_by(RunRow.created_at.desc())
    else:
        query = select(RunRow).order_by(RunRow.created_at.desc())
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.where(RunRow.project_id.in_(allowed))
    query = query.offset(offset).limit(limit)
    runs = db.scalars(query).all()
    eval_engine = EvaluationEngine()

    result = []
    for r in runs:
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
        result.append({
            "id": r.id,
            "project_id": r.project_id,
            "dataset_id": r.dataset_id,
            "system_version": r.system_version,
            "manifest_hash": r.manifest_hash,
            "status": r.status,
            "policy_id": r.policy_id,
            "created_at": r.created_at.isoformat(),
            "trace_count": len(r.traces),
            "summary": summary.model_dump() if summary else None,
        })
    return {"runs": result, "limit": limit, "offset": offset}


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
        return HttpRagAdapter(
            endpoint_url=endpoint_url,
            headers=req.adapter_config.get("headers"),
            timeout_seconds=float(req.adapter_config.get("timeout_seconds", 30.0)),
        )

    if adapter_type_key == "python":
        adapter_name = req.adapter_config.get("adapter_name")
        target_fn = req.adapter_config.get("target_fn")
        if target_fn is not None:
            return PythonRagAdapter(target_fn)
        if adapter_name:
            try:
                fn = PythonAdapterRegistry.get(adapter_name)
                return PythonRagAdapter(fn)
            except ValueError as err:
                raise HTTPException(status_code=422, detail=str(err))
        raise HTTPException(
            status_code=422,
            detail=(
                "Python adapter requires 'adapter_name' in adapter_config referencing "
                "a pre-registered callable in PythonAdapterRegistry."
            ),
        )

    # Any other custom registered adapters
    return AdapterRegistry.get(adapter_type_key, **req.adapter_config)


# Global in-memory evaluation cache shared across runs
_eval_cache: dict[str, MetricResult] = {}


async def _execute_evaluation_run(
    run_id: str,
    req: "CreateRunReq",
    config: RunConfig,
    ds_cases: list[TestCase],
    db_session: Session | None = None,
) -> None:
    """Execute evaluation run with bounded concurrency, strict timeout, fail-fast, and resource lifecycle."""
    eval_engine = EvaluationEngine(cache=_eval_cache)
    attr_engine = FailureAttributionEngine()

    adapter = _resolve_adapter(req)

    # Acquire session (respecting dependency overrides if running in test)
    sess = db_session if db_session is not None else create_session()
    repo = DatabaseRepo(sess)
    repo.update_run_status(run_id, RunStatus.RUNNING)
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

        async def worker():
            nonlocal had_fatal_failure
            while not case_queue.empty():
                if had_fatal_failure and fail_fast:
                    break
                try:
                    case = case_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

                if had_fatal_failure and fail_fast:
                    case_queue.task_done()
                    break

                trace, metrics, diag = await _process_case_safe(case)

                async with results_lock:
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

        if had_fatal_failure and fail_fast:
            repo.update_run_status(run_id, RunStatus.FAILED)
            run_row = sess.get(RunRow, run_id)
            if run_row:
                run_row.failure_reason = "Run aborted due to fatal case failure (fail_fast=true)"
                run_row.failure_type = "FAIL_FAST"
        else:
            repo.update_run_status(run_id, RunStatus.COMPLETED)
        sess.commit()

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
            repo.update_run_status(run_id, RunStatus.FAILED)
            run_row = sess.get(RunRow, run_id)
            if run_row:
                from rag_platform.security import SecretRedactor
                safe_reason = SecretRedactor.redact_text(str(exc))[:500]
                run_row.failure_reason = safe_reason
                run_row.failure_type = exc_type[:64]
            sess.commit()
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


@app.post("/v1/runs")
async def create_run(
    req: CreateRunReq,
    background_tasks: BackgroundTasks,
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
        dataset_version=req.dataset_version or ds.version,
        system_version=req.system_version,
        policy_id=req.policy_id,
        options=RunOptions(
            concurrency=req.concurrency,
            max_cases=getattr(req, "max_cases", None),
            timeout_seconds=getattr(req, "timeout_seconds", 60),
            fail_fast=getattr(req, "fail_fast", False),
            use_cache=getattr(req, "use_cache", True),
        ),
    )

    run = repo.create_run(config, provenance)
    db.commit()

    cases = [db_row_to_test_case(r) for r in ds.cases]

    if req.async_exec:
        # Asynchronous execution with 202 Accepted response
        background_tasks.add_task(_execute_evaluation_run, run.id, req, config, cases)
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
        "id": run.id,
        "project_id": run.project_id,
        "dataset_id": run.dataset_id,
        "system_version": run.system_version,
        "manifest_hash": run.manifest_hash,
        "status": run.status,
        "policy_id": run.policy_id,
        "created_at": run.created_at.isoformat(),
        "trace_count": len(run.traces),
        "summary": summary.model_dump() if summary else None,
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

    items = []
    traces_slice = run.traces[offset : offset + limit]
    for t in traces_slice:
        if failure_only and not t.failure:
            continue

        raw = json.loads(t.raw_trace_json)
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
    return {"run_id": run_id, "traces": items, "limit": limit, "offset": offset, "total": len(run.traces)}


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

    query = select(FailureRow)
    if run_id:
        query = query.where(FailureRow.run_id == run_id)
    elif project_id:
        query = query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id == project_id)
    else:
        allowed = auth.get_authorized_projects()
        if allowed is not None:
            query = query.join(RunRow, FailureRow.run_id == RunRow.id).where(RunRow.project_id.in_(allowed))

    query = query.offset(offset).limit(limit)
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
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness" and m.score is not None), 0.5)
        for t in b_run.traces
    }
    c_case_scores = {
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness" and m.score is not None), 0.5)
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


@app.post("/v1/demo-run")
async def seed_demo_run(
    db: Session = Depends(get_db),
    auth: SecurityContext = Depends(get_auth),
):
    """Convenience helper to bootstrap an evaluation run with real data for dashboard demonstration."""
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
    return HTMLResponse("<h1>RAG Reliability Platform</h1><p>Dashboard static files missing.</p>")
