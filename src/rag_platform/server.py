"""FastAPI REST API control plane and embedded interactive dashboard.

# ponytail: single file for REST endpoints and embedded zero-build web dashboard.
"""

from __future__ import annotations

import json
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rag_platform.adapters import SyntheticRagAdapter, SyntheticRagMode
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.core import generate_id
from rag_platform.db import Base, DatabaseRepo, DatasetRow, ProjectRow, RunRow, TraceRow, create_db_engine
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    Answerability,
    BenchmarkDataset,
    DocumentReference,
    GateStatus,
    MetricFamily,
    MetricResult,
    RagTrace,
    ReleasePolicy,
    RunConfig,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.regression import RegressionEngine

# Initialize database
engine = create_db_engine()
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAG Reliability & Hallucination Platform", version="0.1.0")


def get_db():
    with Session(engine) as session:
        yield session


# --- Schemas ---
class CreateProjectReq(BaseModel):
    name: str
    settings: dict[str, Any] = {}


class CreateDatasetReq(BaseModel):
    project_id: str
    name: str
    version: str
    description: str | None = None
    cases: list[TestCase] = []


class CreateRunReq(BaseModel):
    project_id: str
    dataset_id: str
    system_version: str
    suite: str = "full"
    policy_id: str = "prod-default"
    mock_mode: SyntheticRagMode | None = SyntheticRagMode.PERFECT


class CompareReq(BaseModel):
    baseline_run_id: str
    candidate_run_id: str


# --- REST Endpoints ---
@app.post("/v1/projects")
def create_project(req: CreateProjectReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    proj = repo.create_project(req.name, req.settings)
    db.commit()
    return {"id": proj.id, "name": proj.name}


@app.get("/v1/projects/{project_id}")
def get_project(project_id: str, db: Session = Depends(get_db)):
    proj = db.get(ProjectRow, project_id)
    if not proj:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"id": proj.id, "name": proj.name, "created_at": proj.created_at.isoformat()}


@app.post("/v1/datasets")
def create_dataset(req: CreateDatasetReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    ds = repo.create_dataset(req.project_id, req.name, req.version, req.description)
    if req.cases:
        repo.add_test_cases(ds.id, req.cases)
        repo.publish_dataset(ds.id)
    db.commit()
    return {"id": ds.id, "name": ds.name, "version": ds.version, "status": ds.status, "checksum": ds.checksum_sha256}


@app.post("/v1/runs")
async def create_and_execute_run(req: CreateRunReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    ds_row = db.get(DatasetRow, req.dataset_id)
    if not ds_row:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if ds_row.status != "PUBLISHED":
        raise HTTPException(status_code=400, detail="Dataset must be PUBLISHED to evaluate")

    provenance = RunProvenance(
        dataset_checksum=ds_row.checksum_sha256,
        rag_version=req.system_version,
        model_config_hash="model_default",
        prompt_hash="prompt_v1_hash",
        evaluator_version="1.0.0",
        experiment_hash="exp_default",
    )

    config = RunConfig(
        project_id=req.project_id,
        dataset_id=req.dataset_id,
        dataset_version=ds_row.version,
        system_version=req.system_version,
        policy_id=req.policy_id,
        suite=req.suite,
    )

    run = repo.create_run(config, provenance)
    db.commit()

    # Execute evaluation pipeline
    adapter = SyntheticRagAdapter(req.mock_mode or SyntheticRagMode.PERFECT)
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    cases = [
        TestCase(
            id=r.id,
            question=r.question,
            expected_answer=r.expected_answer,
            expected_facts=json.loads(r.expected_facts_json),
            relevant_documents=[DocumentReference(**d) for d in json.loads(r.relevant_docs_json)],
            answerability=r.answerability,
            tags=json.loads(r.tags_json),
        )
        for r in ds_row.cases
    ]

    traces_with_metrics = []
    repo.update_run_status(run.id, RunStatus.RUNNING)
    db.commit()

    for case in cases:
        trace = await adapter.run(case, config)
        trace.run_id = run.id
        metrics = await eval_engine.evaluate_trace(trace, case)
        attribution = attr_engine.diagnose(trace, case, metrics)
        repo.record_trace(trace, metrics, attribution)
        traces_with_metrics.append((trace, metrics))

    summary = eval_engine.aggregate_run(traces_with_metrics)
    repo.update_run_status(run.id, RunStatus.COMPLETED)
    db.commit()

    return {
        "run_id": run.id,
        "status": RunStatus.COMPLETED.value,
        "manifest_hash": run.manifest_hash,
        "summary": summary.model_dump(),
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "id": run.id,
        "project_id": run.project_id,
        "dataset_id": run.dataset_id,
        "system_version": run.system_version,
        "manifest_hash": run.manifest_hash,
        "status": run.status,
        "policy_id": run.policy_id,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "trace_count": len(run.traces),
    }


@app.get("/v1/runs/{run_id}/traces")
def get_run_traces(run_id: str, failure_only: bool = False, db: Session = Depends(get_db)):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    items = []
    for t in run.traces:
        if failure_only and not t.failure:
            continue
        items.append({
            "trace_id": t.id,
            "test_case_id": t.test_case_id,
            "question": t.question,
            "answer": t.answer,
            "latency_ms": t.latency_ms,
            "metrics": [{"name": m.metric_name, "score": m.score} for m in t.metrics],
            "failure": {
                "type": t.failure.failure_type,
                "confidence": t.failure.confidence,
                "explanation": t.failure.explanation,
            } if t.failure else None,
        })
    return {"run_id": run_id, "traces": items}


@app.post("/v1/compare")
def compare_runs(req: CompareReq, db: Session = Depends(get_db)):
    b_run = db.get(RunRow, req.baseline_run_id)
    c_run = db.get(RunRow, req.candidate_run_id)
    if not b_run or not c_run:
        raise HTTPException(status_code=404, detail="Baseline or candidate run not found")

    # Build metric summaries
    eval_engine = EvaluationEngine()
    b_traces = [
        (RagTrace.model_validate_json(t.raw_trace_json), [
            MetricResult(metric_name=m.metric_name, metric_family=MetricFamily(m.metric_family), score=m.score)
            for m in t.metrics
        ])
        for t in b_run.traces
    ]
    c_traces = [
        (RagTrace.model_validate_json(t.raw_trace_json), [
            MetricResult(metric_name=m.metric_name, metric_family=MetricFamily(m.metric_family), score=m.score)
            for m in t.metrics
        ])
        for t in c_run.traces
    ]

    b_summary = eval_engine.aggregate_run(b_traces)
    c_summary = eval_engine.aggregate_run(c_traces)

    reg_engine = RegressionEngine()
    comparison = reg_engine.compare(b_summary, c_summary, req.baseline_run_id, req.candidate_run_id)
    gate = reg_engine.evaluate_gate(c_summary, ReleasePolicy(), req.candidate_run_id, b_summary, req.baseline_run_id, comparison)

    return {
        "comparison": comparison.model_dump(),
        "gate_result": gate.model_dump(),
    }


# --- Embedded Dashboard UI ---
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>RAG Reliability & Regression Dashboard</title>
        <style>
            :root { --bg: #0f172a; --card: #1e293b; --text: #f8fafc; --accent: #38bdf8; --pass: #22c55e; --fail: #ef4444; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 24px; }
            .header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }
            h1 { margin: 0; font-size: 24px; font-weight: 700; color: var(--accent); }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-bottom: 24px; }
            .card { background: var(--card); border-radius: 8px; padding: 20px; border: 1px solid #334155; }
            .card-title { font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; }
            .card-value { font-size: 28px; font-weight: bold; margin-top: 8px; }
            .pass { color: var(--pass); }
            .fail { color: var(--fail); }
            table { width: 100%; border-collapse: collapse; background: var(--card); border-radius: 8px; overflow: hidden; margin-top: 16px; }
            th, td { text-align: left; padding: 12px 16px; border-bottom: 1px solid #334155; font-size: 14px; }
            th { background: #111827; color: #94a3b8; }
            .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; }
            .badge-fail { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
            .badge-pass { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
        </style>
    </head>
    <body>
        <div class="header">
            <div>
                <h1>RAG Reliability Platform</h1>
                <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 14px;">Evaluation, Hallucination Diagnosis & CI/CD Regression Gate</p>
            </div>
            <div>
                <span class="badge badge-pass" style="font-size: 14px; padding: 6px 14px;">GATE: PASS</span>
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <div class="card-title">Faithfulness (Grounding)</div>
                <div class="card-value pass">94.2%</div>
            </div>
            <div class="card">
                <div class="card-title">Recall @ 5</div>
                <div class="card-value pass">95.0%</div>
            </div>
            <div class="card">
                <div class="card-title">Hallucination Rate</div>
                <div class="card-value pass">2.8%</div>
            </div>
            <div class="card">
                <div class="card-title">P95 Latency</div>
                <div class="card-value">185 ms</div>
            </div>
        </div>

        <div class="card">
            <div class="card-title">Evaluation Traces & Diagnostic Attributions</div>
            <table>
                <thead>
                    <tr>
                        <th>Trace ID</th>
                        <th>Question</th>
                        <th>Latency</th>
                        <th>Faithfulness</th>
                        <th>Diagnosis</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><code>tr_8f1a2c</code></td>
                        <td>What was 2024 gross profit?</td>
                        <td>120ms</td>
                        <td>0.98</td>
                        <td><span class="badge badge-pass">PASS</span></td>
                    </tr>
                    <tr>
                        <td><code>tr_4b9e71</code></td>
                        <td>Where is the Moon server farm?</td>
                        <td>85ms</td>
                        <td>1.00</td>
                        <td><span class="badge badge-pass">VALID ABSTENTION</span></td>
                    </tr>
                    <tr>
                        <td><code>tr_19d44a</code></td>
                        <td>What are the 2024 revenue trends?</td>
                        <td>210ms</td>
                        <td>0.32</td>
                        <td><span class="badge badge-fail">GEN-01: Unsupported Claim</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """
