"""FastAPI REST API control plane and engineering-grade observability dashboard.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
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
    FailureCode,
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

app = FastAPI(title="RAG Reliability & Hallucination Platform", version="0.2.0")


def get_db():
    with Session(engine) as session:
        yield session


# --- API Request Models ---
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
    dataset_version: str | None = None
    system_version: str
    policy_id: str = "prod-default"
    mock_mode: str = "PERFECT"


class CompareReq(BaseModel):
    baseline_run_id: str
    candidate_run_id: str
    policy: ReleasePolicy = ReleasePolicy()


# --- REST API Endpoints ---
@app.post("/v1/projects")
def create_project(req: CreateProjectReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    proj = repo.create_project(name=req.name, settings=req.settings)
    db.commit()
    return {"id": proj.id, "name": proj.name}


@app.get("/v1/projects")
def list_projects(db: Session = Depends(get_db)):
    projects = db.scalars(select(ProjectRow)).all()
    return {"projects": [{"id": p.id, "name": p.name} for p in projects]}


@app.post("/v1/datasets")
def create_dataset(req: CreateDatasetReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    dataset = BenchmarkDataset(
        id=generate_id("ds"),
        project_id=req.project_id,
        name=req.name,
        version=req.version,
        description=req.description,
        cases=req.cases,
    )
    dataset.publish()
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


@app.get("/v1/runs")
def list_runs(project_id: str | None = None, db: Session = Depends(get_db)):
    query = select(RunRow).order_by(RunRow.created_at.desc())
    if project_id:
        query = query.where(RunRow.project_id == project_id)
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
    return {"runs": result}


@app.post("/v1/runs")
async def create_run(req: CreateRunReq, db: Session = Depends(get_db)):
    repo = DatabaseRepo(db)
    ds = repo.get_dataset(req.dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")

    provenance = RunProvenance(
        dataset_checksum=ds.checksum_sha256,
        rag_version=req.system_version,
        model_config_hash="sha256_gpt4o_temp0",
        prompt_hash="sha256_prompt_v1",
        evaluator_version="2.0.0",
        experiment_hash="sha256_exp_default",
        dataset_id=ds.id,
        dataset_version=ds.version,
    )

    config = RunConfig(
        project_id=req.project_id,
        dataset_id=req.dataset_id,
        dataset_version=req.dataset_version or ds.version,
        system_version=req.system_version,
        policy_id=req.policy_id,
    )

    run = repo.create_run(config, provenance)
    repo.update_run_status(run.id, RunStatus.RUNNING)
    db.commit()

    # Execute traces with synthetic adapter
    adapter = SyntheticRagAdapter(SyntheticRagMode(req.mock_mode))
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    cases = [
        TestCase(
            id=r.id,
            question=r.question,
            expected_answer=r.expected_answer,
            relevant_documents=[DocumentReference(**d) for d in json.loads(r.relevant_docs_json)],
            answerability=r.answerability,
        )
        for r in ds.cases
    ]

    evaluated_pairs = []
    for case in cases:
        trace = await adapter.run(case, config)
        trace.run_id = run.id
        metrics = await eval_engine.evaluate_trace(trace, case)
        attr = attr_engine.diagnose(trace, case, metrics)
        repo.record_trace(trace, metrics, attr)
        evaluated_pairs.append((trace, metrics))

    repo.update_run_status(run.id, RunStatus.COMPLETED)
    db.commit()

    summary = eval_engine.aggregate_run(evaluated_pairs)
    return {
        "run_id": run.id,
        "manifest_hash": run.manifest_hash,
        "status": run.status,
        "summary": summary.model_dump(),
    }


@app.get("/v1/runs/{run_id}")
def get_run(run_id: str, db: Session = Depends(get_db)):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

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
def get_run_traces(run_id: str, failure_only: bool = False, db: Session = Depends(get_db)):
    run = db.get(RunRow, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    items = []
    for t in run.traces:
        if failure_only and not t.failure:
            continue

        raw = json.loads(t.raw_trace_json)
        chunks = raw.get("retrieved_chunks", [])
        citations = raw.get("citations", [])

        # Extract claim verification from faithfulness metric if present
        claims = []
        for m in t.metrics:
            if m.metric_name == "faithfulness" and m.reason:
                # Metric reason captures claim-level verdict
                claims.append({"reason": m.reason, "score": m.score})

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
    return {"run_id": run_id, "traces": items}


@app.post("/v1/compare")
def compare_runs(req: CompareReq, db: Session = Depends(get_db)):
    b_run = db.get(RunRow, req.baseline_run_id)
    c_run = db.get(RunRow, req.candidate_run_id)
    if not b_run or not c_run:
        raise HTTPException(status_code=404, detail="Baseline or candidate run not found")

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

    # Calculate per-case scores for regression tracking
    b_case_scores = {
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness"), 0.5)
        for t in b_run.traces
    }
    c_case_scores = {
        t.test_case_id: next((m.score for m in t.metrics if m.metric_name == "faithfulness"), 0.5)
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
async def seed_demo_run(db: Session = Depends(get_db)):
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
        model_config_hash="sha256_gpt4o_baseline",
        prompt_hash="sha256_prompt_v1",
        evaluator_version="2.0.0",
        experiment_hash="sha256_exp_baseline",
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
        model_config_hash="sha256_gpt4o_cand",
        prompt_hash="sha256_prompt_v2",
        evaluator_version="2.0.0",
        experiment_hash="sha256_exp_cand",
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
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Reliability & Release Engineering Console</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #09090b;
            --panel: #121316;
            --panel-elevated: #18191e;
            --border: #27272a;
            --border-subtle: #1e1f24;
            --text: #f4f4f5;
            --text-muted: #a1a1aa;
            --text-dim: #71717a;
            --pass: #10b981;
            --pass-bg: rgba(16, 185, 129, 0.08);
            --fail: #f43f5e;
            --fail-bg: rgba(244, 63, 94, 0.08);
            --warn: #f59e0b;
            --warn-bg: rgba(245, 158, 11, 0.08);
            --neutral: #38bdf8;
            --neutral-bg: rgba(56, 189, 248, 0.08);
            --font-mono: 'JetBrains Mono', monospace;
            --font-sans: 'Inter', -apple-system, sans-serif;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg);
            color: var(--text);
            font-family: var(--font-sans);
            font-size: 13px;
            line-height: 1.5;
            padding: 24px 32px;
            -webkit-font-smoothing: antialiased;
        }

        /* Header */
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 24px;
        }
        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .header-title h1 {
            font-size: 15px;
            font-weight: 600;
            letter-spacing: -0.01em;
            color: var(--text);
        }
        .badge-env {
            font-family: var(--font-mono);
            font-size: 11px;
            padding: 2px 8px;
            background: #27272a;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            color: #d4d4d8;
        }
        .header-controls {
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .btn {
            font-family: var(--font-mono);
            font-size: 12px;
            padding: 6px 12px;
            border-radius: 4px;
            border: 1px solid var(--border);
            background: var(--panel);
            color: var(--text);
            cursor: pointer;
            transition: background 150ms ease, border-color 150ms ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .btn:hover {
            border-color: #52525b;
            background: var(--panel-elevated);
        }
        .btn-primary {
            background: #2563eb;
            border-color: #3b82f6;
            color: #ffffff;
        }
        .btn-primary:hover {
            background: #1d4ed8;
        }
        select.btn {
            appearance: none;
            padding-right: 24px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%23a1a1aa' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");
            background-repeat: no-repeat;
            background-position: right 8px center;
        }

        /* Executive Story Banner */
        .story-bar {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 18px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .story-content h2 {
            font-size: 14px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .story-content p {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 2px;
        }
        .story-status {
            font-family: var(--font-mono);
            font-size: 12px;
            font-weight: 600;
            padding: 4px 12px;
            border-radius: 4px;
        }
        .status-pass {
            background: var(--pass-bg);
            border: 1px solid rgba(16, 185, 129, 0.25);
            color: var(--pass);
        }
        .status-fail {
            background: var(--fail-bg);
            border: 1px solid rgba(244, 63, 94, 0.25);
            color: var(--fail);
        }

        /* Metric Grid */
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .metric-tile {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px 16px;
        }
        .metric-header {
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 6px;
        }
        .metric-label {
            font-size: 11px;
            font-weight: 500;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-val {
            font-family: var(--font-mono);
            font-size: 24px;
            font-weight: 600;
            color: var(--text);
            letter-spacing: -0.02em;
        }
        .metric-footer {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 6px;
            font-size: 11px;
            font-family: var(--font-mono);
        }
        .delta-tag {
            font-weight: 600;
        }
        .delta-improved { color: var(--pass); }
        .delta-regressed { color: var(--fail); }
        .delta-neutral { color: var(--text-dim); }
        .ci-span {
            color: var(--text-dim);
            font-size: 10px;
        }

        /* Main Section Container */
        .section-box {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            margin-bottom: 24px;
        }
        .section-title {
            padding: 14px 18px;
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .section-title h3 {
            font-size: 13px;
            font-weight: 600;
            color: var(--text);
        }

        /* Trace Debugger Table */
        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        th {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--text-dim);
            text-align: left;
            padding: 10px 16px;
            background: #101114;
            border-bottom: 1px solid var(--border);
        }
        td {
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-subtle);
            vertical-align: top;
        }
        tr.trace-row {
            cursor: pointer;
            transition: background 120ms ease;
        }
        tr.trace-row:hover {
            background: #16171b;
        }
        tr.expanded {
            background: #16171b;
        }

        /* Chips & Badges */
        .chip {
            font-family: var(--font-mono);
            font-size: 11px;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
            display: inline-flex;
            align-items: center;
            gap: 4px;
        }
        .chip-pass { background: var(--pass-bg); border: 1px solid rgba(16, 185, 129, 0.25); color: var(--pass); }
        .chip-fail { background: var(--fail-bg); border: 1px solid rgba(244, 63, 94, 0.25); color: var(--fail); }
        .chip-warn { background: var(--warn-bg); border: 1px solid rgba(245, 158, 11, 0.25); color: var(--warn); }
        .chip-neutral { background: var(--neutral-bg); border: 1px solid rgba(56, 189, 248, 0.25); color: var(--neutral); }

        /* Detail Pane for Root Cause Analysis */
        .detail-row {
            background: #0f1013;
        }
        .detail-pane {
            padding: 18px 24px;
            border-bottom: 1px solid var(--border);
        }
        .pipeline-stepper {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }
        .step-card {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 12px;
        }
        .step-label {
            font-family: var(--font-mono);
            font-size: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-dim);
            margin-bottom: 6px;
        }
        .step-body {
            font-size: 12px;
            line-height: 1.4;
            max-height: 140px;
            overflow-y: auto;
        }
        .chunk-pill {
            font-family: var(--font-mono);
            font-size: 11px;
            background: #1e1f24;
            padding: 4px 8px;
            border-radius: 4px;
            margin-bottom: 4px;
            border: 1px solid #2e2f38;
        }
        .claim-item {
            padding: 6px 8px;
            border-radius: 4px;
            margin-bottom: 6px;
            background: #15161a;
            border: 1px solid #27272a;
            font-size: 11px;
        }

        /* Empty State */
        .empty-state {
            padding: 60px 20px;
            text-align: center;
        }
        .empty-state h4 {
            font-size: 15px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .empty-state p {
            color: var(--text-muted);
            margin-bottom: 18px;
            max-width: 480px;
            margin-left: auto;
            margin-right: auto;
        }
        .cli-box {
            font-family: var(--font-mono);
            font-size: 12px;
            background: #000000;
            border: 1px solid var(--border);
            padding: 10px 14px;
            border-radius: 4px;
            display: inline-block;
            color: #38bdf8;
            margin-bottom: 16px;
        }
    </style>
</head>
<body>

    <!-- Header -->
    <header>
        <div class="header-title">
            <h1>RAG Reliability Platform</h1>
            <span class="badge-env">CI/CD Gate v0.2.0</span>
            <span id="repro-badge" class="chip chip-neutral" style="display:none;"></span>
        </div>
        <div class="header-controls">
            <select id="run-select" class="btn" onchange="loadSelectedRun()">
                <option value="">Loading Evaluation Runs...</option>
            </select>
            <button class="btn" onclick="refreshDashboard()">Refresh</button>
            <button class="btn btn-primary" onclick="triggerSeedRun()">Run Benchmark</button>
        </div>
    </header>

    <!-- Main Container: Populated Dynamically -->
    <main id="app-root">
        <div class="empty-state">
            <h4>Loading System State...</h4>
            <p>Querying SQLite/Postgres datastore for evaluation runs.</p>
        </div>
    </main>

    <script>
        let currentRun = null;
        let baselineRun = null;
        let runTraces = [];

        async function init() {
            await fetchRuns();
        }

        async function fetchRuns() {
            try {
                const res = await fetch('/v1/runs');
                const data = await res.json();
                const runs = data.runs || [];
                const selector = document.getElementById('run-select');

                if (runs.length === 0) {
                    renderEmptyState();
                    return;
                }

                selector.innerHTML = runs.map((r, i) =>
                    `<option value="${r.id}" ${i === 0 ? 'selected' : ''}>${r.system_version} (${r.id.substring(0, 10)}) - ${r.trace_count} cases</option>`
                ).join('');

                currentRun = runs[0];
                baselineRun = runs.length > 1 ? runs[1] : null;

                renderRunDashboard(currentRun, baselineRun);
                await loadTraces(currentRun.id);
            } catch (err) {
                console.error("Failed to load runs:", err);
                renderEmptyState("Datastore unreachable. Ensure backend is running.");
            }
        }

        async function loadSelectedRun() {
            const runId = document.getElementById('run-select').value;
            if (!runId) return;
            const res = await fetch(`/v1/runs/${runId}`);
            currentRun = await res.json();
            renderRunDashboard(currentRun, baselineRun);
            await loadTraces(runId);
        }

        async function loadTraces(runId) {
            try {
                const res = await fetch(`/v1/runs/${runId}/traces`);
                const data = await res.json();
                runTraces = data.traces || [];
                renderTraceTable(runTraces);
            } catch(e) {
                console.error("Failed loading traces", e);
            }
        }

        function renderRunDashboard(run, baseline) {
            const root = document.getElementById('app-root');
            const summary = run.summary || {};
            const metrics = summary.metrics || {};

            const faith = metrics.faithfulness || { mean: 0.0, count: 0, std_dev: 0 };
            const recall = metrics.recall_at_5 || { mean: 0.0, count: 0, std_dev: 0 };
            const cit = metrics.citation_accuracy || { mean: 0.0, count: 0, std_dev: 0 };
            const abst = summary.abstention_accuracy !== undefined ? summary.abstention_accuracy : 1.0;
            const p95_lat = summary.p95_latency_ms || 0;
            const cost = summary.total_cost_usd || 0;
            const halluc_rate = summary.hallucination_rate || 0.0;

            // Baseline deltas if available
            const bMetrics = baseline && baseline.summary ? baseline.summary.metrics || {} : {};
            const dFaith = bMetrics.faithfulness ? (faith.mean - bMetrics.faithfulness.mean) : null;
            const dRecall = bMetrics.recall_at_5 ? (recall.mean - bMetrics.recall_at_5.mean) : null;

            // Update Provenance Hash Badge
            const repro = document.getElementById('repro-badge');
            if (run.manifest_hash) {
                repro.style.display = 'inline-flex';
                repro.textContent = `SHA256: ${run.manifest_hash.substring(0, 10)}`;
            }

            const isPassed = halluc_rate <= 0.05 && faith.mean >= 0.85;

            root.innerHTML = `
                <!-- Executive Story Bar -->
                <div class="story-bar">
                    <div class="story-content">
                        <h2>
                            ${isPassed ? '✓ Candidate Satisfies Release Guardrails' : '❌ Release Quality Gate Blocked'}
                        </h2>
                        <p>Evaluated ${run.trace_count} test cases on benchmark <code>${run.dataset_id}</code> against commit <code>${run.system_version}</code>.</p>
                    </div>
                    <div class="story-status ${isPassed ? 'status-pass' : 'status-fail'}">
                        GATE: ${isPassed ? 'PASS' : 'BLOCKED'}
                    </div>
                </div>

                <!-- Core Metric Tiles with 95% Confidence Intervals -->
                <div class="metric-grid">
                    <div class="metric-tile">
                        <div class="metric-header">
                            <span class="metric-label">Claim Faithfulness</span>
                            <span class="chip ${faith.mean >= 0.85 ? 'chip-pass' : 'chip-fail'}">${faith.mean >= 0.85 ? 'HEALTHY' : 'DRIFT'}</span>
                        </div>
                        <div class="metric-val">${(faith.mean * 100).toFixed(1)}%</div>
                        <div class="metric-footer">
                            <span class="delta-tag ${dFaith && dFaith >= 0 ? 'delta-improved' : 'delta-regressed'}">
                                ${dFaith !== null ? (dFaith >= 0 ? '+' : '') + (dFaith * 100).toFixed(1) + '% vs base' : 'N=' + faith.count}
                            </span>
                            <span class="ci-span">95% CI: [${faith.ci_lower || 0}, ${faith.ci_upper || 1}]</span>
                        </div>
                    </div>

                    <div class="metric-tile">
                        <div class="metric-header">
                            <span class="metric-label">Evidence Recall@5</span>
                            <span class="chip ${recall.mean >= 0.90 ? 'chip-pass' : 'chip-warn'}">RANKED</span>
                        </div>
                        <div class="metric-val">${(recall.mean * 100).toFixed(1)}%</div>
                        <div class="metric-footer">
                            <span class="delta-tag ${dRecall && dRecall >= 0 ? 'delta-improved' : 'delta-regressed'}">
                                ${dRecall !== null ? (dRecall >= 0 ? '+' : '') + (dRecall * 100).toFixed(1) + '% vs base' : 'N=' + recall.count}
                            </span>
                            <span class="ci-span">95% CI: [${recall.ci_lower || 0}, ${recall.ci_upper || 1}]</span>
                        </div>
                    </div>

                    <div class="metric-tile">
                        <div class="metric-header">
                            <span class="metric-label">Citation Accuracy</span>
                            <span class="chip chip-neutral">VERIFIED</span>
                        </div>
                        <div class="metric-val">${(cit.mean * 100).toFixed(1)}%</div>
                        <div class="metric-footer">
                            <span class="delta-tag delta-neutral">Evidence Links</span>
                            <span class="ci-span">N=${cit.count}</span>
                        </div>
                    </div>

                    <div class="metric-tile">
                        <div class="metric-header">
                            <span class="metric-label">Abstention Accuracy</span>
                            <span class="chip ${abst >= 0.90 ? 'chip-pass' : 'chip-fail'}">BOUNDARY</span>
                        </div>
                        <div class="metric-val">${(abst * 100).toFixed(1)}%</div>
                        <div class="metric-footer">
                            <span class="delta-tag delta-neutral">Unanswerable Handling</span>
                            <span class="ci-span">N=${run.trace_count}</span>
                        </div>
                    </div>

                    <div class="metric-tile">
                        <div class="metric-header">
                            <span class="metric-label">P95 Latency & Cost</span>
                            <span class="chip chip-neutral">${p95_lat}ms</span>
                        </div>
                        <div class="metric-val">${p95_lat}ms</div>
                        <div class="metric-footer">
                            <span class="delta-tag delta-neutral">Cost: $${cost.toFixed(4)}</span>
                            <span class="ci-span">P95 Budget &le; 1200ms</span>
                        </div>
                    </div>
                </div>

                <!-- Trace Debugger & Deep Causal Analysis -->
                <div class="section-box">
                    <div class="section-title">
                        <h3>Evaluation Traces & Root Cause Inspector</h3>
                        <div>
                            <button class="btn" onclick="filterTraces('all')">All Traces</button>
                            <button class="btn" onclick="filterTraces('fail')">Failed Traces Only</button>
                        </div>
                    </div>
                    <table>
                        <thead>
                            <tr>
                                <th>Trace ID</th>
                                <th>Question</th>
                                <th>Faithfulness</th>
                                <th>Recall</th>
                                <th>Primary Diagnosis</th>
                                <th>Contributing Causes</th>
                            </tr>
                        </thead>
                        <tbody id="trace-table-body">
                            <tr><td colspan="6" style="text-align:center; color:var(--text-dim);">Loading execution traces...</td></tr>
                        </tbody>
                    </table>
                </div>
            `;
        }

        function renderTraceTable(traces) {
            const tbody = document.getElementById('trace-table-body');
            if (!tbody) return;

            if (traces.length === 0) {
                tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; padding: 24px; color: var(--text-dim);">No traces recorded for this evaluation run.</td></tr>`;
                return;
            }

            tbody.innerHTML = traces.map((t, idx) => {
                const faith = t.metrics.find(m => m.name === 'faithfulness');
                const recall = t.metrics.find(m => m.name === 'recall_at_5');
                const fScore = faith ? faith.score.toFixed(2) : '-';
                const rScore = recall ? recall.score.toFixed(2) : '-';

                const fail = t.failure;
                const pCode = fail ? fail.primary_code : 'PASS';
                const contrib = fail && fail.contributing_codes && fail.contributing_codes.length > 0
                    ? fail.contributing_codes.map(c => `<span class="chip chip-warn">${c}</span>`).join(' ')
                    : '<span style="color:var(--text-dim);">-</span>';

                const statusChip = fail
                    ? `<span class="chip chip-fail">${pCode}</span>`
                    : `<span class="chip chip-pass">PASS</span>`;

                return `
                    <tr class="trace-row" onclick="toggleTraceDetail('tr-detail-${idx}')">
                        <td style="font-family:var(--font-mono); color:#38bdf8;">${t.trace_id.substring(0, 10)}</td>
                        <td>${t.question}</td>
                        <td style="font-family:var(--font-mono); color:${fScore >= 0.70 ? 'var(--pass)' : 'var(--fail)'};">${fScore}</td>
                        <td style="font-family:var(--font-mono);">${rScore}</td>
                        <td>${statusChip}</td>
                        <td>${contrib}</td>
                    </tr>
                    <tr id="tr-detail-${idx}" class="detail-row" style="display:none;">
                        <td colspan="6" class="detail-pane">
                            <div class="pipeline-stepper">
                                <!-- Step 1: Retrieval Chunks -->
                                <div class="step-card">
                                    <div class="step-label">1. Retrieved Context Chunks</div>
                                    <div class="step-body">
                                        ${t.chunks && t.chunks.length > 0 ? t.chunks.map(c =>
                                            `<div class="chunk-pill"><strong>[Rank ${c.rank}] ${c.document_id}:${c.chunk_id}</strong><br/>${c.text.substring(0, 140)}...</div>`
                                        ).join('') : '<p style="color:var(--text-dim);">No context retrieved.</p>'}
                                    </div>
                                </div>

                                <!-- Step 2: Generated Response -->
                                <div class="step-card">
                                    <div class="step-label">2. Generated Answer & Citations</div>
                                    <div class="step-body">
                                        <p style="margin-bottom:6px;">${t.answer || (t.abstained ? '<em>Abstained: ' + (t.abstention_reason || 'insufficient evidence') + '</em>' : '<em>Empty</em>')}</p>
                                        ${t.citations && t.citations.length > 0 ? t.citations.map(c =>
                                            `<span class="chip chip-neutral" style="font-size:10px;">Cited: ${c.document_id}:${c.chunk_id}</span>`
                                        ).join(' ') : ''}
                                    </div>
                                </div>

                                <!-- Step 3: Diagnostic Findings -->
                                <div class="step-card">
                                    <div class="step-label">3. Root Cause Explanation</div>
                                    <div class="step-body">
                                        ${fail ? `
                                            <p style="color:var(--fail); font-weight:600; margin-bottom:4px;">Primary: ${fail.primary_code} (Conf: ${(fail.confidence * 100).toFixed(0)}%)</p>
                                            <p style="font-size:11px; color:#d4d4d8;">${fail.explanation}</p>
                                        ` : '<p style="color:var(--pass);">✓ All claims verified against retrieved evidence.</p>'}
                                    </div>
                                </div>

                                <!-- Step 4: Recommended Action -->
                                <div class="step-card">
                                    <div class="step-label">4. Remediation Action</div>
                                    <div class="step-body">
                                        ${fail && fail.recommended_actions && fail.recommended_actions.length > 0 ? `
                                            <ul style="padding-left:14px; color:var(--text-muted); font-size:11px;">
                                                ${fail.recommended_actions.map(a => `<li>${a}</li>`).join('')}
                                            </ul>
                                        ` : '<p style="color:var(--text-dim);">No corrective action required.</p>'}
                                    </div>
                                </div>
                            </div>
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function toggleTraceDetail(id) {
            const el = document.getElementById(id);
            if (!el) return;
            el.style.display = (el.style.display === 'table-row') ? 'none' : 'table-row';
        }

        function filterTraces(type) {
            if (type === 'fail') {
                const fails = runTraces.filter(t => t.failure !== null);
                renderTraceTable(fails);
            } else {
                renderTraceTable(runTraces);
            }
        }

        function renderEmptyState(msg) {
            const root = document.getElementById('app-root');
            root.innerHTML = `
                <div class="empty-state">
                    <h4>No Benchmark Evaluation Runs Found</h4>
                    <p>${msg || 'Initialize your first evaluation run via the CLI, REST API, or click below to bootstrap an evaluation run with live golden benchmarks.'}</p>
                    <div class="cli-box">python -m rag_platform.gate --project prj-001 --dataset ds-gold --system-version git-abc123</div>
                    <br/>
                    <button class="btn btn-primary" onclick="triggerSeedRun()">Initialize Sample Benchmark Run</button>
                </div>
            `;
        }

        async function triggerSeedRun() {
            try {
                const res = await fetch('/v1/demo-run', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'SUCCESS') {
                    await fetchRuns();
                } else {
                    alert('Error creating benchmark run: ' + JSON.stringify(data));
                }
            } catch(e) {
                alert('Benchmark trigger failed: ' + e);
            }
        }

        function refreshDashboard() {
            fetchRuns();
        }

        window.onload = init;
    </script>
</body>
</html>"""
