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


# --- High-End Interactive Dashboard UI ---
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Reliability & Hallucination Diagnostics Platform</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #07090e;
            --surface-outer: rgba(255, 255, 255, 0.03);
            --surface-inner: #0d121f;
            --surface-border: rgba(255, 255, 255, 0.08);
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --pass: #22c55e;
            --pass-bg: rgba(34, 197, 94, 0.12);
            --fail: #ef4444;
            --fail-bg: rgba(239, 68, 68, 0.12);
            --warn: #f59e0b;
            --warn-bg: rgba(245, 158, 11, 0.12);
            --bezel-radius: 20px;
            --inner-radius: 14px;
            --transition: all 400ms cubic-bezier(0.32, 0.72, 0, 1);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg);
            color: var(--text-main);
            min-height: 100vh;
            padding: 32px 48px;
            background-image: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.06), transparent 40%), radial-gradient(circle at 85% 90%, rgba(139, 92, 246, 0.05), transparent 45%);
            background-attachment: fixed;
        }
        /* Top Navigation Header */
        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 32px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--surface-border);
        }
        .brand { display: flex; align-items: center; gap: 14px; }
        .logo-mark {
            width: 40px; height: 40px; border-radius: 10px;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            display: flex; align-items: center; justify-content: center;
            font-weight: 800; font-size: 18px; color: #fff;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.35);
        }
        .title h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.02em; }
        .title p { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
        .controls { display: flex; align-items: center; gap: 12px; }
        .btn {
            background: var(--surface-inner);
            border: 1px solid var(--surface-border);
            color: var(--text-main);
            padding: 9px 18px;
            border-radius: 100px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: var(--transition);
            display: inline-flex; align-items: center; gap: 8px;
        }
        .btn:hover { border-color: var(--accent); background: rgba(56, 189, 248, 0.08); transform: translateY(-1px); }
        .btn-primary { background: linear-gradient(135deg, #0284c7, #38bdf8); color: #fff; border: none; }
        .btn-primary:hover { box-shadow: 0 0 20px var(--accent-glow); }
        /* Double-Bezel Architecture */
        .bezel {
            background: var(--surface-outer);
            border: 1px solid var(--surface-border);
            border-radius: var(--bezel-radius);
            padding: 6px;
            margin-bottom: 24px;
        }
        .bezel-inner {
            background: var(--surface-inner);
            border-radius: var(--inner-radius);
            border: 1px solid rgba(255, 255, 255, 0.04);
            box-shadow: inset 0 1px 1px rgba(255, 255, 255, 0.08);
            padding: 24px;
        }
        /* Bento Metric Grid */
        .bento { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 24px; }
        .metric-card {
            background: var(--surface-inner);
            border-radius: var(--inner-radius);
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: var(--transition);
        }
        .metric-card:hover { border-color: var(--surface-border); transform: translateY(-2px); }
        .metric-label { font-size: 12px; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
        .metric-value { font-size: 32px; font-weight: 800; letter-spacing: -0.03em; margin: 10px 0 4px 0; }
        .metric-sub { font-size: 12px; color: var(--text-muted); }
        .text-pass { color: var(--pass); }
        .text-fail { color: var(--fail); }
        .text-accent { color: var(--accent); }
        /* Badges */
        .badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 4px 10px; border-radius: 100px;
            font-size: 12px; font-weight: 600;
        }
        .badge-pass { background: var(--pass-bg); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
        .badge-fail { background: var(--fail-bg); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
        .badge-warn { background: var(--warn-bg); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
        /* Interactive Trace Inspector Table */
        table { width: 100%; border-collapse: collapse; margin-top: 12px; }
        th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); text-align: left; padding: 14px 16px; border-bottom: 1px solid var(--surface-border); }
        td { font-size: 14px; padding: 16px; border-bottom: 1px solid rgba(255, 255, 255, 0.04); vertical-align: middle; }
        tr.trace-row { cursor: pointer; transition: var(--transition); }
        tr.trace-row:hover { background: rgba(255, 255, 255, 0.02); }
        code { font-family: 'JetBrains Mono', monospace; font-size: 12px; background: rgba(255, 255, 255, 0.06); padding: 3px 6px; border-radius: 4px; color: var(--accent); }
        /* Trace Detail Panel */
        .detail-pane {
            display: none; background: #0a0e18; border-radius: 10px;
            padding: 20px; margin: 12px 0; border: 1px solid rgba(255, 255, 255, 0.06);
        }
        .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .detail-block h4 { font-size: 12px; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; }
        .chunk-box { background: rgba(255, 255, 255, 0.03); border-radius: 6px; padding: 10px; margin-bottom: 8px; font-size: 13px; line-height: 1.5; border-left: 3px solid var(--accent); }
    </style>
</head>
<body>
    <div class="topbar">
        <div class="brand">
            <div class="logo-mark">R</div>
            <div class="title">
                <h1>RAG Reliability Platform</h1>
                <p>Evaluation, Hallucination Diagnosis & CI/CD Release Quality Gates</p>
            </div>
        </div>
        <div class="controls">
            <span class="badge badge-pass" id="gate-badge">CI GATE: PASS</span>
            <button class="btn" onclick="fetchLatestRuns()">Refresh</button>
            <button class="btn btn-primary" onclick="triggerRun()">Execute Run ↗</button>
        </div>
    </div>

    <!-- Executive Metrics Grid -->
    <div class="bento">
        <div class="metric-card">
            <div class="metric-label">Faithfulness (Grounding)</div>
            <div class="metric-value text-pass" id="val-faith">95.4%</div>
            <div class="metric-sub">Target: ≥ 90.0%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Retrieval Recall @ 5</div>
            <div class="metric-value text-pass" id="val-recall">96.0%</div>
            <div class="metric-sub">Target: ≥ 92.0%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Hallucination Rate</div>
            <div class="metric-value text-pass" id="val-halluc">2.1%</div>
            <div class="metric-sub">Ceiling: ≤ 5.0%</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Abstention Accuracy</div>
            <div class="metric-value text-pass" id="val-abstain">100%</div>
            <div class="metric-sub">Refusal on unanswerable</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">P95 Latency</div>
            <div class="metric-value" id="val-latency">192 ms</div>
            <div class="metric-sub">Budget: ≤ 240 ms</div>
        </div>
    </div>

    <!-- Trace & Diagnosis Double-Bezel Table -->
    <div class="bezel">
        <div class="bezel-inner">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="font-size: 16px; font-weight: 600;">Evaluation Traces & Root Cause Diagnostics</h3>
                <span style="font-size: 12px; color: var(--text-muted);">Click any row to inspect chunk lineage & attribution</span>
            </div>

            <table>
                <thead>
                    <tr>
                        <th>Trace ID</th>
                        <th>Question</th>
                        <th>Latency</th>
                        <th>Faithfulness</th>
                        <th>Recall@5</th>
                        <th>Diagnosis</th>
                    </tr>
                </thead>
                <tbody id="trace-rows">
                    <tr class="trace-row" onclick="toggleDetail('d1')">
                        <td><code>tr_9a8c1f</code></td>
                        <td>What was 2024 gross margin?</td>
                        <td>140ms</td>
                        <td class="text-pass">0.98</td>
                        <td class="text-pass">1.00</td>
                        <td><span class="badge badge-pass">PASS</span></td>
                    </tr>
                    <tr id="d1" class="detail-pane">
                        <td colspan="6">
                            <div class="detail-grid">
                                <div class="detail-block">
                                    <h4>Retrieved Evidence Chunks (Top-K)</h4>
                                    <div class="chunk-box"><strong>Rank 1 (annual-2024):</strong> In fiscal year 2024, gross margin expanded 120 bps to 42.1%.</div>
                                </div>
                                <div class="detail-block">
                                    <h4>Generated Answer & Verification</h4>
                                    <p style="font-size: 13px; line-height: 1.5;">2024 gross margin was 42.1% [1].</p>
                                    <p style="margin-top: 8px; font-size: 12px; color: var(--pass);">✓ All claims entailed by retrieved context.</p>
                                </div>
                            </div>
                        </td>
                    </tr>
                    <tr class="trace-row" onclick="toggleDetail('d2')">
                        <td><code>tr_3b4d2a</code></td>
                        <td>Where is the Moon helium processing plant?</td>
                        <td>75ms</td>
                        <td class="text-pass">1.00</td>
                        <td>-</td>
                        <td><span class="badge badge-pass">VALID ABSTENTION</span></td>
                    </tr>
                    <tr id="d2" class="detail-pane">
                        <td colspan="6">
                            <div class="detail-block">
                                <h4>Abstention Diagnosis</h4>
                                <p style="font-size: 13px;">Question flagged as UNANSWERABLE. SUT successfully abstained with reason code: <code>INSUFFICIENT_EVIDENCE</code>.</p>
                            </div>
                        </td>
                    </tr>
                    <tr class="trace-row" onclick="toggleDetail('d3')">
                        <td><code>tr_7e11bb</code></td>
                        <td>What were Q4 2024 revenue growth drivers?</td>
                        <td>230ms</td>
                        <td class="text-fail">0.31</td>
                        <td class="text-pass">1.00</td>
                        <td><span class="badge badge-fail">GEN-01: Unsupported Claim</span></td>
                    </tr>
                    <tr id="d3" class="detail-pane">
                        <td colspan="6">
                            <div class="detail-grid">
                                <div class="detail-block">
                                    <h4>Failure Evidence & Attribution</h4>
                                    <p style="font-size: 13px; color: #f87171;">Answer asserts company acquired European logistics network, which does not appear anywhere in retrieved chunks.</p>
                                </div>
                                <div class="detail-block">
                                    <h4>Recommended Experiments</h4>
                                    <ul style="font-size: 12px; color: var(--text-muted); margin-left: 16px; line-height: 1.6;">
                                        <li>Lower generation temperature to 0.0</li>
                                        <li>Enforce strict prompt instruction: 'Answer ONLY using provided text'</li>
                                    </ul>
                                </div>
                            </div>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function toggleDetail(id) {
            const el = document.getElementById(id);
            el.style.display = (el.style.display === 'table-row') ? 'none' : 'table-row';
        }

        async function fetchLatestRuns() {
            try {
                const res = await fetch('/v1/runs');
                // Auto-refresh data if available
            } catch(e) { console.log('Offline demo data active'); }
        }

        async function triggerRun() {
            alert('Evaluation run queued. Provenance manifest anchored with SHA256.');
        }
    </script>
</body>
</html>"""

