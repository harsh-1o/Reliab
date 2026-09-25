"""Unit tests for Phase 8: FastAPI server endpoints, execution pipeline, and dashboard UI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.adapters import SyntheticRagMode
from rag_platform.db import Base
from rag_platform.server import app, get_db

# Isolated in-memory DB shared across test requests via StaticPool
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


def test_server_project_and_dataset_endpoints():
    from rag_platform.core import generate_id
    proj_name = f"API Eval Test Proj {generate_id()}"
    # 1. Create project
    proj_resp = client.post("/v1/projects", json={"name": proj_name})
    assert proj_resp.status_code == 200
    proj_id = proj_resp.json()["id"]

    # 2. Create and publish dataset with 2 cases
    ds_payload = {
        "project_id": proj_id,
        "name": "api_test_dataset",
        "version": "v1.0",
        "cases": [
            {
                "id": "c1",
                "question": "What is Q1 revenue?",
                "expected_answer": "$10M",
                "expected_facts": ["Q1 was $10M"],
                "relevant_documents": [{"document_id": "doc1"}],
                "answerability": "ANSWERABLE",
            },
            {
                "id": "c2",
                "question": "Where is the CEO's rocket?",
                "expected_answer": None,
                "answerability": "UNANSWERABLE",
            },
        ],
    }
    ds_resp = client.post("/v1/datasets", json=ds_payload)
    assert ds_resp.status_code == 200
    ds_data = ds_resp.json()
    assert ds_data["status"] == "PUBLISHED"
    assert len(ds_data["checksum"]) == 64
    ds_id = ds_data["id"]

    # 3. Create and execute run (PERFECT mock)
    run1_resp = client.post(
        "/v1/runs",
        json={
            "project_id": proj_id,
            "dataset_id": ds_id,
            "system_version": "rag_v1",
            "mock_mode": SyntheticRagMode.PERFECT.value,
        },
    )
    assert run1_resp.status_code == 200
    run1_data = run1_resp.json()
    run1_id = run1_data["run_id"]
    assert run1_data["status"] == "COMPLETED"
    assert len(run1_data["manifest_hash"]) == 64
    assert run1_data["summary"]["total_cases"] == 2

    # 4. Get run by ID
    get_run_resp = client.get(f"/v1/runs/{run1_id}")
    assert get_run_resp.status_code == 200
    assert get_run_resp.json()["id"] == run1_id

    # 5. Get traces
    traces_resp = client.get(f"/v1/runs/{run1_id}/traces")
    assert traces_resp.status_code == 200
    assert len(traces_resp.json()["traces"]) == 2

    # 6. Execute second run (DISTRACTOR mock)
    run2_resp = client.post(
        "/v1/runs",
        json={
            "project_id": proj_id,
            "dataset_id": ds_id,
            "system_version": "rag_v2_regressed",
            "mock_mode": SyntheticRagMode.DISTRACTOR.value,
        },
    )
    assert run2_resp.status_code == 200
    run2_id = run2_resp.json()["run_id"]

    # 7. Compare runs
    compare_resp = client.post(
        "/v1/compare",
        json={"baseline_run_id": run1_id, "candidate_run_id": run2_id},
    )
    assert compare_resp.status_code == 200
    cmp_data = compare_resp.json()
    assert cmp_data["gate_result"]["status"] in ("PASS", "FAIL")

    # 8. Dashboard HTML route
    dash_resp = client.get("/dashboard")
    assert dash_resp.status_code == 200
    assert "RAG Reliability Platform" in dash_resp.text
