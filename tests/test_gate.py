"""Unit tests for Phase 9: CI/CD gate evaluation, exit codes, and JUnit XML reports."""

from __future__ import annotations

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.adapters import SyntheticRagMode
from rag_platform.db import Base, DatabaseRepo
from rag_platform.gate import execute_gate_evaluation, format_junit_xml
from rag_platform.models import (
    DocumentReference,
    GateResult,
    GateStatus,
    GateViolation,
    TestCase,
)


@pytest.fixture
def memory_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def setup_test_benchmark(memory_db: Session):
    repo = DatabaseRepo(memory_db)
    proj = repo.create_project("CI Test Project")
    ds = repo.create_dataset(proj.id, "ci_bench", "v1.0")
    cases = [
        TestCase(
            id="case_1",
            question="What is Q1 revenue?",
            expected_answer="$10.5B",
            relevant_documents=[DocumentReference(document_id="doc_annual")],
        )
    ]
    repo.add_test_cases(ds.id, cases)
    repo.publish_dataset(ds.id)
    memory_db.commit()
    return proj.id, ds.id


def test_format_junit_xml():
    # 1. Passing gate
    pass_gate = GateResult(status=GateStatus.PASS, candidate_run_id="run_1", policy_id="prod")
    xml_pass = format_junit_xml(pass_gate)
    root = ET.fromstring(xml_pass)
    assert root.tag == "testsuites"
    assert root.find(".//testcase") is not None
    assert root.find(".//failure") is None

    # 2. Failing gate
    fail_gate = GateResult(
        status=GateStatus.FAIL,
        candidate_run_id="run_2",
        policy_id="prod",
        violations=[
            GateViolation(
                metric_name="faithfulness",
                candidate_value=0.45,
                threshold=0.90,
                violation_type="THRESHOLD_BREACH",
                message="Faithfulness score 0.45 fell below 0.90 threshold.",
            )
        ],
    )
    xml_fail = format_junit_xml(fail_gate)
    root_fail = ET.fromstring(xml_fail)
    assert root_fail.find(".//failure") is not None
    assert "0.45" in root_fail.find(".//failure").attrib["message"]


def test_execute_gate_pass_and_fail(memory_db: Session, setup_test_benchmark):
    proj_id, ds_id = setup_test_benchmark

    with tempfile.TemporaryDirectory() as tmpdir:
        junit_path = Path(tmpdir) / "junit.xml"

        # Compliant SUT -> returns exit code 0
        code_pass, gate_pass = execute_gate_evaluation(
            db_session=memory_db,
            project_id=proj_id,
            dataset_id=ds_id,
            system_version="git_sha_pass",
            mock_mode=SyntheticRagMode.PERFECT,
            junit_xml_path=str(junit_path),
        )
        assert code_pass == 0
        assert gate_pass.status == GateStatus.PASS
        assert junit_path.exists()

        # Hallucinating SUT -> returns exit code 1 (fails gate)
        code_fail, gate_fail = execute_gate_evaluation(
            db_session=memory_db,
            project_id=proj_id,
            dataset_id=ds_id,
            system_version="git_sha_hallucinating",
            mock_mode=SyntheticRagMode.HALLUCINATING,
            junit_xml_path=str(junit_path),
        )
        assert code_fail == 1
        assert gate_fail.status == GateStatus.FAIL
        assert len(gate_fail.violations) > 0


def test_gate_mode_soft_returns_zero_on_violations(memory_db: Session, setup_test_benchmark):
    """When gate_mode='soft', violations are recorded but exit code is 0 (non-blocking)."""
    proj_id, ds_id = setup_test_benchmark
    code, gate = execute_gate_evaluation(
        db_session=memory_db,
        project_id=proj_id,
        dataset_id=ds_id,
        system_version="git_sha_soft_test",
        mock_mode=SyntheticRagMode.HALLUCINATING,
        gate_mode="soft",
    )
    assert code == 0
    assert gate.status == GateStatus.FAIL
    assert len(gate.violations) > 0


def test_gate_dataset_path_loads_and_evaluates(memory_db: Session):
    """Verify --dataset-path loads test cases from a local JSON file and evaluates them."""
    import json

    from rag_platform.db import RunRow

    with tempfile.TemporaryDirectory() as tmpdir:
        json_file = Path(tmpdir) / "custom_cases.json"
        case_data = [
            {
                "id": "file_case_01",
                "question": "What is the product release date?",
                "expected_answer": "October 15, 2026",
                "expected_facts": ["Release date is October 15, 2026"],
                "relevant_documents": [{"document_id": "doc_roadmap", "chunk_id": "c1"}],
                "answerability": "ANSWERABLE",
                "tags": ["roadmap"],
            }
        ]
        json_file.write_text(json.dumps(case_data), encoding="utf-8")

        code, gate = execute_gate_evaluation(
            db_session=memory_db,
            project_id="proj_file_test",
            dataset_path=str(json_file),
            system_version="git_sha_file_test",
            mock_mode=SyntheticRagMode.PERFECT,
            environment="staging",
            gate_mode="hard",
        )

        assert code == 0
        assert gate.status == GateStatus.PASS

        # Verify candidate run was persisted with expected properties
        run_row = memory_db.get(RunRow, gate.candidate_run_id)
        assert run_row is not None
        assert run_row.status == "COMPLETED"
        assert run_row.system_version == "git_sha_file_test"
        assert run_row.dataset_id.startswith("ds_custom_cases")


def test_gate_dataset_path_nonexistent_raises(memory_db: Session):
    """Passing a non-existent dataset-path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Benchmark dataset file not found"):
        execute_gate_evaluation(
            db_session=memory_db,
            project_id="proj_err",
            dataset_path="/nonexistent/path/to/dataset.json",
        )


def test_gate_cli_main_soft_mode_and_dataset_path(monkeypatch, tmp_path):
    """Test full main() CLI execution with --dataset-path, --environment, and --gate-mode soft."""
    import json
    import sys

    from rag_platform.gate import main

    cases_file = tmp_path / "cli_cases.json"
    cases_file.write_text(
        json.dumps(
            [
                {
                    "id": "cli_c1",
                    "question": "What is status?",
                    "expected_answer": "All systems nominal.",
                    "expected_facts": ["All systems nominal"],
                    "relevant_documents": [{"document_id": "doc1", "chunk_id": "c1"}],
                    "answerability": "ANSWERABLE",
                }
            ]
        ),
        encoding="utf-8",
    )

    test_args = [
        "gate.py",
        "--project", "proj_cli_test",
        "--dataset-path", str(cases_file),
        "--environment", "staging",
        "--gate-mode", "soft",
        "--mock-mode", "HALLUCINATING",
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as exc_info:
        main()

    # In soft gate mode, even with HALLUCINATING mock mode, exit code must be 0
    assert exc_info.value.code == 0


def test_gate_published_dataset_checksum_mismatch_raises_error(memory_db, setup_test_benchmark, tmp_path):
    """When a dataset is already published, supplying a different dataset file with the same
    dataset ID must raise an error instead of silently proceeding with the old dataset.
    """
    import json
    proj_id, ds_id = setup_test_benchmark

    different_file = tmp_path / "different_cases.json"
    different_file.write_text(
        json.dumps([
            {
                "id": "case_diff_1",
                "question": "Different question?",
                "expected_answer": "Different answer.",
                "answerability": "ANSWERABLE",
            }
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already PUBLISHED"):
        execute_gate_evaluation(
            db_session=memory_db,
            project_id=proj_id,
            dataset_id=ds_id,
            dataset_path=str(different_file),
            system_version="v1.0.0",
        )
