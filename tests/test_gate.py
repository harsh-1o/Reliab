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
    ReleasePolicy,
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
