"""CI/CD headless release quality gate CLI and JUnit XML reporter.

# ponytail: single file for argparse CLI, gate evaluator, and JUnit XML generation.
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from rag_platform.adapters import SyntheticRagAdapter, SyntheticRagMode
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.core import generate_id
from rag_platform.db import Base, DatabaseRepo, DatasetRow, ProjectRow, create_db_engine
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    DocumentReference,
    GateResult,
    GateStatus,
    ReleasePolicy,
    RunConfig,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.regression import RegressionEngine


def format_junit_xml(gate_result: GateResult) -> str:
    """Generate standard JUnit XML report for CI/CD test dashboards."""
    suites = ET.Element("testsuites", name="RAG Release Quality Gate")
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=f"Policy_{gate_result.policy_id}",
        tests=str(max(len(gate_result.violations) + 1, 5)),
        failures=str(len(gate_result.violations)),
    )

    if not gate_result.violations:
        tc = ET.SubElement(suite, "testcase", classname="rag.policy", name="ReleaseThresholds", time="0.0")
    else:
        for v in gate_result.violations:
            tc = ET.SubElement(suite, "testcase", classname="rag.policy", name=v.metric_name, time="0.0")
            fail = ET.SubElement(tc, "failure", message=v.message, type=v.violation_type)
            fail.text = f"Candidate Value: {v.candidate_value}, Threshold: {v.threshold}"

    return ET.tostring(suites, encoding="utf-8", xml_declaration=True).decode("utf-8")


def execute_gate_evaluation(
    db_session: Session,
    project_id: str,
    dataset_id: str,
    system_version: str,
    policy: ReleasePolicy | None = None,
    mock_mode: SyntheticRagMode = SyntheticRagMode.PERFECT,
    junit_xml_path: str | None = None,
) -> tuple[int, GateResult]:
    """Execute evaluation run and evaluate against policy gate. Returns (exit_code, GateResult)."""
    repo = DatabaseRepo(db_session)
    active_policy = policy or ReleasePolicy()

    ds_row = db_session.get(DatasetRow, dataset_id)
    if not ds_row:
        raise ValueError(f"Dataset {dataset_id} not found")

    provenance = RunProvenance(
        dataset_checksum=ds_row.checksum_sha256,
        rag_version=system_version,
        model_config_hash="ci_model_default",
        prompt_hash="ci_prompt_default",
        evaluator_version="1.0.0",
        experiment_hash="ci_exp_default",
    )

    config = RunConfig(
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_version=ds_row.version,
        system_version=system_version,
        policy_id=active_policy.policy_id,
    )

    run = repo.create_run(config, provenance)
    repo.update_run_status(run.id, RunStatus.RUNNING)
    db_session.commit()

    adapter = SyntheticRagAdapter(mock_mode)
    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    import json
    cases = [
        TestCase(
            id=r.id,
            question=r.question,
            expected_answer=r.expected_answer,
            relevant_documents=[DocumentReference(**d) for d in json.loads(r.relevant_docs_json)],
            answerability=r.answerability,
        )
        for r in ds_row.cases
    ]

    traces_with_metrics = []
    import asyncio

    async def run_pipeline():
        for case in cases:
            trace = await adapter.run(case, config)
            trace.run_id = run.id
            metrics = await eval_engine.evaluate_trace(trace, case)
            attr = attr_engine.diagnose(trace, case, metrics)
            repo.record_trace(trace, metrics, attr)
            traces_with_metrics.append((trace, metrics))

    asyncio.run(run_pipeline())
    summary = eval_engine.aggregate_run(traces_with_metrics)
    repo.update_run_status(run.id, RunStatus.COMPLETED)
    db_session.commit()

    reg_engine = RegressionEngine()
    gate = reg_engine.evaluate_gate(summary, active_policy, run.id)

    if junit_xml_path:
        xml_content = format_junit_xml(gate)
        p = Path(junit_xml_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(xml_content, encoding="utf-8")

    exit_code = 0 if gate.status == GateStatus.PASS else 1
    return exit_code, gate


def main():
    parser = argparse.ArgumentParser(description="RAG Reliability CI/CD Quality Gate")
    parser.add_argument("--project", required=True, help="Project ID")
    parser.add_argument("--dataset", required=True, help="Published Dataset ID")
    parser.add_argument("--system-version", required=True, help="Candidate RAG Git commit SHA")
    parser.add_argument("--policy", default="prod-default", help="Release policy ID")
    parser.add_argument("--mock-mode", default="PERFECT", choices=[m.value for m in SyntheticRagMode])
    parser.add_argument("--junit-xml", default=None, help="Path to write JUnit XML test results")

    args = parser.parse_args()

    engine = create_db_engine()
    Base.metadata.create_all(bind=engine)

    with Session(engine) as session:
        mode = SyntheticRagMode(args.mock_mode)
        exit_code, gate = execute_gate_evaluation(
            db_session=session,
            project_id=args.project,
            dataset_id=args.dataset,
            system_version=args.system_version,
            mock_mode=mode,
            junit_xml_path=args.junit_xml,
        )

    print("\n" + "=" * 60)
    print(f"RAG CI/CD RELEASE QUALITY GATE: [{gate.status.value}]")
    print(f"Candidate Run ID: {gate.candidate_run_id}")
    print(f"Policy: {gate.policy_id}")
    print("=" * 60)

    if gate.violations:
        print(f"\nVIOLATIONS DETECTED ({len(gate.violations)}):")
        for v in gate.violations:
            print(f"  ❌ [{v.metric_name}]: {v.message}")
    else:
        print("\n✅ All quality thresholds and regression budgets satisfied.")

    print("=" * 60 + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
