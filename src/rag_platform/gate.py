"""CI/CD headless release quality gate CLI and JUnit XML reporter."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from rag_platform.adapters import SyntheticRagAdapter, SyntheticRagMode
from rag_platform.attribution import FailureAttributionEngine
from rag_platform.db import DatabaseRepo, DatasetRow, DatasetStatus, ProjectRow, TestCaseRow, create_db_engine
from rag_platform.evaluators import EvaluationEngine
from rag_platform.models import (
    Answerability,
    DocumentReference,
    GateResult,
    GateStatus,
    GateViolation,
    ReleasePolicy,
    RunConfig,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.regression import RegressionEngine


# ---------------------------------------------------------------------------
# Canonical TestCase reconstruction (mirrors server.db_row_to_test_case)
# Used by the CLI gate runner to preserve all test case fields.
# ---------------------------------------------------------------------------
def _gate_db_row_to_test_case(row: Any) -> TestCase:
    """Convert a TestCaseRow to a TestCase, preserving all fields including expected_facts."""
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


def format_junit_xml(gate_result: GateResult) -> str:
    """Generate standard JUnit XML report for CI/CD test dashboards with accurate 1:1 test accounting."""
    suites = ET.Element("testsuites", name="Reliab Release Quality Gate")
    suite = ET.SubElement(
        suites,
        "testsuite",
        name=f"Policy_{gate_result.policy_id}",
    )

    # Core release gate policy checks evaluated by Reliab
    standard_checks = [
        ("faithfulness", "Claim Faithfulness Threshold"),
        ("recall_at_5", "Evidence Retrieval Recall@5"),
        ("citation_accuracy", "Citation Grounding Accuracy"),
        ("hallucination_rate", "Hallucination Rate Cap"),
        ("abstention_accuracy", "Abstention & Refusal Quality"),
    ]

    violations_by_metric: dict[str, list[GateViolation]] = {}
    for v in gate_result.violations:
        violations_by_metric.setdefault(v.metric_name, []).append(v)

    test_count = 0
    failure_count = len(gate_result.violations)

    for metric_key, check_label in standard_checks:
        test_count += 1
        tc = ET.SubElement(
            suite,
            "testcase",
            classname=f"reliab.policy.{gate_result.policy_id}",
            name=f"{metric_key} ({check_label})",
            time="0.0",
        )
        if metric_key in violations_by_metric:
            for v in violations_by_metric[metric_key]:
                fail = ET.SubElement(tc, "failure", message=v.message, type=v.violation_type)
                fail.text = f"Candidate Value: {v.candidate_value}, Threshold: {v.threshold}, Operator: {v.operator}"

    # Handle any additional custom metric policies or regression budget violations
    for metric_name, v_list in violations_by_metric.items():
        if metric_name not in [k for k, _ in standard_checks]:
            test_count += 1
            tc = ET.SubElement(
                suite,
                "testcase",
                classname=f"reliab.policy.{gate_result.policy_id}",
                name=metric_name,
                time="0.0",
            )
            for v in v_list:
                fail = ET.SubElement(tc, "failure", message=v.message, type=v.violation_type)
                fail.text = f"Candidate Value: {v.candidate_value}, Threshold: {v.threshold}, Operator: {v.operator}"

    suite.set("tests", str(test_count))
    suite.set("failures", str(failure_count))

    return ET.tostring(suites, encoding="utf-8", xml_declaration=True).decode("utf-8")


def bootstrap_ci_database(db_session: Session, project_id: str, dataset_id: str) -> None:
    """Ensure project and benchmark dataset exist in database before evaluation."""
    from rag_platform.models import compute_dataset_checksum

    project = db_session.get(ProjectRow, project_id)
    if not project:
        project = ProjectRow(
            id=project_id,
            name=f"CI Project {project_id}",
            settings_json=json.dumps({"ci_bootstrap": True}),
        )
        db_session.add(project)
        db_session.commit()

    ds_row = db_session.get(DatasetRow, dataset_id)
    if not ds_row or not ds_row.cases:
        cases = [
            TestCase(
                id="case_001_retrieval_facts",
                question="What was Acme Corp's total Q3 revenue and year-over-year growth?",
                expected_answer="Acme Corp's Q3 revenue was $142.5 million, representing a 14% year-over-year growth.",
                expected_facts=["Acme Corp Q3 revenue was $142.5 million", "Year-over-year growth was 14%"],
                relevant_documents=[
                    DocumentReference(
                        document_id="doc_q3_report",
                        chunk_id="chunk_01",
                    )
                ],
                answerability=Answerability.ANSWERABLE,
                tags=["financial", "factual"],
            ),
            TestCase(
                id="case_002_multi_hop",
                question="What is the CEO of Acme Corp's background prior to taking the role?",
                expected_answer="Before becoming CEO of Acme Corp in 2021, Jane Doe was VP of Operations at Globex.",
                expected_facts=["Jane Doe became CEO in 2021", "Prior role was VP of Operations at Globex"],
                relevant_documents=[
                    DocumentReference(
                        document_id="doc_exec_bio",
                        chunk_id="chunk_02",
                    )
                ],
                answerability=Answerability.ANSWERABLE,
                tags=["multi-hop", "executive"],
            ),
            TestCase(
                id="case_003_unanswerable_security",
                question="What is the database root password for the internal production cluster?",
                expected_answer="",
                expected_facts=[],
                relevant_documents=[],
                answerability=Answerability.UNANSWERABLE,
                tags=["unanswerable", "security"],
            ),
            TestCase(
                id="case_004_refund_policy",
                question="What is the standard refund window for software license renewals?",
                expected_answer="Customers can receive refunds within 30 days of the renewal date.",
                expected_facts=["Refunds are permitted within 30 days of renewal"],
                relevant_documents=[
                    DocumentReference(
                        document_id="doc_tos",
                        chunk_id="chunk_03",
                    )
                ],
                answerability=Answerability.ANSWERABLE,
                tags=["policy", "semantic-equivalence"],
            ),
            TestCase(
                id="case_005_numerical_conflict",
                question="What was the total capital expenditure in 2023?",
                expected_answer="Capital expenditure for 2023 was $80M.",
                expected_facts=["Capital expenditure was $80M in 2023"],
                relevant_documents=[
                    DocumentReference(
                        document_id="doc_capex",
                        chunk_id="chunk_04",
                    )
                ],
                answerability=Answerability.ANSWERABLE,
                tags=["numerical", "finance"],
            ),
            TestCase(
                id="case_006_unsupported_claim_detection",
                question="What were the headcount additions in 2023?",
                expected_answer="Acme Corp added 450 engineers in 2023.",
                expected_facts=["Added 450 engineers in 2023"],
                relevant_documents=[
                    DocumentReference(
                        document_id="doc_hr",
                        chunk_id="chunk_05",
                    )
                ],
                answerability=Answerability.ANSWERABLE,
                tags=["claims", "hr"],
            ),
        ]
        checksum = compute_dataset_checksum(cases)
        if not ds_row:
            ds_row = DatasetRow(
                id=dataset_id,
                project_id=project_id,
                name="Locked Gold CI Benchmark",
                version="1.0.0",
                description="Golden locked benchmark dataset for regression and gate evaluation.",
                status=DatasetStatus.PUBLISHED.value,
                checksum_sha256=checksum,
            )
            db_session.add(ds_row)
        else:
            ds_row.checksum_sha256 = checksum
            ds_row.status = DatasetStatus.PUBLISHED.value
        for c in cases:
            tc_row = db_session.get(TestCaseRow, (c.id, dataset_id))
            if not tc_row:
                tc_row = TestCaseRow(
                    id=c.id,
                    dataset_id=dataset_id,
                    question=c.question,
                    expected_answer=c.expected_answer,
                    expected_facts_json=json.dumps(c.expected_facts),
                    relevant_docs_json=json.dumps([d.model_dump() for d in c.relevant_documents]),
                    answerability=c.answerability.value,
                    tags_json=json.dumps(c.tags),
                    metadata_json=json.dumps(c.metadata),
                )
                db_session.add(tc_row)
            else:
                tc_row.dataset_id = dataset_id
                tc_row.question = c.question
                tc_row.expected_answer = c.expected_answer
                tc_row.expected_facts_json = json.dumps(c.expected_facts)
                tc_row.relevant_docs_json = json.dumps([d.model_dump() for d in c.relevant_documents])
                tc_row.answerability = c.answerability.value
                tc_row.tags_json = json.dumps(c.tags)
                tc_row.metadata_json = json.dumps(c.metadata)
        db_session.commit()


def load_dataset_from_file(
    file_path: str | Path,
    project_id: str,
    dataset_id: str | None = None,
) -> tuple[str, list[TestCase]]:
    """Load benchmark test cases from a local .json, .jsonl, or .csv file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark dataset file not found: {file_path}")

    cases: list[TestCase] = []
    ext = path.suffix.lower()

    if ext == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    cases.append(TestCase.model_validate_json(line))
    elif ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            cases = [TestCase.model_validate(c) for c in data]
        elif isinstance(data, dict):
            if "cases" in data and isinstance(data["cases"], list):
                cases = [TestCase.model_validate(c) for c in data["cases"]]
            else:
                cases = [TestCase.model_validate(data)]
        else:
            raise ValueError(f"Unrecognized JSON structure in dataset file: {file_path}")
    elif ext == ".csv":
        from rag_platform.datasets import import_dataset_csv
        imported_ds = import_dataset_csv(path, project_id=project_id, name=path.stem, version="1.0.0")
        cases = imported_ds.cases
    else:
        raise ValueError(f"Unsupported benchmark dataset format '{ext}'. Expected .json, .jsonl, or .csv")

    if not cases:
        raise ValueError(f"No test cases loaded from benchmark dataset file: {file_path}")

    actual_dataset_id = dataset_id or f"ds_{path.stem}"
    return actual_dataset_id, cases


def execute_gate_evaluation(
    db_session: Session,
    project_id: str,
    dataset_id: str | None = None,
    system_version: str = "HEAD",
    policy: ReleasePolicy | None = None,
    mock_mode: SyntheticRagMode = SyntheticRagMode.PERFECT,
    junit_xml_path: str | None = None,
    bootstrap: bool = False,
    adapter_type: str = "synthetic",
    endpoint_url: str | None = None,
    dataset_path: str | None = None,
    environment: str = "production",
    gate_mode: str = "hard",
) -> tuple[int, GateResult]:
    """Execute evaluation run and evaluate against policy gate. Returns (exit_code, GateResult)."""
    from rag_platform.models import compute_dataset_checksum

    # Ensure project exists
    project = db_session.get(ProjectRow, project_id)
    if not project:
        project = ProjectRow(
            id=project_id,
            name=f"CI Project {project_id}",
            settings_json=json.dumps({"ci_bootstrap": True}),
        )
        db_session.add(project)
        db_session.commit()

    if dataset_path:
        actual_ds_id, loaded_cases = load_dataset_from_file(dataset_path, project_id, dataset_id)
        dataset_id = actual_ds_id
        checksum = compute_dataset_checksum(loaded_cases)
        ds_row = db_session.get(DatasetRow, dataset_id)
        if not ds_row:
            ds_row = DatasetRow(
                id=dataset_id,
                project_id=project_id,
                name=f"File Imported: {Path(dataset_path).stem}",
                version="1.0.0",
                description=f"Loaded from {dataset_path}",
                status=DatasetStatus.PUBLISHED.value,
                checksum_sha256=checksum,
            )
            db_session.add(ds_row)
        else:
            ds_row.checksum_sha256 = checksum
            ds_row.status = DatasetStatus.PUBLISHED.value
        for c in loaded_cases:
            tc_row = db_session.get(TestCaseRow, (c.id, dataset_id))
            if not tc_row:
                tc_row = TestCaseRow(
                    id=c.id,
                    dataset_id=dataset_id,
                    question=c.question,
                    expected_answer=c.expected_answer,
                    expected_facts_json=json.dumps(c.expected_facts),
                    relevant_docs_json=json.dumps([d.model_dump() for d in c.relevant_documents]),
                    answerability=c.answerability.value,
                    tags_json=json.dumps(c.tags),
                    metadata_json=json.dumps(c.metadata),
                )
                db_session.add(tc_row)
            else:
                tc_row.question = c.question
                tc_row.expected_answer = c.expected_answer
                tc_row.expected_facts_json = json.dumps(c.expected_facts)
                tc_row.relevant_docs_json = json.dumps([d.model_dump() for d in c.relevant_documents])
                tc_row.answerability = c.answerability.value
                tc_row.tags_json = json.dumps(c.tags)
                tc_row.metadata_json = json.dumps(c.metadata)
        db_session.commit()
    elif bootstrap:
        if not dataset_id:
            dataset_id = "ci_bench"
        bootstrap_ci_database(db_session, project_id, dataset_id)

    if not dataset_id:
        raise ValueError("Either dataset_id or dataset_path must be specified.")

    repo = DatabaseRepo(db_session)
    active_policy = policy or ReleasePolicy()

    ds_row = db_session.get(DatasetRow, dataset_id)
    if not ds_row:
        raise ValueError(f"Dataset {dataset_id} not found. Use --bootstrap or --dataset-path to provide benchmark cases.")

    actual_adapter_type = "http" if (adapter_type == "http" and endpoint_url) else f"synthetic:{mock_mode.value}"
    provenance = RunProvenance(
        dataset_checksum=ds_row.checksum_sha256,
        dataset_id=dataset_id,
        dataset_version=ds_row.version,
        rag_version=system_version,
        model_name="rag-benchmark-engine",
        model_version="1.0.0",
        evaluator_version="2.0.0",
        adapter_type=actual_adapter_type,
        environment_info={
            "ci": "true",
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "environment": environment,
            "gate_mode": gate_mode,
        },
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

    adapter: Any
    if adapter_type == "http" and endpoint_url:
        from rag_platform.adapters import HttpRagAdapter
        adapter = HttpRagAdapter(endpoint_url=endpoint_url)
    else:
        adapter = SyntheticRagAdapter(mock_mode)

    eval_engine = EvaluationEngine()
    attr_engine = FailureAttributionEngine()

    cases = [_gate_db_row_to_test_case(r) for r in ds_row.cases]

    traces_with_metrics = []

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

    if gate.status == GateStatus.FAIL and gate_mode == "soft":
        import logging
        logging.getLogger("rag_platform.gate").warning(
            "Gate violations detected, but gate-mode is 'soft' (non-blocking). Exiting with code 0."
        )
        exit_code = 0
    elif gate.status == GateStatus.FAIL:
        exit_code = 1
    else:
        exit_code = 0

    return exit_code, gate


def main():
    parser = argparse.ArgumentParser(description="Reliab CI/CD Quality Gate & Platform Self-Test")
    parser.add_argument("--project", "--project-id", dest="project", default="reliab-ci-project", help="Project ID")
    parser.add_argument("--dataset", "--dataset-id", dest="dataset", default=None, help="Published Dataset ID in database")
    parser.add_argument("--dataset-path", default=None, help="Filesystem path to load benchmark dataset from (.json, .jsonl, .csv)")
    parser.add_argument("--system-version", default="HEAD", help="Candidate RAG Git commit SHA")
    parser.add_argument("--policy", default="prod-default", help="Release policy ID")
    parser.add_argument("--adapter-type", default="synthetic", choices=["synthetic", "http"], help="Adapter type (Points 15)")
    parser.add_argument("--endpoint-url", default=None, help="HTTP SUT endpoint URL when evaluating real RAG system")
    parser.add_argument("--mock-mode", default="PERFECT", choices=[m.value for m in SyntheticRagMode])
    parser.add_argument("--junit-xml", default=None, help="Path to write JUnit XML test results")
    parser.add_argument("--environment", default="production", help="Deployment environment (e.g. production, staging, development)")
    parser.add_argument("--gate-mode", default="hard", choices=["hard", "soft"], help="Gate enforcement mode: 'hard' exits with code 1 on violations; 'soft' logs violations as warnings and exits 0")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        default=False,
        help="Initialize database and seed baseline golden benchmark dataset if not already present",
    )

    args = parser.parse_args()

    if not args.dataset and not args.dataset_path:
        if args.bootstrap:
            args.dataset = "ci_bench"
        else:
            parser.error("Either --dataset or --dataset-path must be specified.")

    engine = create_db_engine()
    if args.bootstrap:
        from rag_platform.db import init_db
        init_db(engine, allow_non_memory=True)

    with Session(engine) as session:
        mode = SyntheticRagMode(args.mock_mode)
        exit_code, gate = execute_gate_evaluation(
            db_session=session,
            project_id=args.project,
            dataset_id=args.dataset,
            system_version=args.system_version,
            mock_mode=mode,
            junit_xml_path=args.junit_xml,
            bootstrap=args.bootstrap,
            adapter_type=args.adapter_type,
            endpoint_url=args.endpoint_url,
            dataset_path=args.dataset_path,
            environment=args.environment,
            gate_mode=args.gate_mode,
        )

    mode_label = "Platform Self-Test (Synthetic)" if args.adapter_type == "synthetic" else "Production SUT Release Gate"
    print("\n" + "=" * 60)
    print(f"RELIAB CI/CD RELEASE QUALITY GATE: [{gate.status.value}] ({mode_label})")
    print(f"Candidate Run ID: {gate.candidate_run_id}")
    print(f"Policy: {gate.policy_id}")
    print(f"Environment: {args.environment}")
    print(f"Gate Mode: {args.gate_mode.upper()}")
    print("=" * 60)

    if gate.violations:
        print(f"\nVIOLATIONS DETECTED ({len(gate.violations)}):")
        for v in gate.violations:
            print(f"  [FAIL] [{v.metric_name}]: {v.message}")
        if args.gate_mode == "soft":
            print("\n[WARN] Gate mode is 'soft'; violations are non-blocking. Exiting 0.")
    else:
        print("\n[PASS] All quality thresholds and regression budgets satisfied.")

    print("=" * 60 + "\n")
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
