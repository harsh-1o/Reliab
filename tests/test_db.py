"""Unit tests for Phase 2: database models, repositories, and immutability constraints."""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_platform.core import ImmutabilityError
from rag_platform.db import (
    Base,
    DatabaseRepo,
)
from rag_platform.models import (
    Answerability,
    Citation,
    DatasetStatus,
    DocumentReference,
    FailureAttribution,
    FailureCode,
    MetricFamily,
    MetricResult,
    RagTrace,
    RetrievedChunk,
    RunConfig,
    RunProvenance,
    RunStatus,
    Severity,
    TestCase,
)


@pytest.fixture
def session():
    """In-memory SQLite database session fixture."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    with Session(engine) as sess:
        yield sess


def test_project_and_dataset_lifecycle(session: Session):
    repo = DatabaseRepo(session)
    proj = repo.create_project("Customer Support SUT")
    assert proj.id.startswith("proj_")

    ds = repo.create_dataset(proj.id, "q3_support_eval", "v1.0")
    assert ds.status == DatasetStatus.DRAFT.value

    cases = [
        TestCase(
            id="c1",
            question="What is the refund policy?",
            expected_answer="30 days full refund.",
            expected_facts=["30-day refund window"],
            relevant_documents=[DocumentReference(document_id="doc_refund")],
        ),
        TestCase(
            id="c2",
            question="Where is the Mars office?",
            expected_answer=None,
            answerability=Answerability.UNANSWERABLE,
        ),
    ]

    repo.add_test_cases(ds.id, cases)
    published = repo.publish_dataset(ds.id)

    assert published.status == DatasetStatus.PUBLISHED.value
    assert len(published.checksum_sha256) == 64

    # Immutability test: cannot add cases to published dataset
    with pytest.raises(ImmutabilityError):
        repo.add_test_cases(ds.id, [TestCase(id="c3", question="Another question?")])


def test_run_creation_and_provenance_immutability(session: Session):
    repo = DatabaseRepo(session)
    proj = repo.create_project("Finance RAG")
    ds = repo.create_dataset(proj.id, "annual_reports", "v1")
    repo.add_test_cases(ds.id, [TestCase(id="c1", question="2024 Revenue?", expected_answer="$10B")])
    repo.publish_dataset(ds.id)

    provenance = RunProvenance(
        dataset_checksum=ds.checksum_sha256,
        rag_version="git_sha_8a12bc",
        model_config_hash="cfg_hash_1",
        prompt_hash="prm_hash_1",
        evaluator_version="1.0.0",
        experiment_hash="exp_topk5_bm25",
    )

    config = RunConfig(
        project_id=proj.id,
        dataset_id=ds.id,
        dataset_version=ds.version,
        system_version="v2.1-prod",
    )

    run = repo.create_run(config, provenance)
    assert run.manifest_hash == provenance.manifest_hash
    assert run.status == RunStatus.CREATED.value

    # Record trace
    trace = RagTrace(
        trace_id="tr_101",
        run_id=run.id,
        test_case_id="c1",
        question="2024 Revenue?",
        answer="$10B [1]",
        latency_ms=210,
        retrieved_chunks=[RetrievedChunk(document_id="annual", chunk_id="p1", rank=1, text="2024: $10B")],
        citations=[Citation(claim_id="cl1", claim_text="$10B", document_id="annual", chunk_id="p1")],
    )

    metrics = [
        MetricResult(metric_name="faithfulness", metric_family=MetricFamily.GENERATION, score=1.0),
        MetricResult(metric_name="recall_at_k", metric_family=MetricFamily.RETRIEVAL, score=1.0),
    ]

    attribution = FailureAttribution(
        trace_id="tr_101",
        failure_type=FailureCode.OPS_01,
        severity=Severity.LOW,
        confidence=0.1,
        explanation="No failure",
    )

    trace_row = repo.record_trace(trace, metrics, attribution)
    assert trace_row.id == "tr_101"

    # Complete run
    completed_run = repo.update_run_status(run.id, RunStatus.COMPLETED)
    assert completed_run.finished_at is not None

    # Immutability test: cannot modify completed run or add traces
    with pytest.raises(ImmutabilityError):
        repo.update_run_status(run.id, RunStatus.RUNNING)

    with pytest.raises(ImmutabilityError):
        repo.record_trace(trace)


def test_create_run_enforces_dataset_provenance_invariants():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        repo = DatabaseRepo(session)

        proj_a = repo.create_project("Project A")
        proj_b = repo.create_project("Project B")
        ds = repo.create_dataset(proj_a.id, "ds_provenance", "v1.0")
        repo.add_test_cases(ds.id, [TestCase(id="c1", question="Q", expected_answer="A")])
        repo.publish_dataset(ds.id)
        session.commit()

        correct_prov = RunProvenance(dataset_checksum=ds.checksum_sha256, rag_version="v1")
        correct_config = RunConfig(
            project_id=proj_a.id,
            dataset_id=ds.id,
            dataset_version="v1.0",
            system_version="sys1",
        )

        # 1. Project ID mismatch (Project B tries to evaluate Project A's dataset)
        with pytest.raises(ValueError, match="belongs to project"):
            repo.create_run(
                correct_config.model_copy(update={"project_id": proj_b.id}),
                correct_prov,
            )

        # 2. Dataset version mismatch
        with pytest.raises(ValueError, match="version is 'v1.0', but requested dataset_version is 'v2.0'"):
            repo.create_run(
                correct_config.model_copy(update={"dataset_version": "v2.0"}),
                correct_prov,
            )

        # 3. Provenance checksum mismatch
        with pytest.raises(ValueError, match="dataset checksum .* does not match"):
            repo.create_run(
                correct_config,
                correct_prov.model_copy(update={"dataset_checksum": "corrupted_checksum"}),
            )

