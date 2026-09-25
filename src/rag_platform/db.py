"""Relational database models, immutability constraints, and CRUD repository.
Supports SQLite and PostgreSQL via SQLAlchemy 2.0 ORM.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship

from rag_platform.core import (
    ImmutabilityError,
    generate_id,
    settings,
    sha256_hash,
)
from rag_platform.security import RecursiveTraceSanitizer
from rag_platform.models import (
    DatasetStatus,
    DocumentReference,
    FailureAttribution,
    MetricResult,
    RagTrace,
    RunConfig,
    RunProvenance,
    RunStatus,
    Severity,
    TestCase,
    compute_dataset_checksum,
)


class Base(DeclarativeBase):
    pass


class ProjectRow(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    settings_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


class DatasetRow(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default=DatasetStatus.DRAFT.value)
    checksum_sha256: Mapped[str] = mapped_column(String(64), default="")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    cases: Mapped[list[TestCaseRow]] = relationship("TestCaseRow", back_populates="dataset", cascade="all, delete-orphan")


class TestCaseRow(Base):
    __tablename__ = "test_cases"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_facts_json: Mapped[str] = mapped_column(Text, default="[]")
    relevant_docs_json: Mapped[str] = mapped_column(Text, default="[]")
    answerability: Mapped[str] = mapped_column(String(32), default="ANSWERABLE")
    tags_json: Mapped[str] = mapped_column(Text, default="[]")
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")

    dataset: Mapped[DatasetRow] = relationship("DatasetRow", back_populates="cases")


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), nullable=False, index=True)
    system_version: Mapped[str] = mapped_column(String(128), nullable=False)
    
    # 6-Dimension Provenance
    dataset_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    rag_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    evaluator_version: Mapped[str] = mapped_column(String(64), nullable=False)
    experiment_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    status: Mapped[str] = mapped_column(String(32), default=RunStatus.CREATED.value)
    policy_id: Mapped[str] = mapped_column(String(64), default="prod-default")
    suite: Mapped[str] = mapped_column(String(64), default="full")
    options_json: Mapped[str] = mapped_column(Text, default="{}")
    # Structured failure info (FIX #6): preserved when a run fails instead of silent exception swallow
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    traces: Mapped[list[TraceRow]] = relationship("TraceRow", back_populates="run", cascade="all, delete-orphan")


class TraceRow(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id"), nullable=False, index=True)
    test_case_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    abstention_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_usd: Mapped[float | None] = mapped_column(Float, nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    raw_trace_json: Mapped[str] = mapped_column(Text, nullable=False)

    run: Mapped[RunRow] = relationship("RunRow", back_populates="traces")
    metrics: Mapped[list[MetricResultRow]] = relationship("MetricResultRow", back_populates="trace", cascade="all, delete-orphan")
    failure: Mapped[FailureRow | None] = relationship("FailureRow", back_populates="trace", uselist=False, cascade="all, delete-orphan")


class MetricResultRow(Base):
    __tablename__ = "metric_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id"), nullable=False, index=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    metric_name: Mapped[str] = mapped_column(String(64), nullable=False)
    metric_family: Mapped[str] = mapped_column(String(32), nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="PASS")
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    evaluator_version: Mapped[str] = mapped_column(String(64), default="1.0.0")
    cached: Mapped[bool] = mapped_column(Boolean, default=False)

    trace: Mapped[TraceRow] = relationship("TraceRow", back_populates="metrics")


class FailureRow(Base):
    __tablename__ = "failures"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(ForeignKey("traces.id"), nullable=False, unique=True)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    failure_type: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), default=Severity.MEDIUM.value)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence_json: Mapped[str] = mapped_column(Text, default="{}")
    override_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    trace: Mapped[TraceRow] = relationship("TraceRow", back_populates="failure")


class ApiKeyRow(Base):
    __tablename__ = "api_keys"

    key_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    client_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    project_roles_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )


# --- Database Operations & Repository ---
def create_db_engine(db_url: str | None = None):
    url = db_url or settings.database_url
    return create_engine(url, echo=False)


def create_session(db_url: str | None = None) -> Session:
    """Create a new SQLAlchemy session connected to the configured database."""
    return Session(create_db_engine(db_url))


def init_db(engine=None) -> None:
    eng = engine or create_db_engine()
    Base.metadata.create_all(bind=eng)


class DatabaseRepo:
    """Repository handling data access with strict immutability checks."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def create_project(
        self,
        name: str,
        settings_dict: dict[str, Any] | None = None,
        settings: dict[str, Any] | None = None,
    ) -> ProjectRow:
        actual_settings = settings if settings is not None else (settings_dict or {})
        proj = ProjectRow(
            id=generate_id("proj"),
            name=name,
            settings_json=json.dumps(actual_settings),
        )
        self.session.add(proj)
        self.session.flush()
        return proj

    def create_dataset(self, project_id: str, name: str, version: str, description: str | None = None) -> DatasetRow:
        ds = DatasetRow(
            id=generate_id("ds"),
            project_id=project_id,
            name=name,
            version=version,
            description=description,
            status=DatasetStatus.DRAFT.value,
        )
        self.session.add(ds)
        self.session.flush()
        return ds

    def add_test_cases(self, dataset_id: str, cases: list[TestCase]) -> list[TestCaseRow]:
        ds = self.session.get(DatasetRow, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found.")
        if ds.status == DatasetStatus.PUBLISHED.value:
            raise ImmutabilityError(f"Cannot add test cases to published dataset {dataset_id}.")

        rows = []
        for case in cases:
            row = TestCaseRow(
                id=case.id,
                dataset_id=dataset_id,
                question=case.question,
                expected_answer=case.expected_answer,
                expected_facts_json=json.dumps(case.expected_facts),
                relevant_docs_json=json.dumps([d.model_dump() for d in case.relevant_documents]),
                answerability=case.answerability.value,
                tags_json=json.dumps(case.tags),
                metadata_json=json.dumps(case.metadata),
            )
            self.session.add(row)
            rows.append(row)
        self.session.flush()
        return rows

    def publish_dataset(self, dataset_id: str) -> DatasetRow:
        ds = self.session.get(DatasetRow, dataset_id)
        if not ds:
            raise ValueError(f"Dataset {dataset_id} not found.")
        if ds.status == DatasetStatus.PUBLISHED.value:
            return ds

        # Load cases and compute deterministic checksum
        cases = []
        for r in ds.cases:
            cases.append(
                TestCase(
                    id=r.id,
                    question=r.question,
                    expected_answer=r.expected_answer,
                    expected_facts=json.loads(r.expected_facts_json),
                    relevant_documents=[DocumentReference(**d) for d in json.loads(r.relevant_docs_json)],
                    answerability=r.answerability,
                    tags=json.loads(r.tags_json),
                )
            )
        if not cases:
            raise ValueError("Cannot publish an empty dataset.")

        ds.checksum_sha256 = compute_dataset_checksum(cases)
        ds.status = DatasetStatus.PUBLISHED.value
        self.session.flush()
        return ds

    def get_dataset(self, dataset_id: str) -> DatasetRow | None:
        return self.session.get(DatasetRow, dataset_id)

    def create_run(self, config: RunConfig, provenance: RunProvenance) -> RunRow:
        ds = self.session.get(DatasetRow, config.dataset_id)
        if not ds:
            raise ValueError(f"Dataset {config.dataset_id} not found.")
        if ds.status != DatasetStatus.PUBLISHED.value:
            raise ValueError(f"Cannot run evaluation against unpublished dataset {config.dataset_id}.")

        run = RunRow(
            id=generate_id("run"),
            project_id=config.project_id,
            dataset_id=config.dataset_id,
            system_version=config.system_version,
            dataset_checksum=provenance.dataset_checksum,
            rag_version=provenance.rag_version,
            model_config_hash=provenance.model_config_hash,
            prompt_hash=provenance.prompt_hash,
            evaluator_version=provenance.evaluator_version,
            experiment_hash=provenance.experiment_hash,
            manifest_hash=provenance.manifest_hash,
            status=RunStatus.CREATED.value,
            policy_id=config.policy_id,
            suite=config.suite,
            options_json=config.options.model_dump_json(),
        )
        self.session.add(run)
        self.session.flush()
        return run

    def record_trace(
        self,
        trace: RagTrace,
        metrics: list[MetricResult] | None = None,
        attribution: FailureAttribution | None = None,
    ) -> TraceRow:
        run = self.session.get(RunRow, trace.run_id)
        if not run:
            raise ValueError(f"Run {trace.run_id} not found.")
        if run.status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value):
            raise ImmutabilityError(f"Cannot add traces to completed/terminal run {trace.run_id}.")

        # Recursively sanitize the entire trace (nested chunks, metadata, telemetry, credentials)
        clean_trace: RagTrace = RecursiveTraceSanitizer.sanitize_trace(trace)

        trace_row = TraceRow(
            id=clean_trace.trace_id,
            run_id=clean_trace.run_id,
            test_case_id=clean_trace.test_case_id,
            question=clean_trace.question,
            answer=clean_trace.answer,
            abstained=clean_trace.abstained,
            abstention_reason=clean_trace.abstention_reason,
            latency_ms=clean_trace.latency_ms,
            input_tokens=clean_trace.input_tokens,
            output_tokens=clean_trace.output_tokens,
            cost_usd=clean_trace.cost_usd,
            model=clean_trace.model,
            error_code=clean_trace.error_code,
            raw_trace_json=clean_trace.model_dump_json(),
        )
        self.session.add(trace_row)
        self.session.flush()

        if metrics:
            for m in metrics:
                m_row = MetricResultRow(
                    id=generate_id("met"),
                    trace_id=clean_trace.trace_id,
                    run_id=clean_trace.run_id,
                    metric_name=m.metric_name,
                    metric_family=m.metric_family.value,
                    score=m.score,
                    status=m.status.value,
                    reason=m.reason,
                    evaluator_version=m.evaluator_version,
                    cached=m.cached,
                )
                self.session.add(m_row)
            self.session.flush()

        if attribution:
            evidence_payload = {
                **attribution.evidence,
                "recommended_actions": attribution.recommended_actions,
            }
            f_row = FailureRow(
                id=generate_id("fail"),
                trace_id=trace.trace_id,
                run_id=trace.run_id,
                failure_type=attribution.failure_type.value,
                severity=attribution.severity.value,
                confidence=attribution.confidence,
                explanation=attribution.explanation,
                evidence_json=json.dumps(evidence_payload),
            )
            self.session.add(f_row)

        self.session.flush()
        return trace_row

    def update_run_status(self, run_id: str, status: RunStatus) -> RunRow:
        run = self.session.get(RunRow, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found.")
        if run.status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value):
            raise ImmutabilityError(f"Run {run_id} is in terminal state ({run.status}) and cannot be modified.")

        run.status = status.value
        if status in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED):
            run.finished_at = datetime.now(timezone.utc)
        self.session.flush()
        return run

    def create_api_key(
        self,
        client_id: str,
        api_key: str | None = None,
        project_roles: dict[str, Any] | None = None,
        is_admin: bool = False,
    ) -> tuple[str, ApiKeyRow]:
        raw_key = api_key or f"rag_{generate_id('key')}"
        key_hash = sha256_hash(raw_key)
        roles = {p: (r.value if hasattr(r, "value") else str(r)) for p, r in (project_roles or {}).items()}
        row = ApiKeyRow(
            key_hash=key_hash,
            client_id=client_id,
            is_admin=is_admin,
            project_roles_json=json.dumps(roles),
        )
        self.session.add(row)
        self.session.flush()
        return raw_key, row

    def get_api_key(self, api_key: str) -> ApiKeyRow | None:
        key_hash = sha256_hash(api_key)
        return self.session.get(ApiKeyRow, key_hash)
