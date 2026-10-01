"""Relational database models, immutability constraints, and CRUD repository.
Supports SQLite and PostgreSQL via SQLAlchemy 2.0 ORM.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    delete,
    func,
    select,
    update,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from rag_platform.core import (
    ImmutabilityError,
    generate_id,
    settings,
    sha256_hash,
)
from rag_platform.models import (
    Answerability,
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
from rag_platform.security import RecursiveTraceSanitizer, SecretRedactor


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
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"), primary_key=True)
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
    __table_args__ = (
        Index("uq_runs_project_idempotency_key", "project_id", "idempotency_key", unique=True),
    )

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
    # Structured failure info: preserved when a run fails instead of silent exception swallow
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    gate_result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    gate_status: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    # Distributed Worker Lease & Ownership Fields (Points 8 & 31)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    lease_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)

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
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=lambda: datetime.now(timezone.utc)
    )
    revoked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )


class EvaluationCacheRow(Base):
    __tablename__ = "evaluation_cache"

    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    result_json: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    last_accessed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )


class SessionRow(Base):
    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    api_key_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    client_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    project_roles_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    expires_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )



# --- Database Operations & Repository ---
_engines: dict[str, Any] = {}
_sessionmakers: dict[str, sessionmaker[Session]] = {}
_db_lock = threading.RLock()


def get_db_engine(db_url: str | None = None):
    """Retrieve or create application-wide singleton database engine with connection pooling."""
    url = db_url or settings.database_url
    with _db_lock:
        if url not in _engines:
            connect_args = {}
            if url.startswith("sqlite"):
                connect_args["check_same_thread"] = False
            _engines[url] = create_engine(url, echo=False, pool_pre_ping=True, connect_args=connect_args)
        return _engines[url]


def get_sessionmaker(db_url: str | None = None) -> sessionmaker[Session]:
    """Retrieve or create application-wide singleton sessionmaker."""
    url = db_url or settings.database_url
    with _db_lock:
        if url not in _sessionmakers:
            eng = get_db_engine(url)
            _sessionmakers[url] = sessionmaker(bind=eng, autoflush=False, expire_on_commit=False)
        return _sessionmakers[url]


def create_db_engine(db_url: str | None = None):
    return get_db_engine(db_url)


def create_session(db_url: str | None = None) -> Session:
    """Create a new SQLAlchemy session using application-wide sessionmaker and shared engine pool."""
    sm = get_sessionmaker(db_url)
    return sm()


def reset_engine_cache() -> None:
    """Dispose and clear cached engines (used in test isolation)."""
    with _db_lock:
        for eng in _engines.values():
            try:
                eng.dispose()
            except Exception:
                pass
        _engines.clear()
        _sessionmakers.clear()


def init_db(engine=None, allow_non_memory: bool = False) -> None:
    """Initialize database tables using Base.metadata.create_all().

    ARCHITECTURAL RULE: Alembic is the authoritative schema owner for production.
    `Base.metadata.create_all()` is strictly reserved for ephemeral in-memory test databases (`sqlite:///:memory:`).
    Production environments must always apply schema changes via `alembic upgrade head`.
    """
    eng = engine or create_db_engine()
    url_str = str(eng.url)
    if not allow_non_memory and ":memory:" not in url_str:
        raise RuntimeError(
            "init_db() / create_all() is strictly prohibited on persistent databases. "
            "Production schema authority is Alembic migrations (`alembic upgrade head`). "
            "Pass allow_non_memory=True if running explicit non-production test harnesses."
        )
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
                    answerability=Answerability(r.answerability),
                    tags=json.loads(r.tags_json),
                    metadata=json.loads(r.metadata_json) if r.metadata_json else {},
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

    def get_dataset_case_count(self, dataset_id: str) -> int:
        """Database-level count avoiding loading full test case objects into memory."""
        return self.session.scalar(
            select(func.count(TestCaseRow.id)).where(TestCaseRow.dataset_id == dataset_id)
        ) or 0

    def get_run_trace_count(self, run_id: str) -> int:
        """Database-level count avoiding loading full trace objects into memory."""
        return self.session.scalar(
            select(func.count(TraceRow.id)).where(TraceRow.run_id == run_id)
        ) or 0

    def create_run(
        self,
        config: RunConfig,
        provenance: RunProvenance,
        initial_status: RunStatus = RunStatus.CREATED,
        options_override_json: str | None = None,
        idempotency_key: str | None = None,
    ) -> RunRow:
        # Dataset invariants are checked before the provenance manifest hash so checksum mismatches retain their specific error contract.
        if idempotency_key:
            existing = self.session.scalar(
                select(RunRow).where(
                    RunRow.project_id == config.project_id,
                    RunRow.idempotency_key == idempotency_key,
                )
            )
            if existing:
                setattr(existing, "_is_existing", True)
                return existing

        ds = self.session.get(DatasetRow, config.dataset_id)
        if not ds:
            raise ValueError(f"Dataset {config.dataset_id} not found.")
        if ds.status != DatasetStatus.PUBLISHED.value:
            raise ValueError(f"Cannot run evaluation against unpublished dataset {config.dataset_id}.")
        if ds.project_id != config.project_id:
            raise ValueError(
                f"Dataset {config.dataset_id} belongs to project '{ds.project_id}', "
                f"not requested project '{config.project_id}'."
            )
        if config.dataset_version and ds.version != config.dataset_version:
            raise ValueError(
                f"Dataset {config.dataset_id} version is '{ds.version}', "
                f"but requested dataset_version is '{config.dataset_version}'."
            )
        if provenance.dataset_checksum != ds.checksum_sha256:
            raise ValueError(
                f"Run provenance dataset checksum '{provenance.dataset_checksum}' does not match "
                f"dataset {config.dataset_id} checksum '{ds.checksum_sha256}'."
            )

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
            status=initial_status.value,
            policy_id=config.policy_id,
            suite=config.suite,
            options_json=options_override_json or config.options.model_dump_json(),
            idempotency_key=idempotency_key,
        )
        if idempotency_key:
            try:
                with self.session.begin_nested():
                    self.session.add(run)
                    self.session.flush()
                setattr(run, "_is_existing", False)
                return run
            except IntegrityError:
                # Concurrent request inserted the same (project_id, idempotency_key)
                if run in self.session:
                    self.session.expunge(run)
                existing = self.session.scalar(
                    select(RunRow).where(
                        RunRow.project_id == config.project_id,
                        RunRow.idempotency_key == idempotency_key,
                    )
                )
                if existing:
                    setattr(existing, "_is_existing", True)
                    return existing
                raise
        else:
            self.session.add(run)
            self.session.flush()
            setattr(run, "_is_existing", False)
            return run

    def purge_expired_data(self, trace_retention_days: int = 90) -> dict[str, int]:
        """Purge historical evaluation traces older than retention threshold."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=trace_retention_days)
        # Find runs finished before cutoff
        expired_runs = self.session.scalars(
            select(RunRow).where(
                RunRow.finished_at.is_not(None),
                RunRow.finished_at < cutoff,
            )
        ).all()
        purged_traces = 0
        for r in expired_runs:
            count = len(r.traces)
            r.traces.clear()
            purged_traces += count
        self.session.flush()
        return {"expired_runs_evaluated": len(expired_runs), "purged_traces": purged_traces}

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

        # Validate uniqueness of trace_id before persistence to protect against primary-key collisions
        if self.session.get(TraceRow, clean_trace.trace_id) is not None:
            clean_trace.trace_id = f"tr_{clean_trace.run_id}_{clean_trace.test_case_id}_{generate_id()}"

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
            clean_evidence = RecursiveTraceSanitizer.sanitize_value(attribution.evidence)
            evidence_payload = {
                **(clean_evidence if isinstance(clean_evidence, dict) else {}),
                "recommended_actions": attribution.recommended_actions,
            }
            f_row = FailureRow(
                id=generate_id("fail"),
                trace_id=trace.trace_id,
                run_id=trace.run_id,
                failure_type=attribution.failure_type.value if attribution.failure_type else "UNKNOWN",
                severity=attribution.severity.value if hasattr(attribution.severity, "value") else str(attribution.severity),
                confidence=attribution.confidence,
                explanation=SecretRedactor.redact_text(attribution.explanation),
                evidence_json=json.dumps(evidence_payload),
            )
            self.session.add(f_row)

        self.session.flush()
        return trace_row

    def cancel_run_atomic(
        self,
        run_id: str,
        reason: str = "Run cancelled by user request.",
    ) -> tuple[bool, RunRow | None]:
        """Atomically transition an active/queued run to CANCELLED state.

        Uses a conditional database UPDATE (WHERE status IN ('CREATED', 'QUEUED', 'RUNNING'))
        to eliminate read-modify-write race conditions against concurrent worker finalization.
        Returns:
            tuple[bool, RunRow | None]: (True if updated, current RunRow)
        """
        now = datetime.now(timezone.utc)
        stmt = (
            update(RunRow)
            .where(
                RunRow.id == run_id,
                RunRow.status.in_([
                    RunStatus.CREATED.value,
                    RunStatus.QUEUED.value,
                    RunStatus.RUNNING.value,
                ]),
            )
            .values(
                status=RunStatus.CANCELLED.value,
                finished_at=now,
                failure_reason=reason,
            )
        )
        res = self.session.execute(stmt)
        self.session.flush()
        self.session.expire_all()
        run = self.session.get(RunRow, run_id)
        return (res.rowcount > 0, run)

    def update_run_status(self, run_id: str, status: RunStatus) -> RunRow:
        if status == RunStatus.CANCELLED:
            success, run = self.cancel_run_atomic(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found.")
            if not success:
                raise ImmutabilityError(f"Run {run_id} is in terminal state ({run.status}) and cannot be modified.")
            return run

        if status in (RunStatus.COMPLETED, RunStatus.FAILED):
            now = datetime.now(timezone.utc)
            stmt = (
                update(RunRow)
                .where(
                    RunRow.id == run_id,
                    RunRow.status.in_([
                        RunStatus.CREATED.value,
                        RunStatus.QUEUED.value,
                        RunStatus.RUNNING.value,
                    ]),
                )
                .values(
                    status=status.value,
                    finished_at=now,
                )
            )
            res = self.session.execute(stmt)
            self.session.flush()
            self.session.expire_all()
            run = self.session.get(RunRow, run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found.")
            if res.rowcount == 0:
                raise ImmutabilityError(f"Run {run_id} is in terminal state ({run.status}) and cannot be modified.")
            return run

        run = self.session.get(RunRow, run_id)
        if not run:
            raise ValueError(f"Run {run_id} not found.")
        if run.status in (RunStatus.COMPLETED.value, RunStatus.FAILED.value, RunStatus.CANCELLED.value):
            raise ImmutabilityError(f"Run {run_id} is in terminal state ({run.status}) and cannot be modified.")

        run.status = status.value
        self.session.flush()
        return run

    def create_api_key(
        self,
        client_id: str,
        api_key: str | None = None,
        project_roles: dict[str, Any] | None = None,
        is_admin: bool = False,
    ) -> tuple[str, ApiKeyRow]:
        from rag_platform.security import generate_secure_api_key
        raw_key = api_key or generate_secure_api_key()
        key_hash = sha256_hash(raw_key)
        roles = {p: (r.value if hasattr(r, "value") else str(r).upper()) for p, r in (project_roles or {}).items()}
        row = ApiKeyRow(
            key_hash=key_hash,
            client_id=client_id,
            is_admin=is_admin,
            project_roles_json=json.dumps(roles),
            created_at=datetime.now(timezone.utc),
            revoked_at=None,
        )
        self.session.add(row)
        self.session.flush()
        return raw_key, row

    def get_api_key(self, api_key: str) -> ApiKeyRow | None:
        key_hash = sha256_hash(api_key)
        return self.session.get(ApiKeyRow, key_hash)

    def list_api_keys(
        self, client_id: str | None = None, include_revoked: bool = False
    ) -> list[ApiKeyRow]:
        """List persisted API keys with optional client filtering and revocation status."""
        stmt = select(ApiKeyRow)
        if client_id is not None:
            stmt = stmt.where(ApiKeyRow.client_id == client_id)
        if not include_revoked:
            stmt = stmt.where(ApiKeyRow.revoked_at.is_(None))
        stmt = stmt.order_by(ApiKeyRow.created_at.desc())
        return list(self.session.scalars(stmt).all())

    def revoke_api_key(self, key_hash: str) -> bool:
        """Revoke an active API key by its hash. Atomically marks revoked_at and purges process cache."""
        from rag_platform.security import ApiKeyRegistry

        now = datetime.now(timezone.utc)
        stmt = (
            update(ApiKeyRow)
            .where(ApiKeyRow.key_hash == key_hash, ApiKeyRow.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        res = self.session.execute(stmt)
        # Invalidate any browser sessions created from this revoked API key
        self.session.execute(delete(SessionRow).where(SessionRow.api_key_hash == key_hash))
        self.session.flush()
        ApiKeyRegistry.invalidate(key_hash)
        return res.rowcount > 0

    def rotate_api_key(self, old_key_hash: str) -> tuple[str, ApiKeyRow]:
        """Atomically revoke an active API key and issue a new credential with the same identity and permissions."""
        from rag_platform.security import ApiKeyRegistry, generate_secure_api_key

        old_row = self.session.get(ApiKeyRow, old_key_hash)
        if not old_row:
            raise ValueError(f"API key with hash '{old_key_hash}' not found.")
        if old_row.revoked_at is not None:
            raise ValueError(f"API key with hash '{old_key_hash}' is already revoked.")

        now = datetime.now(timezone.utc)
        old_row.revoked_at = now
        # Invalidate any browser sessions created from rotated old API key
        self.session.execute(delete(SessionRow).where(SessionRow.api_key_hash == old_key_hash))
        self.session.flush()
        ApiKeyRegistry.invalidate(old_key_hash)

        raw_new_key = generate_secure_api_key()
        new_key_hash = sha256_hash(raw_new_key)
        new_row = ApiKeyRow(
            key_hash=new_key_hash,
            client_id=old_row.client_id,
            is_admin=old_row.is_admin,
            project_roles_json=old_row.project_roles_json,
            created_at=now,
            revoked_at=None,
        )
        self.session.add(new_row)
        self.session.flush()
        return (raw_new_key, new_row)
