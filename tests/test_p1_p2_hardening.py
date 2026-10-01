"""Regression tests for P1/P2/P3 hardening fixes.

Covers:
  P1  Async /v1/runs idempotency bug
  P1  Dataset metadata loss in publish_dataset checksum
  P1  Gate mutation of published datasets
  P2  MRR list-position vs chunk.rank
  P2  Citation doc_map collapse
  P2  Citation validation with wrong chunk_id
  P2  Eval cache key includes rank
  P2  Wilson CI only for binary metrics
  P2  Percentile linear interpolation
"""

from __future__ import annotations

import math
import statistics

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from rag_platform.db import (
    Base,
    DatabaseRepo,
)
from rag_platform.evaluators import (
    BoundedLRUCache,
    CitationSupportMetric,
    EvaluationEngine,
    MeanReciprocalRankMetric,
    wilson_score_interval,
)
from rag_platform.models import (
    Citation,
    DocumentReference,
    MetricFamily,
    MetricResult,
    RagTrace,
    RetrievedChunk,
    RunConfig,
    RunProvenance,
    RunStatus,
    TestCase,
    compute_dataset_checksum,
)


@pytest.fixture()
def db_session():
    """Create an isolated in-memory SQLite database for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    with Session(engine) as sess:
        yield sess


# ─── P1: Async /v1/runs Idempotency Bug ────────────────────────────────────


class TestAsyncRunIdempotency:
    """Verify that new async runs are NOT treated as existing."""

    def test_new_async_run_is_not_existing(self, db_session):
        """A brand new async run with status=QUEUED must have _is_existing=False."""
        repo = DatabaseRepo(db_session)
        proj = repo.create_project(name="test-proj")
        ds = repo.create_dataset(project_id=proj.id, name="ds", version="1.0")
        case = TestCase(id="c1", question="q", expected_answer="a")
        repo.add_test_cases(ds.id, [case])
        repo.publish_dataset(ds.id)
        db_session.commit()

        config = RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="1.0",
            system_version="v1",
        )
        prov = RunProvenance(dataset_checksum=ds.checksum_sha256)

        # Create an async run (status=QUEUED) with an idempotency key
        run = repo.create_run(
            config,
            prov,
            initial_status=RunStatus.QUEUED,
            idempotency_key="idem-key-1",
        )
        db_session.commit()

        assert run.status == RunStatus.QUEUED.value
        assert getattr(run, "_is_existing", False) is False, (
            "Brand new QUEUED run must NOT be marked as existing"
        )

    def test_repeated_idempotency_key_is_existing(self, db_session):
        """A repeated request with the same idempotency key must be flagged."""
        repo = DatabaseRepo(db_session)
        proj = repo.create_project(name="test-proj-2")
        ds = repo.create_dataset(project_id=proj.id, name="ds2", version="1.0")
        case = TestCase(id="c1", question="q", expected_answer="a")
        repo.add_test_cases(ds.id, [case])
        repo.publish_dataset(ds.id)
        db_session.commit()

        config = RunConfig(
            project_id=proj.id,
            dataset_id=ds.id,
            dataset_version="1.0",
            system_version="v1",
        )
        prov = RunProvenance(dataset_checksum=ds.checksum_sha256)

        # First creation
        run1 = repo.create_run(config, prov, initial_status=RunStatus.QUEUED, idempotency_key="ik-2")
        db_session.commit()
        assert getattr(run1, "_is_existing", False) is False

        # Second creation with same key
        run2 = repo.create_run(config, prov, initial_status=RunStatus.QUEUED, idempotency_key="ik-2")
        assert getattr(run2, "_is_existing", False) is True
        assert run2.id == run1.id


# ─── P1: Dataset Metadata in Checksum ──────────────────────────────────────


class TestDatasetMetadataChecksum:
    """Verify metadata participates in the dataset checksum."""

    def test_metadata_changes_checksum(self):
        """Different metadata must produce different checksums."""
        case_a = TestCase(
            id="c1", question="q", expected_answer="a",
            metadata={"domain": "finance"},
        )
        case_b = TestCase(
            id="c1", question="q", expected_answer="a",
            metadata={"domain": "healthcare"},
        )

        checksum_a = compute_dataset_checksum([case_a])
        checksum_b = compute_dataset_checksum([case_b])
        assert checksum_a != checksum_b, "Different metadata must produce different checksums"

    def test_publish_preserves_metadata_in_checksum(self, db_session):
        """publish_dataset must include metadata in checksum computation."""
        repo = DatabaseRepo(db_session)
        proj = repo.create_project(name="meta-proj")
        ds = repo.create_dataset(project_id=proj.id, name="meta-ds", version="1.0")

        case = TestCase(
            id="c1", question="q", expected_answer="a",
            metadata={"routing": "llm-v2"},
        )
        repo.add_test_cases(ds.id, [case])
        repo.publish_dataset(ds.id)
        db_session.commit()

        # Compute expected checksum WITH metadata
        expected_checksum = compute_dataset_checksum([case])
        db_session.refresh(ds)
        assert ds.checksum_sha256 == expected_checksum

    def test_publish_without_metadata_matches_empty(self, db_session):
        """Metadata-less cases should produce same checksum whether {} or missing."""
        case_empty = TestCase(id="c1", question="q", expected_answer="a", metadata={})
        case_default = TestCase(id="c1", question="q", expected_answer="a")
        assert compute_dataset_checksum([case_empty]) == compute_dataset_checksum([case_default])


# ─── P1: Gate Immutability (Published Datasets) ────────────────────────────


class TestGatePublishedImmutability:
    """Verify gate.py does not mutate published datasets."""

    def test_add_cases_to_published_raises(self, db_session):
        """Adding test cases to a PUBLISHED dataset must raise ImmutabilityError."""
        from rag_platform.core import ImmutabilityError

        repo = DatabaseRepo(db_session)
        proj = repo.create_project(name="immut-proj")
        ds = repo.create_dataset(project_id=proj.id, name="immut-ds", version="1.0")
        case = TestCase(id="c1", question="q", expected_answer="a")
        repo.add_test_cases(ds.id, [case])
        repo.publish_dataset(ds.id)
        db_session.commit()

        # Attempting to add cases to published dataset should fail
        with pytest.raises(ImmutabilityError):
            repo.add_test_cases(ds.id, [TestCase(id="c2", question="q2", expected_answer="a2")])


# ─── P2: MRR Uses List Position, Not chunk.rank ────────────────────────────


class TestMRRListPosition:
    """MRR should use the 1-indexed position in the retrieved_chunks list, not chunk.rank."""

    @pytest.mark.asyncio
    async def test_mrr_uses_list_position_not_chunk_rank(self):
        """When list position differs from chunk.rank, MRR should use position."""
        # The gold document is second in the list (position=2) but has rank=5
        chunks = [
            RetrievedChunk(document_id="other", chunk_id="x1", rank=1, text="irrelevant"),
            RetrievedChunk(document_id="doc1", chunk_id="c1", rank=5, text="relevant evidence"),
        ]
        case = TestCase(
            id="t1", question="q",
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="c1")],
        )
        trace = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1",
            question="q", retrieved_chunks=chunks,
        )

        metric = MeanReciprocalRankMetric()
        result = await metric.compute(trace, case)

        # Position is 2, so RR = 1/2 = 0.5
        assert result.score == pytest.approx(0.5, abs=0.001), (
            f"MRR should be 1/2=0.5 (list position), not 1/5=0.2 (chunk.rank). Got {result.score}"
        )


# ─── P2: Citation doc_map Collapse ──────────────────────────────────────────


class TestCitationDocMapCollapse:
    """Verify doc_map preserves all chunks from the same document."""

    @pytest.mark.asyncio
    async def test_multiple_chunks_same_doc_all_checked(self):
        """If two chunks from the same doc, citation should check both, not just the last."""
        chunks = [
            RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1,
                           text="Revenue was $142.5 million in Q3"),
            RetrievedChunk(document_id="doc1", chunk_id="c2", rank=2,
                           text="Headcount increased by 500 employees"),
        ]
        # Citation references doc1 without specifying chunk, and the claim matches c2 not c1
        cit = Citation(
            claim_id="cl1",
            claim_text="Headcount increased by 500 employees",
            document_id="doc1",
            chunk_id="",  # empty = doc-level match
        )
        case = TestCase(id="t1", question="q")
        trace = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1",
            question="q", answer="Headcount increased by 500 employees",
            retrieved_chunks=chunks, citations=[cit],
        )

        metric = CitationSupportMetric()
        result = await metric.compute(trace, case)

        # With the old doc_map={doc1: c2}, only c2 would be checked.
        # With the fix, both c1 AND c2 are checked, and the claim matches c2.
        assert result.score is not None and result.score > 0.0, (
            "Citation should match against all chunks from same doc, not just last"
        )


# ─── P2: Citation Wrong chunk_id Should Not Fall Back to Document ───────────


class TestCitationStrictChunkMatch:
    """Citation with wrong chunk_id must not silently fall back to doc-level."""

    @pytest.mark.asyncio
    async def test_wrong_chunk_id_is_miss(self):
        """Citation referencing non-existent chunk_id should be a miss, not fall back to doc."""
        chunks = [
            RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1,
                           text="Revenue was $142.5 million"),
        ]
        # Citation says chunk_id="c_WRONG" — this is NOT in the retrieved set
        cit = Citation(
            claim_id="cl1",
            claim_text="Revenue was $142.5 million",
            document_id="doc1",
            chunk_id="c_WRONG",  # non-existent
        )
        case = TestCase(id="t1", question="q")
        trace = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1",
            question="q", answer="Revenue was $142.5 million",
            retrieved_chunks=chunks, citations=[cit],
        )

        metric = CitationSupportMetric()
        result = await metric.compute(trace, case)

        assert result.score == 0.0, (
            f"Citation with wrong chunk_id should be a miss (score=0), got {result.score}"
        )


# ─── P2: Eval Cache Key Includes Rank ───────────────────────────────────────


class TestEvalCacheKeyRank:
    """Cache key must change when chunk rank changes."""

    @pytest.mark.asyncio
    async def test_different_ranks_produce_different_cache_keys(self):
        """Same content but different ranks must produce different cache keys."""
        engine = EvaluationEngine(cache=BoundedLRUCache(capacity=100))

        case = TestCase(
            id="t1", question="q",
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="c1")],
        )

        trace_rank1 = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1", question="q",
            answer="answer", retrieved_chunks=[
                RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="evidence"),
            ],
        )
        trace_rank5 = RagTrace(
            trace_id="tr2", run_id="r1", test_case_id="t1", question="q",
            answer="answer", retrieved_chunks=[
                RetrievedChunk(document_id="doc1", chunk_id="c1", rank=5, text="evidence"),
            ],
        )

        metric = engine.metrics[0]  # recall_at_5
        key1 = engine._cache_key(metric, trace_rank1, case)
        key2 = engine._cache_key(metric, trace_rank5, case)

        assert key1 != key2, "Cache keys must differ when chunk rank differs"

    @pytest.mark.asyncio
    async def test_different_answerability_produces_different_cache_keys(self):
        """Cache keys must differ when test case answerability differs."""
        from rag_platform.models import Answerability

        engine = EvaluationEngine(cache=BoundedLRUCache(capacity=100))
        case_ans = TestCase(
            id="t1", question="q", answerability=Answerability.ANSWERABLE,
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="c1")],
        )
        case_unans = TestCase(
            id="t1", question="q", answerability=Answerability.UNANSWERABLE,
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="c1")],
        )
        trace = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1", question="q",
            answer="answer",
            retrieved_chunks=[RetrievedChunk(document_id="doc1", chunk_id="c1", rank=1, text="evidence")],
        )
        metric = engine.metrics[0]
        key1 = engine._cache_key(metric, trace, case_ans)
        key2 = engine._cache_key(metric, trace, case_unans)
        assert key1 != key2, "Cache keys must differ when case answerability differs"

    @pytest.mark.asyncio
    async def test_different_citation_doc_and_span_produce_different_cache_keys(self):
        """Cache keys must differ when citation document_id or span differs."""
        from rag_platform.models import Citation

        engine = EvaluationEngine(cache=BoundedLRUCache(capacity=100))
        case = TestCase(
            id="t1", question="q",
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="c1")],
        )
        trace_doc1 = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1", question="q", answer="answer",
            citations=[Citation(claim_id="cl1", claim_text="Fact", document_id="doc1", chunk_id="c1", span=[0, 10])],
        )
        trace_doc2 = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1", question="q", answer="answer",
            citations=[Citation(claim_id="cl1", claim_text="Fact", document_id="doc2", chunk_id="c1", span=[0, 10])],
        )
        trace_span2 = RagTrace(
            trace_id="tr1", run_id="r1", test_case_id="t1", question="q", answer="answer",
            citations=[Citation(claim_id="cl1", claim_text="Fact", document_id="doc1", chunk_id="c1", span=[15, 25])],
        )
        metric = engine.metrics[0]
        key_doc1 = engine._cache_key(metric, trace_doc1, case)
        key_doc2 = engine._cache_key(metric, trace_doc2, case)
        key_span2 = engine._cache_key(metric, trace_span2, case)

        assert key_doc1 != key_doc2, "Cache keys must differ when citation document_id differs"
        assert key_doc1 != key_span2, "Cache keys must differ when citation span differs"


# ─── P2: Wilson CI Only for Binary Metrics ──────────────────────────────────


class TestStatisticalCI:
    """Verify CI method selection based on metric value distribution."""

    def test_binary_metrics_use_wilson(self):
        """For binary (0/1) scores, Wilson interval should be used."""
        from rag_platform.evaluators import EvaluationEngine

        engine = EvaluationEngine()
        # Create binary traces
        traces = []
        for i in range(10):
            trace = RagTrace(trace_id=f"tr{i}", run_id="r1", test_case_id=f"t{i}", question="q", latency_ms=100)
            metrics = [
                MetricResult(
                    metric_name="abstention_accuracy",
                    metric_family=MetricFamily.ABSTENTION,
                    score=1.0 if i < 8 else 0.0,  # binary: 1.0 or 0.0
                ),
            ]
            traces.append((trace, metrics))

        summary = engine.aggregate_run(traces)
        ms = summary.metrics.get("abstention_accuracy")
        assert ms is not None

        # Wilson CI for p=0.8, n=10 should produce specific bounds
        expected_lower, expected_upper = wilson_score_interval(0.8, 10)
        assert ms.ci_lower == pytest.approx(expected_lower, abs=0.01)
        assert ms.ci_upper == pytest.approx(expected_upper, abs=0.01)

    def test_continuous_metrics_use_normal_ci(self):
        """For continuous scores (not all 0/1), normal CI should be used."""
        from rag_platform.evaluators import EvaluationEngine

        engine = EvaluationEngine()
        # Create continuous traces
        scores = [0.85, 0.92, 0.78, 0.65, 0.95, 0.88, 0.91, 0.73, 0.82, 0.90]
        traces = []
        for i, s in enumerate(scores):
            trace = RagTrace(trace_id=f"tr{i}", run_id="r1", test_case_id=f"t{i}", question="q", latency_ms=100)
            metrics = [
                MetricResult(
                    metric_name="faithfulness",
                    metric_family=MetricFamily.GENERATION,
                    score=s,
                ),
            ]
            traces.append((trace, metrics))

        summary = engine.aggregate_run(traces)
        ms = summary.metrics.get("faithfulness")
        assert ms is not None

        # For continuous metrics, CI should be normal approximation, not Wilson
        mean_val = sum(scores) / len(scores)
        std_dev = math.sqrt(sum((x - mean_val) ** 2 for x in scores) / (len(scores) - 1))
        se = std_dev / math.sqrt(len(scores))
        z = statistics.NormalDist().inv_cdf(0.975)
        expected_lower = round(max(0.0, mean_val - z * se), 4)
        expected_upper = round(min(1.0, mean_val + z * se), 4)

        assert ms.ci_lower == pytest.approx(expected_lower, abs=0.01)
        assert ms.ci_upper == pytest.approx(expected_upper, abs=0.01)


# ─── P2: Percentile Linear Interpolation ────────────────────────────────────


class TestPercentileCalculation:
    """Verify percentile uses standard linear interpolation."""

    def test_percentile_matches_standard(self):
        """Percentile calculation should match numpy-style linear interpolation."""
        from rag_platform.evaluators import EvaluationEngine

        engine = EvaluationEngine()
        scores = [0.1, 0.3, 0.5, 0.7, 0.9]
        traces = []
        for i, s in enumerate(scores):
            trace = RagTrace(trace_id=f"tr{i}", run_id="r1", test_case_id=f"t{i}", question="q", latency_ms=100)
            metrics = [
                MetricResult(
                    metric_name="faithfulness",
                    metric_family=MetricFamily.GENERATION,
                    score=s,
                ),
            ]
            traces.append((trace, metrics))

        summary = engine.aggregate_run(traces)
        ms = summary.metrics["faithfulness"]

        # For sorted [0.1, 0.3, 0.5, 0.7, 0.9], n=5:
        # p50: rank = 0.5 * 4 = 2.0 → sorted[2] = 0.5
        assert ms.p50 == pytest.approx(0.5, abs=0.001)
        # p95: rank = 0.95 * 4 = 3.8 → 0.7 + 0.8*(0.9-0.7) = 0.7 + 0.16 = 0.86
        assert ms.p95 == pytest.approx(0.86, abs=0.001)

    def test_single_value_percentile(self):
        """Single value should return that value for all percentiles."""
        from rag_platform.evaluators import EvaluationEngine

        engine = EvaluationEngine()
        trace = RagTrace(trace_id="tr0", run_id="r1", test_case_id="t0", question="q", latency_ms=100)
        metrics = [
            MetricResult(
                metric_name="faithfulness",
                metric_family=MetricFamily.GENERATION,
                score=0.75,
            ),
        ]

        summary = engine.aggregate_run([(trace, metrics)])
        ms = summary.metrics["faithfulness"]
        assert ms.p50 == pytest.approx(0.75, abs=0.001)
        assert ms.p95 == pytest.approx(0.75, abs=0.001)


# ─── P2: PostgreSQL Dependency ──────────────────────────────────────────────


class TestPostgresDependency:
    """Verify psycopg is declared in pyproject.toml."""

    def test_psycopg_in_pyproject(self):
        """psycopg must be in optional dependencies."""
        import tomllib
        from pathlib import Path

        pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)

        optional_deps = data.get("project", {}).get("optional-dependencies", {})
        postgres_deps = optional_deps.get("postgres", [])
        assert any("psycopg" in dep for dep in postgres_deps), (
            f"psycopg must be in [project.optional-dependencies.postgres], got: {postgres_deps}"
        )
