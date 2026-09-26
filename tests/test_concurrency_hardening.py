"""Real concurrency, race-condition, and lease-loss tests for Reliab.

Validates:
1. Concurrent idempotency: multiple simultaneous requests with the same (project_id, idempotency_key)
   create exactly one run, return the same run ID to all callers, and never raise an unhandled IntegrityError.
2. Stale runner recovery race: atomic conditional updates prevent a live worker that renewed its lease
   from being marked FAILED by a concurrent recovery scan.
3. Lease loss and obsolete worker isolation: an obsolete worker that finishes late after its lease expired
   or was recovered cannot overwrite the recovered FAILED state.
4. Evaluation exception handling: worker properly consumes and propagates task exceptions without leaking asyncio tasks.
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, delete, select, update
from sqlalchemy.orm import Session

from rag_platform.db import (
    Base,
    DatabaseRepo,
    DatasetRow,
    DatasetStatus,
    ProjectRow,
    RunRow,
    TestCaseRow,
)

TestCaseRow.__test__ = False
from rag_platform.models import (
    Answerability,
    DocumentReference,
    RunConfig,
    RunOptions,
    RunProvenance,
    RunStatus,
    TestCase,
)
from rag_platform.worker import DurableRunWorker


@pytest.fixture
def shared_db(tmp_path):
    """Thread-safe multi-connection SQLite database with WAL mode."""
    db_file = tmp_path / "concurrency_test.db"
    engine = create_engine(
        f"sqlite:///{db_file}",
        connect_args={"timeout": 30.0},
    )
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA journal_mode=WAL;")
        conn.exec_driver_sql("PRAGMA busy_timeout=30000;")
    Base.metadata.create_all(bind=engine)

    def session_factory() -> Session:
        return Session(bind=engine)

    # Seed baseline project and published dataset
    with session_factory() as sess:
        proj = ProjectRow(id="proj_concurrency", name="Concurrency Project")
        ds = DatasetRow(
            id="ds_concurrency",
            project_id="proj_concurrency",
            name="Concurrency Dataset",
            version="1.0.0",
            status=DatasetStatus.PUBLISHED.value,
            checksum_sha256="abc123checksum",
        )
        sess.add_all([proj, ds])
        sess.commit()

    try:
        yield engine, session_factory
    finally:
        engine.dispose()


def _make_config() -> RunConfig:
    return RunConfig(
        project_id="proj_concurrency",
        dataset_id="ds_concurrency",
        dataset_version="1.0.0",
        system_version="v0.2.0-test",
        policy_id="prod-default",
        options=RunOptions(),
    )


def _make_provenance() -> RunProvenance:
    return RunProvenance(
        dataset_checksum="abc123checksum",
        rag_version="v0.2.0-test",
        evaluator_version="0.2.0rc1",
    )


class TestConcurrentIdempotency:
    """A. Idempotency concurrency tests."""

    def test_concurrent_idempotency_creates_single_run(self, shared_db):
        """Launch multiple concurrent transactions with the same idempotency key.
        Verify exactly ONE run is created, all callers get the same run ID, and no IntegrityError occurs.
        """
        engine, session_factory = shared_db
        idempotency_key = "idem_race_key_999"
        config = _make_config()
        provenance = _make_provenance()

        num_threads = 8
        results: list[str] = []
        errors: list[Exception] = []

        def worker_task(thread_id: int):
            try:
                with session_factory() as sess:
                    repo = DatabaseRepo(sess)
                    run = repo.create_run(
                        config=config,
                        provenance=provenance,
                        initial_status=RunStatus.QUEUED,
                        idempotency_key=idempotency_key,
                    )
                    sess.commit()
                    return run.id
            except Exception as e:
                errors.append(e)
                raise

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_task, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        # No errors raised
        assert len(errors) == 0, f"Errors occurred during concurrent creation: {errors}"
        assert len(results) == num_threads

        # Every caller received the EXACT SAME run ID
        first_id = results[0]
        assert all(r == first_id for r in results), f"Got divergent run IDs: {set(results)}"

        # Exactly ONE run exists in the database with this idempotency key
        with session_factory() as sess:
            matching_runs = sess.scalars(
                select(RunRow).where(
                    RunRow.project_id == "proj_concurrency",
                    RunRow.idempotency_key == idempotency_key,
                )
            ).all()
            assert len(matching_runs) == 1
            assert matching_runs[0].id == first_id

    def test_null_idempotency_key_permits_multiple_runs(self, shared_db):
        """When idempotency_key is None, multiple distinct runs can be created."""
        engine, session_factory = shared_db
        config = _make_config()
        provenance = _make_provenance()

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            r1 = repo.create_run(config, provenance, idempotency_key=None)
            sess.commit()
            r2 = repo.create_run(config, provenance, idempotency_key=None)
            sess.commit()

            assert r1.id != r2.id

    def test_concurrent_idempotency_api_endpoint(self, shared_db):
        """Concurrently call POST /v1/runs with Idempotency-Key.
        Verify callers get HTTP 200/202, same run ID, and no unhandled 500 IntegrityError.
        """
        engine, session_factory = shared_db
        from fastapi.testclient import TestClient

        from rag_platform.server import app, get_db

        def override_db():
            with session_factory() as sess:
                yield sess

        app.dependency_overrides[get_db] = override_db
        client = TestClient(app)

        idempotency_key = "api_idem_race_key_777"
        payload = {
            "project_id": "proj_concurrency",
            "dataset_id": "ds_concurrency",
            "system_version": "v0.2.0-test",
            "async_exec": True,
        }

        results: list[dict] = []
        errors: list[Exception] = []

        def call_api(thread_idx: int):
            try:
                resp = client.post(
                    "/v1/runs",
                    json=payload,
                    headers={"Idempotency-Key": idempotency_key},
                )
                return resp.status_code, resp.json()
            except Exception as e:
                errors.append(e)
                raise

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(call_api, i) for i in range(5)]
                for f in concurrent.futures.as_completed(futures):
                    code, body = f.result()
                    assert code in (200, 202)
                    results.append(body)

            assert len(errors) == 0
            assert len(results) == 5
            first_run_id = results[0]["run_id"]
            assert all(r["run_id"] == first_run_id for r in results)
        finally:
            app.dependency_overrides.pop(get_db, None)


class TestStaleRecoveryAtomicRace:
    """B. Stale recovery race condition tests."""

    def test_stale_recovery_skips_live_worker_that_renewed_lease(self, shared_db):
        """Deterministically reproduce race between stale recovery and worker renewal.

        Sequence of events:
        1. Recovery selects stale RUNNING row.
        2. Recovery pauses right before executing the conditional UPDATE.
        3. Worker concurrently renews heartbeat and extends lease in the DB, then commits.
        4. Recovery resumes and attempts conditional UPDATE.
        5. Database conditional update matches 0 rows (rowcount == 0).
        6. Run is NOT marked FAILED and remains RUNNING.
        """
        import threading

        engine, session_factory = shared_db
        config = _make_config()
        provenance = _make_provenance()

        now = datetime.now(timezone.utc)
        expired_time = now - timedelta(seconds=60.0)
        past_heartbeat = now - timedelta(seconds=700.0)

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.started_at = past_heartbeat
            run.heartbeat_at = past_heartbeat
            run.worker_id = "worker_live_race"
            run.lease_id = "lease_live_race"
            run.lease_expires_at = expired_time
            sess.commit()
            run_id = run.id

        recovery_paused_before_update = threading.Event()
        worker_renewed_lease = threading.Event()
        recovered_ids: list[str] = []
        recovery_errors: list[Exception] = []

        def recovery_worker():
            try:
                with session_factory() as recovery_sess:
                    orig_execute = recovery_sess.execute

                    def hooked_execute(statement, *args, **kwargs):
                        # Detect the conditional UPDATE statement
                        if hasattr(statement, "is_update") and statement.is_update:
                            # Recovery has finished SELECT and is about to execute conditional UPDATE
                            recovery_paused_before_update.set()
                            # Pause until worker has renewed its lease
                            if not worker_renewed_lease.wait(timeout=10.0):
                                raise TimeoutError("Timed out waiting for worker to renew lease")
                        return orig_execute(statement, *args, **kwargs)

                    recovery_sess.execute = hooked_execute  # type: ignore[method-assign]
                    result = DurableRunWorker.recover_stale_runs(recovery_sess, max_age_seconds=600.0)
                    recovered_ids.extend(result)
            except Exception as e:
                recovery_errors.append(e)

        rec_thread = threading.Thread(target=recovery_worker, daemon=True)
        rec_thread.start()

        # Deterministically wait for recovery to select the stale row and pause
        assert recovery_paused_before_update.wait(timeout=10.0), "Recovery did not reach update pause"

        # Worker concurrently renews its heartbeat and extends its lease
        with session_factory() as worker_sess:
            renew_time = datetime.now(timezone.utc)
            worker_sess.execute(
                update(RunRow)
                .where(RunRow.id == run_id, RunRow.lease_id == "lease_live_race")
                .values(
                    heartbeat_at=renew_time,
                    lease_expires_at=renew_time + timedelta(seconds=120.0),
                )
            )
            worker_sess.commit()

        # Signal recovery thread to proceed with its conditional update
        worker_renewed_lease.set()
        rec_thread.join(timeout=10.0)

        assert not recovery_errors, f"Recovery thread error: {recovery_errors}"
        # Because the conditional update checked lease_expires_at and heartbeat_at, rowcount was 0
        assert run_id not in recovered_ids
        assert len(recovered_ids) == 0

        # Verify the run is STILL RUNNING in the database
        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.RUNNING.value
            assert current.failure_type is None
            hb = current.heartbeat_at
            if hb is not None and hb.tzinfo is None:
                hb = hb.replace(tzinfo=timezone.utc)
            assert hb is not None
            assert hb >= renew_time - timedelta(seconds=1)

    def test_stale_recovery_marks_actually_dead_worker_failed(self, shared_db):
        """Create a RUNNING run with an expired lease that is never renewed.
        Verify recovery atomically marks it FAILED.
        """
        engine, session_factory = shared_db
        config = _make_config()
        provenance = _make_provenance()

        now = datetime.now(timezone.utc)
        expired_time = now - timedelta(seconds=60.0)
        past_heartbeat = now - timedelta(seconds=800.0)

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.started_at = past_heartbeat
            run.heartbeat_at = past_heartbeat
            run.worker_id = "worker_dead"
            run.lease_id = "lease_dead"
            run.lease_expires_at = expired_time
            sess.commit()
            run_id = run.id

        with session_factory() as recovery_sess:
            recovered = DurableRunWorker.recover_stale_runs(recovery_sess, max_age_seconds=600.0)
            assert run_id in recovered

        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.FAILED.value
            assert current.failure_type == "STALE_RUNNER_RECOVERY"


class TestLeaseLossObsoleteWorkerIsolation:
    """C. Lease loss and obsolete worker isolation tests."""

    @pytest.mark.asyncio
    async def test_obsolete_worker_cannot_overwrite_recovered_state(self, shared_db):
        """Worker claims a run.
        Recovery takes over and marks the run FAILED.
        The obsolete worker finishes evaluation late.
        Verify the obsolete worker's final DB writes do NOT overwrite the recovered FAILED state.
        """
        engine, session_factory = shared_db
        from rag_platform.server import CreateRunReq, _execute_evaluation_run

        config = _make_config()
        provenance = _make_provenance()

        # Seed test cases into dataset
        test_case = TestCase(
            id="tc_concurrency_1",
            question="What is the latency limit?",
            expected_answer="100ms",
            expected_facts=["100ms"],
            relevant_documents=[DocumentReference(document_id="doc1", chunk_id="chk1")],
            answerability=Answerability.ANSWERABLE,
        )

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.worker_id = "worker_obsolete"
            run.lease_id = "lease_obsolete_1"
            run.lease_expires_at = datetime.now(timezone.utc) + timedelta(seconds=30.0)
            sess.commit()
            run_id = run.id

        # Simulate recovery marking the run FAILED while worker is processing
        with session_factory() as recovery_sess:
            recovery_sess.execute(
                update(RunRow)
                .where(RunRow.id == run_id)
                .values(
                    status=RunStatus.FAILED.value,
                    finished_at=datetime.now(timezone.utc),
                    failure_type="STALE_RUNNER_RECOVERY",
                    failure_reason="Recovered by watchdog",
                )
            )
            recovery_sess.commit()

        # Now the obsolete worker attempts to finalize the run with expected_lease_id="lease_obsolete_1"
        req = CreateRunReq(
            project_id="proj_concurrency",
            dataset_id="ds_concurrency",
            system_version="v0.2.0-test",
            adapter_type="synthetic",
        )

        with session_factory() as worker_sess:
            await _execute_evaluation_run(
                run_id=run_id,
                req=req,
                config=config,
                ds_cases=[test_case],
                db_session=worker_sess,
                expected_lease_id="lease_obsolete_1",
            )

        # Verify the database status remains FAILED and was NOT overwritten to COMPLETED!
        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.FAILED.value
            assert current.failure_type == "STALE_RUNNER_RECOVERY"
            assert current.failure_reason == "Recovered by watchdog"

    @pytest.mark.asyncio
    async def test_worker_evaluation_exception_consumed_and_persisted(self, shared_db):
        """Verify that an exception inside process_run_by_id is consumed and the run is marked FAILED."""
        engine, session_factory = shared_db
        config = _make_config()
        provenance = _make_provenance()

        # Create run targeting a non-existent dataset to trigger exception
        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.dataset_id = "ds_non_existent"
            run.lease_id = "lease_err_test"
            sess.commit()
            run_id = run.id

        # Execute worker
        with session_factory() as worker_sess:
            result = await DurableRunWorker.process_run_by_id(run_id, db_session=worker_sess)
            assert result is False

        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.FAILED.value
            assert current.failure_type == "DATASET_NOT_FOUND"


@pytest.fixture
def postgres_shared_db():
    """Live PostgreSQL database connection fixture for real multi-connection concurrency testing.

    Activated when POSTGRES_TEST_URL is configured in the environment (e.g. in CI or local Docker).
    Skipped gracefully when no PostgreSQL database is available.
    """
    import os

    pg_url = os.getenv("POSTGRES_TEST_URL")
    if not pg_url:
        pytest.skip("POSTGRES_TEST_URL not set; skipping live PostgreSQL concurrency suite")

    engine = create_engine(pg_url, pool_pre_ping=True)
    Base.metadata.create_all(bind=engine)

    def session_factory() -> Session:
        return Session(bind=engine)

    # Seed baseline project and dataset for concurrency tests
    with session_factory() as sess:
        sess.execute(delete(RunRow))
        sess.execute(delete(TestCaseRow))
        sess.execute(delete(DatasetRow))
        sess.execute(delete(ProjectRow))
        sess.commit()

        proj = ProjectRow(id="proj_pg_concurrency", name="PG Concurrency Project")
        ds = DatasetRow(
            id="ds_pg_concurrency",
            project_id="proj_pg_concurrency",
            name="PG Concurrency Benchmark",
            version="1.0.0",
            status=DatasetStatus.PUBLISHED.value,
            checksum_sha256="fake_pg_checksum",
        )
        sess.add(proj)
        sess.add(ds)
        sess.commit()

    yield engine, session_factory

    with session_factory() as sess:
        sess.execute(delete(RunRow))
        sess.commit()
    engine.dispose()


class TestPostgresConcurrencyHardening:
    """D. Real multi-threaded concurrency and race-condition tests against PostgreSQL."""

    def test_pg_concurrent_idempotency_same_key_multi_threaded(self, postgres_shared_db):
        """Concurrently create runs with the same (project_id, idempotency_key) on PostgreSQL.

        Verifies that:
        1. Exactly 1 row is committed to PostgreSQL.
        2. All concurrent threads receive the identical run ID.
        3. No raw IntegrityError escapes to callers.
        """
        engine, session_factory = postgres_shared_db
        idempotency_key = "pg_race_key_unique_888"

        results: list[str] = []
        errors: list[Exception] = []

        def worker_attempt(thread_idx: int):
            try:
                with session_factory() as sess:
                    repo = DatabaseRepo(sess)
                    config = RunConfig(
                        project_id="proj_pg_concurrency",
                        dataset_id="ds_pg_concurrency",
                        dataset_version="1.0.0",
                        system_version="v0.2.0-pg",
                    )
                    provenance = _make_provenance()
                    run = repo.create_run(config, provenance, idempotency_key=idempotency_key)
                    sess.commit()
                    return run.id
            except Exception as e:
                errors.append(e)
                raise

        num_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_attempt, i) for i in range(num_threads)]
            for f in concurrent.futures.as_completed(futures):
                results.append(f.result())

        assert len(errors) == 0, f"Encountered unexpected errors during concurrent creation: {errors}"
        assert len(results) == num_threads
        first_id = results[0]
        assert all(r_id == first_id for r_id in results), "All threads must receive the identical run ID"

        with session_factory() as verify_sess:
            stmt = select(RunRow).where(
                RunRow.project_id == "proj_pg_concurrency",
                RunRow.idempotency_key == idempotency_key,
            )
            matching_runs = verify_sess.scalars(stmt).all()
            assert len(matching_runs) == 1
            assert matching_runs[0].id == first_id

    def test_pg_stale_recovery_race_with_heartbeat(self, postgres_shared_db):
        """Deterministically reproduce stale recovery race on PostgreSQL using threading.Event.

        Verifies recovery's conditional UPDATE affects 0 rows when worker renews heartbeat.
        """
        import threading

        engine, session_factory = postgres_shared_db
        config = RunConfig(
            project_id="proj_pg_concurrency",
            dataset_id="ds_pg_concurrency",
            dataset_version="1.0.0",
            system_version="v0.2.0-pg",
        )
        provenance = _make_provenance()

        now = datetime.now(timezone.utc)
        expired_time = now - timedelta(seconds=60.0)
        past_heartbeat = now - timedelta(seconds=700.0)

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.started_at = past_heartbeat
            run.heartbeat_at = past_heartbeat
            run.worker_id = "pg_worker_live"
            run.lease_id = "pg_lease_live"
            run.lease_expires_at = expired_time
            sess.commit()
            run_id = run.id

        recovery_paused = threading.Event()
        worker_renewed = threading.Event()
        recovered_ids: list[str] = []
        recovery_errors: list[Exception] = []

        def recovery_worker():
            try:
                with session_factory() as recovery_sess:
                    orig_execute = recovery_sess.execute

                    def hooked_execute(statement, *args, **kwargs):
                        if hasattr(statement, "is_update") and statement.is_update:
                            recovery_paused.set()
                            if not worker_renewed.wait(timeout=10.0):
                                raise TimeoutError("Timed out waiting for worker renewal")
                        return orig_execute(statement, *args, **kwargs)

                    recovery_sess.execute = hooked_execute  # type: ignore[method-assign]
                    result = DurableRunWorker.recover_stale_runs(recovery_sess, max_age_seconds=600.0)
                    recovered_ids.extend(result)
            except Exception as e:
                recovery_errors.append(e)

        rec_thread = threading.Thread(target=recovery_worker, daemon=True)
        rec_thread.start()

        assert recovery_paused.wait(timeout=10.0), "Recovery did not reach update pause on PostgreSQL"

        with session_factory() as worker_sess:
            renew_time = datetime.now(timezone.utc)
            worker_sess.execute(
                update(RunRow)
                .where(RunRow.id == run_id, RunRow.lease_id == "pg_lease_live")
                .values(
                    heartbeat_at=renew_time,
                    lease_expires_at=renew_time + timedelta(seconds=120.0),
                )
            )
            worker_sess.commit()

        worker_renewed.set()
        rec_thread.join(timeout=10.0)

        assert not recovery_errors, f"PostgreSQL recovery error: {recovery_errors}"
        assert run_id not in recovered_ids
        assert len(recovered_ids) == 0

        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.RUNNING.value
            assert current.failure_type is None

    @pytest.mark.asyncio
    async def test_pg_obsolete_worker_cannot_overwrite_recovered_state(self, postgres_shared_db):
        """Verify on PostgreSQL that an obsolete worker cannot resurrect a recovered run to COMPLETED."""
        engine, session_factory = postgres_shared_db
        from rag_platform.server import CreateRunReq, _execute_evaluation_run

        config = RunConfig(
            project_id="proj_pg_concurrency",
            dataset_id="ds_pg_concurrency",
            dataset_version="1.0.0",
            system_version="v0.2.0-pg",
        )
        provenance = _make_provenance()

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.worker_id = "pg_worker_obsolete"
            run.lease_id = "pg_lease_obsolete"
            sess.commit()
            run_id = run.id

        test_case = TestCase(
            id="pg_tc_1",
            question="What is PostgreSQL?",
            expected_answer="An open-source relational database.",
            relevant_documents=[DocumentReference(document_id="pg_doc", chunk_id="c1")],
            answerability=Answerability.ANSWERABLE,
        )

        with session_factory() as recovery_sess:
            recovery_sess.execute(
                update(RunRow)
                .where(RunRow.id == run_id)
                .values(
                    status=RunStatus.FAILED.value,
                    finished_at=datetime.now(timezone.utc),
                    failure_type="STALE_RUNNER_RECOVERY",
                    failure_reason="Recovered by watchdog",
                )
            )
            recovery_sess.commit()

        req = CreateRunReq(
            project_id="proj_pg_concurrency",
            dataset_id="ds_pg_concurrency",
            system_version="v0.2.0-pg",
            adapter_type="synthetic",
        )

        with session_factory() as worker_sess:
            await _execute_evaluation_run(
                run_id=run_id,
                req=req,
                config=config,
                ds_cases=[test_case],
                db_session=worker_sess,
                expected_lease_id="pg_lease_obsolete",
            )

        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status == RunStatus.FAILED.value
            assert current.failure_type == "STALE_RUNNER_RECOVERY"

    @pytest.mark.asyncio
    async def test_pg_server_execute_evaluation_run_lease_invalidation(self, postgres_shared_db):
        """Simulate lease invalidation during evaluation on PostgreSQL.

        When expected_lease_id does not match the active DB lease, _execute_evaluation_run must
        immediately abort without modifying status or committing traces.
        """
        engine, session_factory = postgres_shared_db
        from rag_platform.server import CreateRunReq, _execute_evaluation_run

        config = RunConfig(
            project_id="proj_pg_concurrency",
            dataset_id="ds_pg_concurrency",
            dataset_version="1.0.0",
            system_version="v0.2.0-pg",
        )
        provenance = _make_provenance()

        with session_factory() as sess:
            repo = DatabaseRepo(sess)
            run = repo.create_run(config, provenance, initial_status=RunStatus.RUNNING)
            run.worker_id = "pg_worker_real"
            run.lease_id = "pg_lease_real"
            sess.commit()
            run_id = run.id

        test_case = TestCase(
            id="pg_tc_inv",
            question="What is lease invalidation?",
            expected_answer="Safety barrier.",
            relevant_documents=[DocumentReference(document_id="pg_doc", chunk_id="c1")],
            answerability=Answerability.ANSWERABLE,
        )

        req = CreateRunReq(
            project_id="proj_pg_concurrency",
            dataset_id="ds_pg_concurrency",
            system_version="v0.2.0-pg",
            adapter_type="synthetic",
        )

        # Worker calls _execute_evaluation_run with a stale/wrong lease id
        with session_factory() as worker_sess:
            await _execute_evaluation_run(
                run_id=run_id,
                req=req,
                config=config,
                ds_cases=[test_case],
                db_session=worker_sess,
                expected_lease_id="pg_stale_lease_999",
            )

        # Run in DB should not be COMPLETED
        with session_factory() as check_sess:
            current = check_sess.get(RunRow, run_id)
            assert current is not None
            assert current.status != RunStatus.COMPLETED.value
            assert current.lease_id == "pg_lease_real"

