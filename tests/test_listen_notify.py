"""Comprehensive distributed worker LISTEN/NOTIFY test suite.

Verifies:
1. Queued run is committed before NOTIFY
2. Notification is published without hijacking caller transactions
3. Listener receives notification
4. Listener safely wakes asyncio worker across thread boundary via loop.call_soon_threadsafe
5. Worker fetches and claims the run
6. Duplicate notification does not duplicate execution
7. Missed notification is recovered through adaptive polling fallback
8. Listener disconnect and reconnect works
9. Listener shutdown does not hang the application
10. Multiple workers can listen safely
"""

import asyncio
from datetime import datetime, timezone
import threading
import time
from typing import Any
import uuid

import pytest
from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from rag_platform.db import Base, DatasetRow, ProjectRow, RunRow
from rag_platform.models import RunStatus
from rag_platform.worker import DurableRunWorker, WorkerNotificationBus


def create_test_run_row(run_id: str, proj_id: str = "proj_notify_test", ds_id: str = "ds_notify_test") -> RunRow:
    return RunRow(
        id=run_id,
        project_id=proj_id,
        dataset_id=ds_id,
        system_version="v1",
        dataset_checksum="a" * 64,
        rag_version="rag_v1",
        model_config_hash="m" * 64,
        prompt_hash="p" * 64,
        evaluator_version="2.0.0",
        experiment_hash="e" * 64,
        manifest_hash="b" * 64,
        policy_id="prod-default",
        status=RunStatus.QUEUED.value,
        options_json='{"mock_mode": "PERFECT", "adapter_type": "synthetic"}',
    )


@pytest.fixture
def memory_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = lambda: Session(engine)

    # Seed test project and dataset
    with session_factory() as sess:
        proj = ProjectRow(id="proj_notify_test", name="Notify Test Project")
        sess.add(proj)
        ds = DatasetRow(
            id="ds_notify_test",
            project_id=proj.id,
            name="Notify Dataset",
            version="1.0",
            status="PUBLISHED",
            checksum_sha256="a" * 64,
        )
        sess.add(ds)
        sess.commit()

    return engine, session_factory


class TestListenNotifyThreadSafetyAndDistributedWorker:

    @pytest.mark.asyncio
    async def test_queued_run_committed_notify_and_worker_fetches(self, memory_db):
        """1, 2, 5: Queued run is committed, notification dispatched, and worker fetches/claims it."""
        engine, session_factory = memory_db
        run_id = f"run_{uuid.uuid4().hex[:8]}"

        notify_event = WorkerNotificationBus.subscribe()
        try:
            with session_factory() as sess:
                # 1. Commit queued run to database
                run = create_test_run_row(run_id)
                sess.add(run)
                sess.commit()

                # 2. Dispatch notification (helper does NOT hijack caller's session)
                WorkerNotificationBus.notify_new_run(run_id, db_session=sess)

            # 4. Listener safely wakes the worker's asyncio event
            await asyncio.wait_for(notify_event.wait(), timeout=1.0)
            assert notify_event.is_set()

            # 5. Worker fetches and claims the run
            with session_factory() as sess:
                claimed = DurableRunWorker.claim_next_run(sess, worker_id="worker_alpha")
                assert claimed is not None
                assert claimed.id == run_id
                assert claimed.status == RunStatus.RUNNING.value
                assert claimed.worker_id == "worker_alpha"
                assert claimed.lease_id is not None
        finally:
            WorkerNotificationBus.unsubscribe(notify_event)

    def test_notify_does_not_commit_caller_unrelated_transaction(self, memory_db):
        """Invariant: notify_new_run must NOT unexpectedly call db_session.commit()."""
        engine, session_factory = memory_db
        with session_factory() as sess:
            # Create a dirty uncommitted record
            uncommitted_proj = ProjectRow(id="proj_dirty_test", name="Dirty Project")
            sess.add(uncommitted_proj)
            assert uncommitted_proj in sess.new

            # Calling notify_new_run must not commit this record
            WorkerNotificationBus.notify_new_run("dummy_run_id", db_session=sess)

            # Record must STILL be uncommitted in the session
            assert uncommitted_proj in sess.new
            sess.rollback()

        with session_factory() as sess:
            # Verify record was rolled back and never committed by notify_new_run
            assert sess.get(ProjectRow, "proj_dirty_test") is None

    @pytest.mark.asyncio
    async def test_cross_thread_listener_safely_wakes_asyncio_worker(self):
        """3, 4: Background OS listener thread wakes worker loop via loop.call_soon_threadsafe."""
        loop = asyncio.get_running_loop()
        worker_event = WorkerNotificationBus.subscribe(loop=loop)
        call_soon_called = threading.Event()

        # Intercept call_soon_threadsafe to verify it is explicitly called across the thread boundary
        orig_call_soon = loop.call_soon_threadsafe

        def _instrumented_call_soon(callback, *args):
            call_soon_called.set()
            return orig_call_soon(callback, *args)

        loop.call_soon_threadsafe = _instrumented_call_soon

        try:
            def _background_os_listener():
                # OS background thread executes notification
                WorkerNotificationBus.notify_new_run("run_cross_thread_payload")

            bg_thread = threading.Thread(target=_background_os_listener, name="test-bg-pg-listener")
            bg_thread.start()
            bg_thread.join(timeout=2.0)

            # Await worker wakeup in asyncio event loop
            await asyncio.wait_for(worker_event.wait(), timeout=1.0)
            assert worker_event.is_set()
            assert call_soon_called.is_set(), "loop.call_soon_threadsafe must be invoked to bridge OS thread"
        finally:
            loop.call_soon_threadsafe = orig_call_soon
            WorkerNotificationBus.unsubscribe(worker_event)

    @pytest.mark.asyncio
    async def test_multiple_workers_listen_safely_and_clean_prune(self):
        """10: Multiple workers listen safely, unsubscribe cleanly, and closed loops are pruned."""
        ev1 = WorkerNotificationBus.subscribe()
        ev2 = WorkerNotificationBus.subscribe()
        ev3 = WorkerNotificationBus.subscribe()

        assert not ev1.is_set() and not ev2.is_set() and not ev3.is_set()

        WorkerNotificationBus.notify_new_run("broadcast_run")

        await asyncio.sleep(0.01)
        assert ev1.is_set()
        assert ev2.is_set()
        assert ev3.is_set()

        # Unsubscribe ev2
        WorkerNotificationBus.unsubscribe(ev2)
        ev1.clear()
        ev2.clear()
        ev3.clear()

        WorkerNotificationBus.notify_new_run("broadcast_run_2")
        await asyncio.sleep(0.01)
        assert ev1.is_set()
        assert not ev2.is_set(), "Unsubscribed worker must not receive notification"
        assert ev3.is_set()

        WorkerNotificationBus.unsubscribe(ev1)
        WorkerNotificationBus.unsubscribe(ev3)

    @pytest.mark.asyncio
    async def test_duplicate_notification_does_not_duplicate_execution(self, memory_db):
        """6: Duplicate notifications do not cause duplicate execution across concurrent workers."""
        engine, session_factory = memory_db
        run_id = f"run_dup_{uuid.uuid4().hex[:8]}"

        with session_factory() as sess:
            run = create_test_run_row(run_id)
            sess.add(run)
            sess.commit()

        # Fire duplicate notifications
        WorkerNotificationBus.notify_new_run(run_id)
        WorkerNotificationBus.notify_new_run(run_id)

        # Worker 1 and Worker 2 concurrently attempt to claim
        with session_factory() as sess1, session_factory() as sess2:
            claim1 = DurableRunWorker.claim_next_run(sess1, worker_id="worker_1")
            claim2 = DurableRunWorker.claim_next_run(sess2, worker_id="worker_2")

        # Exactly one worker wins; the other receives None
        claimed = [c for c in (claim1, claim2) if c is not None]
        assert len(claimed) == 1
        assert claimed[0].id == run_id
        assert claimed[0].status == RunStatus.RUNNING.value

    @pytest.mark.asyncio
    async def test_missed_notification_recovered_through_adaptive_polling(self, memory_db):
        """7: Missed notification is safely recovered through background polling loop."""
        engine, session_factory = memory_db
        run_id = f"run_missed_{uuid.uuid4().hex[:8]}"

        with session_factory() as sess:
            run = create_test_run_row(run_id)
            sess.add(run)
            sess.commit()

        # NOTE: We do NOT call notify_new_run! The notification is completely dropped.
        # The adaptive polling loop must discover and process the run automatically.
        stop_event = asyncio.Event()
        worker_task = asyncio.create_task(
            DurableRunWorker.run_worker_loop(
                stop_event=stop_event,
                min_interval=0.02,
                max_interval=0.05,
                stale_check_interval=60.0,
                session_factory=session_factory,
            )
        )

        # Wait until the run is claimed or processed by the polling fallback
        for _ in range(50):
            with session_factory() as sess:
                check_run = sess.get(RunRow, run_id)
                if check_run and check_run.status in (RunStatus.COMPLETED.value, RunStatus.RUNNING.value):
                    break
            await asyncio.sleep(0.05)

        stop_event.set()
        await asyncio.wait_for(worker_task, timeout=2.0)

        with session_factory() as sess:
            final_run = sess.get(RunRow, run_id)
            assert final_run.status in (RunStatus.COMPLETED.value, RunStatus.RUNNING.value)

    def test_listener_shutdown_does_not_hang(self):
        """9: Listener background thread shuts down promptly when stop event is signaled."""
        stop_event = threading.Event()

        class MockPgDialect:
            name = "postgresql"

        class MockEngine:
            dialect = MockPgDialect()

            def raw_connection(self):
                class MockConn:
                    autocommit = True
                    def cursor(self):
                        class MockCursor:
                            def execute(self, stmt): pass
                        return MockCursor()
                    def notifies(self, timeout=0.5):
                        # Generator that blocks for up to timeout, then yields nothing
                        stop_event.wait(timeout)
                        return []
                    def close(self): pass
                return MockConn()

        thread = WorkerNotificationBus.start_postgres_listener(MockEngine(), stop_event=stop_event)
        assert thread is not None
        assert thread.is_alive()

        # Signal stop
        start_t = time.monotonic()
        stop_event.set()
        thread.join(timeout=2.0)
        duration = time.monotonic() - start_t

        assert not thread.is_alive(), "Listener thread must terminate cleanly without hanging"
        assert duration < 1.5, f"Listener shutdown took too long ({duration:.2f}s)"

    def test_listener_reconnects_on_connection_error(self):
        """8: Listener recovers and reconnects after a connection error."""
        stop_event = threading.Event()
        connection_attempts = 0
        notified = threading.Event()

        class MockPgDialect:
            name = "postgresql"

        class MockEngine:
            dialect = MockPgDialect()

            def raw_connection(self):
                nonlocal connection_attempts
                connection_attempts += 1
                if connection_attempts == 1:
                    # First connection fails
                    raise ConnectionResetError("Simulated connection reset")

                # Second connection succeeds
                class MockNotification:
                    payload = "reconnected_run_payload"

                class MockConn:
                    autocommit = True
                    def cursor(self):
                        class MockCursor:
                            def execute(self, stmt): pass
                        return MockCursor()
                    def notifies(self, timeout=0.5):
                        if not notified.is_set():
                            notified.set()
                            return [MockNotification()]
                        stop_event.wait(timeout)
                        return []
                    def close(self): pass
                return MockConn()

        thread = WorkerNotificationBus.start_postgres_listener(MockEngine(), stop_event=stop_event)
        try:
            # Wait for reconnect and notification
            assert notified.wait(timeout=3.0), "Listener should reconnect and process notifications"
            assert connection_attempts >= 2
        finally:
            stop_event.set()
            thread.join(timeout=2.0)
