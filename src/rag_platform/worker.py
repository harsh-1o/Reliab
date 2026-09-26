"""Durable database-backed execution worker for asynchronous evaluation runs.

Provides atomic run claiming, safe state transitions, stale runner recovery,
and survival across API process restarts without requiring heavyweight message queues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rag_platform.db import DatasetRow, RunRow, create_session
from rag_platform.models import (
    RunConfig,
    RunOptions,
    RunStatus,
)

logger = logging.getLogger("rag_platform.worker")


class WorkerNotificationBus:
    """Notification bus for waking worker instances on new run events.

    Combines in-process asyncio.Event broadcasting, thread-safe cross-thread scheduling,
    and PostgreSQL LISTEN/NOTIFY support to eliminate unnecessary polling
    load in high-concurrency deployments.
    """
    _subscribers: dict[asyncio.Event, asyncio.AbstractEventLoop | None] = {}
    _lock = threading.Lock()

    @classmethod
    def subscribe(cls, loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Event:
        """Register an asyncio worker subscriber, capturing its event loop for thread-safe wakeup."""
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
        event = asyncio.Event()
        with cls._lock:
            cls._subscribers[event] = loop
        return event

    @classmethod
    def unsubscribe(cls, event: asyncio.Event) -> None:
        """Unregister a subscriber and release references."""
        with cls._lock:
            cls._subscribers.pop(event, None)

    @classmethod
    def notify_new_run(cls, run_id: str, db_session: Session | None = None, engine: Any | None = None) -> None:
        """Broadcast notification of a new queued run.

        1. Wakes all local in-process asyncio subscribers using thread-safe loop scheduling
           (`loop.call_soon_threadsafe(event.set)`), safely bridging OS listener threads to worker event loops.
        2. If connected to PostgreSQL, broadcasts `NOTIFY rag_runs_channel, :run_id` on a standalone
           autocommit connection without hijacking or committing caller transactions.
        """
        with cls._lock:
            subscribers_snapshot = list(cls._subscribers.items())

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for ev, loop in subscribers_snapshot:
            if loop is not None:
                if loop.is_closed():
                    with cls._lock:
                        cls._subscribers.pop(ev, None)
                    continue
                try:
                    if current_loop is loop:
                        ev.set()
                    else:
                        loop.call_soon_threadsafe(ev.set)
                except RuntimeError:
                    with cls._lock:
                        cls._subscribers.pop(ev, None)
            else:
                try:
                    ev.set()
                except Exception:
                    with cls._lock:
                        cls._subscribers.pop(ev, None)

        target_engine = engine
        if target_engine is None and db_session is not None:
            try:
                target_engine = db_session.get_bind()
            except Exception:
                target_engine = None

        if target_engine is not None:
            try:
                dialect = getattr(getattr(target_engine, "dialect", None), "name", "")
                if dialect == "postgresql":
                    from sqlalchemy import text
                    with target_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                        conn.execute(text("NOTIFY rag_runs_channel, :run_id"), {"run_id": run_id})
            except Exception as exc:
                logger.debug("PostgreSQL NOTIFY failed or not supported: %s", exc)

    @classmethod
    def start_postgres_listener(cls, engine: Any, stop_event: threading.Event | None = None) -> threading.Thread | None:
        """Start a background daemon thread that executes LISTEN rag_runs_channel on PostgreSQL.

        When running against PostgreSQL, wakes up worker instances across processes/nodes
        with zero latency when any API node executes NOTIFY rag_runs_channel.
        Cross-thread wakeups are routed safely via loop.call_soon_threadsafe.
        If connected to SQLite, gracefully defers to the adaptive polling loop.
        """
        if not engine:
            return None
        dialect = getattr(getattr(engine, "dialect", None), "name", "")
        if dialect != "postgresql":
            logger.debug("Database dialect '%s' does not use PostgreSQL LISTEN/NOTIFY; using adaptive polling.", dialect)
            return None

        thread_stop = stop_event or threading.Event()

        def _listen_worker():
            channel = "rag_runs_channel"
            while not thread_stop.is_set():
                raw_conn = None
                try:
                    raw_conn = engine.raw_connection()
                    if hasattr(raw_conn, "autocommit"):
                        raw_conn.autocommit = True
                    elif hasattr(raw_conn, "set_isolation_level"):
                        raw_conn.set_isolation_level(0)

                    cursor = raw_conn.cursor()
                    cursor.execute(f"LISTEN {channel};")
                    logger.info("Active PostgreSQL LISTEN listener established on '%s'", channel)

                    while not thread_stop.is_set():
                        if hasattr(raw_conn, "notifies"):
                            notifies_attr = getattr(raw_conn, "notifies")
                            if callable(notifies_attr):
                                for n in raw_conn.notifies(timeout=0.5):
                                    cls.notify_new_run(getattr(n, "payload", "pg_notify"))
                                    if thread_stop.is_set():
                                        break
                            elif isinstance(notifies_attr, list):
                                import select
                                if select.select([raw_conn], [], [], 0.5) != ([], [], []):
                                    raw_conn.poll()
                                    while raw_conn.notifies:
                                        n = raw_conn.notifies.pop(0)
                                        cls.notify_new_run(getattr(n, "payload", "pg_notify"))
                        else:
                            if thread_stop.wait(0.5):
                                break
                except Exception as exc:
                    if not thread_stop.is_set():
                        logger.debug("PostgreSQL LISTEN loop error: %s (reconnecting in 1s)", exc)
                        thread_stop.wait(1.0)
                finally:
                    if raw_conn:
                        try:
                            raw_conn.close()
                        except Exception:
                            pass

        listener_thread = threading.Thread(target=_listen_worker, daemon=True, name="pg-listen-listener")
        listener_thread.start()
        return listener_thread


class DurableRunWorker:
    """Database-backed worker that polls, claims, and processes queued runs."""

    @classmethod
    def claim_next_run(
        cls,
        sess: Session,
        worker_id: str | None = None,
        lease_duration_seconds: float = 30.0,
    ) -> RunRow | None:
        """Atomically claim the oldest QUEUED run in the database with worker lease ownership."""
        stmt = (
            select(RunRow.id)
            .where(RunRow.status == RunStatus.QUEUED.value)
            .order_by(RunRow.created_at.asc())
            .limit(1)
        )
        candidate_id = sess.scalar(stmt)
        if not candidate_id:
            return None

        now = datetime.now(timezone.utc)
        w_id = worker_id or f"worker_{uuid.uuid4().hex[:8]}"
        lease_id = str(uuid.uuid4())
        lease_expires = now + timedelta(seconds=lease_duration_seconds)

        res = sess.execute(
            update(RunRow)
            .where(RunRow.id == candidate_id, RunRow.status == RunStatus.QUEUED.value)
            .values(
                status=RunStatus.RUNNING.value,
                started_at=now,
                heartbeat_at=now,
                worker_id=w_id,
                lease_id=lease_id,
                lease_expires_at=lease_expires,
            )
        )
        if res.rowcount == 0:
            sess.rollback()
            return None

        sess.commit()
        claimed = sess.get(RunRow, candidate_id)
        if claimed:
            logger.info("DurableRunWorker %s claimed run_id=%s (lease_id=%s)", w_id, claimed.id, lease_id)
        return claimed

    @classmethod
    def recover_stale_runs(
        cls,
        sess: Session,
        max_age_seconds: float = 600.0,
    ) -> list[str]:
        """Recover runs whose worker heartbeat is stale or lease has expired."""
        now = datetime.now(timezone.utc)
        stmt = select(RunRow).where(RunRow.status == RunStatus.RUNNING.value)
        running_runs = sess.scalars(stmt).all()
        recovered_ids: list[str] = []

        for run in running_runs:
            last_activity = run.heartbeat_at or run.started_at or run.created_at
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)
            age = (now - last_activity).total_seconds()

            is_lease_expired = run.lease_expires_at is not None and (
                (run.lease_expires_at.replace(tzinfo=timezone.utc) if run.lease_expires_at.tzinfo is None else run.lease_expires_at) < now
            )

            if age > max_age_seconds or is_lease_expired:
                run.status = RunStatus.FAILED.value
                run.finished_at = now
                run.failure_type = "STALE_RUNNER_RECOVERY"
                run.failure_reason = (
                    f"Run abandoned or worker died while running (last heartbeat was {int(age)}s ago > {int(max_age_seconds)}s limit)."
                )
                recovered_ids.append(run.id)
                logger.warning(
                    "Recovered stale run_id=%s: heartbeat age=%ds exceeded timeout=%ds (lease_expired=%s)",
                    run.id, int(age), int(max_age_seconds), is_lease_expired,
                )

        if recovered_ids:
            sess.commit()
        return recovered_ids

    @classmethod
    async def process_run_by_id(
        cls,
        run_id: str,
        db_session: Session | None = None,
    ) -> bool:
        """Execute a claimed run and abort promptly if its worker lease is lost."""
        owned_session = False
        sess = db_session if db_session is not None else create_session()
        if db_session is None:
            owned_session = True

        run = sess.get(RunRow, run_id)
        if not run:
            logger.error("Run %s not found in datastore", run_id)
            if owned_session:
                sess.close()
            return False

        active_lease_id = run.lease_id
        lease_lost_event = asyncio.Event()
        consecutive_hb_errors = 0

        async def _heartbeat_loop(target_run_id: str, target_lease_id: str | None, interval: float = 3.0):
            nonlocal consecutive_hb_errors
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        with create_session() as hb_sess:
                            now_hb = datetime.now(timezone.utc)
                            new_expiry = now_hb + timedelta(seconds=30.0)
                            stmt = update(RunRow).where(
                                RunRow.id == target_run_id,
                                RunRow.status == RunStatus.RUNNING.value,
                            )
                            if target_lease_id:
                                stmt = stmt.where(RunRow.lease_id == target_lease_id)
                            res = hb_sess.execute(stmt.values(heartbeat_at=now_hb, lease_expires_at=new_expiry))
                            hb_sess.commit()
                            if res.rowcount == 0:
                                logger.error(
                                    "Worker lease lost: run_id=%s lease_id=%s was invalidated or recovered.",
                                    target_run_id, target_lease_id,
                                )
                                lease_lost_event.set()
                                return
                            consecutive_hb_errors = 0
                    except Exception as hb_exc:
                        consecutive_hb_errors += 1
                        logger.warning(
                            "Heartbeat failure for run %s (%d consecutive): %s",
                            target_run_id, consecutive_hb_errors, hb_exc,
                        )
                        if consecutive_hb_errors >= 5:
                            logger.error("Run %s exceeded consecutive heartbeat errors; releasing lease.", target_run_id)
                            lease_lost_event.set()
                            return
            except asyncio.CancelledError:
                pass

        async def _fail_if_still_owner(reason: str) -> None:
            """Transition the run to FAILED only if this worker still owns the lease."""
            try:
                with create_session() as owner_sess:
                    now = datetime.now(timezone.utc)
                    stmt = (
                        update(RunRow)
                        .where(
                            RunRow.id == run_id,
                            RunRow.status == RunStatus.RUNNING.value,
                        )
                    )
                    if active_lease_id:
                        stmt = stmt.where(RunRow.lease_id == active_lease_id)
                    result = owner_sess.execute(
                        stmt.values(
                            status=RunStatus.FAILED.value,
                            finished_at=now,
                            failure_type="WORKER_LEASE_LOST",
                            failure_reason=reason[:500],
                        )
                    )
                    owner_sess.commit()
                    if result.rowcount:
                        logger.warning("Run %s failed because worker lease was lost.", run_id)
            except Exception:
                logger.exception("Unable to persist lease-loss state for run %s", run_id)

        heartbeat_task = asyncio.create_task(_heartbeat_loop(run_id, active_lease_id))

        try:
            from rag_platform.server import _execute_evaluation_run, db_row_to_test_case

            ds = sess.get(DatasetRow, run.dataset_id)
            if not ds:
                run.status = RunStatus.FAILED.value
                run.failure_reason = f"Dataset {run.dataset_id} not found"
                run.failure_type = "DATASET_NOT_FOUND"
                sess.commit()
                return False

            cases = [db_row_to_test_case(r) for r in ds.cases]
            raw_options = json.loads(run.options_json) if run.options_json else {}
            from rag_platform.server import CreateRunReq

            req_kwargs = {
                "project_id": run.project_id,
                "dataset_id": run.dataset_id,
                "system_version": run.system_version,
                "policy_id": run.policy_id,
            }
            req_kwargs.update(raw_options)
            req = CreateRunReq(**req_kwargs)

            config = RunConfig(
                project_id=run.project_id,
                dataset_id=run.dataset_id,
                dataset_version=ds.version,
                system_version=run.system_version,
                policy_id=run.policy_id,
                options=RunOptions(
                    max_cases=getattr(req, "max_cases", None),
                    concurrency=getattr(req, "concurrency", 5),
                    timeout_seconds=getattr(req, "timeout_seconds", 60.0),
                    fail_fast=getattr(req, "fail_fast", False),
                    use_cache=getattr(req, "use_cache", True),
                ),
            )

            # Race the evaluation against lease-loss detection. If ownership is lost,
            # cancel the in-flight evaluation so an obsolete worker does not continue
            # doing external work after another worker has recovered the run.
            evaluation_task = asyncio.create_task(
                _execute_evaluation_run(
                    run_id=run.id,
                    req=req,
                    config=config,
                    ds_cases=cases,
                    db_session=sess,
                )
            )
            lease_task = asyncio.create_task(lease_lost_event.wait())

            done, pending = await asyncio.wait(
                {evaluation_task, lease_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if lease_task in done and lease_lost_event.is_set():
                evaluation_task.cancel()
                try:
                    await evaluation_task
                except asyncio.CancelledError:
                    pass
                await _fail_if_still_owner("Worker heartbeat/lease was lost while evaluation was running.")
                return False

            lease_task.cancel()
            try:
                await lease_task
            except asyncio.CancelledError:
                pass

            # A lease can be invalidated immediately after evaluation completes.
            # Guard the final handoff so an obsolete worker never reports success.
            with create_session() as verify_sess:
                current = verify_sess.get(RunRow, run_id)
                if not current or current.status != RunStatus.COMPLETED.value:
                    if not current or current.status == RunStatus.FAILED.value:
                        return False
                    if active_lease_id and current.lease_id != active_lease_id:
                        return False
            return True
        except asyncio.CancelledError:
            await _fail_if_still_owner("Worker evaluation task was cancelled.")
            raise
        except Exception as exc:
            logger.exception("Failed processing run %s: %s", run_id, exc)
            try:
                run = sess.get(RunRow, run_id)
                if run and (not active_lease_id or run.lease_id == active_lease_id):
                    run.status = RunStatus.FAILED.value
                    run.failure_reason = str(exc)[:500]
                    run.failure_type = type(exc).__name__[:64]
                    sess.commit()
            except Exception:
                pass
            return False
        finally:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
            if owned_session:
                sess.close()

    @classmethod
    async def process_next_queued_run(cls, db_session: Session | None = None) -> bool:
        """Poll and process next queued run if available."""
        owned_session = False
        sess = db_session if db_session is not None else create_session()
        if db_session is None:
            owned_session = True

        try:
            claimed = cls.claim_next_run(sess)
            if not claimed:
                return False
            return await cls.process_run_by_id(claimed.id, db_session=sess)
        finally:
            if owned_session:
                sess.close()

    @classmethod
    async def run_worker_loop(
        cls,
        stop_event: asyncio.Event | None = None,
        min_interval: float = 0.1,
        max_interval: float = 5.0,
        backoff_factor: float = 1.5,
        stale_check_interval: float = 60.0,
        session_factory: Any | None = None,
    ) -> None:
        """Adaptive worker daemon loop with push-notification wakeup."""
        notify_event = WorkerNotificationBus.subscribe()
        current_interval = min_interval
        last_stale_check = time.monotonic()
        pg_stop_event = threading.Event()

        try:
            from rag_platform.db import get_db_engine
            sample_engine = session_factory().get_bind() if session_factory else get_db_engine()
            WorkerNotificationBus.start_postgres_listener(sample_engine, pg_stop_event)
        except Exception as exc:
            logger.debug("PostgreSQL LISTEN initialization skipped: %s", exc)

        try:
            while not (stop_event and stop_event.is_set()):
                now_mono = time.monotonic()
                if now_mono - last_stale_check >= stale_check_interval:
                    try:
                        if session_factory:
                            with session_factory() as sess:
                                cls.recover_stale_runs(sess)
                        else:
                            with create_session() as sess:
                                cls.recover_stale_runs(sess)
                    except Exception as exc:
                        logger.warning("Periodic stale run recovery encountered error: %s", exc)
                    last_stale_check = now_mono

                if session_factory:
                    with session_factory() as sess:
                        did_work = await cls.process_next_queued_run(db_session=sess)
                else:
                    did_work = await cls.process_next_queued_run()

                if did_work:
                    current_interval = min_interval
                    continue

                if stop_event and stop_event.is_set():
                    break

                notify_event.clear()
                try:
                    if stop_event:
                        done, pending = await asyncio.wait(
                            [asyncio.create_task(notify_event.wait()), asyncio.create_task(stop_event.wait())],
                            timeout=current_interval,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for t in pending:
                            t.cancel()
                        if stop_event.is_set():
                            break
                    else:
                        await asyncio.wait_for(notify_event.wait(), timeout=current_interval)
                    current_interval = min_interval
                except asyncio.TimeoutError:
                    current_interval = min(current_interval * backoff_factor, max_interval)
        finally:
            pg_stop_event.set()
            WorkerNotificationBus.unsubscribe(notify_event)


def main() -> None:
    """CLI entry point for running standalone durable evaluation worker process."""
    import argparse
    parser = argparse.ArgumentParser(description="Reliab Standalone Durable Evaluation Worker")
    parser.add_argument("--min-interval", type=float, default=0.1, help="Minimum polling interval in seconds")
    parser.add_argument("--max-interval", type=float, default=5.0, help="Maximum polling interval in seconds")
    parser.add_argument("--stale-check", type=float, default=60.0, help="Interval for stale run recovery")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting standalone DurableRunWorker process...")

    stop_event = asyncio.Event()

    def _handle_exit(sig, frame):
        logger.info("Termination signal received. Shutting down worker gracefully...")
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    try:
        asyncio.run(
            DurableRunWorker.run_worker_loop(
                stop_event=stop_event,
                min_interval=args.min_interval,
                max_interval=args.max_interval,
                stale_check_interval=args.stale_check,
            )
        )
    except KeyboardInterrupt:
        logger.info("Worker stopped by user.")
    logger.info("DurableRunWorker process terminated.")


if __name__ == "__main__":
    main()
