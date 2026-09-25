"""Durable database-backed execution worker for asynchronous evaluation runs.

Provides atomic run claiming, safe state transitions, stale runner recovery,
and survival across API process restarts without requiring heavyweight message queues.
"""

import asyncio
from datetime import datetime, timezone
import json
import logging
import threading
import time

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

    Combines in-process asyncio.Event broadcasting, thread-safe callbacks,
    and PostgreSQL LISTEN/NOTIFY support to eliminate unnecessary polling
    load in high-concurrency deployments.
    """
    _subscribers: list[asyncio.Event] = []
    _lock = threading.Lock()

    @classmethod
    def subscribe(cls) -> asyncio.Event:
        event = asyncio.Event()
        with cls._lock:
            cls._subscribers.append(event)
        return event

    @classmethod
    def unsubscribe(cls, event: asyncio.Event) -> None:
        with cls._lock:
            if event in cls._subscribers:
                cls._subscribers.remove(event)

    @classmethod
    def notify_new_run(cls, run_id: str, db_session: Session | None = None) -> None:
        """Broadcast notification of a new queued run.

        1. Sets all local in-process asyncio subscriber events.
        2. If connected to PostgreSQL, executes `NOTIFY rag_runs_channel, :run_id`.
        """
        with cls._lock:
            for ev in list(cls._subscribers):
                ev.set()

        # PostgreSQL LISTEN/NOTIFY push notification for multi-node deployments
        if db_session is not None:
            try:
                bind = db_session.get_bind()
                if bind and bind.dialect.name == "postgresql":
                    from sqlalchemy import text
                    db_session.execute(text("NOTIFY rag_runs_channel, :run_id"), {"run_id": run_id})
                    db_session.commit()
            except Exception as exc:
                logger.debug("PostgreSQL NOTIFY failed or not supported: %s", exc)


class DurableRunWorker:
    """Database-backed worker that polls, claims, and processes queued runs."""

    @classmethod
    def claim_next_run(cls, sess: Session) -> RunRow | None:
        """Atomically claim the oldest QUEUED run in the database.

        Sets started_at and heartbeat_at immediately upon claiming.
        Uses an atomic conditional UPDATE (`status='RUNNING' WHERE id=:id AND status='QUEUED'`)
        guaranteeing that exactly one worker claims a run even across multiple concurrent processes.
        """
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
        # Atomic conditional update claiming run and stamping execution start and initial heartbeat
        res = sess.execute(
            update(RunRow)
            .where(RunRow.id == candidate_id, RunRow.status == RunStatus.QUEUED.value)
            .values(
                status=RunStatus.RUNNING.value,
                started_at=now,
                heartbeat_at=now,
            )
        )
        if res.rowcount == 0:
            # Concurrently claimed by another worker
            sess.rollback()
            return None

        sess.commit()
        claimed = sess.get(RunRow, candidate_id)
        if claimed:
            logger.info("DurableRunWorker successfully claimed run_id=%s (started_at=%s)", claimed.id, now.isoformat())
        return claimed

    @classmethod
    def recover_stale_runs(
        cls,
        sess: Session,
        max_age_seconds: float = 600.0,
    ) -> list[str]:
        """Identify and recover runs stuck in RUNNING state beyond max_age_seconds without a heartbeat.

        Uses heartbeat_at rather than created_at so runs queued for long durations
        or legitimately long-running active suites are not mistakenly marked stale.
        """
        now = datetime.now(timezone.utc)
        stmt = select(RunRow).where(RunRow.status == RunStatus.RUNNING.value)
        running_runs = sess.scalars(stmt).all()
        recovered_ids: list[str] = []

        for run in running_runs:
            last_activity = run.heartbeat_at or run.started_at or run.created_at
            if last_activity.tzinfo is None:
                last_activity = last_activity.replace(tzinfo=timezone.utc)
            age = (now - last_activity).total_seconds()
            if age > max_age_seconds:
                run.status = RunStatus.FAILED.value
                run.finished_at = now
                run.failure_type = "STALE_RUNNER_RECOVERY"
                run.failure_reason = (
                    f"Run abandoned or worker died while running (last heartbeat was {int(age)}s ago > {int(max_age_seconds)}s limit)."
                )
                recovered_ids.append(run.id)
                logger.warning(
                    "Recovered stale run_id=%s: heartbeat age=%ds exceeded timeout=%ds",
                    run.id, int(age), int(max_age_seconds),
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
        """Execute a specific run that has already been claimed, with live heartbeat updates."""
        import asyncio
        owned_session = False
        sess = db_session if db_session is not None else create_session()
        if db_session is None:
            owned_session = True

        async def _heartbeat_loop(target_run_id: str, interval: float = 3.0):
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        with create_session() as hb_sess:
                            hb_sess.execute(
                                update(RunRow)
                                .where(RunRow.id == target_run_id, RunRow.status == RunStatus.RUNNING.value)
                                .values(heartbeat_at=datetime.now(timezone.utc))
                            )
                            hb_sess.commit()
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        heartbeat_task = asyncio.create_task(_heartbeat_loop(run_id))

        try:
            from rag_platform.server import _execute_evaluation_run, db_row_to_test_case

            run = sess.get(RunRow, run_id)
            if not run:
                logger.error("Run %s not found in datastore", run_id)
                return False

            ds = sess.get(DatasetRow, run.dataset_id)
            if not ds:
                run.status = RunStatus.FAILED.value
                run.failure_reason = f"Dataset {run.dataset_id} not found"
                run.failure_type = "DATASET_NOT_FOUND"
                sess.commit()
                return False

            cases = [db_row_to_test_case(r) for r in ds.cases]

            # Reconstruct request parameters from options_json
            raw_options = json.loads(run.options_json) if run.options_json else {}
            from rag_platform.server import CreateRunReq

            # Reconstruct typed req and config
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
                dataset_version=run.dataset_checksum,
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

            await _execute_evaluation_run(
                run_id=run.id,
                req=req,
                config=config,
                ds_cases=cases,
                db_session=sess,
            )
            return True
        except Exception as exc:
            logger.exception("Failed processing run %s: %s", run_id, exc)
            try:
                run = sess.get(RunRow, run_id)
                if run:
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
    ) -> None:
        """Adaptive worker daemon loop with push-notification wakeup.

        - When runs are queued: Processes continuously without delay.
        - When idle: Backs off exponentially up to max_interval to eliminate heavy DB polling.
        - On new run notification (WorkerNotificationBus): Immediately wakes up with 0 latency.
        - Periodically checks for and recovers stale runs.
        """
        notify_event = WorkerNotificationBus.subscribe()
        current_interval = min_interval
        last_stale_check = time.monotonic()

        try:
            while not (stop_event and stop_event.is_set()):
                now_mono = time.monotonic()
                if now_mono - last_stale_check >= stale_check_interval:
                    try:
                        with create_session() as sess:
                            cls.recover_stale_runs(sess)
                    except Exception as exc:
                        logger.warning("Periodic stale run recovery encountered error: %s", exc)
                    last_stale_check = now_mono

                did_work = await cls.process_next_queued_run()
                if did_work:
                    # Work was found and processed, immediately check for more
                    current_interval = min_interval
                    continue

                # No work available right now: wait on notification event with adaptive backoff
                notify_event.clear()
                try:
                    await asyncio.wait_for(notify_event.wait(), timeout=current_interval)
                    # Woken up immediately by push notification
                    current_interval = min_interval
                except asyncio.TimeoutError:
                    # Idle timeout expired without notification: back off exponentially to reduce DB polling
                    current_interval = min(current_interval * backoff_factor, max_interval)
        finally:
            WorkerNotificationBus.unsubscribe(notify_event)

