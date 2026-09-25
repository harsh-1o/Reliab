"""Durable database-backed execution worker for asynchronous evaluation runs.

Provides atomic run claiming, safe state transitions, stale runner recovery,
and survival across API process restarts without requiring heavyweight message queues.
"""

from datetime import datetime, timezone
import json
import logging

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from rag_platform.db import DatasetRow, RunRow, create_session
from rag_platform.models import (
    RunConfig,
    RunOptions,
    RunStatus,
)

logger = logging.getLogger("rag_platform.worker")


class DurableRunWorker:
    """Database-backed worker that polls, claims, and processes queued runs."""

    @classmethod
    def claim_next_run(cls, sess: Session) -> RunRow | None:
        """Atomically claim the oldest QUEUED run in the database.

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

        # Atomic conditional update
        res = sess.execute(
            update(RunRow)
            .where(RunRow.id == candidate_id, RunRow.status == RunStatus.QUEUED.value)
            .values(status=RunStatus.RUNNING.value)
        )
        if res.rowcount == 0:
            # Concurrently claimed by another worker
            sess.rollback()
            return None

        sess.commit()
        claimed = sess.get(RunRow, candidate_id)
        if claimed:
            logger.info("DurableRunWorker successfully claimed run_id=%s", claimed.id)
        return claimed

    @classmethod
    def recover_stale_runs(
        cls,
        sess: Session,
        max_age_seconds: float = 600.0,
    ) -> list[str]:
        """Identify and recover runs stuck in RUNNING state beyond max_age_seconds.

        Marks abandoned runs as FAILED with an explicit recovery reason so they
        do not block systems or mislead users.
        """
        now = datetime.now(timezone.utc)
        stmt = select(RunRow).where(RunRow.status == RunStatus.RUNNING.value)
        running_runs = sess.scalars(stmt).all()
        recovered_ids: list[str] = []

        for run in running_runs:
            created = run.created_at
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age = (now - created).total_seconds()
            if age > max_age_seconds:
                run.status = RunStatus.FAILED.value
                run.finished_at = now
                run.failure_type = "STALE_RUNNER_RECOVERY"
                run.failure_reason = (
                    f"Run abandoned or worker died while running (age: {int(age)}s > {int(max_age_seconds)}s limit)."
                )
                recovered_ids.append(run.id)
                logger.warning(
                    "Recovered stale run_id=%s: age=%ds exceeded timeout=%ds",
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
        """Execute a specific run that has already been claimed."""
        owned_session = False
        sess = db_session if db_session is not None else create_session()
        if db_session is None:
            owned_session = True

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
