"""Structured JSON logging with request and evaluation run context propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

# Context variables for trace and run correlation
current_project_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("project_id", default=None)
current_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)


class StructuredJsonFormatter(logging.Formatter):
    """Standard-library JSON formatter with contextvars injection."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Inject contextual IDs if active
        proj = current_project_id.get()
        if proj:
            payload["project_id"] = proj
        run = current_run_id.get()
        if run:
            payload["run_id"] = run
        tr = current_trace_id.get()
        if tr:
            payload["trace_id"] = tr

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)

        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure root logger with stdout handler and structured JSON formatting."""
    root = logging.getLogger("rag_platform")
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s in %(name)s: %(message)s")
        )

    root.addHandler(handler)
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced platform logger."""
    return logging.getLogger(f"rag_platform.{name}")
