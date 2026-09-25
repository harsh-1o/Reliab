"""Application configuration with environment variable overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    """Platform configuration settings."""

    # Environment & Logging
    env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_json: bool = field(default_factory=lambda: os.getenv("LOG_JSON", "true").lower() == "true")

    # Database: SQLite fallback for local zero-infra, PostgreSQL for production
    # ponytail: sqlite works out of the box with zero external daemon; asyncpg for production
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "sqlite+aiosqlite:///./rag_platform.db"
        )
    )

    # Queue: memory (default zero-infra) or redis
    queue_backend: str = field(
        default_factory=lambda: os.getenv("QUEUE_BACKEND", "memory")
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # Evaluator Defaults & Guardrails
    default_max_cases_per_run: int = int(os.getenv("DEFAULT_MAX_CASES", "500"))
    default_timeout_seconds: int = int(os.getenv("DEFAULT_TIMEOUT_SECONDS", "60"))
    default_max_concurrency: int = int(os.getenv("DEFAULT_MAX_CONCURRENCY", "5"))

    # Evaluator Cache
    evaluator_cache_enabled: bool = (
        os.getenv("EVALUATOR_CACHE_ENABLED", "true").lower() == "true"
    )


# Global settings singleton
settings = Settings()
