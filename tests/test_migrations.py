"""Migration tests for Alembic schema upgrades."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


def test_alembic_upgrade_head_on_fresh_sqlite(monkeypatch, tmp_path):
    """Applying all migrations should succeed on a fresh SQLite database."""
    db_path = tmp_path / "migration.db"
    database_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", database_url)

    repo_root = Path(__file__).resolve().parents[1]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))

    command.upgrade(config, "head")

    engine = create_engine(database_url)
    inspector = inspect(engine)
    run_columns = {column["name"] for column in inspector.get_columns("runs")}
    assert {"failure_reason", "failure_type"}.issubset(run_columns)
    run_indexes = {index["name"] for index in inspector.get_indexes("runs")}
    assert {"ix_runs_project_id", "ix_runs_dataset_id"}.issubset(run_indexes)
