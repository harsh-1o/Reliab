"""007_unique_run_idempotency

Revision ID: g123456789af
Revises: f123456789ae
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "g123456789af"
down_revision: Union[str, None] = "f123456789ae"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_INDEX_NAME = "uq_runs_project_idempotency_key"


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    indexes = {idx["name"] for idx in insp.get_indexes("runs")}

    # NULL idempotency keys remain unrestricted on PostgreSQL and SQLite, while
    # non-NULL keys become unique per project at the database level.
    if _INDEX_NAME not in indexes:
        op.create_index(
            _INDEX_NAME,
            "runs",
            ["project_id", "idempotency_key"],
            unique=True,
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    indexes = {idx["name"] for idx in insp.get_indexes("runs")}
    if _INDEX_NAME in indexes:
        op.drop_index(_INDEX_NAME, table_name="runs")
