"""006_worker_leases_and_idempotency

Revision ID: f123456789ae
Revises: e123456789ad
Create Date: 2026-09-26 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f123456789ae'
down_revision: Union[str, None] = 'e123456789ad'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("runs")}

    if "worker_id" not in cols:
        op.add_column("runs", sa.Column("worker_id", sa.String(length=64), nullable=True))
    if "lease_id" not in cols:
        op.add_column("runs", sa.Column("lease_id", sa.String(length=64), nullable=True))
        op.create_index(op.f("ix_runs_lease_id"), "runs", ["lease_id"], unique=False)
    if "lease_expires_at" not in cols:
        op.add_column("runs", sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
    if "idempotency_key" not in cols:
        op.add_column("runs", sa.Column("idempotency_key", sa.String(length=128), nullable=True))
        op.create_index(op.f("ix_runs_idempotency_key"), "runs", ["idempotency_key"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("runs")}

    if "idempotency_key" in cols:
        op.drop_index(op.f("ix_runs_idempotency_key"), table_name="runs")
        op.drop_column("runs", "idempotency_key")
    if "lease_expires_at" in cols:
        op.drop_column("runs", "lease_expires_at")
    if "lease_id" in cols:
        op.drop_index(op.f("ix_runs_lease_id"), table_name="runs")
        op.drop_column("runs", "lease_id")
    if "worker_id" in cols:
        op.drop_column("runs", "worker_id")
