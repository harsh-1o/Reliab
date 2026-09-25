"""004_run_heartbeat_and_summary

Revision ID: d123456789ac
Revises: c123456789ab
Create Date: 2026-09-25 23:05:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd123456789ac'
down_revision: Union[str, None] = 'c123456789ab'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "runs" in tables:
        existing_cols = {col["name"] for col in insp.get_columns("runs")}
        
        if "started_at" not in existing_cols:
            op.add_column("runs", sa.Column("started_at", sa.DateTime(timezone=True), nullable=True))
        if "heartbeat_at" not in existing_cols:
            op.add_column("runs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
            op.create_index(op.f("ix_runs_heartbeat_at"), "runs", ["heartbeat_at"], unique=False)
        if "summary_json" not in existing_cols:
            op.add_column("runs", sa.Column("summary_json", sa.Text(), nullable=True))
        if "gate_result_json" not in existing_cols:
            op.add_column("runs", sa.Column("gate_result_json", sa.Text(), nullable=True))
        if "gate_status" not in existing_cols:
            op.add_column("runs", sa.Column("gate_status", sa.String(length=32), nullable=True))
            op.create_index(op.f("ix_runs_gate_status"), "runs", ["gate_status"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "runs" in tables:
        existing_cols = {col["name"] for col in insp.get_columns("runs")}
        existing_indexes = {idx["name"] for idx in insp.get_indexes("runs")}

        if op.f("ix_runs_gate_status") in existing_indexes:
            op.drop_index(op.f("ix_runs_gate_status"), table_name="runs")
        if op.f("ix_runs_heartbeat_at") in existing_indexes:
            op.drop_index(op.f("ix_runs_heartbeat_at"), table_name="runs")

        if "gate_status" in existing_cols:
            op.drop_column("runs", "gate_status")
        if "gate_result_json" in existing_cols:
            op.drop_column("runs", "gate_result_json")
        if "summary_json" in existing_cols:
            op.drop_column("runs", "summary_json")
        if "heartbeat_at" in existing_cols:
            op.drop_column("runs", "heartbeat_at")
        if "started_at" in existing_cols:
            op.drop_column("runs", "started_at")
