"""002_failure_columns_and_indexes

Revision ID: b756d62b0781
Revises: d99e0c4aa5df
Create Date: 2026-09-25 18:37:47.354675

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b756d62b0781'
down_revision: Union[str, None] = 'd99e0c4aa5df'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "runs" in tables:
        run_cols = {c["name"] for c in insp.get_columns("runs")}
        if "failure_reason" not in run_cols:
            op.add_column("runs", sa.Column("failure_reason", sa.Text(), nullable=True))
        if "failure_type" not in run_cols:
            op.add_column("runs", sa.Column("failure_type", sa.String(length=64), nullable=True))

    for table_name, index_name, columns in [
        ("datasets", op.f("ix_datasets_project_id"), ["project_id"]),
        ("metric_results", op.f("ix_metric_results_trace_id"), ["trace_id"]),
        ("runs", op.f("ix_runs_dataset_id"), ["dataset_id"]),
        ("runs", op.f("ix_runs_project_id"), ["project_id"]),
        ("traces", op.f("ix_traces_run_id"), ["run_id"]),
        ("traces", op.f("ix_traces_test_case_id"), ["test_case_id"]),
    ]:
        if table_name in tables:
            existing_indexes = {idx["name"] for idx in insp.get_indexes(table_name)}
            if index_name not in existing_indexes:
                op.create_index(index_name, table_name, columns, unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    for table_name, index_name in [
        ("traces", op.f("ix_traces_test_case_id")),
        ("traces", op.f("ix_traces_run_id")),
        ("runs", op.f("ix_runs_project_id")),
        ("runs", op.f("ix_runs_dataset_id")),
        ("metric_results", op.f("ix_metric_results_trace_id")),
        ("datasets", op.f("ix_datasets_project_id")),
    ]:
        if table_name in tables:
            existing_indexes = {idx["name"] for idx in insp.get_indexes(table_name)}
            if index_name in existing_indexes:
                op.drop_index(index_name, table_name=table_name)

    if "runs" in tables:
        run_cols = {c["name"] for c in insp.get_columns("runs")}
        if "failure_type" in run_cols:
            op.drop_column("runs", "failure_type")
        if "failure_reason" in run_cols:
            op.drop_column("runs", "failure_reason")
