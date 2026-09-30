"""001_initial_schema

Revision ID: d99e0c4aa5df
Revises:
Create Date: 2026-09-25 17:43:53.210135

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd99e0c4aa5df'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()

    from rag_platform.db import Base
    if "projects" not in tables:
        Base.metadata.create_all(bind=bind)
    else:
        cols = [c["name"] for c in insp.get_columns("metric_results")]
        with op.batch_alter_table('metric_results') as batch_op:
            if "status" not in cols:
                batch_op.add_column(sa.Column('status', sa.String(length=32), nullable=False, server_default='PASS'))
            batch_op.alter_column('score', existing_type=sa.FLOAT(), nullable=True)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = insp.get_table_names()
    if "metric_results" in tables:
        cols = [c["name"] for c in insp.get_columns("metric_results")]
        with op.batch_alter_table('metric_results') as batch_op:
            batch_op.alter_column('score', existing_type=sa.FLOAT(), nullable=False)
            if "status" in cols:
                batch_op.drop_column('status')

