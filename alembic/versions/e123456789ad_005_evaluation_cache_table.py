"""005_evaluation_cache_table

Revision ID: e123456789ad
Revises: d123456789ac
Create Date: 2026-09-25 23:25:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e123456789ad'
down_revision: Union[str, None] = 'd123456789ac'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "evaluation_cache" not in tables:
        op.create_table(
            "evaluation_cache",
            sa.Column("cache_key", sa.String(length=64), primary_key=True),
            sa.Column("result_json", sa.Text(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            op.f("ix_evaluation_cache_last_accessed_at"),
            "evaluation_cache",
            ["last_accessed_at"],
            unique=False,
        )


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "evaluation_cache" in tables:
        op.drop_index(op.f("ix_evaluation_cache_last_accessed_at"), table_name="evaluation_cache")
        op.drop_table("evaluation_cache")
