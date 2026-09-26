"""008_api_key_revocation

Revision ID: h123456789ag
Revises: g123456789af
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "h123456789ag"
down_revision: Union[str, None] = "g123456789af"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "api_keys" in tables:
        cols = {c["name"] for c in insp.get_columns("api_keys")}
        if "revoked_at" not in cols:
            op.add_column("api_keys", sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "api_keys" in tables:
        cols = {c["name"] for c in insp.get_columns("api_keys")}
        if "revoked_at" in cols:
            op.drop_column("api_keys", "revoked_at")
