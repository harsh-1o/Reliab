"""009_sessions_table

Revision ID: i123456789ah
Revises: h123456789ag
"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "i123456789ah"
down_revision: Union[str, None] = "h123456789ag"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "sessions" not in tables:
        op.create_table(
            "sessions",
            sa.Column("session_id", sa.String(128), primary_key=True),
            sa.Column("client_id", sa.String(64), nullable=False),
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("project_roles_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_sessions_client_id", "sessions", ["client_id"])
        op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "sessions" in tables:
        op.drop_index("ix_sessions_expires_at", table_name="sessions")
        op.drop_index("ix_sessions_client_id", table_name="sessions")
        op.drop_table("sessions")
