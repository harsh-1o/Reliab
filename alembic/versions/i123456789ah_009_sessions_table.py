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
            sa.Column("token_hash", sa.String(64), primary_key=True),
            sa.Column("api_key_hash", sa.String(64), nullable=True),
            sa.Column("client_id", sa.String(64), nullable=False),
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("project_roles_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index("ix_sessions_client_id", "sessions", ["client_id"])
        op.create_index("ix_sessions_api_key_hash", "sessions", ["api_key_hash"])
        op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])
    else:
        # If sessions table pre-exists (e.g. from Base.metadata.create_all on fresh SQLite),
        # verify schema integrity: do not silently accept a partially created or inconsistent schema.
        existing_cols = {c["name"] for c in insp.get_columns("sessions")}
        required_cols = {
            "token_hash",
            "api_key_hash",
            "client_id",
            "is_admin",
            "project_roles_json",
            "created_at",
            "expires_at",
        }
        missing_cols = required_cols - existing_cols
        if missing_cols:
            raise RuntimeError(
                f"Partially created sessions table detected! Missing required columns: {missing_cols}. "
                "Inconsistent schemas cannot be silently accepted."
            )
        # Ensure all required indexes exist
        existing_indexes = {idx["name"] for idx in insp.get_indexes("sessions")}
        if "ix_sessions_client_id" not in existing_indexes:
            op.create_index("ix_sessions_client_id", "sessions", ["client_id"])
        if "ix_sessions_api_key_hash" not in existing_indexes:
            op.create_index("ix_sessions_api_key_hash", "sessions", ["api_key_hash"])
        if "ix_sessions_expires_at" not in existing_indexes:
            op.create_index("ix_sessions_expires_at", "sessions", ["expires_at"])


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "sessions" in tables:
        existing_indexes = {idx["name"] for idx in insp.get_indexes("sessions")}
        if "ix_sessions_expires_at" in existing_indexes:
            op.drop_index("ix_sessions_expires_at", table_name="sessions")
        if "ix_sessions_api_key_hash" in existing_indexes:
            op.drop_index("ix_sessions_api_key_hash", table_name="sessions")
        if "ix_sessions_client_id" in existing_indexes:
            op.drop_index("ix_sessions_client_id", table_name="sessions")
        op.drop_table("sessions")
