"""003_api_keys

Revision ID: c123456789ab
Revises: b756d62b0781
Create Date: 2026-09-25 22:20:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c123456789ab'
down_revision: Union[str, None] = 'b756d62b0781'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "api_keys" not in tables:
        op.create_table(
            "api_keys",
            sa.Column("key_hash", sa.String(length=64), primary_key=True, nullable=False),
            sa.Column("client_id", sa.String(length=64), nullable=False),
            sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("project_roles_json", sa.Text(), nullable=False, server_default="{}"),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        )
        op.create_index(op.f("ix_api_keys_client_id"), "api_keys", ["client_id"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    tables = set(insp.get_table_names())

    if "api_keys" in tables:
        existing_indexes = {idx["name"] for idx in insp.get_indexes("api_keys")}
        if op.f("ix_api_keys_client_id") in existing_indexes:
            op.drop_index(op.f("ix_api_keys_client_id"), table_name="api_keys")
        op.drop_table("api_keys")
