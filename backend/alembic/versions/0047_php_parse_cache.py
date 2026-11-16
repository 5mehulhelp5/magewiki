"""php_parse_cache -- content-hash-keyed cache of catalog_module_classes'
tree-sitter parse of one PHP/phtml file, so a file whose content hasn't
changed since it was last parsed (anywhere, by any module) never needs to be
reparsed. Mirrors analysis_cache's content-hash-keyed design, minus the
cost/token columns (a deterministic parser result, not a billable LLM call).

Revision ID: 0047
Revises: 0046
Create Date: 2026-11-16

"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0047"
down_revision: str | None = "0046"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "php_parse_cache",
        sa.Column("content_hash", sa.Text, primary_key=True),
        sa.Column("entries", postgresql.JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("php_parse_cache")
