"""create li_document table

Revision ID: e382e4ca5166
Revises:
Create Date: 2025-08-20 03:34:28.256288
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e382e4ca5166"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")

    op.create_table(
        "li_document",
        sa.Column("node_id", sa.String, primary_key=True),  # no autoincrement
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("metadata", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        schema="public",
    )

    op.execute(
        "CREATE INDEX li_document_embedding_cosine_idx "
        "ON public.li_document USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);"
    )

    op.execute(
        "CREATE INDEX li_document_metadata_gin_idx "
        "ON public.li_document USING gin (metadata);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS public.li_document_metadata_gin_idx;")
    op.execute("DROP INDEX IF EXISTS public.li_document_embedding_cosine_idx;")
    op.execute("DROP INDEX IF EXISTS public.li_document_embedding_l2_idx;")
    op.drop_table("li_document", schema="public")
