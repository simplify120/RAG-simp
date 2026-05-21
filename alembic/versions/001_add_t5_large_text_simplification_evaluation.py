"""Add t5_large_text_simplification_evaluation table

Revision ID: 001
Revises:
Create Date: 2025-03-09

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "t5_large_text_simplification_evaluation",
        sa.Column("id", sa.UUID(), server_default=sa.text("gen_random_uuid()"), nullable=False),
        sa.Column("item_id", sa.UUID(), nullable=False),
        sa.Column("input_text", sa.Text(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("sari", sa.Float(), nullable=True),
        sa.Column("bertscore_f1", sa.Float(), nullable=True),
        sa.Column("fkgl_input", sa.Float(), nullable=True),
        sa.Column("fkgl_output", sa.Float(), nullable=True),
        sa.Column("delta_fkgl", sa.Float(), nullable=True),
        sa.Column("fre_input", sa.Float(), nullable=True),
        sa.Column("fre_output", sa.Float(), nullable=True),
        sa.Column("fre_delta", sa.Float(), nullable=True),
        sa.Column("bleu", sa.Float(), nullable=True),
        sa.Column("perplexity", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["item_id"], ["dataset_items.item_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_t5_large_text_simplification_evaluation_item_id",
        "t5_large_text_simplification_evaluation",
        ["item_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_t5_large_text_simplification_evaluation_item_id",
        table_name="t5_large_text_simplification_evaluation",
    )
    op.drop_table("t5_large_text_simplification_evaluation")
