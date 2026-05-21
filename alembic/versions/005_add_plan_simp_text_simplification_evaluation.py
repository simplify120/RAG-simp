"""Add plan_simp_text_simplification_evaluation table

Revision ID: 005
Revises: 004
Create Date: 2026-03-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "plan_simp_text_simplification_evaluation",
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
        sa.Column("lens", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["item_id"], ["dataset_items.item_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_plan_simp_text_simplification_evaluation_item_id",
        "plan_simp_text_simplification_evaluation",
        ["item_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_plan_simp_text_simplification_evaluation_item_id",
        table_name="plan_simp_text_simplification_evaluation",
    )
    op.drop_table("plan_simp_text_simplification_evaluation")
