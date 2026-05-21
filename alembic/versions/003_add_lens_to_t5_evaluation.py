"""Add lens column to t5_large_text_simplification_evaluation

Revision ID: 003
Revises: 002
Create Date: 2026-03-12

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "t5_large_text_simplification_evaluation",
        sa.Column("lens", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("t5_large_text_simplification_evaluation", "lens")
