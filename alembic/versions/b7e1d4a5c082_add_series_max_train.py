"""add max_train to series

Revision ID: b7e1d4a5c082
Revises: c8af9aefda76
Create Date: 2026-09-04 00:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b7e1d4a5c082'
down_revision: str | Sequence[str] | None = 'c8af9aefda76'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    # Nullable: a series without a cap trains on everything before the cutoff,
    # which is what every series ingested before this revision did.
    op.add_column('series', sa.Column('max_train', sa.Integer(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('series', 'max_train')
