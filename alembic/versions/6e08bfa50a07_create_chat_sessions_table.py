"""create chat_sessions table

Revision ID: 6e08bfa50a07
Revises: 4d85c6a365e3
Create Date: 2026-05-06 19:22:43.995515

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '6e08bfa50a07'
down_revision: Union[str, None] = '4d85c6a365e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('chat_sessions',
    sa.Column('id', sa.UUID(), nullable=False),
    sa.Column('user_id', sa.String(), nullable=True),
    sa.Column('title', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
    sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    op.drop_table('chat_sessions')