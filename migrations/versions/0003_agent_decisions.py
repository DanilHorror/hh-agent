"""Решения человека по черновикам агента (для метрик агента)

Таблицы самого LangGraph (checkpoints, checkpoint_writes, checkpoint_blobs, checkpoint_migrations)
создаёт библиотека при запуске агента — PostgresSaver.setup(); здесь их нет.

Revision ID: 0003_agent_decisions
Revises: 0002_vacancy_chunks
Create Date: 2026-10-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0003_agent_decisions"
down_revision = "0002_vacancy_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "agent_decisions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vacancy_id", sa.String(), nullable=False),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("feedback", sa.Text()),
        sa.Column("relevance_score", sa.Float()),
        sa.Column("revision", sa.Integer(), server_default="0"),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_agent_decisions_id", "agent_decisions", ["id"])
    op.create_index("ix_agent_decisions_vacancy_id", "agent_decisions", ["vacancy_id"])


def downgrade() -> None:
    op.drop_table("agent_decisions")
