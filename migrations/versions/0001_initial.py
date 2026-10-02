"""Исходная схема: вакансии, сырые тексты для анализа рынка, ключевые слова ИИ

Повторяет таблицы, которые бот раньше создавал сам (Base.metadata.create_all) в SQLite.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-02
"""
from alembic import op
import sqlalchemy as sa

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

# SQLAlchemy хранит Enum по именам членов VacancyStatus
VACANCY_STATUS = sa.Enum(
    "NEW", "NOTIFIED", "CONFIRMED", "SENT", "SKIPPED", "NEEDS_MANUAL_REVIEW", name="vacancystatus"
)


def upgrade() -> None:
    op.create_table(
        "vacancies",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vacancy_id", sa.String(), nullable=False),
        sa.Column("title", sa.String(), nullable=False),
        sa.Column("company", sa.String(), nullable=False),
        sa.Column("url", sa.String(), nullable=False),
        sa.Column("description_snippet", sa.Text()),
        sa.Column("status", VACANCY_STATUS),
        sa.Column("relevance_score", sa.Float()),
        sa.Column("cover_letter_draft", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.Column("sent_at", sa.DateTime()),
    )
    op.create_index("ix_vacancies_id", "vacancies", ["id"])
    op.create_index("ix_vacancies_vacancy_id", "vacancies", ["vacancy_id"], unique=True)
    # Частые выборки бота: подтверждённые вакансии и отправленные за сегодня
    op.create_index("ix_vacancies_status", "vacancies", ["status"])

    op.create_table(
        "vacancy_raw",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vacancy_id", sa.String(), nullable=False),
        sa.Column("title", sa.String()),
        sa.Column("company", sa.String()),
        sa.Column("url", sa.String()),
        sa.Column("skills_json", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("scraped_at", sa.DateTime()),
    )
    op.create_index("ix_vacancy_raw_id", "vacancy_raw", ["id"])
    op.create_index("ix_vacancy_raw_vacancy_id", "vacancy_raw", ["vacancy_id"], unique=True)

    op.create_table(
        "vacancy_keywords",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("vacancy_id", sa.String(), nullable=False),
        sa.Column("keywords_json", sa.Text()),
        sa.Column("model", sa.String()),
        sa.Column("extracted_at", sa.DateTime()),
    )
    op.create_index("ix_vacancy_keywords_id", "vacancy_keywords", ["id"])
    op.create_index("ix_vacancy_keywords_vacancy_id", "vacancy_keywords", ["vacancy_id"], unique=True)


def downgrade() -> None:
    op.drop_table("vacancy_keywords")
    op.drop_table("vacancy_raw")
    op.drop_table("vacancies")
    VACANCY_STATUS.drop(op.get_bind(), checkfirst=True)
