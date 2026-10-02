"""Фрагменты вакансий для RAG: текст, метаданные, вектор (pgvector) и полнотекстовый индекс

Только для PostgreSQL: в SQLite нет ни векторов, ни tsvector, там миграция ничего не делает.

- embedding vector(1024) — эмбеддинг модели deepvk/USER-bge-m3;
- HNSW-индекс по косинусному расстоянию — быстрый приближённый поиск ближайших соседей;
- content_tsv — полнотекстовый индекс с русской морфологией (для гибридного поиска);
- metadata JSONB — компания, зарплата, город, формат работы и т.п. для фильтров.

Revision ID: 0002_vacancy_chunks
Revises: 0001_initial
Create Date: 2026-10-02
"""
from alembic import op

revision = "0002_vacancy_chunks"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

EMBEDDING_DIM = 1024


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    if not _is_postgres():
        return
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(f"""
        CREATE TABLE vacancy_chunks (
            id              BIGSERIAL PRIMARY KEY,
            vacancy_id      VARCHAR NOT NULL REFERENCES vacancy_raw (vacancy_id) ON DELETE CASCADE,
            chunk_index     INTEGER NOT NULL,
            section         VARCHAR(32),
            content         TEXT NOT NULL,
            metadata        JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            embedding       vector({EMBEDDING_DIM}),
            embedding_model VARCHAR(100),
            content_tsv     tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED,
            created_at      TIMESTAMP NOT NULL DEFAULT now(),
            CONSTRAINT uq_vacancy_chunks_vacancy_chunk UNIQUE (vacancy_id, chunk_index)
        )
    """)
    op.execute(
        "CREATE INDEX ix_vacancy_chunks_embedding_hnsw ON vacancy_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )
    op.execute("CREATE INDEX ix_vacancy_chunks_content_tsv ON vacancy_chunks USING gin (content_tsv)")
    op.execute("CREATE INDEX ix_vacancy_chunks_metadata ON vacancy_chunks USING gin (metadata)")


def downgrade() -> None:
    if not _is_postgres():
        return
    op.execute("DROP TABLE IF EXISTS vacancy_chunks")
