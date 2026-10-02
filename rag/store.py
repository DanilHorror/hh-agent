"""Работа с таблицей vacancy_chunks в Postgres: запись фрагментов и два вида поиска.

Векторы передаются строкой '[0.1, 0.2, ...]' и приводятся к типу vector в самом SQL —
отдельный Python-пакет pgvector не нужен.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy import text

import database

from .chunking import Chunk


@dataclass
class Hit:
    chunk_id: int
    vacancy_id: str
    section: str
    content: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0          # оценка того поиска, который нашёл фрагмент
    rerank_score: float | None = None


def require_postgres() -> None:
    if database.IS_SQLITE:
        raise RuntimeError(
            "RAG работает только на Postgres (pgvector). Добавьте в .env "
            "DATABASE_URL=postgresql+asyncpg://hh:hh_local_dev@localhost:5433/hh_agent")


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def _row_to_hit(row) -> Hit:
    meta = row.metadata
    if isinstance(meta, str):
        meta = json.loads(meta)
    return Hit(chunk_id=row.id, vacancy_id=row.vacancy_id, section=row.section or "",
               content=row.content, metadata=meta or {}, score=float(row.score or 0.0))


async def vacancies_to_index(rebuild: bool = False) -> list[dict]:
    """Вакансии из vacancy_raw, у которых ещё нет фрагментов (или все — при rebuild)."""
    require_postgres()
    where = "" if rebuild else "WHERE NOT EXISTS (SELECT 1 FROM vacancy_chunks c WHERE c.vacancy_id = r.vacancy_id)"
    async with database.engine.connect() as conn:
        rows = (await conn.execute(text(
            f"SELECT r.vacancy_id, r.title, r.company, r.url, r.skills_json, r.description "
            f"FROM vacancy_raw r {where} ORDER BY r.id"))).mappings().all()
    result = []
    for r in rows:
        try:
            skills = json.loads(r["skills_json"]) if r["skills_json"] else []
        except json.JSONDecodeError:
            skills = []
        result.append({**dict(r), "skills": skills})
    return result


async def replace_chunks(vacancy_id: str, chunks: list[Chunk], vectors: list[list[float]], model: str) -> None:
    """Перезаписать фрагменты одной вакансии (в одной транзакции)."""
    async with database.engine.begin() as conn:
        await conn.execute(text("DELETE FROM vacancy_chunks WHERE vacancy_id = :vid"), {"vid": vacancy_id})
        for chunk, vector in zip(chunks, vectors):
            await conn.execute(text(
                "INSERT INTO vacancy_chunks (vacancy_id, chunk_index, section, content, metadata, embedding, "
                "embedding_model) VALUES (:vid, :idx, :section, :content, CAST(:meta AS jsonb), "
                "CAST(CAST(:emb AS text) AS vector), :model)"),
                {"vid": vacancy_id, "idx": chunk.chunk_index, "section": chunk.section, "content": chunk.content,
                 "meta": json.dumps(chunk.metadata, ensure_ascii=False), "emb": vector_literal(vector),
                 "model": model})


async def stats() -> dict:
    require_postgres()
    async with database.engine.connect() as conn:
        row = (await conn.execute(text(
            "SELECT count(*) AS chunks, count(DISTINCT vacancy_id) AS vacancies FROM vacancy_chunks"))).one()
        raw = (await conn.execute(text("SELECT count(*) FROM vacancy_raw"))).scalar_one()
    return {"chunks": row.chunks, "vacancies": row.vacancies, "vacancy_raw": raw}


def _filters(sections: list[str] | None, params: dict) -> str:
    if not sections:
        return ""
    params["sections"] = list(sections)
    return " AND c.section = ANY(:sections)"


async def vector_search(query_vector: list[float], k: int = 20, sections: list[str] | None = None) -> list[Hit]:
    """Поиск по смыслу: ближайшие по косинусному расстоянию (оператор <=>, индекс HNSW)."""
    require_postgres()
    params = {"q": vector_literal(query_vector), "k": k}
    where = _filters(sections, params)
    sql = (
        "SELECT c.id, c.vacancy_id, c.section, c.content, c.metadata, "
        "1 - (c.embedding <=> CAST(CAST(:q AS text) AS vector)) AS score "
        f"FROM vacancy_chunks c WHERE c.embedding IS NOT NULL{where} "
        "ORDER BY c.embedding <=> CAST(CAST(:q AS text) AS vector) LIMIT :k"
    )
    async with database.engine.begin() as conn:
        # Больше кандидатов при обходе графа HNSW — точнее при фильтрах, цена — миллисекунды
        await conn.execute(text("SET LOCAL hnsw.ef_search = 100"))
        rows = (await conn.execute(text(sql), params)).all()
    return [_row_to_hit(r) for r in rows]


async def text_search(query: str, k: int = 20, sections: list[str] | None = None) -> list[Hit]:
    """Поиск по словам: полнотекстовый индекс Postgres с русской морфологией.

    Слова вопроса объединяются через ИЛИ, а ранжирование (ts_rank_cd) поднимает фрагменты,
    где совпало больше слов и они стоят ближе друг к другу."""
    require_postgres()
    params = {"query": query, "k": k}
    where = _filters(sections, params)
    sql = (
        "WITH q AS (SELECT to_tsquery('russian', "
        "  replace(plainto_tsquery('russian', :query)::text, ' & ', ' | ')) AS tsq) "
        "SELECT c.id, c.vacancy_id, c.section, c.content, c.metadata, "
        "ts_rank_cd(c.content_tsv, q.tsq) AS score "
        f"FROM vacancy_chunks c, q WHERE q.tsq IS NOT NULL AND c.content_tsv @@ q.tsq{where} "
        "ORDER BY score DESC LIMIT :k"
    )
    async with database.engine.connect() as conn:
        rows = (await conn.execute(text(sql), params)).all()
    return [_row_to_hit(r) for r in rows]
