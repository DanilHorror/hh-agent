"""Гибридный поиск: смысл (pgvector) + слова (полнотекстовый Postgres) → RRF → реранкер.

Режимы (нужны для сравнения на шаге оценки):
    vector         — только поиск по смыслу
    text           — только поиск по словам
    hybrid         — оба, объединены RRF
    hybrid_rerank  — оба + перепроверка топа cross-encoder'ом (режим по умолчанию)
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from . import embeddings, store
from .store import Hit

MODES = ("vector", "text", "hybrid", "hybrid_rerank")
RRF_K = 60            # стандартная константа RRF: сглаживает разницу между первыми местами
CANDIDATES = 20       # сколько фрагментов берёт каждый поиск и сколько видит реранкер


def rrf_merge(*ranked_lists: list[Hit], k: int = RRF_K) -> list[Hit]:
    """Reciprocal Rank Fusion: у фрагмента складывается 1/(k + место) из каждого списка.

    Объединяем по местам, а не по очкам: косинус и текстовый ранг в разных шкалах,
    приводить их друг к другу не нужно."""
    scores: dict[int, float] = {}
    best: dict[int, Hit] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(hit.chunk_id, hit)
    merged = sorted(best.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for hit in merged:
        hit.score = scores[hit.chunk_id]
    return merged


def unique_vacancies(hits: list[Hit]) -> list[str]:
    """ID вакансий в порядке первого появления — для метрик на уровне вакансий."""
    seen, order = set(), []
    for hit in hits:
        if hit.vacancy_id not in seen:
            seen.add(hit.vacancy_id)
            order.append(hit.vacancy_id)
    return order


@dataclass
class SearchResult:
    hits: list[Hit]
    mode: str
    timings: dict


async def search(query: str, k: int = 5, mode: str = "hybrid_rerank",
                 sections: list[str] | None = None, candidates: int = CANDIDATES) -> SearchResult:
    if mode not in MODES:
        raise ValueError(f"mode должен быть одним из {MODES}")
    timings = {}
    started = time.time()

    vector_hits: list[Hit] = []
    text_hits: list[Hit] = []
    if mode != "text":
        t = time.time()
        # Модель синхронная — считаем в отдельном потоке, чтобы не блокировать бота
        query_vector = (await asyncio.to_thread(embeddings.embed, [query]))[0]
        timings["embed"] = time.time() - t
        t = time.time()
        vector_hits = await store.vector_search(query_vector, candidates, sections)
        timings["vector"] = time.time() - t
    if mode != "vector":
        t = time.time()
        text_hits = await store.text_search(query, candidates, sections)
        timings["text"] = time.time() - t

    if mode == "vector":
        hits = vector_hits
    elif mode == "text":
        hits = text_hits
    else:
        hits = rrf_merge(vector_hits, text_hits)[:candidates]

    if mode == "hybrid_rerank" and hits:
        t = time.time()
        scores = await asyncio.to_thread(embeddings.rerank_scores, query, [h.content for h in hits])
        for hit, score in zip(hits, scores):
            hit.rerank_score = score
        hits = sorted(hits, key=lambda h: h.rerank_score, reverse=True)
        timings["rerank"] = time.time() - t

    timings["total"] = time.time() - started
    return SearchResult(hits=hits[:k], mode=mode, timings=timings)
