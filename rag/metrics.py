"""Метрики качества поиска на уровне вакансий."""
from __future__ import annotations


def recall_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """Доля нужных вакансий, попавших в первые k. Для вопроса с одной нужной вакансией — 0 или 1."""
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & relevant) / len(relevant)


def hit_at_k(ranked: list[str], relevant: set[str], k: int = 5) -> float:
    """1, если в первых k есть хотя бы одна нужная вакансия."""
    return 1.0 if set(ranked[:k]) & relevant else 0.0


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """1 / место первой нужной вакансии (1 — первая, 0.5 — вторая, 0 — не нашлась)."""
    for position, vacancy_id in enumerate(ranked, start=1):
        if vacancy_id in relevant:
            return 1.0 / position
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
