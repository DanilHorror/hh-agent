"""Метрики оценки вакансий против разметки человека.

Модель ставит вакансии оценку 0–100; вакансия «предлагается» человеку, если оценка ≥ порога.
Разметка: fit (подходит) / not_fit (не подходит); unsure в расчёт не входит.

    precision — из предложенных моделью вакансий какая доля действительно подходит (меньше мусора в Telegram);
    recall    — из подходящих вакансий какую долю модель предложила (меньше пропущенных хороших);
    F1        — баланс двух метрик.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from llm.providers.base import LLMProvider, LLMResponse


@dataclass
class Confusion:
    tp: int = 0   # модель предложила, человеку подходит
    fp: int = 0   # модель предложила, человеку не подходит (мусор в Telegram)
    fn: int = 0   # модель не предложила, а человеку подходит (пропущенная вакансия)
    tn: int = 0   # модель не предложила, человеку не подходит

    @property
    def precision(self) -> Optional[float]:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> Optional[float]:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> Optional[float]:
        p, r = self.precision, self.recall
        if not p or not r:
            return 0.0 if p is not None and r is not None else None
        return 2 * p * r / (p + r)

    @property
    def accuracy(self) -> Optional[float]:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total else None


def confusion(pairs: list[tuple[int, str]], threshold: int) -> Confusion:
    """pairs: (оценка модели, метка человека fit|not_fit)."""
    c = Confusion()
    for score, label in pairs:
        predicted = score >= threshold
        actual = label == "fit"
        if predicted and actual:
            c.tp += 1
        elif predicted:
            c.fp += 1
        elif actual:
            c.fn += 1
        else:
            c.tn += 1
    return c


def sweep(pairs: list[tuple[int, str]], thresholds=range(30, 95, 5)) -> list[tuple[int, Confusion]]:
    return [(t, confusion(pairs, t)) for t in thresholds]


def best_threshold(pairs: list[tuple[int, str]], thresholds=range(30, 95, 5)) -> tuple[int, Confusion]:
    """Порог с наибольшим F1; при равенстве — с большей precision (меньше мусора), затем более строгий."""
    return max(sweep(pairs, thresholds), key=lambda tc: (tc[1].f1 or 0.0, tc[1].precision or 0.0, tc[0]))


def separation(pairs: list[tuple[int, str]]) -> Optional[float]:
    """Вероятность, что случайная подходящая вакансия получила оценку выше случайной неподходящей
    (ROC AUC). 0.5 — модель не различает, 1.0 — различает идеально; не зависит от порога."""
    fits = [s for s, label in pairs if label == "fit"]
    others = [s for s, label in pairs if label == "not_fit"]
    if not fits or not others:
        return None
    wins = sum(1.0 if f > o else 0.5 if f == o else 0.0 for f in fits for o in others)
    return wins / (len(fits) * len(others))


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    models: set = field(default_factory=set)


class RecordingProvider(LLMProvider):
    """Обёртка над провайдером: пропускает вызовы как есть и считает токены и время."""

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.name = inner.name
        self.usage = Usage()

    @property
    def model(self) -> str:  # type: ignore[override]
        return self.inner.model

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3,
                       max_tokens=1500) -> LLMResponse:
        started = time.time()
        response = await self.inner.complete(messages, schema, json_mode=json_mode,
                                             temperature=temperature, max_tokens=max_tokens)
        self.usage.calls += 1
        self.usage.seconds += time.time() - started
        self.usage.prompt_tokens += response.prompt_tokens or 0
        self.usage.completion_tokens += response.completion_tokens or 0
        self.usage.models.add(response.model)
        return response

    async def aclose(self) -> None:
        await self.inner.aclose()
