"""Схемы запросов и ответов API. FastAPI проверяет по ним входящие данные (ошибка 422 с описанием
вместо падения внутри) и строит по ним документацию на /docs."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

SearchMode = Literal["vector", "text", "hybrid", "hybrid_rerank"]


class SearchHit(BaseModel):
    vacancy_id: str
    title: str
    company: str
    url: str
    section: str = Field(description="Раздел вакансии: requirements, duties, conditions…")
    snippet: str = Field(description="Текст фрагмента")
    score: float = Field(description="Оценка поиска (реранкер или RRF)")


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    hits: list[SearchHit]
    took_ms: int


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500,
                          examples=["В каких вакансиях нужно тестировать AI-агентов или LLM?"])
    k: int = Field(default=6, ge=1, le=10, description="Сколько фрагментов отдать модели")


class Source(BaseModel):
    n: int = Field(description="Номер в ответе: [1], [2]…")
    vacancy_id: str
    title: str
    company: str
    url: str
    section: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    timings: dict[str, float]


class EvaluateRequest(BaseModel):
    title: str = Field(min_length=2, max_length=300, examples=["QA Automation Engineer (Python)"])
    company: str = Field(default="", max_length=300)
    description: str = Field(min_length=50, max_length=20000, description="Полный текст вакансии")


class EvaluateResponse(BaseModel):
    relevance_score: int = Field(ge=0, le=100)
    reasoning: str
    cover_letter: str


class StatsResponse(BaseModel):
    vacancies_by_status: dict[str, int]
    vacancy_texts: int
    rag_chunks: Optional[int] = None
    agent: dict


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: str
    llm: str
    llm_provider: str


class GoldenItem(BaseModel):
    vacancy_id: str
    title: str
    company: str
    url: str
    description: str
    label: Optional[Literal["fit", "not_fit", "unsure"]] = None
    note: str = ""


class GoldenSet(BaseModel):
    total: int
    labeled: int
    fit: int
    not_fit: int
    unsure: int
    items: list[GoldenItem]


class LabelRequest(BaseModel):
    label: Optional[Literal["fit", "not_fit", "unsure"]] = Field(
        description="fit — подходит, not_fit — не подходит, unsure — не уверен; null — снять отметку")
    note: str = Field(default="", max_length=500)
