"""Логика инструментов MCP-сервера — без зависимости от пакета mcp, чтобы проверять её тестами.

Инструменты повторяют HTTP API (api/main.py), но в виде, удобном модели-клиенту: короткие поля,
ограниченная длина текстов, понятные ошибки вместо исключений.
"""
from __future__ import annotations

import json
from collections import Counter

MAX_K = 10
SNIPPET = 400


async def search_vacancies(query: str, k: int = 5) -> dict:
    from rag import search as rag_search
    from rag.answer import merge_by_vacancy

    query = (query or "").strip()
    if len(query) < 2:
        return {"error": "Пустой запрос: опишите, что искать (технология, обязанность, условие)."}
    k = max(1, min(int(k), MAX_K))
    result = await rag_search.search(query, k=k * 2, mode="hybrid_rerank")
    hits = merge_by_vacancy(result.hits)[:k]
    return {"query": query, "found": len(hits), "vacancies": [
        {"title": h.metadata.get("title", ""), "company": h.metadata.get("company", ""),
         "url": h.metadata.get("url", ""), "section": h.section, "snippet": h.content[:SNIPPET]}
        for h in hits]}


async def ask_market(question: str) -> dict:
    from rag import answer as rag_answer

    question = (question or "").strip()
    if len(question) < 3:
        return {"error": "Задайте вопрос о рынке вакансий, например: «что требуют от QA Automation на Python?»"}
    result = await rag_answer.answer(question)
    return {"question": question, "answer": result.text,
            "sources": [{"n": c["n"], "title": c["title"], "company": c["company"], "url": c["url"]}
                        for c in rag_answer.source_cards(result.sources)]}


async def evaluate_vacancy(title: str, description: str, company: str = "") -> dict:
    from llm.brain import evaluate_and_draft

    if not (description or "").strip():
        return {"error": "Нужен текст вакансии (description)."}
    result = await evaluate_and_draft({"title": title or "—", "company": company or "—",
                                       "description": description})
    if result.get("error"):
        return {"error": "Модель не ответила. Проверьте, что запущена модель из LLM_PROVIDER (для Ollama: ollama ps)."}
    out = {"relevance_score": int(result["relevance_score"]), "reasoning": result.get("reasoning", ""),
           "cover_letter": result.get("cover_letter", "")}
    if result.get("letter_warning"):
        out["warning"] = result["letter_warning"]
    return out


def count_skills(keyword_lists: list[list[str]], contains: str = "", limit: int = 20) -> list[dict]:
    """Сколько вакансий упоминают каждый навык (один раз на вакансию)."""
    counter: Counter = Counter()
    for keywords in keyword_lists:
        counter.update({k.strip().lower() for k in keywords if isinstance(k, str) and k.strip()})
    needle = (contains or "").strip().lower()
    # при равном числе вакансий — по алфавиту, чтобы выдача не менялась от запуска к запуску
    items = sorted(((k, n) for k, n in counter.items() if not needle or needle in k), key=lambda kn: (-kn[1], kn[0]))
    total = len(keyword_lists)
    return [{"skill": k, "vacancies": n, "share": round(n / total, 3) if total else 0.0} for k, n in items[:limit]]


async def market_stats(top: int = 20, contains: str = "") -> dict:
    """Размер базы и самые частые навыки — на вопросы «что чаще всего требуют» RAG по фрагментам не отвечает."""
    from sqlalchemy import func, select

    import database

    top = max(1, min(int(top), 100))
    async with database.async_session() as session:
        texts = (await session.execute(select(func.count()).select_from(database.VacancyRaw))).scalar_one()
        rows = (await session.execute(select(database.VacancyKeywords.keywords_json))).scalars().all()
    lists = []
    for raw in rows:
        try:
            value = json.loads(raw or "[]")
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            lists.append(value)
    return {"vacancies_with_text": texts, "vacancies_with_skills": len(lists),
            "top_skills": count_skills(lists, contains, top)}
