"""HTTP API hh-агента (FastAPI).

    venv\\Scripts\\python.exe -m uvicorn api.main:app --port 8000
    Документация и кнопки «Try it out»: http://localhost:8000/docs

Эндпоинты:
    GET  /health       — живы ли база и модель
    GET  /search       — гибридный поиск по фрагментам вакансий (RAG, шаг 3)
    POST /ask          — ответ о рынке вакансий со ссылками
    POST /ask/stream   — то же, но по частям (Server-Sent Events)
    POST /evaluate     — оценка вакансии и черновик письма (шаги 1 и 4)
    GET  /stats        — сколько вакансий, фрагментов и решений по агенту
    GET  /eval/golden            — эталонный набор вакансий для разметки
    PUT  /eval/golden/{id}       — отметить вакансию: подходит / не подходит / не уверен
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, select, text

import database
from agent.metrics import summarize
from config import settings
from llm.brain import check_ollama_health, evaluate_and_draft
from llm.providers import LLMError
from rag import answer as rag_answer
from quality import golden
from rag import search as rag_search

from .schemas import (AskRequest, AskResponse, EvaluateRequest, EvaluateResponse, GoldenItem, GoldenSet,
                      HealthResponse, LabelRequest, SearchHit, SearchMode, SearchResponse, StatsResponse)

logger = logging.getLogger("API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Модели RAG — на процессоре по умолчанию: видеокарту занимает Ollama (см. RAG_BOT_DEVICE)
    os.environ.setdefault("RAG_DEVICE", settings.rag_bot_device)
    await database.init_db()
    yield


app = FastAPI(
    title="hh-agent API",
    description="Поиск и вопросы по базе вакансий (RAG), оценка вакансии и черновик сопроводительного письма.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(LLMError)
async def llm_error_handler(request: Request, exc: LLMError):
    # Модель недоступна — это не ошибка клиента: 503 с понятным текстом вместо трейса
    return JSONResponse(status_code=503, content={"detail": f"Модель недоступна ({exc.kind}): {exc.detail[:300]}"})


@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    # Например: RAG без Postgres или без установленного sentence-transformers
    return JSONResponse(status_code=503, content={"detail": str(exc)[:500]})


@app.get("/health", response_model=HealthResponse, tags=["service"])
async def health():
    try:
        async with database.engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:  # noqa: BLE001
        db_status = f"error: {type(e).__name__}"
    alive, err = await check_ollama_health()
    llm_status = "ok" if alive else f"error: {err[:100]}"
    status = "ok" if db_status == "ok" and alive else "degraded"
    return HealthResponse(status=status, database=db_status, llm=llm_status, llm_provider=settings.llm_provider)


@app.get("/search", response_model=SearchResponse, tags=["rag"])
async def search(q: str = Query(min_length=2, max_length=300, description="Что искать"),
                 k: int = Query(default=5, ge=1, le=20),
                 mode: SearchMode = Query(default="hybrid_rerank")):
    started = time.time()
    result = await rag_search.search(q, k=k, mode=mode)
    hits = [SearchHit(vacancy_id=h.vacancy_id, title=h.metadata.get("title", ""),
                      company=h.metadata.get("company", ""), url=h.metadata.get("url", ""),
                      section=h.section, snippet=h.content[:600],
                      score=h.rerank_score if h.rerank_score is not None else h.score)
            for h in result.hits]
    return SearchResponse(query=q, mode=mode, hits=hits, took_ms=int((time.time() - started) * 1000))


@app.post("/ask", response_model=AskResponse, tags=["rag"])
async def ask(body: AskRequest):
    result = await rag_answer.answer(body.question, k=body.k)
    return AskResponse(question=body.question, answer=result.text,
                       sources=rag_answer.source_cards(result.sources),
                       timings={k: round(v, 3) for k, v in result.timings.items() if isinstance(v, (int, float))})


@app.post("/ask/stream", tags=["rag"],
          responses={200: {"content": {"text/event-stream": {}},
                           "description": "События: sources, token (много раз), done; при сбое — error"}})
async def ask_stream(body: AskRequest):
    async def events():
        try:
            async for event in rag_answer.answer_stream(body.question, k=body.k):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as e:  # noqa: BLE001 — поток уже начат, статус не поменять: сообщаем событием
            logger.error(f"/ask/stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)[:300]}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/evaluate", response_model=EvaluateResponse, tags=["llm"])
async def evaluate(body: EvaluateRequest):
    result = await evaluate_and_draft({"title": body.title, "company": body.company or "—",
                                       "description": body.description})
    if result.get("error"):
        raise HTTPException(status_code=503, detail="Модель не ответила. Проверьте, что Ollama запущена: ollama ps")
    return EvaluateResponse(relevance_score=int(result["relevance_score"]), reasoning=result["reasoning"],
                            cover_letter=result["cover_letter"])


@app.get("/stats", response_model=StatsResponse, tags=["service"])
async def stats():
    async with database.async_session() as session:
        rows = (await session.execute(
            select(database.Vacancy.status, func.count()).group_by(database.Vacancy.status))).all()
        texts = (await session.execute(select(func.count()).select_from(database.VacancyRaw))).scalar_one()
    by_status = {(status.name if status is not None else "UNKNOWN"): count for status, count in rows}
    chunks = None
    if not database.IS_SQLITE:
        async with database.engine.connect() as conn:
            chunks = (await conn.execute(text("SELECT count(*) FROM vacancy_chunks"))).scalar_one()
    agent = summarize(await database.get_agent_decisions())
    return StatsResponse(vacancies_by_status=by_status, vacancy_texts=texts, rag_chunks=chunks, agent=agent)


_golden_lock = asyncio.Lock()   # разметку пишет один запрос за раз — файл не перепутается


@app.get("/eval/golden", response_model=GoldenSet, tags=["eval"])
async def get_golden():
    items = golden.load(golden.GOLDEN_PATH)
    if not items:
        raise HTTPException(status_code=404, detail="Набор не собран. Выполните scripts\\eval_build_golden.py")
    return GoldenSet(**golden.progress(items), items=items)


@app.put("/eval/golden/{vacancy_id}", response_model=GoldenItem, tags=["eval"])
async def label_golden(vacancy_id: str, body: LabelRequest):
    async with _golden_lock:
        items = golden.load(golden.GOLDEN_PATH)
        try:
            item = golden.set_label(items, vacancy_id, body.label, body.note)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Вакансии {vacancy_id} нет в наборе") from None
        golden.save(items, golden.GOLDEN_PATH)
    return item
