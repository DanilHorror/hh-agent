"""Эндпоинты API без базы, моделей и сети: поиск, RAG и оценка подменены заглушками."""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from api import main as api  # noqa: E402
from llm.providers import LLMError, LLMProvider, LLMResponse  # noqa: E402
from rag import answer as rag_answer  # noqa: E402
from rag.search import SearchResult  # noqa: E402
from rag.store import Hit  # noqa: E402

HITS = [Hit(1, "101", "requirements", "QA — A. Требования:\nPlaywright, TypeScript",
            {"title": "QA", "company": "A", "url": "https://hh.ru/vacancy/101"}, score=0.03, rerank_score=0.98),
        Hit(2, "102", "duties", "SDET — B. Обязанности:\nавтотесты на Python",
            {"title": "SDET", "company": "B", "url": "https://hh.ru/vacancy/102"}, score=0.02)]


@pytest.fixture
def client(monkeypatch):
    async def no_init():
        return None

    monkeypatch.setattr(api.database, "init_db", no_init)
    with TestClient(api.app) as test_client:
        yield test_client


def test_search_returns_hits(client, monkeypatch):
    async def fake_search(query, k, mode):
        return SearchResult(hits=HITS[:k], mode=mode, timings={})

    monkeypatch.setattr(api.rag_search, "search", fake_search)
    response = client.get("/search", params={"q": "playwright typescript", "k": 2})
    assert response.status_code == 200
    body = response.json()
    assert [h["vacancy_id"] for h in body["hits"]] == ["101", "102"]
    assert body["hits"][0]["score"] == 0.98          # у первого есть оценка реранкера
    assert body["hits"][1]["score"] == 0.02          # у второго — оценка поиска
    assert body["mode"] == "hybrid_rerank"


def test_search_validates_input(client):
    assert client.get("/search", params={"q": "x"}).status_code == 422            # слишком короткий запрос
    assert client.get("/search", params={"q": "qa", "mode": "magic"}).status_code == 422


def test_ask_returns_answer_with_sources(client, monkeypatch):
    async def fake_answer(question, k):
        return rag_answer.Answer(question, "Playwright нужен в [1].", HITS, {"total": 1.23456})

    monkeypatch.setattr(api.rag_answer, "answer", fake_answer)
    response = client.post("/ask", json={"question": "Где нужен Playwright?"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Playwright") and body["sources"][0]["n"] == 1
    assert body["sources"][0]["url"] == "https://hh.ru/vacancy/101"
    assert body["timings"]["total"] == 1.235


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_llm_unavailable_is_503_not_traceback(client, monkeypatch):
    async def broken_answer(question, k):
        raise LLMError("network", "connection refused", "ollama")

    monkeypatch.setattr(api.rag_answer, "answer", broken_answer)
    response = client.post("/ask", json={"question": "Где нужен Playwright?"})
    assert response.status_code == 503
    assert "Модель недоступна" in response.json()["detail"]


def test_ask_stream_sends_sources_tokens_done(client, monkeypatch):
    async def fake_stream(question, k):
        yield {"type": "sources", "sources": rag_answer.source_cards(HITS)}
        for piece in ["Playwright ", "нужен ", "в [1]."]:
            yield {"type": "token", "text": piece}
        yield {"type": "done", "timings": {}}

    monkeypatch.setattr(api.rag_answer, "answer_stream", fake_stream)
    with client.stream("POST", "/ask/stream", json={"question": "Где нужен Playwright?"}) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in response.iter_lines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["sources", "token", "token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Playwright нужен в [1]."


def test_evaluate(client, monkeypatch):
    async def fake_evaluate(vacancy):
        assert vacancy["title"] == "QA Automation"
        return {"relevance_score": 77, "reasoning": "стек совпадает", "cover_letter": "Добрый день!"}

    monkeypatch.setattr(api, "evaluate_and_draft", fake_evaluate)
    response = client.post("/evaluate", json={"title": "QA Automation", "description": "Python, pytest, API. " * 5})
    assert response.status_code == 200 and response.json()["relevance_score"] == 77


def test_evaluate_model_error_is_503(client, monkeypatch):
    async def failed(vacancy):
        return {"relevance_score": 0, "reasoning": "", "cover_letter": "", "error": True}

    monkeypatch.setattr(api, "evaluate_and_draft", failed)
    response = client.post("/evaluate", json={"title": "QA", "description": "x" * 60})
    assert response.status_code == 503


def test_evaluate_rejects_short_description(client):
    assert client.post("/evaluate", json={"title": "QA", "description": "коротко"}).status_code == 422


def test_docs_page_is_available(client):
    assert client.get("/docs").status_code == 200
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/search", "/ask", "/ask/stream", "/evaluate", "/stats", "/health"} <= set(paths)


# --- потоковый режим провайдеров и RAG ---------------------------------------------------

class OneShotProvider(LLMProvider):
    name, model = "fake", "fake-1"

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        return LLMResponse("весь ответ сразу", self.name, self.model)


def test_default_stream_yields_whole_answer():
    async def collect():
        return [piece async for piece in OneShotProvider().stream([{"role": "user", "content": "hi"}])]

    assert asyncio.run(collect()) == ["весь ответ сразу"]


def test_answer_stream_without_hits_says_no_answer(monkeypatch):
    async def empty_search(question, k, mode):
        return SearchResult(hits=[], mode=mode, timings={})

    monkeypatch.setattr(rag_answer, "search", empty_search)

    async def collect():
        return [event async for event in rag_answer.answer_stream("что угодно")]

    events = asyncio.run(collect())
    assert [e["type"] for e in events] == ["sources", "token", "done"]
    assert events[1]["text"] == rag_answer.NO_ANSWER
