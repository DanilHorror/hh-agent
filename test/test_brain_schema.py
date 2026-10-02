"""Разбор и проверка ответа модели при оценке вакансии — без запущенной Ollama:
клиент подменён заглушкой. Запуск: python -m pytest test/test_brain_schema.py"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import brain  # noqa: E402
from llm import providers  # noqa: E402
from llm.providers import ollama_provider  # noqa: E402

VACANCY = {"title": "Python Developer", "company": "Acme", "description": "FastAPI, PostgreSQL, LLM"}
GOOD = {"relevance_score": 72, "reasoning": "Совпадает стек", "cover_letter": "Добрый день!\nМой опыт подходит."}


class FakeClient:
    """Отдаёт заранее заданные ответы по очереди и запоминает, с каким format её звали."""

    script: list = []
    formats: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def chat(self, model, messages, format, options):
        FakeClient.formats.append(format)
        outcome = FakeClient.script.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return {"message": {"content": outcome}, "prompt_eval_count": 4000, "eval_count": 300}


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    FakeClient.script = []
    FakeClient.formats = []
    monkeypatch.setattr(ollama_provider.ollama, "AsyncClient", FakeClient)
    # Тест не зависит от LLM_PROVIDER в .env и не создаёт файл-замок в папке проекта
    providers.set_provider(ollama_provider.OllamaProvider(
        model="test-model", host="http://localhost:11434", timeout=5, num_ctx=8192, use_lock=False))
    yield
    providers.set_provider(None)


def run(coro):
    return asyncio.run(coro)


def test_parse_valid_json_by_schema():
    result, how = brain.parse_evaluation(json.dumps(GOOD, ensure_ascii=False))
    assert how == "schema" and result["relevance_score"] == 72


def test_parse_out_of_range_score_is_clamped():
    result, how = brain.parse_evaluation('{"relevance_score": 150, "reasoning": "x", "cover_letter": "y"}')
    assert how == "json" and result["relevance_score"] == 100


def test_parse_broken_json_falls_back_to_regex():
    result, how = brain.parse_evaluation('{"relevance_score": 55, "reasoning": "обрыв...')
    assert how == "regex" and result["relevance_score"] == 55


def test_schema_is_sent_to_ollama():
    FakeClient.script = [json.dumps(GOOD, ensure_ascii=False)]
    result = run(brain.evaluate_and_draft(VACANCY))
    assert result["relevance_score"] == 72
    assert FakeClient.formats[0] == brain.EVALUATION_SCHEMA


def test_old_ollama_falls_back_to_plain_json():
    FakeClient.script = [Exception("invalid format: expected \"json\""), json.dumps(GOOD, ensure_ascii=False)]
    result = run(brain.evaluate_and_draft(VACANCY))
    assert result["relevance_score"] == 72
    assert FakeClient.formats == [brain.EVALUATION_SCHEMA, "json"]


def test_broken_answer_triggers_one_retry():
    FakeClient.script = ['{"relevance_score": 40, "reasoning": "сломано', json.dumps(GOOD, ensure_ascii=False)]
    result = run(brain.evaluate_and_draft(VACANCY))
    assert result["relevance_score"] == 72
    assert len(FakeClient.formats) == 2


def test_placeholders_removed_from_letter():
    bad = dict(GOOD, cover_letter="Добрый день, [Имя HR-специалиста]!\nЯ подхожу для [Название компании].")
    FakeClient.script = [json.dumps(bad, ensure_ascii=False)]
    result = run(brain.evaluate_and_draft(VACANCY))
    assert "[" not in result["cover_letter"]
