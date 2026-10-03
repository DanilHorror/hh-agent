"""Инструменты MCP-сервера: проверки входа, формат ответа, подсчёт навыков."""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server import tools  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def test_empty_inputs_return_readable_errors():
    assert "error" in run(tools.search_vacancies(" "))
    assert "error" in run(tools.ask_market(""))
    assert "error" in run(tools.evaluate_vacancy("QA", "  "))


def test_count_skills_once_per_vacancy_with_filter():
    lists = [["Python", "SQL", "python"], ["Java", "SQL"], ["Python", "Docker"]]
    top = tools.count_skills(lists, limit=3)
    assert top[0] == {"skill": "python", "vacancies": 2, "share": round(2 / 3, 3)}
    assert {"skill": "sql", "vacancies": 2, "share": round(2 / 3, 3)} in top
    assert [s["skill"] for s in tools.count_skills(lists, contains="doc")] == ["docker"]


def test_evaluate_vacancy_shapes_result(monkeypatch):
    from llm import brain

    async def fake(vacancy, guard=True):
        assert vacancy["company"] == "—"
        return {"relevance_score": 72, "reasoning": "подходит", "cover_letter": "Письмо",
                "letter_warning": "из письма убрано: ссылка"}

    monkeypatch.setattr(brain, "evaluate_and_draft", fake)
    out = run(tools.evaluate_vacancy("QA", "Текст вакансии"))
    assert out == {"relevance_score": 72, "reasoning": "подходит", "cover_letter": "Письмо",
                   "warning": "из письма убрано: ссылка"}


def test_evaluate_vacancy_model_down(monkeypatch):
    from llm import brain

    async def down(vacancy, guard=True):
        return {"relevance_score": 0, "reasoning": "", "cover_letter": "", "error": True}

    monkeypatch.setattr(brain, "evaluate_and_draft", down)
    assert "error" in run(tools.evaluate_vacancy("QA", "Текст"))


def test_server_registers_tools():
    pytest.importorskip("mcp")
    from mcp_server import server

    for name in ("search_vacancies", "ask_market", "evaluate_vacancy", "market_stats"):
        assert callable(getattr(server, name))
