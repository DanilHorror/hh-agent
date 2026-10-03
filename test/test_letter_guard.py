"""Страж письма: проверка → исправление → повторная проверка, с заглушками вместо моделей."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import brain, letter_guard  # noqa: E402
from llm.providers.base import LLMProvider, LLMResponse  # noqa: E402
from quality import letters  # noqa: E402


def run(coro):
    return asyncio.run(coro)


def fake_check(answers):
    """Судья-заглушка: по очереди возвращает заданные списки выдумок."""
    calls = []

    async def check(text):
        calls.append(text)
        return answers[min(len(calls) - 1, len(answers) - 1)]
    return check, calls


async def fix_v2(text, invented, cliches):
    return text + " (исправлено)"


def test_clean_letter_is_not_touched():
    check, calls = fake_check([[]])
    result = run(letter_guard.guard_letter("Письмо", check=check, fix=fix_v2, find_cliches=lambda t: []))
    assert result.clean and result.rounds == 0 and result.letter == "Письмо" and result.warning == ""
    assert len(calls) == 1


def test_invented_fact_fixed_on_second_round():
    check, calls = fake_check([["Kafka"], []])
    result = run(letter_guard.guard_letter("Письмо", check=check, fix=fix_v2, find_cliches=lambda t: []))
    assert result.clean and result.rounds == 1 and result.letter == "Письмо (исправлено)"
    assert result.history[0]["invented"] == ["Kafka"] and len(calls) == 2


def test_gives_up_after_max_rounds_with_warning():
    check, calls = fake_check([["Kafka"]])
    result = run(letter_guard.guard_letter("Письмо", check=check, fix=fix_v2, find_cliches=lambda t: [],
                                           max_rounds=2))
    assert not result.clean and result.rounds == 2 and len(calls) == 3
    assert result.warning == "Возможно, этого нет в резюме: Kafka"


def test_cliches_trigger_fix_but_not_warning():
    check, _ = fake_check([[]])
    result = run(letter_guard.guard_letter("Хочу внести свой вклад", check=check, fix=fix_v2,
                                           find_cliches=letters.find_cliches, max_rounds=1))
    assert not result.clean and result.cliches == ["внести вклад"] and result.warning == ""


def test_empty_fix_keeps_letter():
    check, _ = fake_check([["Kafka"]])

    async def empty_fix(text, invented, cliches):
        return "  "
    result = run(letter_guard.guard_letter("Письмо", check=check, fix=empty_fix, find_cliches=lambda t: [],
                                           max_rounds=1))
    assert result.letter == "Письмо"


class Scripted(LLMProvider):
    """Модель-заглушка: отвечает заранее заданными текстами по очереди."""
    name, model = "fake", "fake-1"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        self.prompts.append(messages[0]["content"])
        return LLMResponse(self.answers.pop(0), self.name, self.model)


def test_guard_cover_letter_wires_judge_and_writer(monkeypatch):
    monkeypatch.setattr(brain.settings, "letter_guard", True)
    monkeypatch.setattr(brain.settings, "letter_guard_rounds", 2)
    judge = Scripted(['{"claims": [{"claim": "Kafka", "evidence": "", "supported": false}]}',
                      '{"claims": [{"claim": "Selenium", "evidence": "Selenium", "supported": true}]}'])
    writer = Scripted(["Здравствуйте! Пишу автотесты на Selenium."])
    monkeypatch.setattr(brain, "get_provider", lambda: writer)
    result = run(brain.guard_cover_letter({"title": "QA"}, "Здравствуйте! Работаю с Kafka.", judge=judge))
    assert result.clean and result.rounds == 1
    assert "Selenium" in result.letter and "Kafka" not in result.letter
    assert "Kafka" in writer.prompts[0]          # исправлять просили ровно найденное


def test_guard_off_returns_none():
    assert run(brain.guard_cover_letter({"title": "QA"}, "Письмо")) is None   # conftest выключил стража
