"""Граф агента на заглушках: без модели, Telegram и базы. Состояние — в памяти (InMemorySaver)."""
import asyncio
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from agent.graph import AgentDeps, MAX_REVISIONS  # noqa: E402
from agent.runner import VacancyAgent  # noqa: E402

VACANCY = {"vacancy_id": "777", "title": "QA Automation", "company": "Acme", "url": "u", "description": "Python"}


def run(coro):
    return asyncio.run(coro)


class Recorder:
    """Заглушки зависимостей графа + журнал того, что граф с ними делал."""

    def __init__(self, score=80, error=False):
        self.score, self.error = score, error
        self.notified, self.statuses, self.decisions, self.rewrites = [], [], [], []

    async def evaluate(self, vacancy):
        if self.error:
            return {"relevance_score": 0, "reasoning": "", "cover_letter": "", "error": True}
        return {"relevance_score": self.score, "reasoning": "стек совпадает", "cover_letter": "Письмо v1"}

    async def notify(self, state):
        self.notified.append(state["cover_letter"])

    async def set_status(self, vacancy_id, status, **fields):
        self.statuses.append(status)

    async def record_decision(self, vacancy_id, decision, feedback, score, revision):
        self.decisions.append((decision, feedback, revision))

    async def rewrite(self, vacancy, letter, feedback):
        self.rewrites.append(feedback)
        return f"Письмо v{len(self.rewrites) + 1}"

    def agent(self):
        deps = AgentDeps(evaluate=self.evaluate, notify=self.notify, set_status=self.set_status,
                         record_decision=self.record_decision, rewrite=self.rewrite, min_score=60)
        return VacancyAgent(deps, InMemorySaver())


def test_low_score_is_skipped_without_telegram():
    rec = Recorder(score=30)
    assert run(rec.agent().start(VACANCY)) == "skipped_low_score"
    assert rec.notified == [] and rec.statuses == ["SKIPPED"]


def test_good_vacancy_waits_for_human_then_approve():
    rec = Recorder(score=85)
    agent = rec.agent()

    async def scenario():
        first = await agent.start(VACANCY)
        assert await agent.waiting("777")
        return first, await agent.resume("777", "approve")

    first, final = run(scenario())
    assert first == "waiting" and final == "approved"
    assert rec.notified == ["Письмо v1"]
    assert rec.statuses == ["NOTIFIED", "CONFIRMED"]
    assert rec.decisions == [("approve", "", 0)]


def test_rewrite_sends_new_letter_and_waits_again():
    rec = Recorder()
    agent = rec.agent()

    async def scenario():
        await agent.start(VACANCY)
        after_rewrite = await agent.resume("777", "rewrite", "короче")
        final = await agent.resume("777", "skip")
        return after_rewrite, final

    after_rewrite, final = run(scenario())
    assert after_rewrite == "waiting" and final == "skipped_by_user"
    assert rec.notified == ["Письмо v1", "Письмо v2"]
    assert rec.decisions == [("rewrite", "короче", 0), ("skip", "", 1)]


def test_rewrite_limit():
    rec = Recorder()
    agent = rec.agent()

    async def scenario():
        await agent.start(VACANCY)
        for i in range(MAX_REVISIONS + 1):
            await agent.resume("777", "rewrite", f"правка {i}")
        return await agent.resume("777", "approve")

    assert run(scenario()) == "approved"
    assert len(rec.rewrites) == MAX_REVISIONS          # лишняя правка не вызывает модель
    assert len(rec.notified) == MAX_REVISIONS + 2       # но письмо показывается ещё раз


def test_model_error_keeps_vacancy_for_retry():
    rec = Recorder(error=True)
    agent = rec.agent()
    assert run(agent.start(VACANCY)) == "error"
    assert rec.statuses == [] and rec.notified == []
    rec.error = False   # в следующем цикле модель ответила — тот же поток доходит до черновика
    assert run(agent.start(VACANCY)) == "waiting"


def test_second_click_is_ignored():
    rec = Recorder()
    agent = rec.agent()

    async def scenario():
        await agent.start(VACANCY)
        await agent.resume("777", "approve")
        return await agent.resume("777", "skip")

    assert run(scenario()) == "not_waiting"
    assert rec.decisions == [("approve", "", 0)]


def test_unknown_vacancy_is_not_waiting():
    assert run(Recorder().agent().resume("nope", "approve")) == "not_waiting"


def _load_metrics_module():
    import importlib.util

    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "agent_metrics.py")
    spec = importlib.util.spec_from_file_location("agent_metrics", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_metrics_counts_first_draft_and_agreement():
    m = _load_metrics_module()

    d = lambda vid, decision, score: SimpleNamespace(vacancy_id=vid, decision=decision, relevance_score=score)  # noqa: E731
    result = m.summarize([
        d("1", "approve", 90),
        d("2", "rewrite", 70), d("2", "approve", 70),
        d("3", "skip", 65),
        d("4", "rewrite", 80),               # ещё ждёт решения — в итог не входит
    ])
    assert result["decided"] == 3 and result["approved"] == 2 and result["skipped"] == 1
    assert result["approved_first_draft"] == 1
    assert result["avg_rewrites_before_approve"] == 0.5
    assert result["agreement"] == pytest.approx(2 / 3)
    assert result["avg_score_approved"] == 80 and result["avg_score_skipped"] == 65
