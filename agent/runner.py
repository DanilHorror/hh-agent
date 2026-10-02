"""Запуск и возобновление графа для конкретных вакансий.

    start(vacancy)              — новая вакансия: прогнать граф до паузы или до конца
    resume(vacancy_id, action)  — решение из Telegram: продолжить поток этой вакансии
"""
from __future__ import annotations

import asyncio
import logging

from langgraph.types import Command

from .graph import AgentDeps, build_graph, thread_config

logger = logging.getLogger("Agent")

ACTIONS = ("approve", "rewrite", "skip")


class VacancyAgent:
    def __init__(self, deps: AgentDeps, checkpointer):
        self.graph = build_graph(deps, checkpointer)
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, vacancy_id: str) -> asyncio.Lock:
        return self._locks.setdefault(vacancy_id, asyncio.Lock())

    async def start(self, vacancy: dict) -> str:
        """Прогнать новую вакансию. Возвращает outcome или 'waiting' (ждёт решения человека)."""
        config = thread_config(vacancy["vacancy_id"])
        async with self._lock(vacancy["vacancy_id"]):
            state = await self.graph.ainvoke({"vacancy": vacancy}, config)
        return self._outcome(state)

    async def waiting(self, vacancy_id: str) -> bool:
        snapshot = await self.graph.aget_state(thread_config(vacancy_id))
        return bool(snapshot.next) and "wait_decision" in snapshot.next

    async def resume(self, vacancy_id: str, action: str, feedback: str = "") -> str:
        """Решение человека. Возвращает outcome, 'waiting' или 'not_waiting' (уже решено / нет такой)."""
        if action not in ACTIONS:
            raise ValueError(f"action должен быть одним из {ACTIONS}")
        async with self._lock(vacancy_id):
            if not await self.waiting(vacancy_id):
                return "not_waiting"
            logger.info(f"Решение по вакансии {vacancy_id}: {action}" + (f" — «{feedback}»" if feedback else ""))
            state = await self.graph.ainvoke(Command(resume={"action": action, "feedback": feedback}),
                                             thread_config(vacancy_id))
        return self._outcome(state)

    @staticmethod
    def _outcome(state: dict) -> str:
        if state.get("__interrupt__"):
            return "waiting"
        return state.get("outcome", "done")
