"""Граф обработки одной вакансии (LangGraph).

    START → evaluate ─(ошибка модели)→ END            (вакансия остаётся NEW, повтор в следующем цикле)
                     ─(оценка < порога)→ skip_low → END
                     ─(оценка ≥ порога)→ notify → wait_decision
    wait_decision ─(«Подходит»)→ approve → END
                  ─(«Переписать: …»)→ rewrite → notify → wait_decision …
                  ─(«Пропустить»)→ user_skip → END

У каждой вакансии свой поток (thread_id = vacancy-<id>). На wait_decision граф останавливается
(interrupt): состояние сохраняется в checkpointer, а бот продолжает искать другие вакансии.
Нажатие кнопки в Telegram возобновляет именно этот поток с того же места (Command(resume=...)).

Узлы не знают про Telegram, базу и модель напрямую — всё приходит через deps (AgentDeps),
поэтому граф тестируется с заглушками.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

logger = logging.getLogger("Agent")

MAX_REVISIONS = 3   # сколько раз можно попросить переписать письмо


class VacancyState(TypedDict, total=False):
    vacancy: dict           # vacancy_id, title, company, url, source, description
    score: int
    reasoning: str
    cover_letter: str
    revisions: int          # сколько раз письмо переписывали
    decision: str           # approve | rewrite | skip — последнее решение человека
    feedback: str           # замечание к письму
    outcome: str            # error | skipped_low_score | approved | skipped_by_user


@dataclass
class AgentDeps:
    """Всё внешнее, что нужно графу. В бою — настоящие функции, в тестах — заглушки."""
    evaluate: Callable[[dict], Awaitable[dict]]                       # → {relevance_score, reasoning, cover_letter[, error]}
    notify: Callable[[VacancyState], Awaitable[None]]                  # отправить черновик с кнопками
    set_status: Callable[..., Awaitable[None]]                         # (vacancy_id, status_name, **поля)
    record_decision: Callable[..., Awaitable[None]]                    # (vacancy_id, decision, feedback, score, revision)
    rewrite: Callable[[dict, str, str], Awaitable[str]]                # (vacancy, письмо, замечание) → новое письмо
    min_score: int = 60


def build_graph(deps: AgentDeps, checkpointer=None):
    async def evaluate(state: VacancyState) -> dict:
        vacancy = state["vacancy"]
        result = await deps.evaluate(vacancy)
        if result.get("error"):
            logger.warning(f"LLM не ответила по вакансии {vacancy['vacancy_id']}, повторю в следующем цикле")
            return {"outcome": "error"}
        logger.info(f"Оценка вакансии {vacancy['vacancy_id']}: {result['relevance_score']}/100")
        # outcome сбрасываем: поток вакансии мог остаться от прошлой попытки, закончившейся ошибкой
        return {"score": int(result["relevance_score"]), "reasoning": result.get("reasoning", ""),
                "cover_letter": result.get("cover_letter", ""), "revisions": 0, "outcome": "",
                "decision": "", "feedback": ""}

    def route_after_evaluate(state: VacancyState) -> str:
        if state.get("outcome") == "error":
            return END
        return "notify" if state["score"] >= deps.min_score else "skip_low"

    async def skip_low(state: VacancyState) -> dict:
        await deps.set_status(state["vacancy"]["vacancy_id"], "SKIPPED", relevance_score=state["score"])
        return {"outcome": "skipped_low_score"}

    async def notify(state: VacancyState) -> dict:
        await deps.notify(state)
        await deps.set_status(state["vacancy"]["vacancy_id"], "NOTIFIED", relevance_score=state["score"],
                              cover_letter_draft=state["cover_letter"])
        return {}

    def wait_decision(state: VacancyState) -> dict:
        # Здесь граф останавливается и ждёт кнопку в Telegram. При возобновлении узел выполняется
        # заново, и interrupt() возвращает присланное решение — поэтому побочных действий в узле нет.
        answer = interrupt({"vacancy_id": state["vacancy"]["vacancy_id"], "revision": state.get("revisions", 0)})
        return {"decision": answer.get("action", "skip"), "feedback": (answer.get("feedback") or "").strip()}

    def route_decision(state: VacancyState) -> str:
        decision = state.get("decision")
        if decision == "approve":
            return "approve"
        if decision == "rewrite" and state.get("feedback") and state.get("revisions", 0) < MAX_REVISIONS:
            return "rewrite"
        if decision == "rewrite":
            return "notify"      # пустое замечание или лимит правок — показываем письмо ещё раз
        return "user_skip"

    async def approve(state: VacancyState) -> dict:
        vacancy_id = state["vacancy"]["vacancy_id"]
        await deps.record_decision(vacancy_id, "approve", "", state["score"], state.get("revisions", 0))
        await deps.set_status(vacancy_id, "CONFIRMED", cover_letter_draft=state["cover_letter"])
        return {"outcome": "approved"}

    async def user_skip(state: VacancyState) -> dict:
        vacancy_id = state["vacancy"]["vacancy_id"]
        await deps.record_decision(vacancy_id, "skip", "", state["score"], state.get("revisions", 0))
        await deps.set_status(vacancy_id, "SKIPPED")
        return {"outcome": "skipped_by_user"}

    async def rewrite(state: VacancyState) -> dict:
        vacancy_id = state["vacancy"]["vacancy_id"]
        await deps.record_decision(vacancy_id, "rewrite", state["feedback"], state["score"], state.get("revisions", 0))
        letter = await deps.rewrite(state["vacancy"], state["cover_letter"], state["feedback"])
        return {"cover_letter": letter, "revisions": state.get("revisions", 0) + 1}

    builder = StateGraph(VacancyState)
    builder.add_node("evaluate", evaluate)
    builder.add_node("skip_low", skip_low)
    builder.add_node("notify", notify)
    builder.add_node("wait_decision", wait_decision)
    builder.add_node("approve", approve)
    builder.add_node("user_skip", user_skip)
    builder.add_node("rewrite", rewrite)

    builder.add_edge(START, "evaluate")
    builder.add_conditional_edges("evaluate", route_after_evaluate, ["notify", "skip_low", END])
    builder.add_edge("skip_low", END)
    builder.add_edge("notify", "wait_decision")
    builder.add_conditional_edges("wait_decision", route_decision, ["approve", "rewrite", "notify", "user_skip"])
    builder.add_edge("rewrite", "notify")
    builder.add_edge("approve", END)
    builder.add_edge("user_skip", END)
    return builder.compile(checkpointer=checkpointer)


def thread_config(vacancy_id: str) -> dict:
    return {"configurable": {"thread_id": f"vacancy-{vacancy_id}"}}


def mermaid(graph) -> Optional[str]:
    """Схема графа в mermaid — для README."""
    try:
        return graph.get_graph().draw_mermaid()
    except Exception:  # noqa: BLE001
        return None
