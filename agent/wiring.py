"""Подключение графа к боту: настоящая модель, база и Telegram."""
from __future__ import annotations

from config import database_url, settings
from database import VacancyStatus, save_agent_decision, update_vacancy_status
from llm.brain import evaluate_and_draft, rewrite_letter_guarded

from .checkpoint import make_checkpointer
from .graph import AgentDeps, VacancyState
from .runner import VacancyAgent


async def create_agent(notifier) -> tuple[VacancyAgent, callable]:
    async def notify(state: VacancyState) -> None:
        vacancy = state["vacancy"]
        await notifier.send_for_approval(
            vacancy["vacancy_id"], vacancy,
            {"relevance_score": state["score"], "reasoning": state.get("reasoning", ""),
             "cover_letter": state["cover_letter"], "letter_warning": state.get("letter_warning", "")},
            revision=state.get("revisions", 0),
        )

    async def set_status(vacancy_id: str, status: str, **fields) -> None:
        await update_vacancy_status(vacancy_id=vacancy_id, status=VacancyStatus[status], **fields)

    deps = AgentDeps(
        evaluate=evaluate_and_draft,
        notify=notify,
        set_status=set_status,
        record_decision=save_agent_decision,
        rewrite=rewrite_letter_guarded,
        min_score=settings.min_relevance_score,
    )
    checkpointer, close = await make_checkpointer(database_url())
    return VacancyAgent(deps, checkpointer), close
