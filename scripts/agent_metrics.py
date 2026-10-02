"""Метрики агента по решениям человека в Telegram.

    venv\\Scripts\\python.exe scripts\\agent_metrics.py

Считает:
    - сколько черновиков принято с первой версии письма (без «Переписать»);
    - сколько раз в среднем просили переписать письмо до принятия;
    - согласие модели с человеком: среди вакансий, которые модель сочла подходящими (оценка ≥ порога),
      какую долю человек тоже принял, и средняя оценка модели у принятых и пропущенных;
    - сколько вакансий ещё ждут решения.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.metrics import summarize  # noqa: E402
from database import Vacancy, VacancyStatus, async_session, get_agent_decisions  # noqa: E402
from sqlalchemy import func, select  # noqa: E402


def _fmt(value, pct=False):
    if value is None:
        return "—"
    return f"{value:.0%}" if pct else (f"{value:.1f}" if isinstance(value, float) else str(value))


async def main() -> int:
    decisions = await get_agent_decisions()
    async with async_session() as session:
        waiting = (await session.execute(
            select(func.count()).select_from(Vacancy).where(Vacancy.status == VacancyStatus.NOTIFIED))).scalar_one()
    m = summarize(decisions)
    print(f"Решений по вакансиям: {m['decided']} (принято {m['approved']}, пропущено {m['skipped']}); "
          f"без решения в Telegram: {waiting} (включая отправленные до запуска агента, без кнопок)")
    if m["approved"]:
        print(f"Принято с первой версии письма: {m['approved_first_draft']} из {m['approved']} "
              f"({m['approved_first_draft'] / m['approved']:.0%})")
        print(f"Переписываний до принятия в среднем: {_fmt(m['avg_rewrites_before_approve'])}")
    print(f"Согласие с моделью (из предложенных моделью вакансий принято): {_fmt(m['agreement'], pct=True)}")
    print(f"Средняя оценка модели: у принятых {_fmt(m['avg_score_approved'])}, "
          f"у пропущенных {_fmt(m['avg_score_skipped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
