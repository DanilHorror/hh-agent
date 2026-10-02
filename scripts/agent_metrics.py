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
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import Vacancy, VacancyStatus, async_session, get_agent_decisions  # noqa: E402
from sqlalchemy import func, select  # noqa: E402


def summarize(decisions: list) -> dict:
    """decisions — записи agent_decisions по порядку. Итог по каждой вакансии — последнее решение."""
    by_vacancy = defaultdict(list)
    for d in decisions:
        by_vacancy[d.vacancy_id].append(d)

    final = {vid: items[-1] for vid, items in by_vacancy.items() if items[-1].decision in ("approve", "skip")}
    approved = [d for d in final.values() if d.decision == "approve"]
    skipped = [d for d in final.values() if d.decision == "skip"]
    rewrites_before_approve = [sum(1 for x in by_vacancy[d.vacancy_id] if x.decision == "rewrite") for d in approved]

    def avg(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    return {
        "decided": len(final),
        "approved": len(approved),
        "skipped": len(skipped),
        "approved_first_draft": sum(1 for r in rewrites_before_approve if r == 0),
        "avg_rewrites_before_approve": avg(rewrites_before_approve),
        "agreement": len(approved) / len(final) if final else None,
        "avg_score_approved": avg([d.relevance_score for d in approved]),
        "avg_score_skipped": avg([d.relevance_score for d in skipped]),
    }


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
