"""Метрики агента по решениям человека (таблица agent_decisions). Без зависимостей от LangGraph —
используются и скриптом scripts/agent_metrics.py, и API (/stats)."""
from collections import defaultdict


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
