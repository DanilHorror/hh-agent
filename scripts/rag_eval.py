"""Оценка качества поиска на размеченных вопросах (evals/rag_questions.json).

Для каждого вопроса известны правильные вакансии. Сравниваем четыре режима поиска:
    Hit@5        — в какой доле вопросов хотя бы одна правильная вакансия попала в первые 5;
    Precision@5  — какая доля из первых 5 вакансий правильная;
    MRR          — насколько высоко стоит первая правильная (1 — всегда первая).

    venv\\Scripts\\python.exe scripts\\rag_eval.py
    venv\\Scripts\\python.exe scripts\\rag_eval.py --modes hybrid hybrid_rerank --show-misses
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from rag.metrics import hit_at_k, mean, reciprocal_rank  # noqa: E402
from rag.search import CANDIDATES, MODES, search, unique_vacancies  # noqa: E402

QUESTIONS = os.path.join(ROOT, "evals", "rag_questions.json")
RESULTS = os.path.join(ROOT, "evals", "rag_results.md")
K = 5


def precision_at_k(ranked: list[str], relevant: set[str], k: int = K) -> float:
    top = ranked[:k]
    return len(set(top) & relevant) / len(top) if top else 0.0


async def evaluate(questions: list[dict], modes: list[str], show_misses: bool) -> list[dict]:
    rows = []
    for mode in modes:
        hits, precisions, rrs, seconds = [], [], [], []
        by_type: dict[str, list[float]] = {}
        misses = []
        for q in questions:
            relevant = set(map(str, q["relevant"]))
            started = time.time()
            result = await search(q["question"], k=CANDIDATES, mode=mode)
            seconds.append(time.time() - started)
            ranked = unique_vacancies(result.hits)
            hits.append(hit_at_k(ranked, relevant, K))
            precisions.append(precision_at_k(ranked, relevant, K))
            rrs.append(reciprocal_rank(ranked, relevant))
            by_type.setdefault(q.get("type", "other"), []).append(rrs[-1])
            if hits[-1] == 0:
                misses.append(q)
        rows.append({"mode": mode, "hit": mean(hits), "precision": mean(precisions), "mrr": mean(rrs),
                     "mrr_keyword": mean(by_type.get("keyword", [])),
                     "mrr_paraphrase": mean(by_type.get("paraphrase", [])),
                     "latency": sorted(seconds)[len(seconds) // 2]})
        print(f"{mode:15} Hit@5 {mean(hits):.2f}  Precision@5 {mean(precisions):.2f}  MRR {mean(rrs):.2f}  "
              f"(keyword {rows[-1]['mrr_keyword']:.2f} / paraphrase {rows[-1]['mrr_paraphrase']:.2f})  "
              f"медиана {rows[-1]['latency']:.2f} с")
        if show_misses:
            for q in misses:
                print(f"    промах: {q['id']}: {q['question']}")
    return rows


def write_report(rows: list[dict], n_questions: int) -> None:
    lines = [
        "# Качество поиска RAG",
        "",
        f"Дата: {date.today().isoformat()}. Вопросов: {n_questions} (`evals/rag_questions.json`). "
        "Метрики на уровне вакансий, топ-5.",
        "",
        "| Режим | Hit@5 | Precision@5 | MRR | MRR: с названием технологии | MRR: перефразированные | Медиана, с |",
        "|---|---|---|---|---|---|---|",
    ]
    names = {"vector": "Только векторы", "text": "Только слова (полнотекстовый)",
             "hybrid": "Гибрид (RRF)", "hybrid_rerank": "Гибрид + реранкер"}
    for r in rows:
        lines.append(f"| {names.get(r['mode'], r['mode'])} | {r['hit']:.2f} | {r['precision']:.2f} | "
                     f"{r['mrr']:.2f} | {r['mrr_keyword']:.2f} | {r['mrr_paraphrase']:.2f} | {r['latency']:.2f} |")
    lines += ["", "Разметка: правильные вакансии отобраны по ключевым словам и частично проверены вручную "
              "(`evals/rag_questions_review.md`): 20 вопросов с названием технологии и 20 перефразированных. "
              "Ключевые слова в разметке дают фору поиску по словам, но он всё равно слабее: слова вроде "
              "«тестирование» есть почти в каждом фрагменте и размывают ранжирование."]
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nТаблица сохранена: {RESULTS}")


async def main(modes: list[str], show_misses: bool) -> int:
    with open(QUESTIONS, encoding="utf-8") as f:
        questions = [q for q in json.load(f) if q.get("relevant")]
    print(f"Вопросов с разметкой: {len(questions)}\n")
    rows = await evaluate(questions, modes, show_misses)
    write_report(rows, len(questions))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--show-misses", action="store_true", help="показать вопросы, где ничего не нашлось")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.modes, args.show_misses)))
