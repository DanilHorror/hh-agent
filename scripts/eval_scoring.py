"""Точность оценки вакансий и сравнение моделей на эталонном наборе (evals/golden_vacancies.json).

    venv\\Scripts\\python.exe scripts\\eval_scoring.py                          модель из .env
    venv\\Scripts\\python.exe scripts\\eval_scoring.py --providers ollama gigachat
    venv\\Scripts\\python.exe scripts\\eval_scoring.py --providers gigachat --rerun   пересчитать заново
    venv\\Scripts\\python.exe scripts\\eval_scoring.py --providers ollama --tag criteria   отдельный эксперимент

Каждый прогон (модель + тег) кэшируется отдельно, в итоговую таблицу попадают все прогоны — так видно,
как менялось качество от эксперимента к эксперименту.

Для каждой модели прогоняет оценку (тот же промпт, что в боте) по размеченным вакансиям и считает
precision / recall / F1 при пороге из .env и при лучшем пороге, а также AUC — насколько модель вообще
отличает подходящие вакансии от неподходящих. Ответы моделей кэшируются в evals/scoring_runs/,
прерванный прогон продолжается с того же места.
Итоговая таблица — evals/scoring_results.md.
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

from config import settings  # noqa: E402
from llm import brain  # noqa: E402
from llm.providers import create_provider, set_provider  # noqa: E402
from quality import golden  # noqa: E402
from quality.scoring import RecordingProvider, best_threshold, confusion, separation, sweep  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "evals", "scoring_runs")
RESULTS = os.path.join(ROOT, "evals", "scoring_results.md")


def _pct(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def run_key(name: str, tag: str) -> str:
    return f"{name}-{tag}" if tag else name


async def run_provider(name: str, items: list[dict], rerun: bool, tag: str = "") -> dict:
    """Оценить все вакансии моделью name. Возвращает {vacancy_id: результат} + статистику."""
    try:
        provider = RecordingProvider(create_provider(name))
    except (ValueError, ImportError) as e:
        print(f"[{name}] пропускаю: {e}")
        return {}
    os.makedirs(RUNS_DIR, exist_ok=True)
    cache_path = os.path.join(RUNS_DIR, f"{run_key(name, tag)}.json")
    cache = {}
    if os.path.exists(cache_path) and not rerun:
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
    base = cache.get("usage", {})   # токены прошлых запусков (из кэша)
    set_provider(provider)
    try:
        todo = [i for i in items if i["vacancy_id"] not in cache.get("results", {})]
        results = cache.get("results", {})
        print(f"[{name}] {provider.model}: в кэше {len(results)}, осталось {len(todo)}")
        for n, item in enumerate(todo, start=1):
            started = time.time()
            result = await brain.evaluate_and_draft(
                {"title": item["title"], "company": item["company"], "description": item["description"]})
            results[item["vacancy_id"]] = {
                "score": result.get("relevance_score", 0), "error": bool(result.get("error")),
                "seconds": round(time.time() - started, 2), "reasoning": result.get("reasoning", ""),
                "cover_letter": result.get("cover_letter", ""),
            }
            print(f"  {n}/{len(todo)} {item['title'][:50]} → {results[item['vacancy_id']]['score']}")
            cache = {"provider": name, "tag": tag, "model": provider.model, "results": results,
                     "usage": {"calls": base.get("calls", 0) + provider.usage.calls,
                               "prompt_tokens": base.get("prompt_tokens", 0) + provider.usage.prompt_tokens,
                               "completion_tokens": base.get("completion_tokens", 0) + provider.usage.completion_tokens}}
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=1)
    finally:
        set_provider(None)
        await provider.aclose()
    return cache


def summarize(name: str, run: dict, items: list[dict]) -> dict:
    labels = {i["vacancy_id"]: i["label"] for i in items}
    results = run.get("results", {})
    ok = {vid: r for vid, r in results.items() if not r["error"] and vid in labels}
    pairs = [(ok[vid]["score"], labels[vid]) for vid in ok]
    at_current = confusion(pairs, settings.min_relevance_score)
    best_t, best = best_threshold(pairs)
    seconds = sorted(r["seconds"] for r in ok.values())
    usage = run.get("usage", {})
    return {
        "name": name, "model": run.get("model", "?"), "n": len(pairs),
        "errors": sum(1 for r in results.values() if r["error"]),
        "auc": separation(pairs), "current": at_current, "best_t": best_t, "best": best,
        "median_s": seconds[len(seconds) // 2] if seconds else None,
        "tokens_per_call": ((usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)) /
                            usage["calls"]) if usage.get("calls") else None,
        "sweep": sweep(pairs), "pairs": pairs,
    }


def write_report(rows: list[dict], items: list[dict]) -> None:
    progress = golden.progress(items)
    t = settings.min_relevance_score
    lines = [
        "# Точность оценки вакансий",
        "",
        f"Дата: {date.today().isoformat()}. Эталонный набор: {progress['total']} вакансий, размечено вручную "
        f"{progress['fit']} «подходит», {progress['not_fit']} «не подходит» ({progress['unsure']} «не уверен» "
        f"не учитываются). Промпт тот же, что в боте.",
        "",
        f"| Модель | AUC | Precision @{t} | Recall @{t} | F1 @{t} | Лучший порог | Precision | Recall | F1 | "
        f"Медиана, с | Токенов на вакансию | Ошибок |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        c, b = r["current"], r["best"]
        lines.append(
            f"| {r['name']} ({r['model']}) | {_pct(r['auc'])} | {_pct(c.precision)} | {_pct(c.recall)} | "
            f"{_pct(c.f1)} | {r['best_t']} | {_pct(b.precision)} | {_pct(b.recall)} | {_pct(b.f1)} | "
            f"{r['median_s'] or '—'} | {int(r['tokens_per_call']) if r['tokens_per_call'] else '—'} | {r['errors']} |")
    lines += ["", "AUC — насколько модель отличает подходящие вакансии от неподходящих независимо от порога "
              "(0.5 — не отличает, 1.0 — идеально).", ""]
    for r in rows:
        lines += [f"## Порог → метрики: {r['name']}", "", "| Порог | Предложено | Precision | Recall | F1 |",
                  "|---|---|---|---|---|"]
        for threshold, c in r["sweep"]:
            lines.append(f"| {threshold} | {c.tp + c.fp} | {_pct(c.precision)} | {_pct(c.recall)} | {_pct(c.f1)} |")
        lines.append("")
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nТаблица: {RESULTS}")


def all_runs() -> list[tuple[str, dict]]:
    runs = []
    for filename in sorted(os.listdir(RUNS_DIR)) if os.path.isdir(RUNS_DIR) else []:
        if filename.endswith(".json"):
            with open(os.path.join(RUNS_DIR, filename), encoding="utf-8") as f:
                runs.append((filename[:-5], json.load(f)))
    return runs


async def main(providers: list[str], rerun: bool, tag: str = "") -> int:
    items = [i for i in golden.load(golden.GOLDEN_PATH) if i.get("label") in ("fit", "not_fit")]
    if not items:
        print("Нет размеченных вакансий (fit / not_fit). Сначала разметьте набор на демо-странице.")
        return 1
    for name in providers:
        await run_provider(name, items, rerun, tag)
    rows = [summarize(key, run, items) for key, run in all_runs()]
    if not rows:
        return 1
    print()
    for r in rows:
        c, b = r["current"], r["best"]
        print(f"{r['name']:22} AUC {_pct(r['auc'])} | порог {settings.min_relevance_score}: P {_pct(c.precision)} "
              f"R {_pct(c.recall)} F1 {_pct(c.f1)} | лучший порог {r['best_t']}: P {_pct(b.precision)} "
              f"R {_pct(b.recall)} F1 {_pct(b.f1)} | медиана {r['median_s']} с")
    write_report(rows, golden.load(golden.GOLDEN_PATH))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--providers", nargs="+", default=[settings.llm_provider],
                        help="ollama gigachat yandexgpt openai")
    parser.add_argument("--rerun", action="store_true", help="не брать ответы из кэша")
    parser.add_argument("--tag", default="", help="название эксперимента: отдельный кэш и строка в таблице")
    parser.add_argument("--no-company-filter", action="store_true",
                        help="не применять список компаний из criteria.txt (оценивать только промпт)")
    args = parser.parse_args()
    if args.no_company_filter:
        brain.APPLY_COMPANY_FILTER = False
    sys.exit(asyncio.run(main(args.providers, args.rerun, args.tag)))
