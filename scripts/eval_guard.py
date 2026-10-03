"""Прогнать стража письма по готовым письмам из прогона оценки и сохранить исправленные.

    venv\\Scripts\\python.exe scripts\\eval_guard.py                                   прогоны по умолчанию
    venv\\Scripts\\python.exe scripts\\eval_guard.py --runs gigachat-letters-v2

Берёт письма из evals/scoring_runs/<прогон>.json. Письмо исправляет та же модель, что его написала
(ollama-… → ollama, gigachat-… → gigachat), проверяет судья LETTER_JUDGE_PROVIDER (по умолчанию gigachat).
Результат — evals/scoring_runs/<прогон>-guard.json в том же формате, поэтому его можно проверить как обычный прогон:

    venv\\Scripts\\python.exe scripts\\eval_letters.py --runs gigachat-letters-v2 gigachat-letters-v2-guard

Внимание: судья внутри стража и судья в eval_letters — одна модель, поэтому её оценка «после» завышена.
Честная цифра — по эталонной разметке evals/letter_labels.json.
"""
import argparse
import asyncio
import json
import os
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import settings  # noqa: E402
from llm import brain  # noqa: E402
from llm.providers import create_provider, set_provider  # noqa: E402
from quality import golden  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "evals", "scoring_runs")
DEFAULT_RUNS = ["ollama-letters-v2", "gigachat-letters-v2"]


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


async def guard_run(run: str, vacancies: dict, judge) -> dict:
    source = load_json(os.path.join(RUNS_DIR, f"{run}.json"), None)
    if source is None:
        print(f"нет прогона evals/scoring_runs/{run}.json")
        return {}
    out_path = os.path.join(RUNS_DIR, f"{run}-guard.json")
    out = load_json(out_path, {"provider": source.get("provider"), "model": source.get("model"),
                               "source_run": run, "results": {}})
    done = out["results"]
    todo = [(vid, r) for vid, r in source["results"].items() if not r.get("error") and vid not in done]
    writer = create_provider(source.get("provider") or run.split("-")[0])
    set_provider(writer)
    print(f"[страж] {run}: пишет {writer.name}/{writer.model}, судья {judge.name}/{judge.model}; "
          f"готово {len(done)}, осталось {len(todo)}")
    try:
        for n, (vid, r) in enumerate(todo, start=1):
            vacancy = vacancies.get(vid, {})
            started = time.time()
            checked = await brain.guard_cover_letter(
                {"title": vacancy.get("title", ""), "company": vacancy.get("company", "")}, r["cover_letter"], judge=judge)
            if checked is None:
                print(f"  {n}/{len(todo)} {vid}: страж не сработал, пропускаю")
                continue
            done[vid] = {**r, "cover_letter": checked.letter, "original_letter": r["cover_letter"],
                         "guard": checked.as_dict(), "guard_seconds": round(time.time() - started, 1)}
            state = "чисто" if checked.clean else (checked.warning or "штампы: " + ", ".join(checked.cliches))
            print(f"  {n}/{len(todo)} {vacancy.get('title', vid)[:40]} → исправлений {checked.rounds}, {state[:100]}")
            save_json(out_path, out)
    finally:
        set_provider(None)
        await writer.aclose()
    return out


def summary(run: str, out: dict) -> str:
    items = list(out.get("results", {}).values())
    if not items:
        return f"{run}: нет данных"
    first_clean = sum(1 for i in items if i["guard"]["rounds"] == 0 and i["guard"]["clean"])
    clean = sum(1 for i in items if i["guard"]["clean"])
    no_invented = sum(1 for i in items if not i["guard"]["invented"])
    seconds = statistics.median(i["guard_seconds"] for i in items)
    return (f"{run:22} писем {len(items)} | сразу чистых {first_clean} | после стража без выдумок (по судье) "
            f"{no_invented} | полностью чистых {clean} | медиана {seconds} с на письмо")


async def main(runs: list[str]) -> int:
    settings.letter_guard = True
    vacancies = {i["vacancy_id"]: i for i in golden.load(golden.GOLDEN_PATH)}
    judge = create_provider(settings.letter_judge_provider)
    outs = {}
    try:
        for run in runs:
            outs[run] = await guard_run(run, vacancies, judge)
    finally:
        await judge.aclose()
    print()
    for run, out in outs.items():
        print(summary(run, out))
    print("\nДальше: scripts/eval_letters.py --runs " + " ".join(f"{r} {r}-guard" for r in runs))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.runs)))
