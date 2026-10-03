"""Проверка сопроводительных писем из прогонов оценки (evals/scoring_runs/).

    venv\\Scripts\\python.exe scripts\\eval_letters.py                                   прогоны по умолчанию
    venv\\Scripts\\python.exe scripts\\eval_letters.py --runs ollama-companies gigachat-companies
    venv\\Scripts\\python.exe scripts\\eval_letters.py --no-judge                        только проверки кодом
    venv\\Scripts\\python.exe scripts\\eval_letters.py --rejudge                         судить заново
    venv\\Scripts\\python.exe scripts\\eval_letters.py --judge-version v1                старый судья (общий вердикт)

Письма генерирует scripts/eval_scoring.py (тот же промпт, что в боте) — здесь они только проверяются:
кодом (заглушки, фамилия, штампы, длина, куски резюме) и моделью-судьёй (выдуманные факты).
Вердикты судьи кэшируются в evals/letter_judgments/ (v2 — в подпапке v2/), отчёт — evals/letters_results.md.
Качество судьи считается по эталонной разметке выдумок evals/letter_labels.json
(просмотреть и поправить разметку: scripts/eval_judge_review.py).
"""
import argparse
import asyncio
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from llm import brain  # noqa: E402
from llm.providers import create_provider  # noqa: E402
from llm.providers.base import with_retries  # noqa: E402
from quality import golden, letters  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "evals", "scoring_runs")
JUDGE_DIR = os.path.join(ROOT, "evals", "letter_judgments")
LABELS_PATH = os.path.join(ROOT, "evals", "letter_labels.json")
RESULTS = os.path.join(ROOT, "evals", "letters_results.md")
DEFAULT_RUNS = ["ollama-companies", "gigachat-companies"]


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def judge_cache_path(run: str, version: str) -> str:
    return os.path.join(JUDGE_DIR, f"{run}.json") if version == "v1" else os.path.join(JUDGE_DIR, version, f"{run}.json")


def run_letters(run: str) -> dict[str, str]:
    data = load_json(os.path.join(RUNS_DIR, f"{run}.json"), None)
    if data is None:
        raise FileNotFoundError(f"нет прогона evals/scoring_runs/{run}.json — сначала scripts/eval_scoring.py")
    return {vid: r.get("cover_letter", "") for vid, r in data.get("results", {}).items() if not r.get("error")}


async def judge_run(run: str, texts: dict[str, str], titles: dict[str, str], judge_name: str, rejudge: bool,
                    version: str = "v2") -> dict:
    path = judge_cache_path(run, version)
    judge_fn = letters.JUDGES[version]
    cache = {} if rejudge else load_json(path, {})
    results = cache.get("results", {})
    todo = [vid for vid in texts if vid not in results]
    if not todo:
        return results
    provider = create_provider(judge_name)
    print(f"[судья {version} {provider.name}/{provider.model}] {run}: в кэше {len(results)}, осталось {len(todo)}")
    try:
        for n, vid in enumerate(todo, start=1):
            try:
                j = await with_retries(lambda: judge_fn(
                    provider, brain.USER_RESUME, titles.get(vid, ""), texts[vid]))
            except Exception as e:  # noqa: BLE001 — один сбой не должен останавливать весь прогон
                print(f"  {n}/{len(todo)} {vid}: ошибка судьи ({type(e).__name__}: {str(e)[:120]}), пропускаю")
                continue
            results[vid] = j.model_dump()
            mark = "выдумка: " + "; ".join(j.invented_facts)[:120] if j.verdict == "invented" else "ok"
            print(f"  {n}/{len(todo)} {titles.get(vid, vid)[:40]} → {mark}")
            save_json(path, {"run": run, "judge": f"{provider.name}/{provider.model}", "version": version,
                             "results": results})
    finally:
        await provider.aclose()
    return results


def judge_quality_lines(runs: list[str]) -> list[str]:
    """Сравнить судей (v1, v2) с эталонной разметкой выдумок."""
    labels = load_json(LABELS_PATH, {})
    if not labels:
        return ["Эталонной разметки выдумок нет (evals/letter_labels.json) — качество судьи не посчитано."]
    lines = ["| Судья | Писем | Нашёл выдумок (recall) | Обвинил зря (1 − precision) | Совпадение с разметкой |",
             "|---|---|---|---|---|"]
    for version in letters.JUDGES:
        q = letters.JudgeQuality()
        for run in runs:
            part = letters.judge_quality(labels.get(run, {}), load_json(judge_cache_path(run, version), {}).get("results", {}))
            q.tp, q.fp, q.fn, q.tn = q.tp + part.tp, q.fp + part.fp, q.fn + part.fn, q.tn + part.tn
        total = q.tp + q.fp + q.fn + q.tn
        if not total:
            continue
        lines.append(f"| {version} | {total} | {pct(q.tp, q.tp + q.fn)} | {pct(q.fp, q.tp + q.fp)} | "
                     f"{pct(q.tp + q.tn, total)} |")
    truth = [v for run in runs for v in labels.get(run, {}).values()]
    lines += ["", f"По разметке выдумки есть в {pct(sum(v['invented'] for v in truth), len(truth))} писем."]
    for run in runs:
        part = labels.get(run, {})
        if part:
            lines.append(f"- {run}: {pct(sum(v['invented'] for v in part.values()), len(part))}")
    return lines


def pct(part: int, total: int) -> str:
    return f"{part}/{total} ({part / total:.0%})" if total else "—"


def write_report(rows: list[tuple[str, letters.LetterStats]], judge: str | None) -> None:
    lines = [
        "# Проверка сопроводительных писем", "",
        f"Дата: {date.today().isoformat()}. Письма — из прогонов `scripts/eval_scoring.py` "
        f"(эталонный набор, тот же промпт, что в боте). Судья выдумок: {judge or 'не запускался'}.", "",
        "| Прогон | Писем | Прошли проверки кодом | Выдумки (судья) | Прошли всё | Медиана длины |",
        "|---|---|---|---|---|---|",
    ]
    for run, s in rows:
        median = sorted(s.lengths)[len(s.lengths) // 2] if s.lengths else "—"
        lines.append(f"| {run} | {s.total} | {pct(s.passed_checks, s.total)} | "
                     f"{pct(s.invented, s.judged) if s.judged else '—'} | "
                     f"{pct(s.passed_all, s.total) if s.judged else '—'} | {median} |")
    lines += ["", "## Какие проверки не прошли (число писем)", "",
              "| Прогон | " + " | ".join(letters.CHECK_NAMES.values()) + " |",
              "|---" * (len(letters.CHECK_NAMES) + 1) + "|"]
    for run, s in rows:
        lines.append(f"| {run} | " + " | ".join(str(s.by_check.get(k, 0)) for k in letters.CHECK_NAMES) + " |")
    lines += ["", "## Частые штампы", ""]
    for run, s in rows:
        top = sorted(s.cliches.items(), key=lambda kv: -kv[1])[:8]
        lines.append(f"- **{run}**: " + (", ".join(f"«{k}» — {v}" for k, v in top) if top else "нет"))
    lines += ["", "## Насколько можно верить судье", ""] + judge_quality_lines([r for r, _ in rows])
    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nОтчёт: {RESULTS}")


async def main(runs: list[str], judge_name: str | None, rejudge: bool, version: str = "v2") -> int:
    titles = {i["vacancy_id"]: i["title"] for i in golden.load(golden.GOLDEN_PATH)}
    rows = []
    for run in runs:
        try:
            texts = run_letters(run)
        except FileNotFoundError as e:
            print(e)
            continue
        judgments = await judge_run(run, texts, titles, judge_name, rejudge, version) if judge_name else {}
        rows.append((run, letters.summarize(texts, judgments, brain.USER_NAME, brain.USER_SURNAME)))
    if not rows:
        return 1
    print()
    for run, s in rows:
        judged = f" | выдумки {pct(s.invented, s.judged)} | прошли всё {pct(s.passed_all, s.total)}" if s.judged else ""
        print(f"{run:24} проверки кодом {pct(s.passed_checks, s.total)}{judged}")
        print(f"{'':24} не прошли: " + ", ".join(f"{letters.CHECK_NAMES[k]} {v}" for k, v in s.by_check.items()))
    write_report(rows, f"{judge_name} ({version})" if judge_name else None)
    print("\n".join(judge_quality_lines([r for r, _ in rows])))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="имена файлов из evals/scoring_runs без .json")
    parser.add_argument("--judge", default="gigachat", help="модель-судья (по умолчанию gigachat)")
    parser.add_argument("--no-judge", action="store_true", help="только проверки кодом")
    parser.add_argument("--rejudge", action="store_true", help="не брать вердикты судьи из кэша")
    parser.add_argument("--judge-version", default="v2", choices=sorted(letters.JUDGES),
                        help="v1 — общий вердикт, v2 — проверка каждого утверждения (по умолчанию)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args.runs, None if args.no_judge else args.judge, args.rejudge, args.judge_version)))
