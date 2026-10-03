"""Просмотреть эталонную разметку выдумок в письмах (evals/letter_labels.json) и поправить её.

    venv\\Scripts\\python.exe scripts\\eval_judge_review.py               10 случайных писем
    venv\\Scripts\\python.exe scripts\\eval_judge_review.py --n 20

По этой разметке считается качество судьи в scripts/eval_letters.py. Скрипт показывает письмо и
выдумки из разметки; ответ y — разметка верна, n — неверна (тогда выдумки можно вписать заново
или оставить пустыми), s — пропустить, q — выйти. Проверенные письма помечаются и не повторяются.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from quality import golden  # noqa: E402

RUNS_DIR = os.path.join(ROOT, "evals", "scoring_runs")
LABELS_PATH = os.path.join(ROOT, "evals", "letter_labels.json")


def load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels: dict) -> None:
    tmp = LABELS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=1)
    os.replace(tmp, LABELS_PATH)


def pick(labels: dict, n: int, seed: int = 42) -> list[tuple[str, str]]:
    """Поровну писем «с выдумкой» и «чистых» среди ещё не проверенных."""
    todo = [(run, vid, v) for run, part in labels.items() for vid, v in part.items() if not v.get("reviewed")]
    rnd = random.Random(seed)
    rnd.shuffle(todo)
    invented = [(r, v) for r, v, x in todo if x["invented"]]
    clean = [(r, v) for r, v, x in todo if not x["invented"]]
    chosen = invented[:n // 2] + clean[:n - n // 2]
    chosen += [x for x in invented + clean if x not in chosen][:n - len(chosen)]
    rnd.shuffle(chosen)
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    labels = load_json(LABELS_PATH, {})
    if not labels:
        print("Нет разметки evals/letter_labels.json.")
        return 1
    titles = {i["vacancy_id"]: i["title"] for i in golden.load(golden.GOLDEN_PATH)}
    chosen = pick(labels, args.n)
    if not chosen:
        print("Все письма уже проверены.")
        return 0
    print("Ответы: y — разметка верна, n — неверна, s — пропустить, q — выход.")
    checked = wrong = 0
    for k, (run, vid) in enumerate(chosen, start=1):
        label = labels[run][vid]
        letter = load_json(os.path.join(RUNS_DIR, f"{run}.json"), {}).get("results", {}).get(vid, {}).get("cover_letter", "")
        print("\n" + "=" * 80)
        print(f"[{k}/{len(chosen)}] {run} · {titles.get(vid, vid)}")
        print("-" * 80)
        print(letter)
        print("-" * 80)
        if label["invented"]:
            print("РАЗМЕТКА: есть выдумки:")
            for fact in label["facts"]:
                print(f"  • {fact}")
        else:
            print("РАЗМЕТКА: выдумок нет.")
        while True:
            answer = input("Разметка верна? [y/n/s/q]: ").strip().lower()
            if answer in ("y", "n", "s", "q", "д", "н"):
                break
        if answer == "q":
            break
        if answer == "s":
            continue
        checked += 1
        if answer in ("n", "н"):
            wrong += 1
            facts = input("Выдумки через «;» (пусто — выдумок нет): ").strip()
            label["facts"] = [f.strip() for f in facts.split(";") if f.strip()]
            label["invented"] = bool(label["facts"])
        label["reviewed"] = True
        save_labels(labels)
    if checked:
        print(f"\nПроверено {checked}, исправлено {wrong}. Файл: {LABELS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
