"""Собрать эталонный набор вакансий для разметки (evals/golden_vacancies.json).

    venv\\Scripts\\python.exe scripts\\eval_build_golden.py            50 вакансий
    venv\\Scripts\\python.exe scripts\\eval_build_golden.py --n 60
    venv\\Scripts\\python.exe scripts\\eval_build_golden.py --force    пересобрать (разметка пропадёт)

Берёт вакансии с полным текстом и раскладывает поровну по оценке, которую им раньше дала модель
(0–39, 40–59, 60–79, 80–100), чтобы в наборе были и явные, и пограничные случаи.
Размечать — на демо-странице, вкладка «Разметка».
"""
import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text  # noqa: E402

import database  # noqa: E402
from quality import golden  # noqa: E402


async def fetch_rows() -> list[dict]:
    sql = text(
        "SELECT r.vacancy_id, r.title, r.company, r.url, r.description, v.relevance_score AS score "
        "FROM vacancy_raw r LEFT JOIN vacancies v ON v.vacancy_id = r.vacancy_id")
    async with database.engine.connect() as conn:
        rows = (await conn.execute(sql)).mappings().all()
    await database.engine.dispose()
    return [dict(r) for r in rows]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true", help="перезаписать существующий набор")
    args = parser.parse_args()

    existing = golden.load()
    if existing and not args.force:
        p = golden.progress(existing)
        print(f"Набор уже есть: {p['total']} вакансий, размечено {p['labeled']}. "
              f"Чтобы пересобрать (разметка пропадёт), добавьте --force.")
        return 1

    rows = asyncio.run(fetch_rows())
    items = golden.build_sample(rows, n=args.n, seed=args.seed)
    golden.save(items)
    scored = sum(1 for r in rows if r.get("score") is not None)
    print(f"Вакансий с текстом: {len(rows)} (с оценкой модели: {scored}). В набор взято: {len(items)}.")
    print(f"Файл: {golden.GOLDEN_PATH}")
    print("Дальше: демо-страница → вкладка «Разметка».")
    return 0


if __name__ == "__main__":
    sys.exit(main())
