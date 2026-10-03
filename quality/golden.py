"""Эталонный набор вакансий (golden set): человек отметил, подходит ли вакансия.

Файл evals/golden_vacancies.json хранится в git вместе с текстами вакансий — оценку можно повторить
через месяц, даже если в базе всё поменялось. Оценку модели в набор не кладём специально:
при разметке человек её не видит и не подстраивается под неё.
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN_PATH = os.path.join(ROOT, "evals", "golden_vacancies.json")
LABELS = ("fit", "not_fit", "unsure")

# Корзины по оценке модели — чтобы в набор попали и «уверенно да», и «пограничные», и «уверенно нет»
BUCKETS = [(0, 40), (40, 60), (60, 80), (80, 101)]


def load(path: str = GOLDEN_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save(items: list[dict], path: str = GOLDEN_PATH) -> None:
    """Запись через временный файл: оборванная запись не испортит разметку."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def build_sample(rows: list[dict], n: int = 50, seed: int = 42) -> list[dict]:
    """Выбрать n вакансий поровну из корзин по оценке модели; недостающее — из вакансий без оценки.

    rows: vacancy_id, title, company, url, description, score (может быть None)."""
    rng = random.Random(seed)
    rows = [r for r in rows if (r.get("description") or "").strip()]
    scored = [r for r in rows if r.get("score") is not None]
    unscored = [r for r in rows if r.get("score") is None]

    per_bucket = n // len(BUCKETS)
    chosen: list[dict] = []
    leftovers: list[dict] = []
    for low, high in BUCKETS:
        bucket = [r for r in scored if low <= r["score"] < high]
        rng.shuffle(bucket)
        chosen.extend(bucket[:per_bucket])
        leftovers.extend(bucket[per_bucket:])
    rng.shuffle(leftovers)
    rng.shuffle(unscored)
    for pool in (leftovers, unscored):
        while len(chosen) < n and pool:
            chosen.append(pool.pop())
    rng.shuffle(chosen)   # порядок разметки не должен подсказывать оценку модели
    return [{"vacancy_id": str(r["vacancy_id"]), "title": r.get("title") or "", "company": r.get("company") or "",
             "url": r.get("url") or "", "description": r["description"], "label": None, "note": ""}
            for r in chosen[:n]]


def set_label(items: list[dict], vacancy_id: str, label: Optional[str], note: str = "") -> dict:
    if label is not None and label not in LABELS:
        raise ValueError(f"label должен быть одним из {LABELS}")
    for item in items:
        if item["vacancy_id"] == vacancy_id:
            item["label"] = label
            item["note"] = note
            return item
    raise KeyError(vacancy_id)


def progress(items: list[dict]) -> dict:
    labeled = [i for i in items if i.get("label")]
    return {"total": len(items), "labeled": len(labeled),
            "fit": sum(1 for i in labeled if i["label"] == "fit"),
            "not_fit": sum(1 for i in labeled if i["label"] == "not_fit"),
            "unsure": sum(1 for i in labeled if i["label"] == "unsure")}
