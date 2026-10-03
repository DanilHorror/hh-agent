"""Критерии кандидата: чего он точно не хочет, что ему важно, какой формат работы.

Файл criteria.txt (рядом с resume.txt, в git не попадает; пример — criteria.example.txt) делится на разделы:

    [нет]        — темы, при которых вакансия не подходит, даже если стек совпадает (1С, C#, гемблинг…)
    [плюс]       — что делает вакансию особенно подходящей
    [формат]     — удалёнка / гибрид / офис
    [компании]   — компании, к которым кандидат относится скептически

Разделы нет / плюс / формат уходят в промпт оценки. Список компаний проверяется кодом, а не моделью:
сравнить название компании надёжнее строкой, чем просить 7B-модель не ошибиться.
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRITERIA_PATH = os.path.join(ROOT, "criteria.txt")
SECTIONS = ("нет", "плюс", "формат", "компании")
COMPANY_SCORE_CAP = 15   # потолок оценки для компаний из списка


def parse(text: str) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {name: [] for name in SECTIONS}
    current = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):   # комментарий — только целая строка: в «C#» решётка значимая
            continue
        header = re.fullmatch(r"\[(.+)\]", line)
        if header:
            name = header.group(1).strip().lower()
            current = name if name in result else None
            continue
        if current:
            result[current].append(line.lstrip("-•* ").strip())
    return result


def load(path: str = CRITERIA_PATH) -> dict[str, list[str]]:
    try:
        with open(path, encoding="utf-8") as f:
            return parse(f.read())
    except FileNotFoundError:
        return {name: [] for name in SECTIONS}


def prompt_block(criteria: dict[str, list[str]]) -> str:
    """Раздел промпта с критериями; пустая строка, если критериев нет."""
    parts = []
    if criteria.get("нет"):
        parts.append("НЕ ПОДХОДИТ (если это основная работа в вакансии — оценка 0–20, даже если стек совпадает):\n"
                     + "\n".join(f"- {x}" for x in criteria["нет"]))
    if criteria.get("плюс"):
        parts.append("ОСОБЕННО ПОДХОДИТ (повышай оценку):\n" + "\n".join(f"- {x}" for x in criteria["плюс"]))
    if criteria.get("формат"):
        parts.append("ФОРМАТ РАБОТЫ:\n" + "\n".join(f"- {x}" for x in criteria["формат"]))
    if not parts:
        return ""
    return ("КРИТЕРИИ КАНДИДАТА (важнее, чем простое совпадение навыков с резюме):\n"
            + "\n\n".join(parts)
            + "\n\nШКАЛА: 0–20 — есть пункт из «не подходит»; 30–50 — QA-вакансия без пунктов из «особенно подходит»"
              " или с неподходящим форматом; 60–80 — стек и формат подходят; 80–100 — подходит и есть пункты"
              " из «особенно подходит». Используй всю шкалу, не ставь всем 80–90.\n")


def _norm(name: str) -> str:
    name = name.lower().replace("ё", "е")
    name = re.sub(r"\b(ооо|оао|зао|пао|ао|ип|гк)\b", " ", name)
    return re.sub(r"[^\w]+", " ", name).strip()


def blocked_company(company: str, criteria: dict[str, list[str]]) -> str | None:
    """Название из списка [компании], если компания вакансии в нём есть (сравнение по словам)."""
    normalized = f" {_norm(company or '')} "
    for entry in criteria.get("компании", []):
        key = _norm(entry)
        if key and f" {key} " in normalized:
            return entry
    return None
