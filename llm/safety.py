"""Защита от мусора и prompt injection в тексте вакансий (INPUT_DEFENSE).

Три слоя — ни один не надёжен в одиночку:
    1. вход: мусор отсекается кодом до вызова модели, строки с явными командами для ИИ вырезаются;
    2. промпт: текст вакансии помечен как данные постороннего человека, правило повторено после текста
       (см. llm/brain.py);
    3. выход: из письма вырезаются предложения со ссылками, контактами, зарплатой и «работаю бесплатно» —
       в письме кандидата им взяться неоткуда, а значит, их подсунул текст вакансии.
"""
from __future__ import annotations

import re

# ---------- 1. мусор вместо вакансии ----------
# Пороги подобраны так, чтобы ни одна из 50 вакансий эталонного набора не была отсечена
# (у реальных: от 1400 символов, от 159 слов, разнообразие слов ≥ 0.68, доля букв ≥ 0.8).
MIN_CHARS = 300
MIN_WORDS = 40
MIN_UNIQUE_RATIO = 0.4    # доля разных слов среди первых 200: повторы «string string» дают ~0
MIN_ALPHA_RATIO = 0.5     # доля букв в тексте: эмодзи, символы, разметка

_WORD = re.compile(r"[A-Za-zА-Яа-яЁё]{2,}")


def junk_reason(text: str) -> str | None:
    """Почему текст не похож на вакансию, или None, если похож."""
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        return f"слишком короткий текст ({len(text)} символов)"
    words = [w.lower() for w in _WORD.findall(text)]
    if len(words) < MIN_WORDS:
        return f"почти нет слов ({len(words)})"
    head = words[:200]
    if len(set(head)) / len(head) < MIN_UNIQUE_RATIO:
        return "текст состоит из повторов"
    if sum(c.isalpha() for c in text) / len(text) < MIN_ALPHA_RATIO:
        return "мало букв: символы, разметка или эмодзи"
    return None


# ---------- 1. явные команды для ИИ внутри вакансии ----------
_INJECTION = [re.compile(p, re.IGNORECASE) for p in (
    r"игнорир\w*\s+(все\s+)?(предыдущ|прошл|ранее|выше)",
    r"ignore\s+(all\s+)?(the\s+)?(previous|prior|above)",
    r"(инструкц|указани|примечани|заметк|сообщени)\w*\s+(для|к)\s+(ии|ai|нейросет|языков|модел|ассистент|бот)",
    r"(для|к)\s+(нейросет|языков\w*\s+модел|llm|gpt|chatgpt)\w*",
    r"(note|instruction|message)s?\s+(to|for)\s+(the\s+)?(ai|assistant|model|llm|bot)",
    r"^\W*(system|assistant|система)\s*:",
    r"relevance_score|cover_letter",
    r"автоматическ\w+\s+отклик",
)]


def strip_injections(text: str) -> tuple[str, list[str]]:
    """Убрать строки с явными командами для ИИ. Возвращает (текст, вырезанные строки)."""
    kept, removed = [], []
    for line in (text or "").split("\n"):
        if any(p.search(line) for p in _INJECTION):
            removed.append(line.strip())
        else:
            kept.append(line)
    return "\n".join(kept), removed


def wrap_vacancy(text: str) -> str:
    """Не дать тексту вакансии «закрыть» тег <vacancy> и выйти из рамки данных."""
    return re.sub(r"</?\s*vacancy\s*>", "", text or "", flags=re.IGNORECASE)


# ---------- 3. проверка письма на выходе ----------
_LETTER_RISKS = {
    "ссылка": re.compile(r"https?://|www\.|\b[\w-]+\.(ru|com|io|net|org|рф)/", re.IGNORECASE),
    "телефон": re.compile(r"\+?\d[\d\s()\-]{8,}\d"),
    "почта": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    "зарплата": re.compile(r"зарплат|оклад|₽|\bруб(\.|л)", re.IGNORECASE),
    "работа бесплатно": re.compile(r"бесплатн|неоплачиваем|без оплаты|безвозмездн|без зарплат", re.IGNORECASE),
}
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n")


def sanitize_letter(letter: str) -> tuple[str, list[str]]:
    """Вырезать из письма предложения со ссылками, контактами, зарплатой, согласием работать бесплатно.
    Возвращает (письмо, что вырезано)."""
    if not letter:
        return letter, []
    found: list[str] = []
    out_lines = []
    for line in letter.split("\n"):
        parts = re.split(r"(?<=[.!?])\s+", line)
        kept = []
        for part in parts:
            hits = [name for name, rx in _LETTER_RISKS.items() if rx.search(part)]
            if hits:
                found.extend(h for h in hits if h not in found)
            else:
                kept.append(part)
        if kept or not line.strip():
            out_lines.append(" ".join(kept))
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip()
    return text, found
