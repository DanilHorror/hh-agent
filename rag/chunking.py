"""Нарезка текста вакансии на смысловые фрагменты.

В текстах hh.ru разделы почти всегда начинаются со строки-заголовка с двоеточием:
«Требования:», «Чем предстоит заниматься:», «Мы предлагаем:». Режем по ним и определяем тип раздела,
длинные разделы дробим по абзацам. Маленький фрагмент про одну тему находится поиском точнее,
чем вакансия целиком.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_CHARS = 900       # примерно 200–250 токенов: фрагмент про одну тему
MIN_CHARS = 120       # совсем короткие куски приклеиваем к соседнему
HEADER_MAX_LEN = 60

# Тип раздела по ключевым словам заголовка (порядок важен: «будет плюсом» раньше «требований»)
SECTION_RULES = [
    ("plus", ("плюс", "преимуществ", "желательно", "приветству")),
    ("stack", ("стек", "технологи", "инструмент")),
    ("duties", ("обязанност", "задач", "предстоит", "заниматься", "делать", "чем ты будешь", "функционал",
                "вы будете", "ты будешь")),
    ("requirements", ("требован", "ожида", "важно", "ждем", "ждём", "пожелани", "нужно", "необходим",
                      "кого мы ищем", "ищем", "навык", "опыт", "обязательно", "кандидат", "видеть", "soft skills")),
    ("conditions", ("услови", "предлага", "предлож", "предостав", "обеспечива", "почему", "бонус", "льгот",
                    "мы даем", "мы даём", "плюшк", "компенсац", "дмс", "забот", "обучение и развитие")),
    ("about", ("о компании", "о проекте", "о нас", "о команде", "кто мы", "о продукте", "ценности")),
]

SECTION_NAMES = {
    "intro": "Описание",
    "duties": "Обязанности",
    "requirements": "Требования",
    "plus": "Будет плюсом",
    "conditions": "Условия",
    "stack": "Стек",
    "about": "О компании",
    "other": "Другое",
    "skills": "Ключевые навыки",
}

_EMOJI_AND_BULLETS = re.compile(r"^[\W_]+", re.UNICODE)


@dataclass
class Chunk:
    vacancy_id: str
    chunk_index: int
    section: str
    content: str            # текст, который индексируется и показывается модели
    metadata: dict = field(default_factory=dict)


def _clean(text: str) -> str:
    text = text.replace("\xa0", " ").replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def classify_header(line: str) -> str | None:
    """Если строка похожа на заголовок раздела — вернуть его тип, иначе None."""
    stripped = line.strip()
    if not stripped.endswith(":") or len(stripped) > HEADER_MAX_LEN:
        return None
    words = _EMOJI_AND_BULLETS.sub("", stripped).rstrip(":").strip().lower()
    if not words or len(words.split()) > 8:
        return None
    for section, keys in SECTION_RULES:
        if any(k in words for k in keys):
            return section
    return "other"


def split_sections(description: str) -> list[tuple[str, str]]:
    """[(тип раздела, текст)] в порядке следования. Текст до первого заголовка — intro."""
    sections: list[tuple[str, list[str]]] = [("intro", [])]
    for line in _clean(description).split("\n"):
        kind = classify_header(line)
        if kind is not None:
            sections.append((kind, []))
        else:
            sections[-1][1].append(line)
    result = []
    for kind, lines in sections:
        text = "\n".join(lines).strip()
        if text:
            result.append((kind, text))
    return result


def _split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Длинный раздел — на куски по границам абзацев и строк, не длиннее max_chars."""
    if len(text) <= max_chars:
        return [text]
    parts, current = [], ""
    for piece in re.split(r"(?<=\n)", text):
        if len(piece) > max_chars:  # одна огромная строка — режем по предложениям
            for sentence in re.split(r"(?<=[.!?;])\s+", piece):
                if current and len(current) + len(sentence) > max_chars:
                    parts.append(current.strip())
                    current = ""
                current += sentence + " "
            continue
        if current and len(current) + len(piece) > max_chars:
            parts.append(current.strip())
            current = ""
        current += piece
    if current.strip():
        parts.append(current.strip())
    return [p for p in parts if p]


def _merge_small(pieces: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Короткий кусок приклеиваем к предыдущему того же раздела или к следующему."""
    merged: list[tuple[str, str]] = []
    for kind, text in pieces:
        if merged and len(text) < MIN_CHARS and merged[-1][0] == kind \
                and len(merged[-1][1]) + len(text) <= MAX_CHARS + MIN_CHARS:
            merged[-1] = (kind, merged[-1][1] + "\n" + text)
        else:
            merged.append((kind, text))
    return merged


def chunk_vacancy(vacancy_id: str, title: str, company: str, url: str,
                  description: str, skills: list[str] | None = None) -> list[Chunk]:
    """Фрагменты одной вакансии. В начало каждого добавлен заголовок «должность — компания. Раздел»,
    чтобы фрагмент был понятен сам по себе (и модели, и поиску)."""
    skills = [s for s in (skills or []) if s]
    pieces: list[tuple[str, str]] = []
    for kind, text in split_sections(description or ""):
        pieces.extend((kind, part) for part in _split_long(text))
    pieces = _merge_small(pieces)
    if skills:
        pieces.append(("skills", ", ".join(skills)))

    base_meta = {"title": title, "company": company, "url": url, "skills": skills}
    chunks = []
    for index, (kind, text) in enumerate(pieces):
        header = f"{title} — {company}. {SECTION_NAMES.get(kind, kind)}:"
        chunks.append(Chunk(
            vacancy_id=vacancy_id,
            chunk_index=index,
            section=kind,
            content=f"{header}\n{text}",
            metadata=dict(base_meta),
        ))
    return chunks
