"""Извлечение ключевиков из вакансии с помощью ИИ (Ollama).

ИИ читает ПОЛНЫЙ текст вакансии и сам выписывает, что работодатель хочет видеть у кандидата:
технологии, инструменты, практики, знания. В отличие от словаря (analytics/keywords.py)
он находит и то, чего в словаре нет, и требования, написанные обычными предложениями.

Модуль не импортирует ollama на верхнем уровне — вызов модели передаётся функцией chat_fn,
поэтому логику можно тестировать без Ollama (см. test/test_llm_keywords.py).
"""
import json
import re
from typing import Callable, List, Optional

from analytics.keywords import clean_description

# Сколько символов описания отдаём модели. Хватает для самых длинных вакансий на hh.ru;
# усечение — только защита от аномально огромных страниц.
MAX_TEXT_CHARS = 14000

# Размер контекстного окна модели (в токенах). У Ollama по умолчанию окно маленькое (2048–4096),
# и длинный промпт молча обрезается — из-за этого часть вакансии «пропадает».
NUM_CTX = 8192

MAX_KEYWORDS = 30

PROMPT = """Ты помогаешь анализировать рынок вакансий. Прочитай текст вакансии целиком и выпиши КЛЮЧЕВЫЕ СЛОВА: то, что работодатель хочет видеть у кандидата и чем предстоит заниматься.

ВКЛЮЧАЙ:
- технологии, инструменты, языки программирования, протоколы, стандарты, форматы (например: Postman, PostgreSQL, Docker, CI/CD, REST API, Swagger, TCP/IP);
- практики и методики (например: "автоматизация тестирования", "регрессионное тестирование", "тест-дизайн", "написание тест-кейсов");
- предметные знания и области (например: "мобильные приложения", "безопасность", "сетевые протоколы");
- требуемый язык, если он указан (например: "английский язык").

НЕ ВКЛЮЧАЙ:
- условия работы: зарплату, график, формат, офис, ДМС, льготы, бонусы, обучение за счёт компании;
- описание компании и продукта;
- общие личные качества (ответственность, коммуникабельность, стрессоустойчивость);
- количество лет опыта и название должности целиком.

ПРАВИЛА:
1. Каждое ключевое слово — от 1 до 4 слов. Названия технологий пиши так, как принято (Postman, PostgreSQL, CI/CD, REST API).
2. Практики пиши в именительном падеже, строчными буквами: "регрессионное тестирование", а не "регрессионного тестирования".
3. Одно и то же не повторяй разными словами. Не выдумывай: только то, что явно есть в тексте.
4. Не больше {max_keywords} ключевых слов, самые важные в начале.
5. Ответ — СТРОГО валидный JSON вида {{"keywords": ["...", "..."]}} и больше ничего.

ВАКАНСИЯ
Должность: {title}
{tags_line}Текст:
{text}
"""


def build_prompt(title: str, description: str, tags: Optional[List[str]] = None) -> str:
    text = clean_description(description or "").strip()[:MAX_TEXT_CHARS]
    tags_line = f"Ключевые навыки, указанные работодателем: {', '.join(tags)}\n" if tags else ""
    return PROMPT.format(max_keywords=MAX_KEYWORDS, title=title or "—", tags_line=tags_line, text=text)


def parse_keywords(raw: str) -> List[str]:
    """Достать список ключевиков из ответа модели, даже если JSON слегка сломан."""
    raw = (raw or "").strip()
    if not raw:
        return []

    # 1. Нормальный JSON
    for candidate in (raw, *re.findall(r"\{.*\}", raw, re.DOTALL)[:1]):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, dict):
            for key in ("keywords", "ключевые_слова", "ключевики", "skills"):
                if isinstance(data.get(key), list):
                    return _clean_list(data[key])
            # Единственное значение-список под любым другим ключом
            lists = [v for v in data.values() if isinstance(v, list)]
            if lists:
                return _clean_list(lists[0])
        elif isinstance(data, list):
            return _clean_list(data)

    # 2. Спасательный вариант: вытащить строки в кавычках из первого [...]
    m = re.search(r"\[(.*)", raw, re.DOTALL)
    if m:
        return _clean_list(re.findall(r'"([^"\n]{1,80})"', m.group(1)))
    return []


def _clean_list(items) -> List[str]:
    out, seen = [], set()
    for it in items:
        if not isinstance(it, str):
            continue
        s = re.sub(r"\s+", " ", it).strip().strip("-•*·\"'«»").strip()
        if not s or len(s) > 60 or len(s.split()) > 6:
            continue
        if s.lower() in seen:
            continue
        seen.add(s.lower())
        out.append(s)
    return out[:MAX_KEYWORDS]


def extract_keywords(title: str, description: str, tags: Optional[List[str]] = None,
                     chat_fn: Optional[Callable] = None, model: Optional[str] = None) -> List[str]:
    """Спросить у модели ключевики для одной вакансии. Блокирующий вызов.

    chat_fn(model=..., messages=..., options=..., format=...) — по умолчанию ollama.chat.
    Бросает исключение, если модель недоступна (вызывающий код решает, что делать).
    """
    if chat_fn is None:
        import ollama
        chat_fn = ollama.chat
    if model is None:
        from config import settings
        model = settings.ollama_model

    response = chat_fn(
        model=model,
        messages=[{"role": "user", "content": build_prompt(title, description, tags)}],
        options={"temperature": 0.0, "num_ctx": NUM_CTX},
        format="json",
    )
    raw = response["message"]["content"] if not isinstance(response, str) else response
    return parse_keywords(raw)
