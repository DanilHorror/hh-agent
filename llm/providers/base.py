"""Общий интерфейс LLM-провайдера.

Остальной код (оценка вакансий, ответы на вопросы, дальше — RAG и агент) знает только
LLMProvider.complete() и LLMResponse. Какая модель отвечает — Ollama, GigaChat, YandexGPT
или любой OpenAI-совместимый сервис — решает строка LLM_PROVIDER в .env.
"""
from __future__ import annotations

import asyncio
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional, Type, TypeVar

from pydantic import BaseModel

Messages = list[dict]  # [{"role": "system" | "user" | "assistant", "content": "..."}]


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class LLMError(Exception):
    """Классифицированная ошибка провайдера.

    kind: auth | no_tokens | rate_limit | network | timeout | bad_request | unknown.
    retryable — имеет ли смысл повторить тот же запрос через паузу.
    """

    def __init__(self, kind: str, detail: str, provider: str = "", retryable: Optional[bool] = None):
        self.kind = kind
        self._retryable = retryable
        self.detail = detail
        self.provider = provider
        super().__init__(f"[{provider}] {kind}: {detail}" if provider else f"{kind}: {detail}")

    @property
    def retryable(self) -> bool:
        if self._retryable is not None:
            return self._retryable
        return self.kind in ("rate_limit", "network")


def classify_http_status(status: int, body: str, provider: str) -> LLMError:
    """Единая классификация HTTP-кодов для облачных провайдеров."""
    short = body[:300]
    if status in (401, 403):
        return LLMError("auth", f"ключ недействителен или нет доступа (HTTP {status}): {short}", provider)
    if status == 402:
        return LLMError("no_tokens", f"закончился баланс или квота (HTTP 402): {short}", provider)
    if status == 429:
        return LLMError("rate_limit", f"превышен лимит запросов (HTTP 429): {short}", provider)
    if status >= 500:
        return LLMError("network", f"сервер недоступен (HTTP {status}): {short}", provider)
    return LLMError("bad_request", f"HTTP {status}: {short}", provider)


class LLMProvider(ABC):
    """Один метод — один запрос к модели.

    schema — Pydantic-класс, если нужен ответ строго по схеме (structured output). Провайдер
    передаёт схему модели тем способом, который она понимает (format в Ollama, response_format
    в OpenAI-совместимых API, jsonSchema в YandexGPT, инструкция в промпте у GigaChat).
    Проверку ответа схемой делает вызывающий код: провайдер возвращает только текст.
    json_mode — нужен любой валидный JSON без конкретной схемы.
    """

    name: str = "base"
    model: str = ""

    @abstractmethod
    async def complete(
        self,
        messages: Messages,
        schema: Optional[Type[BaseModel]] = None,
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        ...

    async def stream(self, messages: Messages, *, temperature: float = 0.3,
                     max_tokens: int = 1500) -> AsyncIterator[str]:
        """Ответ по частям по мере генерации (для чата и SSE).

        По умолчанию — весь ответ одним куском: провайдер без потокового режима тоже работает."""
        response = await self.complete(messages, temperature=temperature, max_tokens=max_tokens)
        yield response.text

    async def aclose(self) -> None:
        """Освободить ресурсы (клиенты, соединения). По умолчанию ничего."""


T = TypeVar("T")

MAX_ATTEMPTS = 3
BACKOFF_BASE = 1.0


async def with_retries(
    call: Callable[[], Awaitable[T]],
    *,
    attempts: int = MAX_ATTEMPTS,
    base_delay: float = BACKOFF_BASE,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    log: Callable[[str], None] = print,
) -> T:
    """Повтор при временных сбоях (сеть, 5xx, 429) с растущей паузой и случайной добавкой.

    Пауза: base * 2^(n-1) + 0..30% — чтобы два процесса не повторяли запрос синхронно.
    Ошибки ключа, баланса и формата запроса не повторяются: повтор их не исправит.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await call()
        except LLMError as e:
            if not e.retryable or attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            delay += random.uniform(0, delay * 0.3)
            log(f"[LLM] {e.provider}: {e.kind}, повтор {attempt + 1}/{attempts} через {delay:.1f} с")
            await sleep(delay)
    raise AssertionError("unreachable")


def schema_instruction(schema: Type[BaseModel]) -> str:
    """Текстовая инструкция со схемой — для моделей, которым схему нельзя передать параметром."""
    import json

    return (
        "Верни ТОЛЬКО JSON-объект, соответствующий этой JSON-схеме, без пояснений и markdown:\n"
        + json.dumps(schema.model_json_schema(), ensure_ascii=False)
    )
