"""GigaChat (Сбер).

Использует тот же клиент, что и анализ рынка (analytics/llm_gigachat.py): каскад моделей при
исчерпании токенов (HTTP 402), разбор ошибок по кодам, повторы при 429 и сбоях сети, один клиент
на модель без повторной авторизации на каждый запрос. SDK синхронный, поэтому вызов уходит
в отдельный поток и не блокирует Telegram-бота.
"""
from __future__ import annotations

import asyncio
from typing import Optional, Type

from pydantic import BaseModel

from analytics.llm_gigachat import GigaChatBackend, GigaChatFailure

from .base import LLMError, LLMProvider, LLMResponse, Messages, schema_instruction

# Ошибки, которые клиент GigaChat уже повторял сам, — снаружи не повторяем.
_KIND_MAP = {
    "no_tokens": "no_tokens",
    "auth": "auth",
    "rate_limit": "rate_limit",
    "network": "network",
    "ssl": "auth",
    "unknown": "unknown",
}


def _with_instruction(messages: Messages, instruction: str) -> Messages:
    """GigaChat принимает системное сообщение только первым — дописываем инструкцию в него."""
    messages = [dict(m) for m in messages]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
        return messages
    return [{"role": "system", "content": instruction}, *messages]


class GigaChatProvider(LLMProvider):
    name = "gigachat"

    def __init__(self, backend: GigaChatBackend):
        self._backend = backend
        # Параметры генерации лежат на общем клиенте, поэтому вызовы идут по одному.
        self._lock = asyncio.Lock()

    @classmethod
    def from_env(cls, base_dir: str) -> "GigaChatProvider":
        return cls(GigaChatBackend.from_env(base_dir))

    @property
    def model(self) -> str:  # type: ignore[override]
        return self._backend.current_model

    async def complete(
        self,
        messages: Messages,
        schema: Optional[Type[BaseModel]] = None,
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        # Передать JSON-схему параметром в GigaChat можно только через function calling;
        # для оценки вакансии хватает инструкции в промпте + проверки ответа схемой в brain.py.
        if schema is not None:
            messages = _with_instruction(messages, schema_instruction(schema))
        elif json_mode:
            messages = _with_instruction(messages, "Верни ТОЛЬКО валидный JSON без пояснений и markdown.")

        async with self._lock:
            # 0 у облачных API иногда считается невалидным значением температуры.
            self._backend.generation = {"temperature": max(temperature, 0.0001), "max_tokens": max_tokens}
            try:
                result = await asyncio.to_thread(self._backend, messages=messages)
            except GigaChatFailure as f:
                raise LLMError(_KIND_MAP.get(f.kind, "unknown"), f.detail, self.name, retryable=False) from f
            finally:
                self._backend.generation = {}
            prompt_tokens, completion_tokens = self._backend.last_usage
            model = self._backend.used_model

        return LLMResponse(
            text=(result.get("message", {}).get("content") or "").strip(),
            provider=self.name,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def aclose(self) -> None:
        self._backend.close()
