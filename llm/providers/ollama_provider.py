"""Локальная модель через Ollama."""
from __future__ import annotations

import asyncio
import contextlib
import os
import time
from typing import Optional, Type

import ollama
from pydantic import BaseModel

from .base import LLMError, LLMProvider, LLMResponse, Messages

# Файл-замок: если запущены два скрипта (hh и Habr), к Ollama обращается только один за раз.
# Два одновременных запроса к 7B-модели тормозят друг друга и могут "подвесить" обоих.
OLLAMA_LOCK_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ollama.lock"
)


@contextlib.asynccontextmanager
async def ollama_lock(stale_after: float):
    """Межпроцессный замок на файле. Замок старше stale_after секунд считается брошенным (процесс убили)."""
    fd = None
    announced = False
    while fd is None:
        try:
            fd = os.open(OLLAMA_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(OLLAMA_LOCK_PATH) > stale_after:
                    os.remove(OLLAMA_LOCK_PATH)
                    continue
            except OSError:
                pass
            if not announced:
                print("[LLM] ⏳ Ollama занята другим окном, жду своей очереди...")
                announced = True
            await asyncio.sleep(1.0)
    try:
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(OLLAMA_LOCK_PATH)
        except OSError:
            pass


def _field(response, name: str, default=None):
    """Ответ ollama — dict в старых версиях клиента и объект в новых."""
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _content(response) -> str:
    message = _field(response, "message")
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return (getattr(message, "content", "") or "").strip()


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, model: str, host: str, timeout: float, num_ctx: int, use_lock: bool = True):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.num_ctx = num_ctx
        self.use_lock = use_lock

    def _warn_if_truncated(self, prompt_tokens: Optional[int]) -> None:
        """Ollama молча обрезает промпт длиннее num_ctx. Если модель прочитала почти весь
        контекст, часть резюме или вакансии могла в него не попасть."""
        if prompt_tokens is not None and prompt_tokens >= self.num_ctx * 0.95:
            print(
                f"[LLM] ⚠️ Промпт занял {prompt_tokens} из {self.num_ctx} токенов контекста — "
                "вероятно, он обрезан. Увеличьте OLLAMA_NUM_CTX в .env."
            )

    async def complete(
        self,
        messages: Messages,
        schema: Optional[Type[BaseModel]] = None,
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        # Асинхронный клиент: синхронный ollama.chat блокировал весь процесс (и Telegram-бота).
        client = ollama.AsyncClient(host=self.host, timeout=self.timeout)
        options = {
            "temperature": temperature,
            # Резюме + вакансия + инструкции ~5 тыс. токенов; при контексте по умолчанию
            # начало промпта обрезается. num_predict не даёт модели генерировать бесконечно.
            "num_ctx": self.num_ctx,
            "num_predict": max_tokens,
        }
        # Structured output: JSON-схема ответа. Ollama на каждом шаге генерации запрещает
        # токены, которые нарушают схему. Старые версии понимают только format='json'.
        fmt = schema.model_json_schema() if schema else ("json" if json_mode else None)

        async def ask(fmt_value):
            kwargs = {"model": self.model, "messages": messages, "options": options}
            kwargs["format"] = fmt_value if fmt_value is not None else ""
            return await asyncio.wait_for(client.chat(**kwargs), timeout=self.timeout + 10)

        lock = ollama_lock(stale_after=self.timeout + 60) if self.use_lock else contextlib.nullcontext()
        try:
            async with lock:
                try:
                    response = await ask(fmt)
                except Exception as e:  # noqa: BLE001
                    if schema is None or "format" not in str(e).lower():
                        raise
                    print("[LLM] Эта версия Ollama не поддерживает JSON-схему, использую format='json'")
                    response = await ask("json")
        except (asyncio.TimeoutError, TimeoutError) as e:
            # Таймаут локальной модели не повторяем: следующая попытка ждала бы столько же.
            raise LLMError("timeout", f"модель не ответила за {self.timeout} с", self.name) from e
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            name = type(e).__name__
            kind = "network" if any(s in name for s in ("Connect", "Timeout", "Network")) else "unknown"
            raise LLMError(kind, f"{name}: {e}", self.name) from e

        prompt_tokens = _field(response, "prompt_eval_count")
        self._warn_if_truncated(prompt_tokens)
        return LLMResponse(
            text=_content(response),
            provider=self.name,
            model=self.model,
            prompt_tokens=prompt_tokens,
            completion_tokens=_field(response, "eval_count"),
        )

    async def health(self, timeout: float = 5.0) -> tuple[bool, str]:
        """Быстрая проверка, что Ollama отвечает (без генерации)."""
        try:
            client = ollama.AsyncClient(host=self.host, timeout=timeout)
            await asyncio.wait_for(client.list(), timeout=timeout)
            return True, ""
        except Exception as e:  # noqa: BLE001
            return False, str(e)
