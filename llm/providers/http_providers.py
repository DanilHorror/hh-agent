"""Облачные модели по HTTP: любой OpenAI-совместимый API и YandexGPT.

OpenAI-совместимый формат (/v1/chat/completions) понимают Groq, OpenRouter, DeepSeek, vLLM,
LM Studio и др., поэтому одна реализация закрывает их все. YandexGPT использует свой формат
Foundation Models API.

HTTP-запрос вынесен в объект transport: в тестах он подменяется заглушкой, сеть не нужна.
"""
from __future__ import annotations

import json
from typing import Optional, Protocol, Type

from pydantic import BaseModel

from .base import (
    LLMError,
    LLMProvider,
    LLMResponse,
    Messages,
    classify_http_status,
    schema_instruction,
)


class Transport(Protocol):
    async def post(self, url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, str]:
        ...

    async def aclose(self) -> None:
        ...


class HttpxTransport:
    """Один httpx-клиент на провайдера: соединение переиспользуется между запросами."""

    def __init__(self):
        self._client = None

    async def post(self, url: str, headers: dict, payload: dict, timeout: float) -> tuple[int, str]:
        import httpx  # приходит вместе с пакетом ollama

        if self._client is None:
            self._client = httpx.AsyncClient()
        response = await self._client.post(url, headers=headers, json=payload, timeout=timeout)
        return response.status_code, response.text

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


def _network_error(e: Exception, provider: str) -> LLMError:
    name = type(e).__name__
    if any(s in name for s in ("Timeout", "Connect", "Network", "Protocol", "ReadError")):
        return LLMError("network", f"{name}: {e}", provider)
    return LLMError("unknown", f"{name}: {e}", provider)


def _with_instruction(messages: Messages, instruction: str) -> Messages:
    messages = [dict(m) for m in messages]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0]['content']}\n\n{instruction}"
        return messages
    return [{"role": "system", "content": instruction}, *messages]


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class _HttpProvider(LLMProvider):
    def __init__(self, timeout: float, transport: Optional[Transport] = None):
        self.timeout = timeout
        self.transport = transport or HttpxTransport()
        # Поддерживает ли сервис JSON-схему параметром. Узнаём по первому отказу (HTTP 400)
        # и дальше сразу кладём схему в промпт.
        self.schema_supported = True

    async def _post(self, url: str, headers: dict, payload: dict) -> dict:
        try:
            status, body = await self.transport.post(url, headers, payload, self.timeout)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise _network_error(e, self.name) from e
        if status != 200:
            raise classify_http_status(status, body, self.name)
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise LLMError("unknown", f"ответ не JSON: {body[:200]}", self.name) from e

    async def aclose(self) -> None:
        await self.transport.aclose()


class OpenAICompatibleProvider(_HttpProvider):
    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0,
                 transport: Optional[Transport] = None):
        super().__init__(timeout, transport)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.model = model

    def _payload(self, messages, schema, json_mode, temperature, max_tokens, use_schema_param):
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if schema is not None and use_schema_param:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
            }
        elif schema is not None:
            payload["messages"] = _with_instruction(messages, schema_instruction(schema))
            payload["response_format"] = {"type": "json_object"}
        elif json_mode:
            payload["messages"] = _with_instruction(messages, "Верни ТОЛЬКО валидный JSON.")
            payload["response_format"] = {"type": "json_object"}
        return payload

    async def complete(
        self,
        messages: Messages,
        schema: Optional[Type[BaseModel]] = None,
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        use_param = schema is not None and self.schema_supported
        payload = self._payload(messages, schema, json_mode, temperature, max_tokens, use_param)
        try:
            data = await self._post(self.url, headers, payload)
        except LLMError as e:
            if not (use_param and e.kind == "bad_request"):
                raise
            # Модель не умеет json_schema — повторяем с JSON-режимом и схемой в промпте.
            print(f"[LLM] {self.model}: JSON-схема параметром не поддерживается, передаю её в промпте")
            self.schema_supported = False
            payload = self._payload(messages, schema, json_mode, temperature, max_tokens, False)
            data = await self._post(self.url, headers, payload)

        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("unknown", f"неожиданный ответ: {str(data)[:200]}", self.name) from e
        usage = data.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=data.get("model") or self.model,
            prompt_tokens=_to_int(usage.get("prompt_tokens")),
            completion_tokens=_to_int(usage.get("completion_tokens")),
        )


YANDEX_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"


class YandexGPTProvider(_HttpProvider):
    name = "yandexgpt"

    def __init__(self, api_key: str, folder_id: str, model: str = "yandexgpt/latest",
                 timeout: float = 60.0, url: str = YANDEX_URL, transport: Optional[Transport] = None):
        super().__init__(timeout, transport)
        self.api_key = api_key
        self.folder_id = folder_id
        self.url = url
        # Можно указать полный URI (gpt://<каталог>/yandexgpt/latest) или только имя модели.
        self.model = model if model.startswith("gpt://") else f"gpt://{folder_id}/{model}"

    def _payload(self, messages, schema, json_mode, temperature, max_tokens, use_schema_param):
        if schema is not None and not use_schema_param:
            messages = _with_instruction(messages, schema_instruction(schema))
        payload = {
            "modelUri": self.model,
            "completionOptions": {
                "stream": False,
                "temperature": min(max(temperature, 0.0), 1.0),
                "maxTokens": str(max_tokens),
            },
            "messages": [{"role": m["role"], "text": m["content"]} for m in messages],
        }
        if schema is not None and use_schema_param:
            payload["jsonSchema"] = {"schema": schema.model_json_schema()}
        elif schema is not None or json_mode:
            payload["jsonObject"] = True
        return payload

    async def complete(
        self,
        messages: Messages,
        schema: Optional[Type[BaseModel]] = None,
        *,
        json_mode: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 1500,
    ) -> LLMResponse:
        headers = {"Authorization": f"Api-Key {self.api_key}", "x-folder-id": self.folder_id}
        use_param = schema is not None and self.schema_supported
        payload = self._payload(messages, schema, json_mode, temperature, max_tokens, use_param)
        try:
            data = await self._post(self.url, headers, payload)
        except LLMError as e:
            if not (use_param and e.kind == "bad_request"):
                raise
            print(f"[LLM] {self.model}: jsonSchema не поддерживается, передаю схему в промпте")
            self.schema_supported = False
            payload = self._payload(messages, schema, json_mode, temperature, max_tokens, False)
            data = await self._post(self.url, headers, payload)

        try:
            result = data["result"]
            text = result["alternatives"][0]["message"]["text"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError("unknown", f"неожиданный ответ: {str(data)[:200]}", self.name) from e
        usage = result.get("usage") or {}
        return LLMResponse(
            text=text.strip(),
            provider=self.name,
            model=self.model,
            prompt_tokens=_to_int(usage.get("inputTextTokens")),
            completion_tokens=_to_int(usage.get("completionTokens")),
        )
