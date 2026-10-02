"""Выбор LLM-провайдера по строке LLM_PROVIDER в .env.

    LLM_PROVIDER=ollama     локальная модель (по умолчанию): OLLAMA_MODEL, OLLAMA_HOST
    LLM_PROVIDER=gigachat   GIGACHAT_CREDENTIALS, GIGACHAT_MODEL, GIGACHAT_MODEL_CASCADE
    LLM_PROVIDER=yandexgpt  YANDEX_API_KEY, YANDEX_FOLDER_ID, YANDEX_MODEL
    LLM_PROVIDER=openai     любой OpenAI-совместимый API: LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
"""
from __future__ import annotations

import os
from typing import Optional

from .base import LLMError, LLMProvider, LLMResponse, with_retries

__all__ = ["LLMError", "LLMProvider", "LLMResponse", "with_retries", "create_provider", "get_provider", "PROVIDERS"]

PROVIDERS = ("ollama", "gigachat", "yandexgpt", "openai")

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_current: Optional[LLMProvider] = None


def create_provider(name: Optional[str] = None, settings=None) -> LLMProvider:
    """Собрать провайдера по имени. ValueError с понятным текстом, если он не настроен."""
    if settings is None:
        from config import settings as default_settings

        settings = default_settings
    name = (name or settings.llm_provider or "ollama").strip().lower()

    if name == "ollama":
        from .ollama_provider import OllamaProvider

        return OllamaProvider(
            model=settings.ollama_model,
            host=settings.ollama_host,
            timeout=settings.ollama_timeout,
            num_ctx=settings.ollama_num_ctx,
        )
    if name == "gigachat":
        from .gigachat_provider import GigaChatProvider

        return GigaChatProvider.from_env(_BASE_DIR)
    if name == "yandexgpt":
        from .http_providers import YandexGPTProvider

        if not settings.yandex_api_key or not settings.yandex_folder_id:
            raise ValueError("Для LLM_PROVIDER=yandexgpt задайте в .env YANDEX_API_KEY и YANDEX_FOLDER_ID")
        return YandexGPTProvider(
            api_key=settings.yandex_api_key,
            folder_id=settings.yandex_folder_id,
            model=settings.yandex_model,
            timeout=settings.llm_timeout,
        )
    if name == "openai":
        from .http_providers import OpenAICompatibleProvider

        if not settings.llm_base_url or not settings.llm_model:
            raise ValueError(
                "Для LLM_PROVIDER=openai задайте в .env LLM_BASE_URL (например https://api.groq.com/openai/v1), "
                "LLM_MODEL и LLM_API_KEY")
        return OpenAICompatibleProvider(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            timeout=settings.llm_timeout,
        )
    raise ValueError(f"Неизвестный LLM_PROVIDER={name!r}. Доступны: {', '.join(PROVIDERS)}")


def get_provider() -> LLMProvider:
    """Провайдер из .env, один на процесс: клиенты и соединения переиспользуются."""
    global _current
    if _current is None:
        _current = create_provider()
    return _current


def set_provider(provider: Optional[LLMProvider]) -> None:
    """Подменить провайдера (тесты, сравнение моделей). None — снова читать из .env."""
    global _current
    _current = provider
