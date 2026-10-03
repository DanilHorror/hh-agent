"""Трассировка вызовов моделей в Langfuse (self-hosted, docker-compose.langfuse.yml).

Включается ключами LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY в .env; без них или без пакета langfuse
всё работает как раньше — функции модуля превращаются в пустые обёртки. Ошибка Langfuse не должна
мешать боту, поэтому всё, что касается отправки трасс, обёрнуто в try.

    with tracing.trace("vacancy", session_id=vacancy_id):            # общая трасса одной вакансии
        with tracing.span("evaluate_vacancy", input=...) as s:        # шаг
            with tracing.generation("evaluate", model=..., input=messages) as g:   # вызов модели
                g.update(output=text, usage_details={...})
"""
from __future__ import annotations

import contextlib
import os

from config import settings

_client = None
_state = "unknown"   # unknown | on | off


class _Noop:
    def update(self, **kwargs) -> None:
        pass


def client():
    """Клиент Langfuse или None, если трассировка выключена."""
    global _client, _state
    if _state == "unknown":
        _state = "off"
        if settings.langfuse_public_key and settings.langfuse_secret_key:
            try:
                # SDK читает настройки из окружения; pydantic-settings их туда не выгружает
                os.environ.setdefault("LANGFUSE_PUBLIC_KEY", settings.langfuse_public_key)
                os.environ.setdefault("LANGFUSE_SECRET_KEY", settings.langfuse_secret_key)
                os.environ.setdefault("LANGFUSE_HOST", settings.langfuse_host)
                os.environ.setdefault("LANGFUSE_BASE_URL", settings.langfuse_host)
                from langfuse import get_client

                _client = get_client()
                _state = "on"
                print(f"[Langfuse] трассировка включена: {settings.langfuse_host}")
            except Exception as e:  # noqa: BLE001 — нет пакета или сервера: работаем без трасс
                print(f"[Langfuse] трассировка выключена: {type(e).__name__}: {e}")
    return _client if _state == "on" else None


def reset() -> None:
    """Сбросить состояние (для тестов)."""
    global _client, _state
    _client, _state = None, "unknown"


@contextlib.contextmanager
def _observation(as_type: str, name: str, **fields):
    lf = client()
    if lf is None:
        yield _Noop()
        return
    try:
        cm = lf.start_as_current_observation(as_type=as_type, name=name, **fields)
        obs = cm.__enter__()
    except Exception:  # noqa: BLE001
        yield _Noop()
        return
    try:
        yield obs
    except BaseException as e:
        try:
            obs.update(level="ERROR", status_message=f"{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            pass
        cm.__exit__(type(e), e, e.__traceback__)
        raise
    else:
        cm.__exit__(None, None, None)


def span(name: str, **fields):
    """Шаг обработки (оценка вакансии, страж, поиск RAG)."""
    return _observation("span", name, **fields)


def generation(name: str, **fields):
    """Один вызов модели: input, output, model, usage_details."""
    return _observation("generation", name, **fields)


@contextlib.contextmanager
def trace(name: str, **attributes):
    """Общие атрибуты трассы (имя, session_id, metadata) для всех вложенных шагов."""
    lf = client()
    if lf is None:
        yield
        return
    try:
        from langfuse import propagate_attributes

        cm = propagate_attributes(trace_name=name, **{k: v for k, v in attributes.items() if v is not None})
        cm.__enter__()
    except Exception:  # noqa: BLE001
        yield
        return
    try:
        yield
    finally:
        cm.__exit__(None, None, None)


def usage(prompt_tokens, completion_tokens) -> dict | None:
    if prompt_tokens is None and completion_tokens is None:
        return None
    return {"input_tokens": prompt_tokens or 0, "output_tokens": completion_tokens or 0}


def flush() -> None:
    lf = client()
    if lf is not None:
        try:
            lf.flush()
        except Exception:  # noqa: BLE001
            pass
