"""Слой LLM-провайдеров без сети: HTTP-запросы и SDK подменены заглушками.
Запуск: python -m pytest test/test_providers.py"""
import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.llm_gigachat import GigaChatBackend, GigaChatFailure  # noqa: E402
from llm import brain, providers  # noqa: E402
from llm.providers import LLMError, LLMProvider, LLMResponse, create_provider, with_retries  # noqa: E402
from llm.providers.gigachat_provider import GigaChatProvider  # noqa: E402
from llm.providers.http_providers import OpenAICompatibleProvider, YandexGPTProvider  # noqa: E402

GOOD = {"relevance_score": 81, "reasoning": "Стек совпадает", "cover_letter": "Добрый день!\nОпыт подходит."}
VACANCY = {"title": "LLM Engineer", "company": "Acme", "description": "RAG, pgvector, LangGraph"}


def run(coro):
    return asyncio.run(coro)


class FakeTransport:
    """Отдаёт заранее заданные (статус, тело) и запоминает запросы."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def post(self, url, headers, payload, timeout):
        self.calls.append({"url": url, "headers": headers, "payload": payload})
        outcome = self.responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        status, body = outcome
        return status, body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)

    async def aclose(self):
        pass


def openai_body(content, prompt_tokens=120, completion_tokens=40):
    return {
        "model": "llama-3.3-70b",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


def settings_stub(**overrides):
    base = dict(
        llm_provider="ollama", llm_timeout=30, llm_base_url="", llm_api_key="", llm_model="",
        yandex_api_key="", yandex_folder_id="", yandex_model="yandexgpt/latest",
        ollama_model="qwen2.5:7b", ollama_host="http://localhost:11434", ollama_timeout=240, ollama_num_ctx=8192,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# --- выбор провайдера -------------------------------------------------------------------

def test_factory_default_is_ollama():
    provider = create_provider(settings=settings_stub())
    assert provider.name == "ollama" and provider.model == "qwen2.5:7b"


def test_factory_openai_requires_url_and_model():
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        create_provider("openai", settings=settings_stub())
    provider = create_provider("openai", settings=settings_stub(
        llm_base_url="https://api.groq.com/openai/v1/", llm_model="llama-3.3-70b", llm_api_key="k"))
    assert provider.url == "https://api.groq.com/openai/v1/chat/completions"


def test_factory_yandex_builds_model_uri():
    provider = create_provider("yandexgpt", settings=settings_stub(yandex_api_key="k", yandex_folder_id="b1g"))
    assert provider.model == "gpt://b1g/yandexgpt/latest"


def test_factory_unknown_name():
    with pytest.raises(ValueError, match="Неизвестный"):
        create_provider("unknown-llm", settings=settings_stub())


# --- OpenAI-совместимый -----------------------------------------------------------------

def test_openai_sends_json_schema_and_reads_usage():
    transport = FakeTransport((200, openai_body(json.dumps(GOOD, ensure_ascii=False))))
    provider = OpenAICompatibleProvider("https://x/v1", "secret", "m", transport=transport)
    response = run(provider.complete([{"role": "user", "content": "оцени"}], brain.VacancyEvaluation))
    sent = transport.calls[0]
    assert sent["headers"]["Authorization"] == "Bearer secret"
    assert sent["payload"]["response_format"]["type"] == "json_schema"
    assert response.prompt_tokens == 120 and response.completion_tokens == 40
    assert json.loads(response.text)["relevance_score"] == 81


def test_openai_falls_back_when_schema_not_supported():
    transport = FakeTransport(
        (400, {"error": {"message": "response_format json_schema is not supported"}}),
        (200, openai_body("{}")),
    )
    provider = OpenAICompatibleProvider("https://x/v1", "k", "m", transport=transport)
    run(provider.complete([{"role": "user", "content": "оцени"}], brain.VacancyEvaluation))
    retry = transport.calls[1]["payload"]
    assert retry["response_format"] == {"type": "json_object"}
    assert retry["messages"][0]["role"] == "system" and "relevance_score" in retry["messages"][0]["content"]
    assert provider.schema_supported is False


def test_openai_auth_error_is_not_retried():
    transport = FakeTransport((401, "invalid api key"))
    provider = OpenAICompatibleProvider("https://x/v1", "bad", "m", transport=transport)
    with pytest.raises(LLMError) as err:
        run(with_retries(lambda: provider.complete([{"role": "user", "content": "hi"}]), sleep=_no_sleep))
    assert err.value.kind == "auth" and len(transport.calls) == 1


# --- YandexGPT --------------------------------------------------------------------------

def test_yandex_request_and_response_format():
    body = {"result": {
        "alternatives": [{"message": {"role": "assistant", "text": json.dumps(GOOD, ensure_ascii=False)}}],
        "usage": {"inputTextTokens": "900", "completionTokens": "150"},
    }}
    transport = FakeTransport((200, body))
    provider = YandexGPTProvider("key", "b1g", transport=transport)
    response = run(provider.complete(
        [{"role": "system", "content": "ты помощник"}, {"role": "user", "content": "оцени"}],
        brain.VacancyEvaluation, temperature=1.5, max_tokens=700))
    sent = transport.calls[0]
    assert sent["headers"] == {"Authorization": "Api-Key key", "x-folder-id": "b1g"}
    payload = sent["payload"]
    assert payload["modelUri"] == "gpt://b1g/yandexgpt/latest"
    assert payload["messages"][1] == {"role": "user", "text": "оцени"}
    assert payload["completionOptions"] == {"stream": False, "temperature": 1.0, "maxTokens": "700"}
    assert "schema" in payload["jsonSchema"]
    assert response.prompt_tokens == 900 and response.completion_tokens == 150


# --- повторы ----------------------------------------------------------------------------

async def _no_sleep(_):
    return None


def test_network_error_is_retried_then_succeeds():
    transport = FakeTransport(ConnectionError("reset"), (503, "busy"), (200, openai_body("ok")))
    provider = OpenAICompatibleProvider("https://x/v1", "k", "m", transport=transport)
    response = run(with_retries(lambda: provider.complete([{"role": "user", "content": "hi"}]),
                                sleep=_no_sleep, log=lambda _: None))
    assert response.text == "ok" and len(transport.calls) == 3


def test_retries_give_up_after_max_attempts():
    transport = FakeTransport((429, "slow down"), (429, "slow down"), (429, "slow down"))
    provider = OpenAICompatibleProvider("https://x/v1", "k", "m", transport=transport)
    with pytest.raises(LLMError) as err:
        run(with_retries(lambda: provider.complete([{"role": "user", "content": "hi"}]),
                         sleep=_no_sleep, log=lambda _: None))
    assert err.value.kind == "rate_limit" and len(transport.calls) == 3


# --- GigaChat ---------------------------------------------------------------------------

def test_gigachat_passes_params_schema_and_usage():
    seen = {}

    def send(model, messages, temperature=None, max_tokens=None):
        seen.update(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        send.last_usage = (1000, 200)
        return json.dumps(GOOD, ensure_ascii=False)

    backend = GigaChatBackend("cred", ["GigaChat-2-Max", "GigaChat-2"], send_fn=send, sleep_fn=lambda _: None)
    provider = GigaChatProvider(backend)
    response = run(provider.complete([{"role": "user", "content": "оцени"}], brain.VacancyEvaluation,
                                     temperature=0.0, max_tokens=900))
    assert seen["temperature"] == 0.0001 and seen["max_tokens"] == 900
    assert seen["messages"][0]["role"] == "system" and "relevance_score" in seen["messages"][0]["content"]
    assert response.prompt_tokens == 1000 and response.model == "GigaChat-2-Max"
    assert backend.generation == {}  # параметры не «прилипают» к следующему вызову анализа рынка


def test_gigachat_failure_becomes_llm_error():
    def send(model, messages, **kwargs):
        raise GigaChatFailure("auth", "ключ истёк")

    provider = GigaChatProvider(GigaChatBackend("cred", ["GigaChat-2"], send_fn=send))
    with pytest.raises(LLMError) as err:
        run(provider.complete([{"role": "user", "content": "hi"}]))
    assert err.value.kind == "auth" and not err.value.retryable


# --- brain.py поверх любого провайдера -------------------------------------------------

class ScriptedProvider(LLMProvider):
    name = "fake"
    model = "fake-1"

    def __init__(self, *texts):
        self.texts = list(texts)
        self.calls = []

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        self.calls.append({"schema": schema, "temperature": temperature})
        return LLMResponse(self.texts.pop(0), self.name, self.model, 10, 5)


@pytest.fixture
def fake_provider():
    def install(*texts):
        provider = ScriptedProvider(*texts)
        providers.set_provider(provider)
        return provider
    yield install
    providers.set_provider(None)


def test_brain_works_with_any_provider(fake_provider):
    provider = fake_provider(json.dumps(GOOD, ensure_ascii=False))
    result = run(brain.evaluate_and_draft(VACANCY))
    assert result["relevance_score"] == 81 and "error" not in result
    assert provider.calls[0]["schema"] is brain.VacancyEvaluation
    assert provider.calls[0]["temperature"] == 0.6


def test_brain_marks_provider_failure_as_error(fake_provider):
    class Broken(ScriptedProvider):
        async def complete(self, *a, **k):
            raise LLMError("auth", "ключ истёк", "fake")

    providers.set_provider(Broken())
    result = run(brain.evaluate_and_draft(VACANCY))
    assert result["error"] is True and result["relevance_score"] == 0


def test_short_answer_uses_provider_and_strips_quotes(fake_provider):
    fake_provider('"Да, работал с Docker два года."')
    answer = run(brain.generate_short_answer("Работали с Docker?", fallback="—"))
    assert answer == "Да, работал с Docker два года."


# --- межпроцессный замок Ollama ----------------------------------------------------------

def test_lock_left_by_dead_process_is_taken_over(tmp_path, monkeypatch):
    from llm.providers import ollama_provider

    lock_path = tmp_path / "ollama.lock"
    lock_path.write_text("999999")  # PID процесса, которого нет
    monkeypatch.setattr(ollama_provider, "OLLAMA_LOCK_PATH", str(lock_path))
    monkeypatch.setattr(ollama_provider, "_pid_alive", lambda pid: False)

    async def take():
        async with ollama_provider.ollama_lock(stale_after=3600):
            return lock_path.read_text()

    assert run(asyncio.wait_for(take(), timeout=5)) == str(os.getpid())
    assert not lock_path.exists()


def test_own_process_is_alive():
    from llm.providers import ollama_provider

    assert ollama_provider._pid_alive(os.getpid()) is True


def test_gigachat_usage_both_sdk_field_names():
    from types import SimpleNamespace
    from analytics.llm_gigachat import _usage_tokens
    assert _usage_tokens(SimpleNamespace(input_tokens=120, output_tokens=30)) == (120, 30)          # новый SDK
    assert _usage_tokens(SimpleNamespace(prompt_tokens=100, completion_tokens=20)) == (100, 20)     # старый SDK
    assert _usage_tokens(None) == (None, None)
