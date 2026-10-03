"""Трассировка в Langfuse: без ключей — пустые обёртки, с клиентом — шаги и вызовы модели попадают в трассу."""
import asyncio
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm import brain, tracing  # noqa: E402
from llm.providers.base import LLMProvider, LLMResponse  # noqa: E402


def test_disabled_without_keys():
    assert tracing.client() is None
    with tracing.span("x", input="y") as s:
        s.update(output="z")                   # пустая обёртка не падает
    with tracing.trace("vacancy", session_id="v-1"):
        pass
    assert tracing.usage(None, None) is None
    assert tracing.usage(10, 5) == {"input_tokens": 10, "output_tokens": 5}


class FakeObservation:
    def __init__(self, log, kind, name, fields):
        self.log, self.record = log, {"type": kind, "name": name, **fields}

    def update(self, **kwargs):
        self.record.update(kwargs)


class FakeLangfuse:
    def __init__(self):
        self.log = []

    @contextlib.contextmanager
    def start_as_current_observation(self, as_type, name, **fields):
        obs = FakeObservation(self.log, as_type, name, fields)
        self.log.append(obs.record)
        yield obs


class Echo(LLMProvider):
    name, model = "fake", "fake-1"

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        return LLMResponse("ответ", self.name, self.model, prompt_tokens=12, completion_tokens=3)


def test_ask_llm_records_generation(monkeypatch):
    fake = FakeLangfuse()
    monkeypatch.setattr(tracing, "client", lambda: fake)
    monkeypatch.setattr(brain, "get_provider", lambda: Echo())
    asyncio.run(brain.ask_llm([{"role": "user", "content": "вопрос"}], name="evaluate"))
    gen = fake.log[0]
    assert gen["type"] == "generation" and gen["name"] == "evaluate" and gen["model"] == "fake-1"
    assert gen["output"] == "ответ" and gen["usage_details"] == {"input_tokens": 12, "output_tokens": 3}


def test_error_marks_observation_and_reraises(monkeypatch):
    fake = FakeLangfuse()
    monkeypatch.setattr(tracing, "client", lambda: fake)
    try:
        with tracing.span("boom"):
            raise ValueError("сломалось")
    except ValueError:
        pass
    else:
        raise AssertionError("исключение должно пробрасываться")
    assert fake.log[0]["level"] == "ERROR" and "сломалось" in fake.log[0]["status_message"]
