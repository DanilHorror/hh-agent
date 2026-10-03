"""Метрики точности оценки вакансий и обёртка учёта токенов."""
import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers.base import LLMProvider, LLMResponse  # noqa: E402
from quality.scoring import RecordingProvider, best_threshold, confusion, separation  # noqa: E402

PAIRS = [(90, "fit"), (75, "fit"), (65, "not_fit"), (55, "fit"), (40, "not_fit"), (20, "not_fit")]


def test_confusion_at_threshold():
    c = confusion(PAIRS, 60)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 2)
    assert c.precision == pytest.approx(2 / 3) and c.recall == pytest.approx(2 / 3)
    assert c.f1 == pytest.approx(2 / 3) and c.accuracy == pytest.approx(4 / 6)


def test_best_threshold_prefers_f1_then_precision():
    threshold, c = best_threshold(PAIRS)
    # пороги 45–55 одинаково хороши (все три подходящие найдены, лишняя одна, F1 0.857) — берём строгий
    assert threshold == 55 and c.recall == 1.0 and c.precision == pytest.approx(0.75)


def test_empty_cases_do_not_crash():
    c = confusion([(10, "not_fit")], 60)
    assert c.precision is None and c.recall is None and c.f1 is None
    assert separation([(10, "not_fit")]) is None


def test_separation_auc():
    assert separation([(90, "fit"), (10, "not_fit")]) == 1.0
    assert separation([(10, "fit"), (90, "not_fit")]) == 0.0
    assert separation([(50, "fit"), (50, "not_fit")]) == 0.5
    # 3 подходящие × 3 неподходящие: «неправильно» упорядочена одна пара из 9 (55 < 65)
    assert separation(PAIRS) == pytest.approx(8 / 9)


class Fake(LLMProvider):
    name, model = "fake", "fake-1"

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        return LLMResponse("{}", self.name, self.model, prompt_tokens=100, completion_tokens=20)


def test_recording_provider_counts_tokens():
    provider = RecordingProvider(Fake())

    async def two_calls():
        await provider.complete([{"role": "user", "content": "a"}])
        await provider.complete([{"role": "user", "content": "b"}])

    asyncio.run(two_calls())
    assert provider.usage.calls == 2
    assert provider.usage.prompt_tokens == 200 and provider.usage.completion_tokens == 40
    assert provider.model == "fake-1" and provider.usage.models == {"fake-1"}
