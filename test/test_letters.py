"""Проверки сопроводительных писем: детерминированные правила, разбор ответа судьи, сводка."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.providers.base import LLMProvider, LLMResponse  # noqa: E402
from quality import letters  # noqa: E402

GOOD = ("Здравствуйте!\n\nМеня зовут Данил Захаров, я QA-инженер с опытом почти 4 года. Автоматизирую "
        "регрессионные UI-сценарии на Java (Selenium, JUnit, TestNG), проверяю REST API через Postman и Newman, "
        "смотрю данные в базе SQL-запросами. В вакансии у вас микросервисы и много интеграций: мой сетевой опыт "
        "помогает быстро находить, на чьей стороне ошибка, по логам и трафику. Готов показать примеры тестов.\n\n"
        "С уважением,\nДанил Захаров")


def test_good_letter_passes():
    assert letters.check_letter(GOOD, "Данил", "Захаров") == {}


def test_cliches_found():
    text = GOOD + " Хочу внести свой вклад и соответствовать высоким стандартам. Буду рад обсудить детали."
    found = letters.check_letter(text, "Данил", "Захаров")["cliche"]
    assert {"внести вклад", "высокие стандарты"} <= set(found)
    assert "буду рад обсудить" not in found   # обычная концовка письма, не штамп


def test_not_only_but_also():
    assert "не только…, но и" in letters.find_cliches("Я умею не только писать тесты, но и читать логи.")


def test_placeholder_surname_length_dump():
    assert "placeholder" in letters.check_letter(GOOD.replace("Здравствуйте!", "Здравствуйте, [Имя HR]!"))
    assert letters.check_letter(GOOD + " Данил Закаряев", "Данил", "Захаров")["surname"] == ["Закаряев"]
    assert "length" in letters.check_letter("Здравствуйте! Прошу рассмотреть.")
    assert "length" in letters.check_letter(GOOD * 3)
    assert "resume_dump" in letters.check_letter(GOOD + "\nЖелаемая зарплата: 150 000 ₽")


def test_empty_and_fallback():
    assert "empty" in letters.check_letter("")
    assert "empty" in letters.check_letter("Здравствуйте! Прошу рассмотреть мое резюме на данную позицию.")


def test_parse_judgment_tolerates_markdown_and_fixes_verdict():
    j = letters.parse_judgment('```json\n{"invented_facts": ["5 лет Kubernetes"], "verdict": "ok"}\n```')
    assert j.verdict == "invented" and j.invented_facts == ["5 лет Kubernetes"]
    assert letters.parse_judgment('{"invented_facts": [], "verdict": "invented"}').verdict == "ok"


class FakeJudge(LLMProvider):
    name, model = "fake", "judge-1"

    def __init__(self):
        self.prompts = []

    async def complete(self, messages, schema=None, *, json_mode=False, temperature=0.3, max_tokens=1500):
        self.prompts.append(messages[0]["content"])
        return LLMResponse('{"invented_facts": ["опыт с Kubernetes"], "verdict": "invented"}', self.name, self.model)


def test_judge_letter_sends_resume_and_letter():
    judge = FakeJudge()
    j = asyncio.run(letters.judge_letter(judge, "РЕЗЮМЕ-ТЕКСТ", "QA", "ПИСЬМО-ТЕКСТ"))
    assert j.verdict == "invented"
    assert "РЕЗЮМЕ-ТЕКСТ" in judge.prompts[0] and "ПИСЬМО-ТЕКСТ" in judge.prompts[0]


def test_summarize():
    texts = {"1": GOOD, "2": GOOD + " Хочу внести свой вклад.", "3": GOOD}
    judgments = {"1": {"verdict": "ok"}, "2": {"verdict": "ok"}, "3": {"verdict": "invented"}}
    s = letters.summarize(texts, judgments, "Данил", "Захаров")
    assert (s.total, s.passed_checks, s.judged, s.invented, s.passed_all) == (3, 2, 3, 1, 1)
    assert s.by_check == {"cliche": 1} and s.cliches == {"внести вклад": 1}


def test_garbage_found():
    assert "garbage" in letters.check_letter(GOOD + " CI/CD流程已结束 {\"cover_letter\": ")


def test_judge_v2_unsupported_claims_are_invented():
    raw = ('{"claims": [{"claim": "Selenium", "evidence": "Selenium, JUnit", "supported": true}, '
           '{"claim": "Kafka", "evidence": "", "supported": false}]}')
    j = letters.parse_claims(raw)
    assert j.verdict == "invented" and j.invented_facts == ["Kafka"]
    assert letters.parse_claims('{"claims": []}').verdict == "ok"


def test_judge_quality():
    labels = {"1": {"invented": True}, "2": {"invented": True}, "3": {"invented": False}, "4": {"invented": False}}
    said = {"1": {"verdict": "invented"}, "2": {"verdict": "ok"}, "3": {"verdict": "invented"}, "4": {"verdict": "ok"}}
    q = letters.judge_quality(labels, said)
    assert (q.tp, q.fn, q.fp, q.tn) == (1, 1, 1, 1)
    assert q.recall == 0.5 and q.precision == 0.5 and q.agreement == 0.5
