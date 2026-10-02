"""Оффлайн-тест ИИ-извлечения ключевиков (Ollama не нужен: вместо неё подставная функция).

Запуск из папки hh_agent:   venv\\Scripts\\python.exe test\\test_llm_keywords.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.keywords import analyze_llm, normalize_keyword
from analytics.llm_keywords import build_prompt, parse_keywords, extract_keywords, NUM_CTX


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False

# --- разбор ответа модели ---
check(parse_keywords('{"keywords": ["Postman", "SQL"]}') == ["Postman", "SQL"], "чистый JSON")
check(parse_keywords('Вот ответ: {"keywords": ["Docker", "Docker", "git"]} Готово') == ["Docker", "git"],
      "JSON с лишним текстом и дублями")
check(parse_keywords('["a b", "c"]') == ["a b", "c"], "JSON-список без объекта")
check(parse_keywords('{"ключевики": ["python"]}') == ["python"], "другое имя ключа")
check(parse_keywords('{"keywords": ["postman", "sql", "rest api"') == ["postman", "sql", "rest api"],
      "оборванный JSON — вытаскиваем что есть")
check(parse_keywords("") == [] and parse_keywords("не знаю") == [], "пустой/бесполезный ответ")
check(parse_keywords('{"keywords": ["' + "очень длинное слово " * 10 + '", "ok"]}') == ["ok"], "слишком длинные отбрасываются")

# --- промпт: текст не теряется, служебные блоки hh.ru вырезаны ---
long_text = "Обязанности:\n" + "\n".join(f"Пункт {i}: тестирование модуля {i}" for i in range(300))
long_text += "\nТребования: SQL и Postman\nЗадайте вопрос работодателю\nКакая оплата труда?"
prompt = build_prompt("QA", long_text, ["Postman", "SQL"])
check("Пункт 299" in prompt and "Postman и" not in prompt.split("Текст:")[0], "длинный текст доходит до модели целиком")
check("Требования: SQL и Postman" in prompt, "хвост описания (требования) на месте")
check("Задайте вопрос работодателю" not in prompt and "Какая оплата труда" not in prompt.split("ВАКАНСИЯ")[1],
      "служебные блоки hh.ru вырезаны")
check("Ключевые навыки, указанные работодателем: Postman, SQL" in prompt, "теги hh добавлены в промпт")

# --- вызов модели ---
calls = []


def fake_chat(**kw):
    calls.append(kw)
    return {"message": {"content": '{"keywords": ["Postman", "тестирования API", "CI-CD"]}'}}


res = extract_keywords("QA", "Текст вакансии", ["SQL"], chat_fn=fake_chat, model="test-model")
check(res == ["Postman", "тестирования API", "CI-CD"], "extract_keywords возвращает ключевики")
check(calls[0]["model"] == "test-model" and calls[0]["format"] == "json", "формат JSON и модель переданы")
check(calls[0]["options"]["num_ctx"] == NUM_CTX >= 8192 and calls[0]["options"]["temperature"] == 0.0,
      "контекстное окно увеличено, температура 0")

# --- нормализация и склейка вариантов ---
check(normalize_keyword("Postman") == "postman", "Postman -> postman")
check(normalize_keyword("CI-CD") == "ci/cd", "CI-CD -> ci/cd")
check(normalize_keyword("опыт") == "" and normalize_keyword("  ") == "" and normalize_keyword("5") == "",
      "мусор отбрасывается")

vacs = [
    {"vacancy_id": "1", "skills": ["Postman", "SQL"]},
    {"vacancy_id": "2", "skills": []},
    {"vacancy_id": "3", "skills": ["python"]},
    {"vacancy_id": "4", "skills": []},   # ИИ по ней ничего не вернул — в подсчёт не входит
]
llm = {
    "1": ["Postman", "тестирование API", "регрессионное тестирование"],
    "2": ["postman", "тестирования API", "CI-CD", "Регрессионного тестирования"],
    "3": ["ci/cd", "тестирование api", "Zabbix"],
}
out = analyze_llm(vacs, llm, min_df=1)
by = {r["keyword"]: r for r in out["rows"]}
check(out["total"] == 3, "считаются только обработанные ИИ вакансии (3 из 4)")
check(by["postman"]["vacancies"] == 2 and by["postman"]["in_tags"] == 1 and by["postman"]["in_llm"] == 2,
      "postman: 2 вакансии, тег у 1, ИИ назвал у 2")
api = [r for r in out["rows"] if "api" in r["keyword"] and "тестирован" in r["keyword"]]
check(len(api) == 1 and api[0]["vacancies"] == 3, "«тестирование API» / «тестирования API» склеены в один (3 вакансии)")
reg = [r for r in out["rows"] if "регрессион" in r["keyword"]]
check(len(reg) == 1 and reg[0]["vacancies"] == 2, "падежные варианты «регрессионное тестирование» склеены")
check(by["ci/cd"]["vacancies"] == 2, "«CI-CD» и «ci/cd» — один ключевик")
check(by["python"]["vacancies"] == 1 and by["python"]["in_tags"] == 1, "тег hh учитывается вместе с ИИ")
check([r["vacancies"] for r in out["rows"]] == sorted([r["vacancies"] for r in out["rows"]], reverse=True),
      "сортировка от большего к меньшему")

# --- канонизация формулировок ИИ ---
from analytics.keywords import keyword_variants
check(keyword_variants("разработка тест-кейсов") == ["тест-кейсы"], "«разработка тест-кейсов» -> тест-кейсы")
check(keyword_variants("test case") == ["тест-кейсы"], "«test case» -> тест-кейсы")
check(keyword_variants("чек-листов") == ["чек-листы"], "«чек-листов» -> чек-листы")
check(keyword_variants("автоматизированное тестирование") == ["автоматизация тестирования"], "автоматизированное тестирование")
check(keyword_variants("автоматические тесты") == ["автоматизация тестирования"], "автоматические тесты")
check(keyword_variants("REST API") == ["rest api"], "REST API не раскладывается на api")
check(keyword_variants("тестирование API") == ["тестирование api"], "«тестирование API» остаётся отдельным ключевиком")
check(keyword_variants("английский — B1 — средний") == ["английский язык"], "английский B1 средний -> английский язык")
check(keyword_variants("ooп") == ["ооп"], "латинская o вместо кириллической исправлена")
check(keyword_variants("QA") == [] and keyword_variants("тестирование") == [] and keyword_variants("QA Engineer") == [],
      "слишком общие ключевики отбрасываются")
check(keyword_variants("napalm") == ["napalm"], "неизвестная технология остаётся как есть")

# --- словарь подстраховывает ИИ ---
vs = [{"vacancy_id": "1", "skills": [], "description": "Составление тест-кейсов и чек-листов, ведение баг-репортов. Postman."},
      {"vacancy_id": "2", "skills": [], "description": "Ищем тестировщика. Нужен Postman и SQL."}]
ll = {"1": ["Postman"], "2": ["SQL"]}
with_dict = {r["keyword"]: r for r in analyze_llm(vs, ll, use_dictionary=True)["rows"]}
no_dict = {r["keyword"]: r for r in analyze_llm(vs, ll, use_dictionary=False)["rows"]}
check("тест-кейсы" in with_dict and "тест-кейсы" not in no_dict, "словарь находит то, что ИИ пропустил")
check(with_dict["postman"]["vacancies"] == 2 and with_dict["postman"]["in_llm"] == 1 and with_dict["postman"]["in_dict"] == 2,
      "postman: ИИ назвал в 1, словарь нашёл в 2, итого 2 вакансии")

print("\nРЕЗУЛЬТАТ:", "ЕСТЬ ОШИБКИ" if check.failed else "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
sys.exit(1 if check.failed else 0)
