"""Оффлайн-тест GigaChat-движка (пакет gigachat и сеть не нужны: вместо SDK подставная функция).

Запуск из папки hh_agent:   venv\\Scripts\\python.exe test\\test_llm_gigachat.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.llm_gigachat import (GigaChatBackend, GigaChatFailure, classify_error,
                                    read_env_file, resolve_cascade)
from analytics.llm_keywords import extract_keywords


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False


class HttpError(Exception):
    def __init__(self, code, text="err"):
        super().__init__(text)
        self.status_code = code


class ConnectTimeout(Exception):
    pass


# --- классификация ошибок ---
check(classify_error(HttpError(402)).kind == "no_tokens", "402 -> no_tokens")
check(classify_error(HttpError(401)).kind == "auth" and classify_error(HttpError(403)).kind == "auth", "401/403 -> auth")
check(classify_error(HttpError(429)).kind == "rate_limit", "429 -> rate_limit")
check(classify_error(HttpError(503)).kind == "network", "5xx -> network")
check(classify_error(HttpError(400)).kind == "unknown", "400 -> unknown")
check(classify_error(ConnectTimeout("x")).kind == "network", "таймаут -> network")
check(classify_error(Exception("[SSL: CERTIFICATE_VERIFY_FAILED] boom")).kind == "ssl", "сертификат -> ssl")
check(classify_error(ValueError("что-то")).kind == "unknown", "прочее -> unknown")

# --- каскад ---
check(resolve_cascade("B", ["A", "B", "C"]) == ["B", "C"], "каскад начинается с выбранной модели")
check(resolve_cascade("X", ["A", "B"]) == ["X", "A", "B"], "модель вне каскада — первой, дальше весь каскад")

# --- .env ---
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, ".env")
    with open(path, "w", encoding="utf-8") as f:
        f.write('# коммент\nGIGACHAT_CREDENTIALS="abc=="\nGIGACHAT_MODEL = GigaChat-2\n\nBAD LINE\n')
    env = read_env_file(path)
    check(env.get("GIGACHAT_CREDENTIALS") == "abc==", "значение в кавычках, знак = внутри значения")
    check(env.get("GIGACHAT_MODEL") == "GigaChat-2", "пробелы вокруг = убираются")
    check(read_env_file(os.path.join(d, "нет.env")) == {}, "нет файла — пустой словарь")

    # from_env без ключа -> понятная ошибка, ничего не падает
    for k in ("GIGACHAT_CREDENTIALS",):
        os.environ.pop(k, None)
    try:
        GigaChatBackend.from_env(d + os.sep + "пусто")
        check(False, "без ключа должна быть ошибка")
    except ValueError as e:
        check("GIGACHAT_CREDENTIALS" in str(e), "без ключа — ValueError с подсказкой")

# --- backend: успешный вызов + совместимость с extract_keywords ---
calls = []


def ok_send(model, messages):
    calls.append(model)
    return 'Вот: {"keywords": ["Postman", "регрессионное тестирование"]}'


gc = GigaChatBackend("k", ["M1", "M2"], send_fn=ok_send, sleep_fn=lambda s: None)
kws = extract_keywords("QA", "Нужен Postman и регрессия", ["SQL"], gc, gc.current_model)
check(kws == ["Postman", "регрессионное тестирование"], "extract_keywords работает через GigaChatBackend")
check(gc.used_model == "M1" and calls == ["M1"], "использована первая модель")

# --- 402: переход на следующую модель и остаёмся на ней ---
seq = []


def send_402(model, messages):
    seq.append(model)
    if model == "M1":
        raise HttpError(402, "Payment Required")
    return '{"keywords": ["sql"]}'


gc = GigaChatBackend("k", ["M1", "M2"], send_fn=send_402, sleep_fn=lambda s: None)
r = gc(messages=[{"role": "user", "content": "x"}])
check(gc.used_model == "M2" and seq == ["M1", "M2"], "402 -> откат на M2")
gc(messages=[{"role": "user", "content": "x"}])
check(seq == ["M1", "M2", "M2"], "после отката M1 больше не дёргаем")

# --- 402 на всех моделях -> GigaChatFailure(no_tokens) ---
gc = GigaChatBackend("k", ["M1"], send_fn=lambda m, msgs: (_ for _ in ()).throw(HttpError(402)), sleep_fn=lambda s: None)
try:
    gc(messages=[])
    check(False, "должно быть исключение")
except GigaChatFailure as e:
    check(e.kind == "no_tokens", "нет токенов нигде -> GigaChatFailure(no_tokens)")

# --- 401 сразу наружу, без повторов ---
n401 = []
gc = GigaChatBackend("k", ["M1", "M2"], send_fn=lambda m, msgs: (n401.append(m), (_ for _ in ()).throw(HttpError(401)))[1],
                     sleep_fn=lambda s: None)
try:
    gc(messages=[])
except GigaChatFailure as e:
    check(e.kind == "auth" and len(n401) == 1, "401 -> сразу auth, без повторов и без отката")

# --- 429 и сеть: повторы с ожиданием, потом успех ---
state = {"n": 0}
sleeps = []


def flaky(model, messages):
    state["n"] += 1
    if state["n"] == 1:
        raise HttpError(429)
    if state["n"] == 2:
        raise ConnectTimeout("t")
    return '{"keywords": ["git"]}'


gc = GigaChatBackend("k", ["M1"], send_fn=flaky, sleep_fn=sleeps.append)
check(gc(messages=[])["message"]["content"] == '{"keywords": ["git"]}' and len(sleeps) == 2, "429 и сеть — повтор с паузой")

# --- сеть не восстанавливается -> ошибка после лимита повторов ---
gc = GigaChatBackend("k", ["M1"], send_fn=lambda m, msgs: (_ for _ in ()).throw(HttpError(503)), sleep_fn=lambda s: None)
try:
    gc(messages=[])
    check(False, "должно быть исключение")
except GigaChatFailure as e:
    check(e.kind == "network", "вечный 5xx -> network после повторов")

# --- пустые ответы: 3 попытки, потом пустой результат (без исключения) ---
empties = []
gc = GigaChatBackend("k", ["M1"], send_fn=lambda m, msgs: (empties.append(1), "  ")[1], sleep_fn=lambda s: None)
check(gc(messages=[])["message"]["content"] == "" and len(empties) == 3, "пустой ответ: 3 попытки, затем пусто")
check(extract_keywords("t", "d", None, gc, "M1") == [], "пустой ответ -> пустой список ключевиков")

if check.failed:
    print("\nЕСТЬ ОШИБКИ")
    sys.exit(1)
print("\nВсе проверки пройдены")
