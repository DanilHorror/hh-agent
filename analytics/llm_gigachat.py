"""GigaChat (Сбер) как движок выделения ключевиков — альтернатива локальной Ollama.

Устроен по образцу call-assistant/src/llm_gigachat.py: SDK `gigachat`, доверенный сертификат
Минцифры через ca_bundle_file, температура ~0, классификация ошибок (402 — кончились токены,
401/403 — ключ, 429 — лимит запросов, 5xx/таймаут — сеть) и каскад моделей: когда у модели
кончились токены, автоматически берём следующую, более лёгкую.

Объект GigaChatBackend вызывается так же, как ollama.chat (см. analytics/llm_keywords.py:
extract_keywords(..., chat_fn=backend)), поэтому промпт и разбор ответа общие для обоих движков.

Настройки берутся из переменных окружения или из файла .env рядом со скриптом:

    GIGACHAT_CREDENTIALS=<ключ авторизации из личного кабинета GigaChat API>   (обязательно)
    GIGACHAT_CA_BUNDLE_FILE=russian_trusted_root_ca.cer   (путь к сертификату Минцифры; относительный
                                                           путь считается от папки скрипта)
    GIGACHAT_MODEL=GigaChat-2-Pro                          (с какой модели начинать; по умолчанию GigaChat-2-Pro)
    GIGACHAT_MODEL_CASCADE=GigaChat-2-Max,GigaChat-2-Pro,GigaChat-2   (порядок отката при нехватке токенов)
    GIGACHAT_TIMEOUT=60                                    (секунд на один запрос)

Пакет gigachat импортируется только при реальном обращении к API — тесты и режим Ollama
работают без него.
"""
import logging
import os
import time
from contextlib import ExitStack
from typing import Callable, Dict, List, Optional

logger = logging.getLogger('GigaChat')

# Как в call-assistant: 0 у облачных API иногда считается невалидным значением.
TEMPERATURE = 0.0001
# Ответ — короткий JSON (до 30 ключевиков), большого запаса не нужно.
MAX_TOKENS = 1024
DEFAULT_TIMEOUT = 60.0
DEFAULT_MODEL = "GigaChat-2-Pro"
DEFAULT_CASCADE = ["GigaChat-2-Max", "GigaChat-2-Pro", "GigaChat-2"]

EMPTY_RETRIES = 3        # сколько раз переспросить, если модель вернула пустой ответ
RATE_LIMIT_RETRIES = 5   # сколько раз подождать и повторить при HTTP 429
NETWORK_RETRIES = 3      # сколько раз повторить при сетевом сбое / 5xx


class GigaChatFailure(Exception):
    """Классифицированная ошибка GigaChat. kind: no_tokens | auth | rate_limit | network | ssl | unknown."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{kind}: {detail}")


def classify_error(e: Exception) -> GigaChatFailure:
    """Превратить исключение SDK в GigaChatFailure. Сама никогда не бросает исключений."""
    if isinstance(e, GigaChatFailure):
        return e
    code = getattr(e, "status_code", None)
    if isinstance(code, int):
        if code == 402:
            return GigaChatFailure("no_tokens", f"токены закончились (HTTP 402): {e}")
        if code in (401, 403):
            return GigaChatFailure("auth", f"ключ недействителен или истёк (HTTP {code}): {e}")
        if code == 429:
            return GigaChatFailure("rate_limit", f"превышен лимит запросов (HTTP 429): {e}")
        if code >= 500:
            return GigaChatFailure("network", f"сервер GigaChat недоступен (HTTP {code}): {e}")
        return GigaChatFailure("unknown", f"HTTP {code}: {e}")
    name = type(e).__name__
    text = str(e)
    if "CERTIFICATE_VERIFY_FAILED" in text or "SSLError" in name or "SSLCertVerification" in name:
        return GigaChatFailure(
            "ssl", "не удалось проверить сертификат сервера GigaChat. Скачайте корневой сертификат Минцифры "
                   "и укажите путь к нему в GIGACHAT_CA_BUNDLE_FILE (как в call-assistant). " + text[:200])
    if "Timeout" in name or "Connect" in name or "Network" in name:
        return GigaChatFailure("network", f"сеть недоступна ({name}): {e}")
    return GigaChatFailure("unknown", f"необработанная ошибка ({name}): {e}")


def read_env_file(path: str) -> Dict[str, str]:
    """Минимальный разбор .env (KEY=VALUE, комментарии #, значения в кавычках). Нет файла — пустой словарь."""
    values: Dict[str, str] = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                values[key.strip()] = value
    except OSError:
        pass
    return values


def resolve_cascade(starting_model: str, cascade: List[str]) -> List[str]:
    """Начать каскад с выбранной модели (как _resolve_gigachat_cascade в call-assistant)."""
    if starting_model in cascade:
        return cascade[cascade.index(starting_model):]
    return [starting_model, *cascade]


def _default_send_fn(credentials: str, ca_bundle_file: Optional[str], timeout: float):
    """Настоящий вызов через SDK gigachat: send(model, messages) -> текст ответа.

    Клиент на каждую модель создаётся один раз и переиспользуется (токен доступа обновляется SDK сам),
    а не создаётся заново на каждую вакансию.
    """
    stack = ExitStack()
    clients: Dict[str, object] = {}

    def send(model: str, messages: List[dict]) -> str:
        from gigachat import GigaChat
        from gigachat.models import ChatCompletionRequest, ChatMessage

        client = clients.get(model)
        if client is None:
            # timeout задаётся на клиенте: в новых версиях SDK у chat.create() своего timeout нет.
            client = stack.enter_context(
                GigaChat(credentials=credentials, model=model, ca_bundle_file=ca_bundle_file, timeout=timeout))
            clients[model] = client
        chat_messages = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        try:
            # Новые версии SDK: параметры генерации лежат в model_options.
            request = ChatCompletionRequest(
                messages=chat_messages,
                model_options={"temperature": TEMPERATURE, "max_tokens": MAX_TOKENS})
        except Exception:  # noqa: BLE001 — старые версии SDK: temperature/max_tokens на верхнем уровне
            request = ChatCompletionRequest(messages=chat_messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        response = client.chat.create(request)
        return response.messages[0].content[0].text

    send.close = stack.close  # type: ignore[attr-defined]
    return send


class GigaChatBackend:
    """Вызывается как ollama.chat: backend(model=..., messages=[...], options=..., format=...)."""

    def __init__(self, credentials: str, models: List[str], ca_bundle_file: Optional[str] = None,
                 timeout: float = DEFAULT_TIMEOUT,
                 send_fn: Optional[Callable[[str, List[dict]], str]] = None,
                 sleep_fn: Callable[[float], None] = time.sleep):
        if not models:
            raise ValueError("список моделей GigaChat пуст")
        self.models = list(models)
        self._idx = 0
        self.used_model = self.models[0]
        self._sleep = sleep_fn
        self._send = send_fn or _default_send_fn(credentials, ca_bundle_file, timeout)

    @classmethod
    def from_env(cls, base_dir: str) -> "GigaChatBackend":
        """Собрать из переменных окружения и .env. ValueError с понятным текстом, если настроено неверно."""
        file_values = read_env_file(os.path.join(base_dir, ".env"))

        def get(name: str, default: str = "") -> str:
            return (os.environ.get(name) or file_values.get(name) or default).strip()

        credentials = get("GIGACHAT_CREDENTIALS")
        if not credentials:
            raise ValueError(
                "Не задан GIGACHAT_CREDENTIALS. Добавьте в файл .env (рядом с analyze_market.py) строку "
                "GIGACHAT_CREDENTIALS=<ключ авторизации из личного кабинета GigaChat API>.")
        try:
            import gigachat  # noqa: F401
        except ImportError:
            raise ValueError("Не установлен пакет gigachat. Выполните: venv\\Scripts\\pip install gigachat") from None

        ca = get("GIGACHAT_CA_BUNDLE_FILE") or None
        if ca:
            ca_path = ca if os.path.isabs(ca) else os.path.join(base_dir, ca)
            if not os.path.isfile(ca_path):
                raise ValueError(f"Файл сертификата GIGACHAT_CA_BUNDLE_FILE не найден: {ca_path}")
            ca = ca_path

        start = get("GIGACHAT_MODEL", DEFAULT_MODEL)
        cascade = [m.strip() for m in get("GIGACHAT_MODEL_CASCADE", ",".join(DEFAULT_CASCADE)).split(",") if m.strip()]
        try:
            timeout = float(get("GIGACHAT_TIMEOUT", str(DEFAULT_TIMEOUT)))
        except ValueError:
            timeout = DEFAULT_TIMEOUT
        return cls(credentials, resolve_cascade(start, cascade), ca_bundle_file=ca, timeout=timeout)

    @property
    def current_model(self) -> str:
        return self.models[self._idx]

    def close(self):
        closer = getattr(self._send, "close", None)
        if closer:
            try:
                closer()
            except Exception:  # noqa: BLE001 — закрытие не должно ронять отчёт
                pass

    def __call__(self, model: Optional[str] = None, messages: Optional[List[dict]] = None,
                 options: Optional[dict] = None, format: Optional[str] = None) -> dict:
        """Один запрос. Параметры model/options/format принимаются для совместимости с ollama.chat и
        игнорируются: модель выбирает каскад, температура и лимит токенов заданы константами выше."""
        messages = messages or []
        empty = rate = net = 0
        while True:
            model_name = self.current_model
            try:
                text = self._send(model_name, messages)
            except Exception as e:  # noqa: BLE001 — классифицируем любую ошибку SDK
                failure = classify_error(e)
                if failure.kind == "no_tokens":
                    if self._idx < len(self.models) - 1:
                        self._idx += 1
                        logger.warning(f"У модели {model_name} кончились токены — перехожу на {self.current_model}")
                        continue
                    raise failure
                if failure.kind == "rate_limit" and rate < RATE_LIMIT_RETRIES:
                    rate += 1
                    logger.warning(f"Лимит запросов GigaChat, жду {10 * rate} с (попытка {rate}/{RATE_LIMIT_RETRIES})")
                    self._sleep(10 * rate)
                    continue
                if failure.kind == "network" and net < NETWORK_RETRIES:
                    net += 1
                    logger.warning(f"Сбой связи с GigaChat, повтор через {5 * net} с ({failure.detail[:120]})")
                    self._sleep(5 * net)
                    continue
                raise failure
            if text and text.strip():
                self.used_model = model_name
                return {"message": {"content": text}}
            empty += 1
            if empty >= EMPTY_RETRIES:
                self.used_model = model_name
                return {"message": {"content": ""}}
