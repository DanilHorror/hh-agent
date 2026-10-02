import json
import re
import os
import time
import asyncio
import contextlib
import ollama
from pydantic import BaseModel, Field, ValidationError
from config import settings

# Файл-замок: если запущены два скрипта (hh и Habr), к Ollama обращается только один за раз.
# Два одновременных запроса к 7B-модели тормозят друг друга и могут "подвесить" обоих.
OLLAMA_LOCK_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ollama.lock")


@contextlib.asynccontextmanager
async def ollama_lock(stale_after: float):
    """Межпроцессный замок на файле. Замок старше stale_after секунд считается брошенным (процесс убили)."""
    fd = None
    announced = False
    while fd is None:
        try:
            fd = os.open(OLLAMA_LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(OLLAMA_LOCK_PATH) > stale_after:
                    os.remove(OLLAMA_LOCK_PATH)
                    continue
            except OSError:
                pass
            if not announced:
                print("[LLM] ⏳ Ollama занята другим окном, жду своей очереди...")
                announced = True
            await asyncio.sleep(1.0)
    try:
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(OLLAMA_LOCK_PATH)
        except OSError:
            pass


class VacancyEvaluation(BaseModel):
    """Схема ответа модели на оценку вакансии. Передаётся в Ollama как format
    (structured output: модель физически не может выдать JSON другой формы)
    и используется для проверки ответа."""
    relevance_score: int = Field(ge=0, le=100, description="Релевантность вакансии резюме, 0-100")
    reasoning: str = Field(description="Короткое обоснование оценки")
    cover_letter: str = Field(description="Черновик сопроводительного письма")


EVALUATION_SCHEMA = VacancyEvaluation.model_json_schema()


def _field(response, name: str, default=None):
    """Ответ ollama — dict в старых версиях клиента и объект в новых."""
    if isinstance(response, dict):
        return response.get(name, default)
    return getattr(response, name, default)


def _content(response) -> str:
    message = _field(response, "message")
    if isinstance(message, dict):
        return (message.get("content") or "").strip()
    return (getattr(message, "content", "") or "").strip()


def parse_evaluation(raw_content: str) -> tuple[dict, str]:
    """Разбирает ответ модели. Возвращает (результат, способ):
    "schema" — ответ прошёл проверку схемой; "json" — валидный JSON, но поля
    пришлось привести вручную; "regex" — JSON сломан, данные вытащены регулярками."""
    match = re.search(r'\{.*\}', raw_content, re.DOTALL)
    json_str = match.group(0) if match else raw_content
    try:
        return VacancyEvaluation.model_validate_json(json_str).model_dump(), "schema"
    except ValidationError:
        pass
    try:
        data = json.loads(json_str)
        if isinstance(data, dict):
            score = data.get("relevance_score", data.get("score", 0))
            try:
                score = max(0, min(100, int(float(score))))
            except (TypeError, ValueError):
                score = 0
            return {
                "relevance_score": score,
                "reasoning": str(data.get("reasoning") or ""),
                "cover_letter": str(data.get("cover_letter") or ""),
            }, "json"
    except json.JSONDecodeError:
        pass
    return extract_from_broken_json(raw_content), "regex"


def _warn_if_truncated(response) -> None:
    """Ollama молча обрезает промпт, длиннее num_ctx. Если модель прочитала
    почти весь контекст, часть резюме или вакансии могла не попасть в него."""
    prompt_tokens = _field(response, "prompt_eval_count")
    output_tokens = _field(response, "eval_count")
    if prompt_tokens is None:
        return
    print(f"[LLM] токены: промпт {prompt_tokens}, ответ {output_tokens} (num_ctx {settings.ollama_num_ctx})")
    if prompt_tokens >= settings.ollama_num_ctx * 0.95:
        print(
            f"[LLM] ⚠️ Промпт занял {prompt_tokens} из {settings.ollama_num_ctx} токенов контекста — "
            "вероятно, он обрезан. Увеличьте OLLAMA_NUM_CTX в .env."
        )


def load_resume() -> str:
    """Загружает резюме из файла resume.txt"""
    resume_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resume.txt")
    try:
        with open(resume_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print("⚠️ Файл resume.txt не найден! Использую резервное резюме.")
        return "DevOps Engineer. Навыки: Linux, Docker, CI/CD, Python, GitHub Actions, Azure."

def extract_user_name(resume_text: str) -> str:
    """
    Извлекает имя из первой строки резюме.
    Формат экспорта с hh.ru: "Фамилия Имя Отчество" (например, "Захаров Данил Алексеевич").
    Так письмо всегда подписывается именем того, чьё резюме сейчас лежит в resume.txt,
    а не захардкоженным именем — удобно, если копию бота настраивают под другого человека.
    """
    first_line = resume_text.strip().split('\n', 1)[0].strip()
    parts = first_line.split()
    if len(parts) >= 2:
        return parts[1]  # Имя — второе слово в "Фамилия Имя Отчество"
    elif parts:
        return parts[0]
    return "Кандидат"

def extract_user_surname(resume_text: str) -> str:
    """Фамилия — первое слово в "Фамилия Имя Отчество" (та же первая строка резюме)."""
    first_line = resume_text.strip().split('\n', 1)[0].strip()
    parts = first_line.split()
    return parts[0] if parts else ""

USER_RESUME = load_resume()
USER_NAME = extract_user_name(USER_RESUME)
USER_SURNAME = extract_user_surname(USER_RESUME)
USER_FULL_NAME = f"{USER_NAME} {USER_SURNAME}".strip()

def strip_placeholders(text: str) -> str:
    """
    Подчищает плейсхолдеры вроде "[Имя HR-специалиста]", "[Название компании]" и т.п.,
    которые модель иногда оставляет незаполненными несмотря на инструкцию в промпте.
    Приветствие "Добрый день, [Имя HR-специалиста]!" превращается в универсальное "Добрый день!",
    а плейсхолдеры в середине текста просто вырезаются с аккуратной чисткой пунктуации/пробелов.
    """
    # "Добрый день, [что-то]!" / "Здравствуйте, [что-то]!" -> без обращения по имени
    text = re.sub(r'([,，])\s*\[[^\]\n]{1,60}\]\s*!', '!', text)
    # Любые оставшиеся [квадратные скобки] вырезаем целиком
    text = re.sub(r'\[[^\]\n]{1,60}\]', '', text)
    # Чистим то, что могло остаться после вырезания: "текст , !" -> "текст!"
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\s+([.,!?])', r'\1', text)
    text = re.sub(r'([.,!?])\1+', r'\1', text)
    return text

def soften_ai_tells(text: str) -> str:
    """
    Подчищает самые узнаваемые ИИ-обороты, которые модель добавляет даже вопреки
    инструкции в промпте (запрет в тексте промпта не гарантирует, что 7B-модель ему следует).
    Основное: конструкция "не только X, но и Y" — один из самых частых и узнаваемых
    маркеров сгенерированного текста, встречается почти в каждом письме.
    """
    # "не только X, но и Y" / "не просто X, а Y" и варианты -> "X и Y"
    text = re.sub(r'не\s+(?:только|просто)\s+(.+?),\s*(?:но\s+и|а\s+также|а)\s+', r'\1 и ', text, flags=re.IGNORECASE)
    return text

def fix_surname(text: str) -> str:
    """
    Модель почти всегда верно пишет имя (оно жёстко задано правилом в промпте), но
    иногда путает или выдумывает фамилию по памяти (например, "Данил Закаряев"
    вместо "Данил Захаров") — 7B-модели это свойственно. Раз имя совпало —
    значит, это точно обращение к кандидату, и мы принудительно подставляем
    настоящую фамилию из резюме вместо того, что придумала модель.
    """
    if not USER_SURNAME:
        return text
    pattern = re.compile(rf'\b{re.escape(USER_NAME)}\s+[А-ЯЁ][а-яё]+\b')
    return pattern.sub(USER_FULL_NAME, text)

def extract_from_broken_json(raw_content: str) -> dict:
    """Спасательный круг: извлекает данные из сломанного JSON с помощью Regex"""
    result = {"relevance_score": 0, "reasoning": "Извлечено через fallback", "cover_letter": ""}
    
    score_match = re.search(r'"(?:relevance_score|relevancy_score|relevance|score)"\s*:\s*(\d+)', raw_content)
    if score_match:
        result["relevance_score"] = int(score_match.group(1))
        
    reasoning_match = re.search(r'"reasoning"\s*:\s*"(.+?)"\s*,', raw_content, re.DOTALL)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).replace('\n', ' ').strip()
        
    cover_match = re.search(r'"cover_letter"\s*:\s*"(.+)"\s*\}', raw_content, re.DOTALL)
    if cover_match:
        text = cover_match.group(1)
        text = text.replace('\n', '\\n').replace('\r', '')
        if text.endswith('"'):
            text = text[:-1]
        result["cover_letter"] = text.strip()
        
    return result

async def check_ollama_health(timeout: float = 5.0) -> tuple[bool, str]:
    """
    Быстрая проверка, что Ollama отвечает по settings.ollama_host (без реальной генерации).
    Возвращает (True, "") если жива, иначе (False, текст_ошибки).
    """
    try:
        client = ollama.AsyncClient(host=settings.ollama_host, timeout=timeout)
        await asyncio.wait_for(client.list(), timeout=timeout)
        return True, ""
    except Exception as e:
        return False, str(e)


async def evaluate_and_draft(vacancy_data: dict) -> dict:
    """Оценивает релевантность и генерирует черновик письма"""
    
    if not vacancy_data.get('description') or vacancy_data['description'] == "Описание отсутствует":
        print("[LLM] ⚠️ Описание вакансии пустое! Невозможно оценить релевантность.")
        return {"relevance_score": 0, "reasoning": "Не удалось извлечь описание", "cover_letter": ""}

    prompt = f"""
Ты — опытный карьерный консультант. Оцени релевантность вакансии и напиши черновик письма.

КРИТИЧЕСКИ ВАЖНЫЕ ПРАВИЛА ФОРМАТА:
1. Твой ответ должен быть СТРОГО валидным JSON.
2. Используй ТОЛЬКО эти ключи: "relevance_score" (число 0-100), "reasoning" (строка), "cover_letter" (строка).
3. В поле "cover_letter" ИСПОЛЬЗУЙ символы \\n для переноса строки. ЗАПРЕЩЕНО использовать реальные переносы строки.
4. В письме используй полное имя "{USER_FULL_NAME}" вместо "[Ваше имя]". Фамилия — ТОЧНО "{USER_SURNAME}", не путай и не придумывай другую.
5. ЗАПРЕЩЕНО оставлять любые плейсхолдеры в квадратных скобках (например, "[Имя HR-специалиста]", "[Название компании]"). Имя получателя и так неизвестно — используй нейтральное приветствие "Добрый день!" или "Здравствуйте!" без обращения по имени.
6. Пиши живым человеческим языком, без узнаваемых ИИ-штампов. КОНКРЕТНО ЗАПРЕЩЕНО:
   - Конструкция "не только X, но и Y" (и варианты "не просто X, а Y") — пиши прямо: "X и Y".
   - Длинное тире (—) как драматическая пауза посреди предложения — используй обычные запятые и точки.
   - Канцеляризмы и отглагольные существительные ("осуществление", "внедрение", "оптимизация процессов") — используй обычные глаголы ("внедрил", "ускорил", "сделал").
   - Клише: "в современном мире", "динамично развивающийся", "с большим интересом", "хотел(а) бы выразить свой интерес", "буду рад(а) обсудить детали при личной встрече", "уверен(а), что мои навыки и опыт будут полезны/ценны вашей команде".
   - Одинаковые формулировки из письма в письмо — пиши каждое письмо своими словами, ориентируясь на конкретную вакансию.
   Пиши так, как реальный кандидат написал бы сам: короче, конкретнее, с разной длиной предложений.

МОЕ РЕЗЮМЕ:
{USER_RESUME}

ВАКАНСИЯ:
Должность: {vacancy_data['title']}
Компания: {vacancy_data['company']}
Описание: {vacancy_data['description']}
"""

    try:
        print(f"[LLM] Отправка запроса к {settings.ollama_model}...")
        started = time.time()
        client = ollama.AsyncClient(host=settings.ollama_host, timeout=settings.ollama_timeout)
        # Асинхронный клиент вместо синхронного ollama.chat: тот блокировал весь процесс
        # (в том числе Telegram-бота) на всё время генерации и не имел таймаута.

        async def ask(messages, fmt):
            return await asyncio.wait_for(
                client.chat(
                    model=settings.ollama_model,
                    messages=messages,
                    # Structured output: JSON-схема ответа. Старые версии Ollama понимают
                    # только format='json' — на них откатываемся ниже.
                    format=fmt,
                    options={
                        # Температуру подняли с 0.3: при низкой модель почти детерминирована и
                        # штампует письма одинаковыми фразами независимо от вакансии. 0.6 даёт
                        # больше лексического разнообразия, оставаясь связным.
                        'temperature': 0.6,
                        # Резюме + вакансия + инструкции ~5 тыс. токенов; при контексте по умолчанию
                        # (2-4 тыс.) начало промпта обрезается. Ограничение num_predict не даёт
                        # модели генерировать бесконечно, если она зациклилась.
                        'num_ctx': settings.ollama_num_ctx,
                        'num_predict': settings.ollama_num_predict,
                    },
                ),
                timeout=settings.ollama_timeout + 10,
            )

        messages = [{'role': 'user', 'content': prompt}]
        async with ollama_lock(stale_after=settings.ollama_timeout + 60):
            try:
                response = await ask(messages, EVALUATION_SCHEMA)
            except Exception as e:
                if "format" not in str(e).lower():
                    raise
                print("[LLM] Эта версия Ollama не поддерживает JSON-схему, использую format='json'")
                response = await ask(messages, 'json')
            _warn_if_truncated(response)
            raw_content = _content(response)
            result, parsed_by = parse_evaluation(raw_content)

            if parsed_by == "regex":
                # Один повтор с текстом ошибки: модели проще исправить конкретный сбой.
                print("[LLM] ⚠️ Ответ не прошёл проверку схемой, прошу модель исправить...")
                retry_messages = messages + [
                    {'role': 'assistant', 'content': raw_content},
                    {'role': 'user', 'content': (
                        'Ответ выше — невалидный JSON. Верни ТОЛЬКО JSON-объект с ключами '
                        '"relevance_score" (целое 0-100), "reasoning" (строка), "cover_letter" (строка).'
                    )},
                ]
                try:
                    retry_response = await ask(retry_messages, EVALUATION_SCHEMA)
                    retry_result, retry_parsed_by = parse_evaluation(_content(retry_response))
                    if retry_parsed_by != "regex":
                        result, parsed_by = retry_result, retry_parsed_by
                except Exception as e:
                    print(f"[LLM] Повтор не удался ({e}), использую извлечённые регулярками данные")
        print(f"[LLM] Ответ получен за {time.time() - started:.0f} сек (разбор: {parsed_by})")

        result["relevance_score"] = int(result.get("relevance_score", 0))
        if "reasoning" not in result or not result["reasoning"]:
            result["reasoning"] = "Модель не предоставила обоснование"
        if "cover_letter" not in result or not result["cover_letter"]:
            result["cover_letter"] = "Здравствуйте! Прошу рассмотреть мое резюме на данную позицию."
        
        # ИСПРАВЛЕНИЕ: Заменяем экранированные \n на реальные переносы строк
        result["cover_letter"] = result["cover_letter"].replace('\\n', '\n')
        
        # ИСПРАВЛЕНИЕ: Заменяем плейсхолдер на полное имя из резюме
        result["cover_letter"] = result["cover_letter"].replace('[Ваше имя]', USER_FULL_NAME)
        result["cover_letter"] = result["cover_letter"].replace('[ваше имя]', USER_FULL_NAME)
        result["cover_letter"] = result["cover_letter"].replace('[Ваше Имя]', USER_FULL_NAME)

        # ИСПРАВЛЕНИЕ: На случай, если модель всё же оставила [Имя HR] / [Название компании] и т.п.
        result["cover_letter"] = strip_placeholders(result["cover_letter"])

        # ИСПРАВЛЕНИЕ: Сглаживаем узнаваемые ИИ-обороты ("не только X, но и Y" и т.п.)
        result["cover_letter"] = soften_ai_tells(result["cover_letter"])

        # ИСПРАВЛЕНИЕ: Принудительно подставляем настоящую фамилию, если модель её перепутала
        result["cover_letter"] = fix_surname(result["cover_letter"])

        print(f"[LLM] ✅ Оценка релевантности: {result['relevance_score']}/100")
        return result
        
    except Exception as e:
        # error=True: вакансия не должна навсегда попадать в SKIPPED из-за сбоя модели — оркестратор повторит её позже
        reason = "таймаут" if isinstance(e, (asyncio.TimeoutError, TimeoutError)) or "timeout" in type(e).__name__.lower() else str(e)
        print(f"❌ Ошибка вызова LLM ({reason}). Проверьте, что Ollama запущена: команда `ollama ps`")
        return {"relevance_score": 0, "reasoning": "Ошибка соединения с Ollama", "cover_letter": "", "error": True}


async def generate_answers_for_questions(questions: list, vacancy_data: dict) -> list:
    """Генерирует ответы на вопросы работодателя на основе резюме."""
    questions_text = "\n".join([f"{q['index']+1}. {q['text']} (тип: {q['type']})" for q in questions])
    
    prompt = f"""
Ты — помощник соискателя. Ответь на вопросы работодателя кратко и по делу.
ОТВЕТ ДОЛЖЕН БЫТЬ СТРОГО валидным JSON-массивом строк (например: ["Ответ 1", "Ответ 2"]).
Не используй реальные переносы строк внутри элементов массива.

МОЕ РЕЗЮМЕ:
{USER_RESUME}

ВОПРОСЫ:
{questions_text}
"""

    try:
        print(f"[LLM] Генерация ответов на {len(questions)} вопросов...")
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{'role': 'user', 'content': prompt}],
            options={'temperature': 0.3, 'format': 'json'}
        )
        
        raw_content = response['message']['content'].strip()
        match = re.search(r'\[.*\]', raw_content, re.DOTALL)
        json_str = match.group(0) if match else raw_content
            
        try:
            answers = json.loads(json_str)
        except json.JSONDecodeError:
            answers = [raw_content]
            
        print(f"[LLM] ✅ Сгенерировано ответов: {len(answers)}")
        return answers
        
    except Exception as e:
        print(f"[LLM] ❌ Ошибка генерации ответов: {e}")
        return [None] * len(questions)