from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import Optional

class Settings(BaseSettings):
    # Telegram (значения задаются только в .env, в коде секретов нет)
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    # Можно указать несколько ID через запятую, например: 111111,222222
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    @property
    def telegram_chat_ids(self) -> list[int]:
        """Список получателей уведомлений (поддержка нескольких ID через запятую в .env)"""
        return [int(cid.strip()) for cid in str(self.telegram_chat_id).split(',') if cid.strip()]

    # LLM: какой провайдер отвечает — ollama | gigachat | yandexgpt | openai (см. llm/providers)
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    # Сколько секунд ждать ответ облачной модели
    llm_timeout: int = Field(default=60, alias="LLM_TIMEOUT")
    # OpenAI-совместимый API (Groq, OpenRouter, DeepSeek, vLLM, LM Studio...)
    llm_base_url: str = Field(default="", alias="LLM_BASE_URL")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    # YandexGPT (Yandex Cloud / AI Studio)
    yandex_api_key: str = Field(default="", alias="YANDEX_API_KEY")
    yandex_folder_id: str = Field(default="", alias="YANDEX_FOLDER_ID")
    yandex_model: str = Field(default="yandexgpt/latest", alias="YANDEX_MODEL")

    # База данных. Пусто — файл SQLite hh_agent.db, как раньше.
    # Postgres из docker-compose.yml: postgresql+asyncpg://hh:hh_local_dev@localhost:5433/hh_agent
    database_url: str = Field(default="", alias="DATABASE_URL")

    # Агент на LangGraph: кнопки «Подходит / Переписать / Пропустить» под черновиками.
    # false — старый режим (уведомление со ссылкой, без кнопок).
    agent_enabled: bool = Field(default=True, alias="AGENT_ENABLED")

    # RAG в Telegram-боте (/ask): на чём считать эмбеддинги и реранкер. По умолчанию процессор:
    # видеокарту в это время занимает Ollama, и две нагрузки на 8 ГБ видеопамяти мешают друг другу.
    # cuda — на видеокарте (быстрее, если памяти хватает).
    rag_bot_device: str = Field(default="cpu", alias="RAG_BOT_DEVICE")

    # Ollama
    ollama_model: str = Field(default="qwen2.5:7b", alias="OLLAMA_MODEL")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    # Сколько секунд ждать ответ модели, прежде чем считать запрос зависшим
    ollama_timeout: int = Field(default=240, alias="OLLAMA_TIMEOUT")
    # Размер контекста: резюме + вакансия + инструкции занимают ~5 тыс. токенов
    ollama_num_ctx: int = Field(default=8192, alias="OLLAMA_NUM_CTX")
    # Максимум токенов в ответе (защита от зацикливания генерации)
    ollama_num_predict: int = Field(default=1500, alias="OLLAMA_NUM_PREDICT")

    # Поведенческие лимиты (п. 6)
    daily_response_limit: int = Field(default=3, alias="DAILY_RESPONSE_LIMIT")
    working_hours_start: int = Field(default=9, alias="WORKING_HOURS_START")
    working_hours_end: int = Field(default=23, alias="WORKING_HOURS_END")

    # Параметры поиска вакансий
    search_query: str = Field(default="Python Developer", alias="SEARCH_QUERY")
    # ID региона(ов) hh.ru (параметр area в ссылке). Можно указать несколько через запятую,
    # например: SEARCH_AREA=113,16,40 — тогда поиск идёт сразу по всем регионам.
    search_area: str = Field(default="1", alias="SEARCH_AREA")

    @property
    def search_areas(self) -> list[str]:
        """Список area-кодов hh.ru для поиска (поддержка нескольких регионов через запятую в .env)."""
        return [a.strip() for a in str(self.search_area).split(',') if a.strip()]
    # "name" — искать только в названии вакансии (рекомендуется, меньше мусора).
    # Оставить пустым — искать по всему тексту вакансии (как раньше, шире, но больше нерелевантного).
    search_field: Optional[str] = Field(default="name", alias="SEARCH_FIELD")
    
    # Необязательные фильтры (можно оставлять пустыми в .env)
    search_experience: Optional[str] = Field(default=None, alias="SEARCH_EXPERIENCE")
    search_employment: Optional[str] = Field(default=None, alias="SEARCH_EMPLOYMENT")
    # Старый фильтр hh.ru "График работы" (fullDay/shift/flexible/remote/flyInFlyOut).
    # remote тут — это ЖЁСТКО только удалённая работа, гибрид под него не попадает.
    # Для удалёнки+гибрида используйте SEARCH_WORK_FORMAT ниже — оставьте это поле пустым.
    search_schedule: Optional[str] = Field(default=None, alias="SEARCH_SCHEDULE")
    # Новый фильтр hh.ru "Формат работы": ON_SITE, REMOTE, HYBRID, FIELD_WORK.
    # Можно несколько через запятую, например: SEARCH_WORK_FORMAT=REMOTE,HYBRID
    search_work_format: str = Field(default="", alias="SEARCH_WORK_FORMAT")

    @property
    def search_work_formats(self) -> list[str]:
        """Список значений work_format для hh.ru (несколько форматов через запятую в .env)."""
        return [w.strip().upper() for w in str(self.search_work_format).split(',') if w.strip()]

    # Habr Career (https://career.habr.com/vacancies) — второй источник вакансий
    habr_enabled: bool = Field(default=True, alias="HABR_ENABLED")
    # ID специализаций Habr (параметр s[] в ссылке): 12,10,13,87 = ручное тестирование, автоматизация,
    # QA-инженер, UX-тестировщик. Пустое значение в .env отключает фильтр.
    habr_specializations: str = Field(default="12,10,13,87", alias="HABR_SPECIALIZATIONS")
    # Дополнительные текстовые запросы через запятую или OR (каждый ищется отдельно, вместе с фильтром).
    # Пусто и заданы специализации — текстового поиска нет; пусто и нет специализаций — берётся SEARCH_QUERY
    habr_queries: str = Field(default="", alias="HABR_QUERIES")
    habr_remote_only: bool = Field(default=False, alias="HABR_REMOTE_ONLY")
    habr_max_pages: int = Field(default=2, alias="HABR_MAX_PAGES")

    max_vacancies_per_cycle: int = Field(default=10, alias="MAX_VACANCIES_PER_CYCLE")
    freshness_days: int = Field(default=3, alias="FRESHNESS_DAYS")
    min_relevance_score: int = Field(default=60, alias="MIN_RELEVANCE_SCORE")
    # Страж письма: судья проверяет черновик на выдумки, модель исправляет (не больше LETTER_GUARD_ROUNDS раз)
    letter_guard: bool = Field(default=True, alias="LETTER_GUARD")
    letter_judge_provider: str = Field(default="gigachat", alias="LETTER_JUDGE_PROVIDER")
    letter_guard_rounds: int = Field(default=2, alias="LETTER_GUARD_ROUNDS")

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

SQLITE_URL = "sqlite+aiosqlite:///./hh_agent.db"
# Postgres из docker-compose.yml (локальный, доступен только с этого компьютера)
LOCAL_POSTGRES_URL = "postgresql+asyncpg://hh:hh_local_dev@localhost:5433/hh_agent"


def database_url() -> str:
    """Адрес базы бота: DATABASE_URL из .env или SQLite по умолчанию."""
    return settings.database_url.strip() or SQLITE_URL


def migration_target_url() -> str:
    """Куда применять миграции и переносить данные: Postgres из .env, иначе локальный из docker-compose.

    Так миграции и перенос можно выполнить до того, как бот переключён на Postgres."""
    url = settings.database_url.strip()
    return url if url.startswith("postgresql") else LOCAL_POSTGRES_URL