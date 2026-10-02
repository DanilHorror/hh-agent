import asyncio
import json
from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import Column, Integer, String, DateTime, Enum, Text, Float
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import inspect, select
import enum

from config import database_url

Base = declarative_base()


def utcnow() -> datetime:
    """Текущее время UTC без часового пояса — в таком виде даты уже лежат в базе."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

class VacancyStatus(enum.Enum):
    NEW = "new"
    NOTIFIED = "notified"
    CONFIRMED = "confirmed"
    SENT = "sent"
    SKIPPED = "skipped"
    NEEDS_MANUAL_REVIEW = "needs_manual_review"

class Vacancy(Base):
    __tablename__ = "vacancies"
    
    id = Column(Integer, primary_key=True, index=True)
    vacancy_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=False)
    company = Column(String, nullable=False)
    url = Column(String, nullable=False)
    description_snippet = Column(Text)
    status = Column(Enum(VacancyStatus), default=VacancyStatus.NEW, index=True)
    relevance_score = Column(Float)
    cover_letter_draft = Column(Text)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    sent_at = Column(DateTime)

class VacancyRaw(Base):
    """Полные «сырые» данные вакансии для анализа рынка (analyze_market.py).

    Отдельная таблица: основной поток бота (таблица vacancies) её не использует.
    """
    __tablename__ = "vacancy_raw"

    id = Column(Integer, primary_key=True, index=True)
    vacancy_id = Column(String, unique=True, index=True, nullable=False)
    title = Column(String)
    company = Column(String)
    url = Column(String)
    skills_json = Column(Text)        # JSON-список тегов блока «Ключевые навыки»
    description = Column(Text)        # полный текст описания, без обрезки
    scraped_at = Column(DateTime, default=utcnow)

class VacancyKeywords(Base):
    """Ключевые слова вакансии, которые выписал ИИ (кэш, чтобы не гонять модель повторно)."""
    __tablename__ = "vacancy_keywords"

    id = Column(Integer, primary_key=True, index=True)
    vacancy_id = Column(String, unique=True, index=True, nullable=False)
    keywords_json = Column(Text)      # JSON-список ключевиков
    model = Column(String)            # какой моделью получено
    extracted_at = Column(DateTime, default=utcnow)

class AgentDecision(Base):
    """Решение человека по черновику агента: принять, переписать или пропустить.

    Из этих записей считаются метрики агента (scripts/agent_metrics.py): доля писем,
    принятых без правок, и совпадение оценки модели с решением человека."""
    __tablename__ = "agent_decisions"

    id = Column(Integer, primary_key=True, index=True)
    vacancy_id = Column(String, index=True, nullable=False)
    decision = Column(String(16), nullable=False)       # approve | rewrite | skip
    feedback = Column(Text)                              # замечание к письму (для rewrite)
    relevance_score = Column(Float)                      # оценка модели на момент решения
    revision = Column(Integer, default=0)                # какая по счёту версия письма
    created_at = Column(DateTime, default=utcnow)


# Движок и сессии. Адрес — DATABASE_URL из .env; без него файл SQLite hh_agent.db, как раньше.
DATABASE_URL = database_url()
IS_SQLITE = DATABASE_URL.startswith("sqlite")
# pool_pre_ping: перед запросом проверить, что соединение с сервером живо (Postgres мог перезапуститься)
engine = create_async_engine(DATABASE_URL, echo=False, **({} if IS_SQLITE else {"pool_pre_ping": True}))
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


def _safe_url(url: str) -> str:
    return url.split("@", 1)[-1] if "@" in url else url


async def init_db():
    """Инициализация базы данных.

    SQLite: таблицы создаются сами, как раньше. Postgres: схемой управляют миграции Alembic,
    поэтому здесь только проверка, что они применены.
    """
    async with engine.begin() as conn:
        if IS_SQLITE:
            await conn.run_sync(Base.metadata.create_all)
            return
        tables = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    if "alembic_version" not in tables:
        raise RuntimeError(
            f"В базе {_safe_url(DATABASE_URL)} нет схемы. Выполните: venv\\Scripts\\python.exe -m alembic upgrade head")

async def save_vacancy(vacancy: Vacancy):
    """Сохранение вакансии в БД"""
    async with async_session() as session:
        session.add(vacancy)
        await session.commit()

async def update_vacancy_status(vacancy_id: str, status: VacancyStatus, **kwargs):
    """Обновление статуса вакансии"""
    async with async_session() as session:
        result = await session.execute(select(Vacancy).where(Vacancy.vacancy_id == vacancy_id))
        vacancy = result.scalar_one_or_none()
        
        if vacancy:
            vacancy.status = status
            vacancy.updated_at = utcnow()
            
            # Обновляем дополнительные поля если переданы
            for key, value in kwargs.items():
                if hasattr(vacancy, key):
                    setattr(vacancy, key, value)
            
            if status == VacancyStatus.SENT:
                vacancy.sent_at = utcnow()
            
            await session.commit()

async def is_vacancy_known(vacancy_id: str) -> bool:
    """Проверка, известна ли вакансия (то есть уже полностью обработана).

    Вакансия со статусом NEW — «недообработанная»: её сохранили, но оценка LLM не завершилась
    (сбой/зависание Ollama, окно закрыли). Такую считаем неизвестной, чтобы обработать её заново,
    иначе она навсегда осталась бы висеть без оценки и письма.
    """
    async with async_session() as session:
        result = await session.execute(select(Vacancy).where(Vacancy.vacancy_id == vacancy_id))
        vacancy = result.scalar_one_or_none()
        return vacancy is not None and vacancy.status != VacancyStatus.NEW

async def get_vacancy_by_id(vacancy_id: str) -> Optional[Vacancy]:
    """Получить вакансию по ID"""
    async with async_session() as session:
        result = await session.execute(select(Vacancy).where(Vacancy.vacancy_id == vacancy_id))
        return result.scalar_one_or_none()

async def get_confirmed_vacancies() -> list:
    """Получить все подтвержденные вакансии для отправки"""
    async with async_session() as session:
        result = await session.execute(
            select(Vacancy).where(Vacancy.status == VacancyStatus.CONFIRMED)
        )
        return result.scalars().all()

async def count_today_sent() -> int:
    """Посчитать количество отправленных вакансий сегодня"""
    async with async_session() as session:
        today = datetime.now().date()
        result = await session.execute(
            select(Vacancy).where(
                Vacancy.status == VacancyStatus.SENT,
                Vacancy.sent_at >= today
            )
        )
        return len(result.scalars().all())


# ---------- Анализ рынка: сырые данные вакансий ----------

async def save_vacancy_raw(vacancy_id: str, title: str, company: str, url: str,
                           skills: list, description: str):
    """Сохранить (или перезаписать) полные данные вакансии."""
    skills_json = json.dumps(skills, ensure_ascii=False)
    async with async_session() as session:
        result = await session.execute(select(VacancyRaw).where(VacancyRaw.vacancy_id == vacancy_id))
        row = result.scalar_one_or_none()
        if row is None:
            row = VacancyRaw(vacancy_id=vacancy_id)
            session.add(row)
        row.title = title
        row.company = company
        row.url = url
        row.skills_json = skills_json
        row.description = description
        row.scraped_at = utcnow()
        await session.commit()

async def get_raw_vacancy_ids() -> set:
    """ID всех вакансий, уже сохранённых для анализа."""
    async with async_session() as session:
        result = await session.execute(select(VacancyRaw.vacancy_id))
        return set(result.scalars().all())

async def get_raw_vacancies(vacancy_ids=None) -> list:
    """Вернуть сохранённые вакансии как список dict.

    vacancy_ids — необязательный набор ID для фильтрации (по умолчанию все).
    """
    async with async_session() as session:
        result = await session.execute(select(VacancyRaw))
        rows = result.scalars().all()

    wanted = set(vacancy_ids) if vacancy_ids is not None else None
    vacancies = []
    for row in rows:
        if wanted is not None and row.vacancy_id not in wanted:
            continue
        vacancies.append({
            "vacancy_id": row.vacancy_id,
            "title": row.title or "",
            "company": row.company or "",
            "url": row.url or "",
            "skills": json.loads(row.skills_json) if row.skills_json else [],
            "description": row.description or "",
        })
    return vacancies


async def save_llm_keywords(vacancy_id: str, keywords: list, model: str):
    """Сохранить (или перезаписать) ключевики, выписанные ИИ для вакансии."""
    async with async_session() as session:
        result = await session.execute(select(VacancyKeywords).where(VacancyKeywords.vacancy_id == vacancy_id))
        row = result.scalar_one_or_none()
        if row is None:
            row = VacancyKeywords(vacancy_id=vacancy_id)
            session.add(row)
        row.keywords_json = json.dumps(keywords, ensure_ascii=False)
        row.model = model
        row.extracted_at = utcnow()
        await session.commit()

async def get_llm_keywords(model_prefix: str = None) -> dict:
    """Все сохранённые ключевики ИИ: {vacancy_id: [ключевики]}.

    model_prefix — вернуть только ответы моделей, имя которых начинается с этой строки
    (например "GigaChat"), чтобы не мешать в один отчёт ответы разных движков.
    """
    async with async_session() as session:
        result = await session.execute(select(VacancyKeywords))
        rows = result.scalars().all()
    if model_prefix:
        rows = [r for r in rows if (r.model or "").startswith(model_prefix)]
    return {r.vacancy_id: (json.loads(r.keywords_json) if r.keywords_json else []) for r in rows}


# ---------- Агент: решения человека ----------

async def save_agent_decision(vacancy_id: str, decision: str, feedback: str = "",
                              relevance_score: float = None, revision: int = 0):
    async with async_session() as session:
        session.add(AgentDecision(vacancy_id=vacancy_id, decision=decision, feedback=feedback or None,
                                  relevance_score=relevance_score, revision=revision))
        await session.commit()


async def get_agent_decisions() -> list:
    async with async_session() as session:
        result = await session.execute(select(AgentDecision).order_by(AgentDecision.id))
        return result.scalars().all()
