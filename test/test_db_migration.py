"""Миграции Alembic и перенос данных без Postgres: обе базы — временные файлы SQLite.

Проверяет, что миграции применяются и откатываются, а скрипт переноса копирует все строки,
сохраняет статусы и даты и не затирает данные без --force.
"""
import asyncio
import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from sqlalchemy import inspect, select  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

import config  # noqa: E402
from database import Base, Vacancy, VacancyKeywords, VacancyRaw, VacancyStatus  # noqa: E402
from migrate_sqlite_to_postgres import MigrationError, migrate  # noqa: E402


def sqlite_url(path) -> str:
    return "sqlite+aiosqlite:///" + str(path).replace("\\", "/")


def alembic_config(url: str) -> Config:
    cfg = Config(os.path.join(ROOT, "alembic.ini"))
    cfg.set_main_option("script_location", os.path.join(ROOT, "migrations"))
    cfg.set_main_option("sqlalchemy.url", url)
    cfg.attributes["configure_logger"] = False
    return cfg


async def tables(url: str) -> set:
    engine = create_async_engine(url)
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
    await engine.dispose()
    return names


async def fill_source(url: str) -> None:
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # По одной строке: у строк разный набор полей
        for row in [
            {"id": 1, "vacancy_id": "101", "title": "QA", "company": "A", "url": "u1",
             "status": VacancyStatus.SKIPPED, "relevance_score": 25.0, "created_at": datetime(2026, 10, 1, 9, 30)},
            {"id": 2, "vacancy_id": "102", "title": "LLM Engineer", "company": "B", "url": "u2",
             "status": VacancyStatus.NOTIFIED, "relevance_score": 81.0, "cover_letter_draft": "Добрый день!"},
            {"id": 5, "vacancy_id": "105", "title": "SDET", "company": "C", "url": "u5",
             "status": VacancyStatus.NEW},
        ]:
            await conn.execute(Vacancy.__table__.insert(), row)
        await conn.execute(VacancyRaw.__table__.insert(), [
            {"id": 1, "vacancy_id": "101", "title": "QA", "description": "Pytest, Selenium", "skills_json": "[]"},
        ])
        await conn.execute(VacancyKeywords.__table__.insert(), [
            {"id": 1, "vacancy_id": "101", "keywords_json": '["pytest"]', "model": "qwen2.5:7b"},
        ])
    await engine.dispose()


@pytest.fixture
def dbs(tmp_path):
    source, target = sqlite_url(tmp_path / "source.db"), sqlite_url(tmp_path / "target.db")
    asyncio.run(fill_source(source))
    command.upgrade(alembic_config(target), "head")
    return source, target


def test_migrations_create_and_drop_schema(tmp_path):
    url = sqlite_url(tmp_path / "schema.db")
    cfg = alembic_config(url)
    command.upgrade(cfg, "head")
    assert {"vacancies", "vacancy_raw", "vacancy_keywords", "alembic_version"} <= asyncio.run(tables(url))
    command.downgrade(cfg, "base")
    assert asyncio.run(tables(url)) == {"alembic_version"}


def test_migration_copies_all_rows_and_values(dbs):
    source, target = dbs
    report = asyncio.run(migrate(source, target, log=lambda _: None))
    assert report == {"vacancies": (3, 3), "vacancy_raw": (1, 1), "vacancy_keywords": (1, 1)}

    async def read():
        engine = create_async_engine(target)
        async with engine.connect() as conn:
            rows = (await conn.execute(select(Vacancy.__table__).order_by(Vacancy.id))).mappings().all()
        await engine.dispose()
        return rows

    rows = asyncio.run(read())
    assert [r["status"] for r in rows] == [VacancyStatus.SKIPPED, VacancyStatus.NOTIFIED, VacancyStatus.NEW]
    assert rows[0]["created_at"] == datetime(2026, 10, 1, 9, 30)
    assert rows[1]["cover_letter_draft"] == "Добрый день!"
    assert rows[2]["id"] == 5


def test_second_run_needs_force(dbs):
    source, target = dbs
    asyncio.run(migrate(source, target, log=lambda _: None))
    with pytest.raises(MigrationError, match="--force"):
        asyncio.run(migrate(source, target, log=lambda _: None))
    report = asyncio.run(migrate(source, target, force=True, log=lambda _: None))
    assert report["vacancies"] == (3, 3)


def test_target_without_schema_is_rejected(tmp_path):
    source = sqlite_url(tmp_path / "source.db")
    asyncio.run(fill_source(source))
    with pytest.raises(MigrationError, match="alembic upgrade head"):
        asyncio.run(migrate(source, sqlite_url(tmp_path / "empty.db"), log=lambda _: None))


def test_migration_target_defaults_to_local_postgres(monkeypatch):
    monkeypatch.setattr(config.settings, "database_url", "")
    assert config.migration_target_url() == config.LOCAL_POSTGRES_URL
    monkeypatch.setattr(config.settings, "database_url", "sqlite+aiosqlite:///./hh_agent.db")
    assert config.migration_target_url() == config.LOCAL_POSTGRES_URL
    monkeypatch.setattr(config.settings, "database_url", "postgresql+asyncpg://u:p@db:5432/x")
    assert config.migration_target_url() == "postgresql+asyncpg://u:p@db:5432/x"
