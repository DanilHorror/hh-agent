"""Перенос данных бота из SQLite (hh_agent.db) в Postgres.

Перед запуском: база Postgres поднята (docker compose up -d) и схема создана
(python -m alembic upgrade head). Бота на время переноса лучше остановить.

    venv\\Scripts\\python.exe scripts\\migrate_sqlite_to_postgres.py           перенос
    venv\\Scripts\\python.exe scripts\\migrate_sqlite_to_postgres.py --force   очистить таблицы
                                                                             в Postgres и перенести заново

Исходная база не меняется. В конце скрипт сверяет число строк и набор vacancy_id в каждой таблице.
Код возврата 0 — всё совпало, 1 — есть расхождения или ошибка.
"""
import argparse
import asyncio
import os
import sys

from sqlalchemy import func, inspect, select, text
from sqlalchemy.ext.asyncio import create_async_engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config import migration_target_url  # noqa: E402
from database import Vacancy, VacancyKeywords, VacancyRaw  # noqa: E402

TABLES = [Vacancy.__table__, VacancyRaw.__table__, VacancyKeywords.__table__]
DEFAULT_SOURCE = "sqlite+aiosqlite:///" + os.path.join(ROOT, "hh_agent.db").replace("\\", "/")
BATCH = 500


class MigrationError(Exception):
    pass


def _safe_url(url: str) -> str:
    """Адрес без пароля — для вывода в консоль."""
    if "@" in url and "://" in url:
        scheme, rest = url.split("://", 1)
        return f"{scheme}://***@{rest.split('@', 1)[1]}"
    return url


async def _count(conn, table) -> int:
    return (await conn.execute(select(func.count()).select_from(table))).scalar_one()


async def _ids(conn, table) -> set:
    return set((await conn.execute(select(table.c.vacancy_id))).scalars().all())


async def migrate(source_url: str, target_url: str, force: bool = False, log=print) -> dict:
    """Перенести три таблицы. Возвращает {таблица: (строк в источнике, строк в цели)}."""
    if source_url == target_url:
        raise MigrationError("источник и цель совпадают")
    source = create_async_engine(source_url)
    target = create_async_engine(target_url)
    report = {}
    try:
        async with target.connect() as conn:
            existing = await conn.run_sync(lambda c: set(inspect(c).get_table_names()))
        if "alembic_version" not in existing:
            raise MigrationError(
                "в целевой базе нет схемы. Сначала выполните: venv\\Scripts\\python.exe -m alembic upgrade head")

        async with source.connect() as src:
            source_tables = await src.run_sync(lambda c: set(inspect(c).get_table_names()))
            data = {}
            for table in TABLES:
                if table.name not in source_tables:
                    log(f"  {table.name}: в исходной базе нет такой таблицы, пропускаю")
                    data[table.name] = []
                    continue
                rows = (await src.execute(select(table))).mappings().all()
                data[table.name] = [dict(r) for r in rows]

        async with target.begin() as dst:
            dialect = dst.dialect.name
            if force:
                for table in reversed(TABLES):
                    if dialect == "postgresql":
                        await dst.execute(text(f"TRUNCATE {table.name} RESTART IDENTITY CASCADE"))
                    else:
                        await dst.execute(table.delete())
            else:
                busy = {t.name: await _count(dst, t) for t in TABLES}
                busy = {k: v for k, v in busy.items() if v}
                if busy:
                    raise MigrationError(
                        f"в целевой базе уже есть данные {busy}. Запустите с --force, чтобы перенести заново")

            for table in TABLES:
                rows = data[table.name]
                for start in range(0, len(rows), BATCH):
                    await dst.execute(table.insert(), rows[start:start + BATCH])
                if dialect == "postgresql":
                    # id переносим как есть, поэтому счётчик автонумерации надо сдвинуть за максимум
                    await dst.execute(text(
                        f"SELECT setval(pg_get_serial_sequence('{table.name}', 'id'), "
                        f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table.name}"))

        ok = True
        async with source.connect() as src, target.connect() as dst:
            for table in TABLES:
                src_count = len(data[table.name])
                dst_count = await _count(dst, table)
                same_ids = {r["vacancy_id"] for r in data[table.name]} == await _ids(dst, table)
                report[table.name] = (src_count, dst_count)
                mark = "OK  " if src_count == dst_count and same_ids else "FAIL"
                ok = ok and mark == "OK  "
                log(f"  {mark} {table.name}: SQLite {src_count} -> Postgres {dst_count}"
                    + ("" if same_ids else " (наборы vacancy_id различаются)"))
        if not ok:
            raise MigrationError("данные перенеслись не полностью")
        return report
    finally:
        await source.dispose()
        await target.dispose()


def main() -> int:
    parser = argparse.ArgumentParser(description="Перенос данных hh-агента из SQLite в Postgres")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="откуда (по умолчанию hh_agent.db)")
    parser.add_argument("--target", default=None, help="куда (по умолчанию DATABASE_URL или локальный Postgres)")
    parser.add_argument("--force", action="store_true", help="очистить таблицы в цели перед переносом")
    args = parser.parse_args()
    target = args.target or migration_target_url()

    print(f"Источник: {_safe_url(args.source)}")
    print(f"Цель:     {_safe_url(target)}")
    try:
        asyncio.run(migrate(args.source, target, force=args.force))
    except MigrationError as e:
        print(f"Ошибка: {e}")
        return 1
    except OSError as e:
        print(f"Не удалось подключиться к базе ({e}). Проверьте, что Postgres запущен: docker compose ps")
        return 1
    print("Готово: все строки перенесены.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
