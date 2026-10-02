"""Где граф хранит состояние вакансий между шагами (checkpointer).

Postgres — основной вариант: ожидающие решения вакансии переживают перезапуск бота и компьютера.

Тонкость Windows: асинхронный драйвер psycopg не работает с ProactorEventLoop, а Playwright на Windows
требует именно его. Поэтому берём синхронный PostgresSaver (пул соединений psycopg) и даём ему
асинхронные методы, которые выполняют синхронные в отдельном потоке (asyncio.to_thread).
Граф и всё остальное остаются асинхронными, цикл событий не блокируется.

Без Postgres (бот на SQLite) — состояние в памяти: работает, но ожидающие вакансии
теряются при перезапуске.
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("Agent")


def _psycopg_url(sqlalchemy_url: str) -> str:
    """postgresql+asyncpg://… → postgresql://… (формат строки подключения psycopg)."""
    scheme, rest = sqlalchemy_url.split("://", 1)
    return "postgresql://" + rest


def _threaded_saver_class():
    from langgraph.checkpoint.postgres import PostgresSaver

    class ThreadedPostgresSaver(PostgresSaver):
        """Синхронный PostgresSaver с асинхронными методами через пул потоков."""

        async def aget_tuple(self, config):
            return await asyncio.to_thread(self.get_tuple, config)

        async def alist(self, config, *, filter=None, before=None, limit=None):
            items = await asyncio.to_thread(
                lambda: list(self.list(config, filter=filter, before=before, limit=limit)))
            for item in items:
                yield item

        async def aput(self, config, checkpoint, metadata, new_versions):
            return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

        async def aput_writes(self, config, writes, task_id, task_path=""):
            return await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

        async def adelete_thread(self, thread_id):
            return await asyncio.to_thread(self.delete_thread, thread_id)

    return ThreadedPostgresSaver


async def make_checkpointer(database_url: str):
    """Вернуть (checkpointer, закрыть()). Таблицы LangGraph в Postgres создаются здесь же."""
    if database_url.startswith("postgresql"):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        pool = ConnectionPool(
            _psycopg_url(database_url), min_size=1, max_size=4, open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        )
        await asyncio.to_thread(pool.open, True, 10.0)   # wait=True, timeout=10 с
        saver = _threaded_saver_class()(pool)
        await asyncio.to_thread(saver.setup)
        logger.info("Состояние агента хранится в Postgres")
        return saver, lambda: pool.close()

    from langgraph.checkpoint.memory import InMemorySaver

    logger.warning("Состояние агента хранится в памяти (нет DATABASE_URL с Postgres): "
                   "ожидающие решения вакансии пропадут при перезапуске бота")
    return InMemorySaver(), lambda: None
