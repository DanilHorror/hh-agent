"""Запуск агента только по Habr Career (hh.ru не трогает)."""
import os
os.environ["AGENT_SOURCE"] = "habr"  # до импорта orchestrator: от этого зависит имя лог-файла

import asyncio
from orchestrator.scheduler import Orchestrator


async def main():
    print("=" * 60)
    print("🚀 Запуск агента поиска работы: Habr Career")
    print("=" * 60)

    orchestrator = Orchestrator(source="habr")
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки. Завершение работы...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
