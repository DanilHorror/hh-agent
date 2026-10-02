"""Запуск агента только по hh.ru (Habr Career не трогает)."""
import os
os.environ["AGENT_SOURCE"] = "hh"  # до импорта orchestrator: от этого зависит имя лог-файла

import asyncio
from orchestrator.scheduler import Orchestrator


async def main():
    print("=" * 60)
    print("🚀 Запуск агента поиска работы: hh.ru")
    print("=" * 60)

    orchestrator = Orchestrator(source="hh")
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки. Завершение работы...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
