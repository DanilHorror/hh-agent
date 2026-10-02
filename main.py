import asyncio
from orchestrator.scheduler import Orchestrator

async def main():
    print("="*60)
    print("🚀 Запуск агента автоматизации поиска работы на hh.ru")
    print("="*60)
    
    orchestrator = Orchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print("\n🛑 Получен сигнал остановки. Завершение работы...")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    asyncio.run(main())