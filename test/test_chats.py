import sys
import os

# Добавляем родительскую папку (hh_agent) в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from browser.worker import BrowserWorker
from chat_monitor.worker import ChatMonitor
from database import init_db

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestChats')

async def test_chat_monitoring():
    """Тест проверки чатов hh.ru"""
    logger.info("="*60)
    logger.info("🧪 ТЕСТ ПРОВЕРКИ ЧАТОВ HH.RU")
    logger.info("="*60)
    
    await init_db()
    
    # ИСПРАВЛЕНИЕ: Принудительно указываем путь к профилю в корневой папке hh_agent
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    profile_dir = os.path.join(root_dir, "browser_profile")
    
    worker = BrowserWorker(user_data_dir=profile_dir)
    
    try:
        await worker.start()
        logger.info("✅ Браузер запущен")
        
        monitor = ChatMonitor(worker)
        
        # 1. Проверяем чаты
        logger.info("\n📬 Шаг 1: Проверяем чаты...")
        result = await monitor.check_messages()
        
        logger.info("\n" + "="*60)
        logger.info("РЕЗУЛЬТАТЫ ПРОВЕРКИ ЧАТОВ")
        logger.info("="*60)
        
        # Новые чаты
        if result['new_chats']:
            logger.info(f"\n💬 НОВЫЕ ЧАТЫ ({len(result['new_chats'])}):")
            for i, chat in enumerate(result['new_chats'], 1):
                logger.info(f"\n{i}. {chat['title']}")
                logger.info(f"   Время: {chat.get('time', 'Неизвестно')}")
                logger.info(f"   Сообщение: {chat.get('last_message', 'Нет текста')[:100]}...")
                logger.info(f"   URL: {chat.get('url', 'N/A')}")
        else:
            logger.info("\n💬 Новых чатов не найдено")
        
        # Отказы
        if result['rejected_chats']:
            logger.info(f"\n❌ ОТКАЗЫ ({len(result['rejected_chats'])}):")
            for i, chat in enumerate(result['rejected_chats'], 1):
                logger.info(f"\n{i}. {chat['title']}")
                logger.info(f"   Время: {chat.get('time', 'Неизвестно')}")
                logger.info(f"   URL: {chat.get('url', 'N/A')}")
        else:
            logger.info("\n❌ Отказов не найдено")
        
        # Ошибки
        if result['errors']:
            logger.info(f"\n⚠️ ОШИБКИ ({len(result['errors'])}):")
            for error in result['errors']:
                logger.error(f"  - {error}")
        
        logger.info("\n" + "="*60)
        
        # 2. Предложить удалить отказы
        if result['rejected_chats']:
            answer = input("\n🗑️ Удалить отказы? (y/n): ").strip().lower()
            if answer == 'y':
                deleted = await monitor.delete_rejected_chats(result['rejected_chats'])
                logger.info(f"\n✅ Удалено чатов: {deleted}/{len(result['rejected_chats'])}")
            else:
                logger.info("Удаление отменено")

    except Exception as e:
        logger.error(f"❌ Ошибка теста чатов: {e}")
        raise
    finally:
        await worker.close()
        logger.info("Браузер закрыт")


if __name__ == "__main__":
    # Ручная проверка с живым браузером и профилем hh.ru; в pytest не запускается (см. pytest.ini)
    asyncio.run(test_chat_monitoring())
