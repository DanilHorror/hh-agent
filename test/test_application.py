import sys
import os

# Добавляем родительскую папку (hh_agent) в путь поиска модулей
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from browser.worker import BrowserWorker
from llm.brain import evaluate_and_draft
from database import init_db

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TestApplication')

async def test_vacancy_application(vacancy_url: str):
    """Тест отклика на конкретную вакансию"""
    logger.info("="*60)
    logger.info("🧪 ТЕСТ ОТПРАВКИ ОТКЛИКА")
    logger.info("="*60)
    logger.info(f"URL вакансии: {vacancy_url}")
    
    await init_db()
    
    # ИСПРАВЛЕНИЕ: Принудительно указываем путь к профилю в корневой папке hh_agent
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    profile_dir = os.path.join(root_dir, "browser_profile")
    
    worker = BrowserWorker(user_data_dir=profile_dir)
    
    try:
        await worker.start()
        logger.info("✅ Браузер запущен")
        
        # 1. Извлекаем данные о вакансии
        logger.info("\n📖 Шаг 1: Извлекаем данные о вакансии...")
        vacancy_data = await worker.extract_vacancy_data(vacancy_url)
        logger.info(f"Вакансия: {vacancy_data['title']}")
        logger.info(f"Компания: {vacancy_data['company']}")
        
        # 2. Генерируем сопроводительное письмо через LLM
        logger.info("\n🧠 Шаг 2: Генерируем сопроводительное письмо...")
        llm_result = await evaluate_and_draft(vacancy_data)
        logger.info(f"Оценка релевантности: {llm_result['relevance_score']}/100")
        logger.info(f"Обоснование: {llm_result['reasoning']}")
        logger.info(f"\n📝 Черновик письма:\n{llm_result['cover_letter']}")
        
        # 3. Анализируем форму отклика
        logger.info("\n🔍 Шаг 3: Анализируем форму отклика...")
        form_analysis = await worker.detect_application_form(vacancy_url)
        logger.info(f"Есть вопросы: {form_analysis['has_custom_questions']}")
        logger.info(f"Можно авто-отклик: {form_analysis['can_auto_respond']}")
        if form_analysis['questions']:
            logger.info(f"Количество вопросов: {len(form_analysis['questions'])}")
            for q in form_analysis['questions']:
                logger.info(f"  - {q['text']} (тип: {q['type']})")
        
        # 4. Отправляем отклик
        logger.info("\n📤 Шаг 4: Отправляем отклик...")
        input("\n⚠️  Нажмите Enter для отправки отклика (или Ctrl+C для отмены)...")
        
        result = await worker.send_application(vacancy_url, llm_result['cover_letter'], form_analysis)
        
        logger.info("\n" + "="*60)
        if result['success']:
            logger.info("✅ ОТКЛИК ОТПРАВЛЕН УСПЕШНО!")
        else:
            logger.error(f"❌ ОШИБКА ОТПРАВКИ: {result['reason']}")
            logger.error(f"Статус: {result['status']}")
        logger.info("="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}", exc_info=True)
    finally:
        await worker.close()
        logger.info("✅ Браузер закрыт")

if __name__ == "__main__":
    print("="*60)
    print("🧪 ТЕСТ ОТПРАВКИ ОТКЛИКА НА hh.ru")
    print("="*60)
    print("\nВведите URL вакансии для тестирования:")
    print("Пример: https://hh.ru/vacancy/130817312")
    print("\n(нажмите Ctrl+C для выхода)\n")
    
    while True:
        try:
            url = input("URL: ").strip()
            if url:
                asyncio.run(test_vacancy_application(url))
                print("\n" + "="*60)
                print("Продолжить тестирование? Введите другой URL или Ctrl+C для выхода")
                print("="*60 + "\n")
        except KeyboardInterrupt:
            print("\n👋 Завершение тестирования...")
            break
        except Exception as e:
            print(f"\n❌ Ошибка: {e}\n")