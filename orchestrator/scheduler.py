import os
import asyncio
import random
import logging
from datetime import datetime
from config import settings
from database import (
    init_db, Vacancy, VacancyStatus, save_vacancy, update_vacancy_status,
    is_vacancy_known, get_vacancy_by_id
)
from browser.worker import BrowserWorker
from browser.habr import HabrScraper, is_habr_id, describe_search as describe_habr_search
from llm.brain import evaluate_and_draft, check_ollama_health
from notifier.telegram import TelegramNotifier

# Какой источник обслуживает этот процесс: "hh", "habr" или "all" (оба, как раньше).
# Выставляется в main_hh.py / main_habr.py через переменную окружения ДО импорта этого модуля.
AGENT_SOURCE = os.environ.get("AGENT_SOURCE", "all").strip().lower()
if AGENT_SOURCE not in ("hh", "habr", "all"):
    AGENT_SOURCE = "all"

# У каждого источника свой лог, чтобы два окна не писали в один файл
LOG_FILE = "habr_agent.log" if AGENT_SOURCE == "habr" else "hh_agent.log"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('Orchestrator')

class Orchestrator:
    def __init__(self, source: str = None):
        logger.info("=" * 60)
        logger.info("Инициализация Orchestrator...")
        self.source = source or AGENT_SOURCE
        self.use_hh = self.source in ("hh", "all")
        self.use_habr = self.source in ("habr", "all") and settings.habr_enabled
        logger.info(f"Источник вакансий: {self.source} (hh.ru: {self.use_hh}, Habr: {self.use_habr})")
        self.is_running = True
        self.notifier = None
        self.worker = None
        self.habr = None
        # None = ещё не проверяли, True/False = последнее известное состояние Ollama.
        # Нужно, чтобы уведомлять в Telegram только при СМЕНЕ состояния, а не на каждом цикле.
        self._ollama_alive = None
        # Агент на LangGraph (граф с паузой на решение в Telegram). None — старый линейный режим.
        self.agent = None
        self._close_agent = lambda: None
        logger.info("Orchestrator инициализирован")

    async def check_ollama_and_alert(self) -> bool:
        """
        Проверяет, отвечает ли Ollama, и шлёт уведомление в Telegram при смене состояния
        (была жива -> упала, или была недоступна -> снова ожила). Без этого сбой Ollama
        можно не замечать часами: оркестратор просто оставляет вакансии в статусе NEW
        и молча повторяет попытку в каждом цикле.
        """
        alive, err = await check_ollama_health()
        if alive and self._ollama_alive is False:
            logger.info("✅ Ollama снова доступна. Оценка вакансий возобновлена.")
            if self.notifier:
                await self.notifier.send_alert("✅ Ollama снова доступна. Оценка вакансий возобновлена.")
        elif not alive and self._ollama_alive is not False:
            logger.error(
                f"❌ Ollama недоступна по адресу {settings.ollama_host} ({err}). "
                f"Вакансии будут копиться со статусом NEW, пока Ollama не запустится."
            )
            if self.notifier:
                await self.notifier.send_alert(
                    "⚠️ <b>Ollama недоступна!</b>\n\n"
                    "Бот не может оценивать вакансии — запустите приложение Ollama на компьютере.\n"
                    f"Адрес: {settings.ollama_host}"
                )
        self._ollama_alive = alive
        return alive

    def shutdown(self):
        logger.info("Получен сигнал остановки. Начинаю корректное завершение...")
        self.is_running = False
        logger.info("Флаг is_running установлен в False")

    def is_working_hours(self) -> bool:
        now = datetime.now()
        # Используем настройки из .env, но с небольшим случайным разбросом для естественности
        start_hour = settings.working_hours_start - random.randint(0, 1)
        end_hour = settings.working_hours_end + random.randint(0, 1)
        is_working = start_hour <= now.hour <= end_hour
        logger.info(f"Проверка рабочего времени: {now.strftime('%H:%M')} (рабочие часы: {start_hour}:00-{end_hour}:00) -> {'ДА' if is_working else 'НЕТ'}")
        return is_working

    async def do_idle_action(self):
        """Холостое действие для маскировки"""
        logger.info("🎭 Начало холостого действия для маскировки...")
        try:
            logger.info("Переход на главную страницу hh.ru...")
            await self.worker.page.goto("https://hh.ru/", wait_until="domcontentloaded")
            logger.info("Страница загружена. Имитация скролла...")
            await self.worker.human_delay(2.0, 4.0)
            await self.worker.human_scroll(max_scrolls=3)
            logger.info("✅ Холостое действие завершено успешно")
        except Exception as e:
            logger.error(f"⚠️ Ошибка холостого действия: {e}")

    async def process_vacancy(self, vacancy_data: dict):
        """Обработка одной вакансии: парсинг + оценка + отправка в Telegram"""
        vacancy_id = vacancy_data['vacancy_id']
        url = vacancy_data['url']
        title = vacancy_data['title']
        
        logger.info(f"🔍 Начало обработки вакансии: {title} (ID: {vacancy_id})")
        
        if await is_vacancy_known(vacancy_id):
            logger.info(f"⏭️ Вакансия {vacancy_id} уже известна, пропускаем")
            return
        
        logger.info(f"Вакансия {vacancy_id} новая. Извлекаю данные...")
        if is_habr_id(vacancy_id):
            detailed_data = await self.habr.extract_vacancy_data(url)
        else:
            detailed_data = await self.worker.extract_vacancy_data(url)
            detailed_data['source'] = 'hh.ru'
        detailed_data['url'] = url
        detailed_data['vacancy_id'] = vacancy_id
        logger.info(f"Данные извлечены: {detailed_data['title']} в {detailed_data['company']}")
        
        # Строка со статусом NEW может остаться от прерванной обработки (см. is_vacancy_known) —
        # тогда повторно не вставляем (vacancy_id уникален), а просто доводим её до конца
        if await get_vacancy_by_id(vacancy_id) is None:
            logger.info(f"Сохранение вакансии {vacancy_id} в базу данных...")
            await save_vacancy(Vacancy(
                vacancy_id=vacancy_id,
                title=detailed_data['title'],
                company=detailed_data['company'],
                url=url,
                description_snippet=detailed_data['description'][:200] + "...",
                status=VacancyStatus.NEW
            ))
        else:
            logger.info(f"Вакансия {vacancy_id} уже есть в базе со статусом NEW (прошлая обработка не завершилась), оцениваю заново")

        if self.agent is not None:
            # Граф: оценка → (пропуск | черновик в Telegram → пауза до решения человека)
            logger.info(f"Вакансия {vacancy_id} передана агенту...")
            outcome = await self.agent.start(detailed_data)
            labels = {"waiting": "черновик в Telegram, ждёт решения", "skipped_low_score": "пропущена (низкая оценка)",
                      "error": "LLM не ответила, оставляю статус NEW и повторю в следующем цикле"}
            logger.info(f"Вакансия {vacancy_id}: {labels.get(outcome, outcome)}")
            if outcome == "error":
                return
        else:
            await self._process_linear(vacancy_id, detailed_data)

        pause = random.uniform(5.0, 15.0)
        logger.info(f"Пауза {pause:.1f} сек перед следующей вакансией...")
        await asyncio.sleep(pause)

    async def _process_linear(self, vacancy_id: str, detailed_data: dict):
        """Старый режим без агента: оценка → уведомление или пропуск."""
        logger.info(f"Отправка вакансии {vacancy_id} на оценку LLM...")
        llm_result = await evaluate_and_draft(detailed_data)
        if llm_result.get("error"):
            # Сбой модели: оставляем статус NEW — вакансия будет обработана в следующем цикле
            logger.warning(f"⚠️ LLM не ответила по вакансии {vacancy_id}. Оставляю статус NEW, повторю в следующем цикле")
            return
        logger.info(f"LLM оценка: {llm_result['relevance_score']}/100")

        if llm_result['relevance_score'] >= settings.min_relevance_score:
            logger.info(f"Оценка {llm_result['relevance_score']} >= порога {settings.min_relevance_score}. Отправляю в Telegram...")
            await self.notifier.send_for_approval(vacancy_id, detailed_data, llm_result)
            await update_vacancy_status(
                vacancy_id=vacancy_id,
                status=VacancyStatus.NOTIFIED,
                relevance_score=llm_result['relevance_score'],
                cover_letter_draft=llm_result['cover_letter']
            )
            logger.info(f"✅ Вакансия {vacancy_id} отправлена в Telegram. Статус: NOTIFIED")
        else:
            logger.info(f"Оценка {llm_result['relevance_score']} < порога {settings.min_relevance_score}. Пропускаем.")
            await update_vacancy_status(vacancy_id=vacancy_id, status=VacancyStatus.SKIPPED)
            logger.info(f"Вакансия {vacancy_id} пропущена. Статус: SKIPPED")

    async def search_cycle(self):
        """Один цикл поиска новых вакансий"""
        logger.info("=" * 60)
        logger.info(f"🔄 Начало цикла поиска. Время: {datetime.now().strftime('%H:%M:%S')}")
        logger.info("=" * 60)

        if not self.is_working_hours():
            logger.info("🌙 Сейчас не рабочее время. Пропуск цикла поиска.")
            return

        await self.check_ollama_and_alert()

        try:
            if self.use_hh and random.random() < 0.3:
                logger.info("Случайный выбор: выполняю холостое действие (30% шанс)...")
                await self.do_idle_action()
            else:
                logger.info("Случайный выбор: пропускаю холостое действие (70% шанс)")

            logger.info("🚀 Запуск автоматического поиска вакансий...")
            vacancies = []

            # Каждый источник в своём try: сбой одного (капча, смена вёрстки) не должен убивать другой
            if self.use_hh:
                try:
                    hh_vacancies = await self.worker.search_vacancies()
                    logger.info(f"hh.ru: найдено {len(hh_vacancies)} новых вакансий")
                    vacancies.extend(hh_vacancies)
                except Exception as e:
                    logger.error(f"❌ Ошибка поиска на hh.ru: {e}", exc_info=True)

            if self.use_habr and self.habr is not None and self.is_running:
                try:
                    habr_vacancies = await self.habr.search_vacancies()
                    logger.info(f"Habr Career: найдено {len(habr_vacancies)} новых вакансий")
                    vacancies.extend(habr_vacancies)
                except Exception as e:
                    logger.error(f"❌ Ошибка поиска на Habr Career: {e}", exc_info=True)

            logger.info(f"Поиск завершен. Найдено {len(vacancies)} вакансий")
            
            if not vacancies:
                logger.info("😴 Свежих вакансий не найдено.")
            else:
                logger.info(f"Начинаю обработку {len(vacancies)} вакансий...")
                random.shuffle(vacancies)
                
                for i, vacancy_data in enumerate(vacancies, 1):
                    if not self.is_running:
                        logger.info("Получен сигнал остановки. Прерываю обработку вакансий.")
                        break
                    logger.info(f"Обработка вакансии {i}/{len(vacancies)}...")
                    await self.process_vacancy(vacancy_data)

        except Exception as e:
            logger.error(f"❌ Ошибка в цикле поиска: {e}", exc_info=True)

    async def start(self):
        logger.info("=" * 60)
        logger.info("🚀 Запуск агента...")
        logger.info("=" * 60)
        logger.info(f"Параметры поиска:")
        logger.info(f"  - Запрос: {settings.search_query}")
        logger.info(f"  - Регион: {settings.search_area}")
        logger.info(f"  - hh.ru: {'ВКЛ' if self.use_hh else 'выкл'}")
        if self.use_habr:
            logger.info(f"  - Habr Career: ВКЛ, {describe_habr_search()}, только удалёнка: {settings.habr_remote_only}")
        else:
            logger.info("  - Habr Career: выкл")
        logger.info(f"  - Рабочие часы: {settings.working_hours_start}:00 - {settings.working_hours_end}:00")
        logger.info(f"  - Макс. вакансий за цикл: {settings.max_vacancies_per_cycle}")
        logger.info(f"  - Порог релевантности: {settings.min_relevance_score}")
        
        logger.info("Инициализация базы данных...")
        await init_db()
        logger.info("База данных инициализирована")
        
        logger.info("Инициализация TelegramNotifier...")
        self.notifier = TelegramNotifier()
        logger.info("TelegramNotifier инициализирован")
        
        logger.info("Инициализация BrowserWorker...")
        # Habr-only процесс получает свой профиль браузера: Chromium не даёт открыть один профиль
        # из двух процессов, а вход в аккаунт Habr для парсинга не нужен.
        profile_dir = "./browser_profile_habr" if self.source == "habr" else "./browser_profile"
        self.worker = BrowserWorker(user_data_dir=profile_dir)
        self.habr = HabrScraper(self.worker)
        logger.info(f"BrowserWorker инициализирован (профиль: {profile_dir})")

        self.notifier.set_browser_worker(self.worker)

        if settings.agent_enabled:
            try:
                from agent.wiring import create_agent
                self.agent, self._close_agent = await create_agent(self.notifier)
                self.notifier.set_agent(self.agent)
                logger.info("Агент LangGraph запущен: под черновиками будут кнопки «Подходит / Переписать / Пропустить»")
            except ImportError as e:
                logger.warning(f"LangGraph не установлен ({e}) — работаю в старом режиме без кнопок")
            except Exception as e:
                logger.error(f"Агент не запустился ({e}) — работаю в старом режиме без кнопок")

        # Long polling у одного токена может вести только один процесс, иначе Telegram отвечает Conflict.
        # Отправлять уведомления можно без polling, поэтому Habr-процесс его не запускает
        # (единственное, что теряется там — ответ на /start).
        notifier_task = None
        if self.source != "habr":
            logger.info("Запуск Telegram-бота в фоновом режиме...")
            notifier_task = asyncio.create_task(self.notifier.run_polling())
            await asyncio.sleep(2)
            logger.info("Telegram-бот запущен")
        else:
            logger.info("Habr-режим: Telegram polling не запускается (только отправка уведомлений)")
        
        logger.info("Запуск браузера...")
        await self.worker.start()
        logger.info("Браузер запущен")

        logger.info("Проверка доступности Ollama...")
        await self.check_ollama_and_alert()

        try:
            logger.info("Вход в основной цикл поиска вакансий...")
            while self.is_running:
                await self.search_cycle()
                
                if not self.is_running:
                    logger.info("Получен сигнал остановки. Выход из основного цикла.")
                    break
                
                # УМНАЯ ПАУЗА: днем чаще, ночью реже
                current_hour = datetime.now().hour
                if 8 <= current_hour <= 22:
                    # Днем и вечером: активный поиск (15-30 минут)
                    sleep_minutes = random.uniform(15.0, 30.0)
                    logger.info("☀️ Дневной режим: короткая пауза")
                else:
                    # Ночью: редкий поиск (60-120 минут), чтобы не грузить систему
                    sleep_minutes = random.uniform(60.0, 120.0)
                    logger.info("🌙 Ночной режим: длинная пауза")
                
                sleep_seconds = sleep_minutes * 60
                next_run = datetime.now().timestamp() + sleep_seconds
                
                logger.info(f"💤 Переход в режим сна до {datetime.fromtimestamp(next_run).strftime('%H:%M:%S')} ({sleep_minutes:.1f} мин)")
                logger.info("(Нажмите Ctrl+C для остановки)")
                
                try:
                    await asyncio.sleep(sleep_seconds)
                except asyncio.CancelledError:
                    logger.info("Получен сигнал отмены во время сна.")
                    break
        finally:
            logger.info("Начало корректного завершения работы...")
            self.is_running = False
            
            logger.info("Остановка Telegram-бота...")
            if notifier_task is not None:
                notifier_task.cancel()
                try:
                    await notifier_task
                except asyncio.CancelledError:
                    pass
            
            try:
                self._close_agent()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии хранилища агента: {e}")

            logger.info("Закрытие браузера...")
            try:
                await self.worker.close()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии браузера: {e}")
            
            logger.info("Закрытие сессии Telegram-бота...")
            try:
                await self.notifier.close()
            except Exception as e:
                logger.warning(f"Ошибка при закрытии Telegram-бота: {e}")
            
            logger.info("=" * 60)
            logger.info("✅ Агент корректно завершил работу")
            logger.info("=" * 60)

if __name__ == "__main__":
    logger.info("Запуск Orchestrator из __main__...")
    orchestrator = Orchestrator()
    try:
        asyncio.run(orchestrator.start())
    except KeyboardInterrupt:
        logger.info("\n🛑 Получен сигнал остановки (Ctrl+C). Завершение работы...")
        orchestrator.shutdown()
    except Exception as e:
        logger.error(f"\n❌ Критическая ошибка: {e}", exc_info=True)