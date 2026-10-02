import asyncio
import html
import logging
from datetime import datetime, timezone
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandObject
from aiogram.types import ForceReply, InlineKeyboardMarkup, InlineKeyboardButton, LinkPreviewOptions
from config import settings
from database import update_vacancy_status, VacancyStatus

logger = logging.getLogger('TelegramNotifier')

# Сколько ждать ответа на /ask: загрузка моделей + поиск + очередь к Ollama (бот в это время может
# оценивать вакансию) + генерация
ASK_TIMEOUT = 300
# Вопросы старше этого (пришли, пока бот был выключен) пропускаем
ASK_MAX_AGE = 120

class TelegramNotifier:
    def __init__(self):
        logger.info("Инициализация TelegramNotifier...")
        self.bot = Bot(token=settings.telegram_bot_token)
        self.chat_ids = settings.telegram_chat_ids
        self.chat_id = self.chat_ids[0]  # используется в cmd_start и т.п. как "основной" чат
        self.dp = Dispatcher()
        self.is_running = False
        self.worker = None
        self._ask_lock = asyncio.Lock()  # /ask обрабатываем по одному: модели и Ollama общие
        self.agent = None                # VacancyAgent: кнопки под черновиками возобновляют его граф
        self._awaiting_feedback: dict[int, str] = {}  # chat_id → вакансия, к которой ждём замечание
        self._setup_handlers()
        logger.info(f"TelegramNotifier инициализирован. Chat ID(ы): {self.chat_ids}")

    def set_agent(self, agent):
        """Агент, которому передаются решения по кнопкам «Подходит / Переписать / Пропустить»."""
        self.agent = agent

    def set_browser_worker(self, worker):
        """Устанавливаем ссылку на браузер для парсинга вакансий"""
        self.worker = worker

    def _setup_handlers(self):
        """Настройка обработчиков команд"""
        logger.info("Настройка обработчиков Telegram-бота...")
        
        @self.dp.message(Command("start"))
        async def cmd_start(message: types.Message):
            logger.info(f"Получена команда /start от пользователя {message.from_user.id}")
            await message.answer(
                "🤖 Бот для поиска работы запущен!\n\n"
                "Вы будете получать уведомления о релевантных вакансиях с автоматически сгенерированными сопроводительными письмами.\n\n"
                "Каждое сообщение содержит:\n"
                "• Описание вакансии и оценку релевантности\n"
                "• Готовое сопроводительное письмо — тапните по нему, чтобы скопировать целиком\n"
                "• Кнопку для перехода на вакансию\n\n"
                "Скопируйте письмо и перейдите по ссылке для отклика!"
            )
            logger.info("Ответ на /start отправлен")

        @self.dp.message(Command("ask"))
        async def cmd_ask(message: types.Message, command: CommandObject):
            """/ask вопрос — ответ по базе вакансий (RAG) со ссылками на вакансии."""
            if message.chat.id not in self.chat_ids:
                return  # отвечаем только своим чатам: вопросы гоняют модели на этом компьютере
            # Telegram досылает сообщения, пришедшие, пока бот был выключен: старые вопросы не обрабатываем
            age = (datetime.now(timezone.utc) - message.date).total_seconds()
            if age > ASK_MAX_AGE:
                logger.info(f"/ask пропущен: сообщение отправлено {age:.0f} с назад, пока бот был выключен")
                return
            if self._ask_lock.locked():
                await message.answer("⏳ Ещё отвечаю на предыдущий вопрос — задайте этот через минуту.")
                return
            question = (command.args or "").strip()
            if not question:
                await message.answer("Напишите вопрос после команды, например:\n"
                                     "/ask в каких вакансиях нужен Playwright на TypeScript?")
                return
            logger.info(f"/ask: {question}")
            await message.answer("🔎 Ищу по базе вакансий...")
            async with self._ask_lock:
                await self._answer_question(message, question)

        @self.dp.callback_query(F.data.startswith("ag:"))
        async def on_decision(callback: types.CallbackQuery):
            """Кнопка под черновиком: ag:<a|r|s>:<vacancy_id>."""
            if callback.message is None or callback.message.chat.id not in self.chat_ids:
                await callback.answer()
                return
            _, code, vacancy_id = callback.data.split(":", 2)
            if self.agent is None:
                await callback.answer("Агент не запущен", show_alert=True)
                return
            if code == "r":
                self._awaiting_feedback[callback.message.chat.id] = vacancy_id
                await callback.answer()
                await callback.message.answer(
                    "✏️ Что поправить в письме? Ответьте одним сообщением, например: "
                    "«короче, упомяни опыт с Playwright».",
                    reply_markup=ForceReply(input_field_placeholder="Что поправить в письме"))
                return
            action = {"a": "approve", "s": "skip"}.get(code)
            if action is None:
                await callback.answer()
                return
            await callback.answer("Принято")
            outcome = await self.agent.resume(vacancy_id, action)
            if outcome == "not_waiting":
                await callback.message.answer("По этой вакансии решение уже принято (или она пришла до запуска агента).")
                return
            # Убираем кнопки решения, оставляем ссылку на вакансию
            link_rows = [row for row in (callback.message.reply_markup.inline_keyboard
                                         if callback.message.reply_markup else []) if row and row[0].url]
            try:
                await callback.message.edit_reply_markup(
                    reply_markup=InlineKeyboardMarkup(inline_keyboard=link_rows) if link_rows else None)
            except Exception as e:  # сообщение могло быть изменено или удалено
                logger.warning(f"Не удалось убрать кнопки: {e}")
            note = "✅ Отмечено как подходящая — откликнитесь по ссылке выше." if action == "approve" \
                else "❌ Вакансия пропущена."
            await callback.message.reply(note)

        @self.dp.message(F.text & ~F.text.startswith("/"))
        async def on_text(message: types.Message):
            """Замечание к письму после кнопки «Переписать»."""
            vacancy_id = self._awaiting_feedback.pop(message.chat.id, None)
            if vacancy_id is None or self.agent is None:
                return
            await message.answer("✍️ Переписываю письмо...")
            try:
                outcome = await self.agent.resume(vacancy_id, "rewrite", message.text)
            except Exception as e:
                logger.error(f"Переписать письмо не удалось: {e}")
                await message.answer(f"⚠️ Не получилось переписать: {html.escape(str(e))[:300]}")
                return
            if outcome == "not_waiting":
                await message.answer("По этой вакансии решение уже принято (или она пришла до запуска агента).")

        logger.info("Все обработчики настроены")

    async def _answer_question(self, message, question: str):
        """Поиск по базе вакансий и ответ со ссылками (вызывается из /ask по одному вопросу за раз)."""
        try:
            import os
            os.environ.setdefault("RAG_DEVICE", settings.rag_bot_device)
            from rag.answer import answer
            result = await asyncio.wait_for(answer(question), timeout=ASK_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error(f"/ask: нет ответа за {ASK_TIMEOUT} с")
            await message.answer(f"⚠️ Не уложился в {ASK_TIMEOUT} с: модель занята оценкой вакансий "
                                 "или видеокарта перегружена. Попробуйте ещё раз позже.")
            return
        except Exception as e:
            logger.error(f"/ask не удался: {e}")
            await message.answer(f"⚠️ Не получилось ответить: {html.escape(str(e))[:300]}")
            return
        lines = [html.escape(result.text[:3000]), ""]
        if result.sources:
            lines.append("<b>Источники:</b>")
            for number, hit in enumerate(result.sources, start=1):
                meta = hit.metadata or {}
                title = html.escape(f"{meta.get('title', '')} — {meta.get('company', '')}")
                url = html.escape(meta.get("url", ""), quote=True)
                lines.append(f'[{number}] <a href="{url}">{title}</a>')
        await message.answer("\n".join(lines), parse_mode="HTML",
                             link_preview_options=LinkPreviewOptions(is_disabled=True))

    async def send_for_approval(self, vacancy_id: str, vacancy_data: dict, llm_result: dict, revision: int = 0):
        """Отправка вакансии в Telegram с сопроводительным письмом и кнопками решения.

        revision > 0 — это переписанная по замечанию версия письма."""
        logger.info(f"Отправка вакансии {vacancy_id} в Telegram...")

        url = vacancy_data.get('url', '#')
        cover_letter = llm_result['cover_letter']
        source = vacancy_data.get('source') or ('Habr Career' if str(vacancy_id).startswith('habr_') else 'hh.ru')

        # Одно сообщение: карточка вакансии + письмо код-блоком (тап = копирование)
        header = (f"✏️ <b>Письмо, версия {revision + 1}</b>" if revision
                  else " <b>Новая релевантная вакансия!</b>")
        text = (
            f"{header} <i>({html.escape(source)})</i>\n\n"
            f" <b>{html.escape(str(vacancy_data['title']))}</b>\n"
            f"🏢 {html.escape(str(vacancy_data['company']))}\n\n"
            f"🧠 <b>Оценка релевантности:</b> {llm_result['relevance_score']}/100\n"
            f"💬 <b>Обоснование:</b> {html.escape(str(llm_result['reasoning']))}\n\n"
            f"📝 <b>Сопроводительное письмо</b> "
            f"<i>(тапните по тексту, чтобы скопировать)</i>\n"
            f"<pre><code>{html.escape(cover_letter)}</code></pre>"
        )

        # Кнопка с прямой ссылкой на вакансию
        rows = []
        if self.agent is not None:
            rows.append([
                InlineKeyboardButton(text="✅ Подходит", callback_data=f"ag:a:{vacancy_id}"),
                InlineKeyboardButton(text="✏️ Переписать", callback_data=f"ag:r:{vacancy_id}"),
                InlineKeyboardButton(text="❌ Пропустить", callback_data=f"ag:s:{vacancy_id}"),
            ])
        rows.append([InlineKeyboardButton(text=f"🔗 Перейти к вакансии на {source}", url=url)])
        keyboard = InlineKeyboardMarkup(inline_keyboard=rows)

        for chat_id in self.chat_ids:
            try:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                    link_preview_options=LinkPreviewOptions(is_disabled=False),
                )
                logger.info(f"✅ Вакансия {vacancy_id} отправлена в Telegram (chat_id={chat_id})")
            except Exception as e:
                logger.error(f" Ошибка отправки вакансии {vacancy_id} в Telegram (chat_id={chat_id}): {e}")

    async def send_alert(self, text: str):
        """Служебное уведомление (не привязанное к конкретной вакансии), например о недоступности Ollama."""
        for chat_id in self.chat_ids:
            try:
                await self.bot.send_message(chat_id=chat_id, text=text, parse_mode="HTML")
                logger.info(f"✅ Служебное уведомление отправлено (chat_id={chat_id})")
            except Exception as e:
                logger.error(f"Ошибка отправки служебного уведомления (chat_id={chat_id}): {e}")

    async def run_polling(self):
        """Запуск бота в фоновом режиме"""
        logger.info("🚀 Запуск Telegram-бота в режиме polling...")
        self.is_running = True
        try:
            await self.dp.start_polling(self.bot)
            logger.info("Telegram-бот завершил работу polling")
        except Exception as e:
            logger.error(f"❌ Ошибка в polling Telegram-бота: {e}")
        finally:
            self.is_running = False

    async def stop_polling(self):
        """Остановка бота"""
        logger.info("Остановка Telegram-бота...")
        self.is_running = False
        await self.bot.session.close()
        logger.info("Telegram-бот остановлен")

    async def close(self):
        """Закрытие сессии бота"""
        logger.info("Закрытие сессии Telegram-бота...")
        try:
            await self.bot.session.close()
            logger.info("Сессия Telegram-бота закрыта")
        except Exception as e:
            logger.warning(f"Ошибка при закрытии сессии Telegram-бота: {e}")