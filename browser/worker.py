import asyncio
import random
import os
import logging
from urllib.parse import urlencode
from playwright.async_api import async_playwright, Page, BrowserContext
from config import settings
from database import is_vacancy_known
from llm.brain import USER_RESUME, generate_short_answer

logger = logging.getLogger('BrowserWorker')

class BrowserWorker:
    def __init__(self, user_data_dir: str = "./browser_profile"):
        self.user_data_dir = os.path.abspath(user_data_dir)
        self.playwright = None
        self.context: BrowserContext = None
        self.page: Page = None

    async def start(self):
        logger.info("Запуск браузера с постоянным профилем...")
        self.playwright = await async_playwright().start()
        self.context = await self.playwright.chromium.launch_persistent_context(
            user_data_dir=self.user_data_dir,
            headless=False,
            viewport={'width': 1920, 'height': 1080},
            locale='ru-RU',
            timezone_id='Europe/Moscow',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            color_scheme='light',
            device_scale_factor=1.0,
        )
        self.page = self.context.pages[0] if self.context.pages else await self.context.new_page()
        logger.info(f"✅ Профиль загружен. Сессия сохранена в: {self.user_data_dir}")

    async def human_delay(self, min_sec: float = 1.5, max_sec: float = 4.0):
        delay = random.uniform(min_sec, max_sec)
        await asyncio.sleep(delay)

    async def human_scroll(self, max_scrolls: int = 5):
        scrolls = random.randint(2, max_scrolls)
        for _ in range(scrolls):
            distance = random.randint(200, 600)
            await self.page.mouse.wheel(0, distance)
            await self.human_delay(0.3, 1.2)

    def build_search_url(self, page: int = 0, query: str = None, freshness_days: int = None) -> str:
        """URL выдачи hh.ru по текущим фильтрам из настроек.

        page — номер страницы выдачи (с 0); query — переопределить поисковый запрос;
        freshness_days — переопределить период (0 = без ограничения по давности).
        """
        days = settings.freshness_days if freshness_days is None else freshness_days
        params = {
            "text": query if query else settings.search_query,
            # Список: если регионов несколько, urlencode(doseq=True) ниже развернёт их
            # в несколько повторяющихся параметров area=... — так понимает hh.ru.
            "area": settings.search_areas,
        }
        if days:
            params["search_period"] = days
        if settings.search_field:
            # Ограничиваем поиск названием вакансии, чтобы не находились вакансии,
            # где искомое слово просто упомянуто где-то в описании (не по теме)
            params["search_field"] = settings.search_field
        if settings.search_experience:
            params["experience"] = settings.search_experience
        if settings.search_employment:
            params["employment"] = settings.search_employment
        if settings.search_schedule:
            params["schedule"] = settings.search_schedule
        if settings.search_work_formats:
            # Новый фильтр hh.ru "Формат работы" (ON_SITE/REMOTE/HYBRID/FIELD_WORK).
            # Список: несколько форматов -> несколько повторяющихся work_format=... (doseq=True)
            params["work_format"] = settings.search_work_formats
        if page > 0:
            params["page"] = page
        return f"https://hh.ru/search/vacancy?{urlencode(params, doseq=True)}"

    async def search_vacancies(self) -> list:
        logger.info(f"🔍 Запуск поиска: {settings.search_query}")
        logger.info(f"Цель: найти {settings.max_vacancies_per_cycle} новых вакансий")

        search_url = self.build_search_url()
        logger.info(f"URL поиска: {search_url}")
        await self.page.goto(search_url, wait_until="domcontentloaded")
        await self.human_delay(2.0, 4.0)
        
        if "captcha" in self.page.url.lower() or "check" in self.page.url.lower():
            logger.warning("⚠️ Обнаружена капча! Пройдите её вручную и нажмите Enter...")
            input()
        
        new_vacancies = []
        all_vacancies = []
        max_scrolls = 10
        scroll_num = 0
        
        logger.info("Начинаю поиск новых вакансий...")
        
        while scroll_num < max_scrolls and len(new_vacancies) < settings.max_vacancies_per_cycle:
            scroll_num += 1
            logger.info(f"Скролл #{scroll_num}... (найдено новых: {len(new_vacancies)}/{settings.max_vacancies_per_cycle})")
            
            vacancy_cards = await self.page.locator(
                '[data-qa="serp-item__title"], .vacancy-serp__item .bloko-column, a[data-qa="serp-item__title"]'
            ).all()
            
            logger.info(f"Найдено карточек на странице: {len(vacancy_cards)}")
            
            for card in vacancy_cards:
                try:
                    title = await card.text_content()
                    href = await card.get_attribute("href")
                    if not title or not href:
                        continue
                    
                    title = title.strip()
                    if href.startswith("/"):
                        href = f"https://hh.ru{href}"
                    if "/vacancy/" not in href:
                        continue
                    
                    vacancy_id = href.split("/vacancy/")[-1].split("?")[0]
                    
                    if any(v['vacancy_id'] == vacancy_id for v in all_vacancies):
                        continue
                    
                    all_vacancies.append({"vacancy_id": vacancy_id, "title": title, "url": href})
                    is_known = await is_vacancy_known(vacancy_id)
                    
                    if is_known:
                        logger.info(f"⏭️ Вакансия {vacancy_id} уже известна: {title}")
                    else:
                        logger.info(f"✅ НОВАЯ вакансия {vacancy_id}: {title}")
                        new_vacancies.append({"vacancy_id": vacancy_id, "title": title, "url": href})
                    
                    if len(new_vacancies) >= settings.max_vacancies_per_cycle:
                        logger.info(f"🎯 Цель достигнута: найдено {len(new_vacancies)} новых вакансий")
                        break
                        
                except Exception as e:
                    logger.error(f"Ошибка парсинга карточки: {e}")
                    continue
            
            if len(new_vacancies) >= settings.max_vacancies_per_cycle:
                break
            
            if len(vacancy_cards) == 0:
                logger.info("Больше нет результатов для загрузки")
                break
            
            await self.human_scroll(max_scrolls=2)
            await self.human_delay(1.5, 3.0)
        
        logger.info("=" * 60)
        logger.info(f"✅ Поиск завершен!")
        logger.info(f"Всего просмотрено вакансий: {len(all_vacancies)}")
        logger.info(f"Найдено НОВЫХ вакансий: {len(new_vacancies)}")
        logger.info(f"Известных вакансий пропущено: {len(all_vacancies) - len(new_vacancies)}")
        logger.info("=" * 60)
        
        return new_vacancies

    async def extract_vacancy_data(self, url: str) -> dict:
        logger.info(f"📖 Извлечение данных: {url}")
        await self.page.goto(url, wait_until="domcontentloaded")
        await self.human_delay(1.5, 3.0)

        if "captcha" in self.page.url.lower() or "check" in self.page.url.lower() or "security" in self.page.url.lower():
            logger.warning("⚠️ Обнаружена страница проверки безопасности hh.ru!")
            logger.info("👉 Пройдите проверку вручную, затем нажмите Enter в консоли...")
            input()
            logger.info("Продолжаем работу...")

        try:
            title = "Неизвестная должность"
            try:
                title = await self.page.locator('h1[data-qa="vacancy-title"], h1.vacancy-title, h1').first.text_content()
            except Exception:
                pass

            company = "Неизвестная компания"
            try:
                company = await self.page.locator('[data-qa="vacancy-company-name"], .vacancy-company-name, a[data-qa="company-name"]').first.text_content()
            except Exception:
                pass

            description = "Описание отсутствует"
            try:
                desc_locator = self.page.locator('[data-qa="vacancy-description"], .g-user-content, .vacancy-description').first
                await desc_locator.wait_for(state="visible", timeout=5000)
                description = await desc_locator.text_content()
            except Exception:
                try:
                    description = await self.page.locator('main').text_content() or await self.page.locator('body').text_content()
                except Exception:
                    pass

            description = description.strip()[:3000] if description else "Описание отсутствует"
            await self.human_scroll(max_scrolls=3)
            await self.human_delay(2.0, 4.0)
            
            return {
                "title": title.strip() if title else "Неизвестная должность",
                "company": company.strip() if company else "Неизвестная компания",
                "description": description
            }
        except Exception as e:
            logger.error(f"❌ Ошибка парсинга страницы: {e}")
            screenshot_path = "debug_error.png"
            await self.page.screenshot(path=screenshot_path)
            logger.info(f"📸 Скриншот сохранен в {screenshot_path}.")
            return {"title": "Ошибка парсинга", "company": "N/A", "description": ""}

    async def detect_application_form(self, url: str) -> dict:
        logger.info(f"🔍 Анализ формы отклика: {url}")
        await self.page.goto(url, wait_until="domcontentloaded")
        await self.human_delay(2.0, 4.0)
        
        try:
            already_applied = await self.page.get_by_text("Вы уже откликнулись").count()
            if already_applied > 0:
                logger.warning("⚠️ Вы уже откликнулись на эту вакансию ранее.")
                return {
                    "has_custom_questions": False,
                    "questions": [],
                    "has_file_upload": False,
                    "can_auto_respond": False,
                    "reason_to_skip": "Вы уже откликнулись на эту вакансию",
                    "form_is_open": False
                }

            respond_button = self.page.locator(
                '[data-qa="vacancy-response-button"], '
                'button:has-text("Откликнуться"), '
                'a:has-text("Откликнуться"), '
                '.vacancy-response-button'
            ).first
            
            try:
                await respond_button.wait_for(state="visible", timeout=10000)
                logger.info("Шаг 1: Кнопка 'Откликнуться' найдена. Кликаем...")
                await respond_button.click()
                await self.human_delay(2.0, 3.0)
            except Exception as e:
                logger.error(f"❌ Не удалось найти или нажать кнопку 'Откликнуться' за 10 секунд.")
                await self.page.screenshot(path="debug_no_apply_button.png")
                logger.info("📸 Скриншот сохранен: debug_no_apply_button.png")
                return {
                    "has_custom_questions": False, "questions": [], "has_file_upload": False,
                    "can_auto_respond": False, "reason_to_skip": "Кнопка отклика не найдена",
                    "form_is_open": False
                }
            
            # Считаем поля РАЗНЫМИ способами (для разных типов форм)
            text_fields = await self.page.locator('input[type="text"], input[type="number"]').count()
            radio_fields = await self.page.locator('input[type="radio"]').count()
            file_fields = await self.page.locator('input[type="file"]').count()
            textareas = await self.page.locator('textarea').count()
            
            # Для компактных форм hh.ru также ищем по классам
            if textareas == 0:
                textareas = await self.page.locator('[class*="textarea"], [data-qa*="textarea"]').count()
            
            logger.info(f"📊 Найдено полей: text={text_fields}, radio={radio_fields}, file={file_fields}, textarea={textareas}")
            
            logger.info("✅ Форма открыта и готова к заполнению")
            
            return {
                "has_custom_questions": (radio_fields > 0 or file_fields > 0),
                "questions": [],
                "has_file_upload": (file_fields > 0),
                "can_auto_respond": (file_fields == 0),
                "form_is_open": True,
                "_field_counts": {
                    "text": text_fields,
                    "radio": radio_fields,
                    "file": file_fields,
                    "textarea": textareas
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа формы: {e}")
            return {"has_custom_questions": False, "questions": [], "has_file_upload": False, "can_auto_respond": True, "form_is_open": False}

    async def send_application(self, url: str, cover_letter: str, form_analysis: dict = None, vacancy_id: str = "") -> dict:
        logger.info(f"📤 ЗАПОЛНЕНИЕ И ОТПРАВКА ОТКЛИКА (форма уже открыта)")
        
        if not form_analysis or not form_analysis.get("form_is_open", False):
            logger.info("⚠️ Форма не открыта, пытаюсь открыть заново...")
            try:
                await self.page.goto(url, wait_until="domcontentloaded")
                await self.human_delay(2.0, 4.0)
                respond_button = self.page.locator('[data-qa="vacancy-response-button"], button:has-text("Откликнуться")').first
                await respond_button.wait_for(state="visible", timeout=10000)
                await respond_button.click()
                await self.human_delay(2.0, 3.0)
            except Exception as e:
                logger.error(f"❌ Не удалось открыть форму: {e}")
                return {"success": False, "status": "CANT_OPEN_FORM", "reason": str(e)}
        
        try:
            external_apply = await self.page.get_by_text("Откликнуться на сайте работодателя").count()
            if external_apply > 0:
                logger.warning("️ Требуется отклик на внешнем сайте.")
                await self.page.keyboard.press("Escape")
                return {"success": False, "status": "EXTERNAL_LINK", "reason": "Отклик на сайте работодателя"}

            if "login" in self.page.url.lower():
                logger.error("❌ Сессия истекла! Требуется вход в аккаунт.")
                return {"success": False, "status": "SESSION_EXPIRED", "reason": "Сессия истекла"}
            
            if form_analysis and not form_analysis.get("can_auto_respond", True):
                await self.page.keyboard.press("Escape")
                return {"success": False, "status": "NEEDS_MANUAL_REVIEW", "reason": "Требуется ручное заполнение"}
            
            # ШАГ 1: Заполняем сопроводительное письмо (УНИВЕРСАЛЬНЫЙ ПОИСК)
            logger.info("Шаг 1: Заполняем сопроводительное письмо...")
            cover_letter_filled = False
            
            try:
                # Метод 1: Ищем все textarea на странице
                all_textareas = await self.page.locator('textarea').all()
                logger.info(f"Найдено textarea (метод 1): {len(all_textareas)}")
                
                for i, textarea in enumerate(all_textareas):
                    try:
                        # Проверяем, не пустое ли уже
                        val = await textarea.input_value()
                        if val and len(val) > 10:
                            logger.info(f"Textarea #{i} уже заполнено, пропускаем")
                            continue
                        
                        # Заполняем
                        await textarea.scroll_into_view_if_needed()
                        await textarea.fill(cover_letter)
                        logger.info(f"✅ Сопроводительное письмо заполнено в textarea #{i}")
                        cover_letter_filled = True
                        break
                    except Exception as e:
                        logger.warning(f"️ Не удалось заполнить textarea #{i}: {e}")
                        continue
                
                # Метод 2: Если не нашли через locator, ищем по placeholder или label
                if not cover_letter_filled:
                    logger.info("Метод 2: Ищем по placeholder/label...")
                    try:
                        # Ищем textarea рядом с текстом "Сопроводительное письмо"
                        cover_letter_area = self.page.locator('textarea[placeholder*="сопровод"], textarea[aria-label*="сопровод"]')
                        count = await cover_letter_area.count()
                        if count > 0:
                            field = cover_letter_area.first
                            await field.scroll_into_view_if_needed()
                            await field.fill(cover_letter)
                            logger.info("✅ Сопроводительное письмо заполнено (метод 2)")
                            cover_letter_filled = True
                    except Exception as e:
                        logger.warning(f"⚠️ Метод 2 не сработал: {e}")
                
                # Метод 3: Ищем по классам hh.ru
                if not cover_letter_filled:
                    logger.info("Метод 3: Ищем по классам hh.ru...")
                    try:
                        hh_textarea = self.page.locator('[class*="textarea"], [data-qa*="cover-letter"]')
                        count = await hh_textarea.count()
                        if count > 0:
                            field = hh_textarea.first
                            await field.scroll_into_view_if_needed()
                            await field.fill(cover_letter)
                            logger.info("✅ Сопроводительное письмо заполнено (метод 3)")
                            cover_letter_filled = True
                    except Exception as e:
                        logger.warning(f"️ Метод 3 не сработал: {e}")
                        
            except Exception as e:
                logger.error(f"❌ Ошибка при заполнении сопроводительного письма: {e}")
            
            if not cover_letter_filled:
                logger.warning("⚠️ СОПРОВОДИТЕЛЬНОЕ ПИСЬМО НЕ ЗАПОЛНЕНО! Это может быть критично.")
                # Делаем скриншот для отладки
                await self.page.screenshot(path="debug_no_cover_letter.png")
                logger.info("📸 Скриншот сохранен: debug_no_cover_letter.png")
            
            # ШАГ 2: Автоматическое заполнение вопросов
            logger.info("Шаг 2: Автоматическое заполнение вопросов...")
            await self._auto_fill_questions()
            
            # ==========================================================
            # 🛑 ПАУЗА ПЕРЕД ОТПРАВКОЙ ДЛЯ ПРОВЕРКИ
            # ==========================================================
            print("\n" + "="*70)
            print(" ПАУЗА: Посмотрите в браузер, как скрипт заполнил поля!")
            print("   Проверьте ответы на вопросы и сопроводительное письмо.")
            print("   Если всё верно, нажмите ENTER в этой консоли для отправки.")
            print("   Если хотите отменить, нажмите Ctrl+C.")
            print("="*70 + "\n")
            input("Нажмите ENTER для отправки отклика (или Ctrl+C для отмены)... ")
            # ==========================================================
            
            # ШАГ 3: Отправляем форму
            logger.info("Шаг 3: Отправляем форму...")
            submit_button = self.page.locator(
                '[data-qa="vacancy-response-popup-form-submit"], '
                'button:has-text("Отправить"), '
                'button:has-text("Откликнуться"),'
                'button[type="submit"]'
            ).first
            
            try:
                await submit_button.wait_for(state="visible", timeout=10000)
                
                is_disabled = await submit_button.get_attribute('disabled')
                if is_disabled is not None:
                    logger.error("❌ Кнопка 'Отправить' заблокирована (disabled).")
                    screenshot_path = "debug_button_disabled.png"
                    await self.page.screenshot(path=screenshot_path)
                    return {"success": False, "status": "BUTTON_DISABLED", "reason": "Кнопка неактивна. Возможно, не заполнено обязательное поле."}
                
                await submit_button.click()
                logger.info("✅ Кнопка 'Отправить' нажата!")
            except Exception as e:
                logger.error(f"❌ Не удалось нажать кнопку отправки: {e}")
                screenshot_path = "debug_respond_error.png"
                await self.page.screenshot(path=screenshot_path)
                return {"success": False, "status": "ERROR", "reason": f"Не удалось нажать кнопку: {str(e)}"}
            
            await self.human_delay(3.0, 5.0)
            
            # ШАГ 4: Проверяем результат
            logger.info("Шаг 4: Проверяем результат...")
            success = await self.page.get_by_text("Отклик отправлен").count() > 0 or \
                      await self.page.get_by_text("Вы откликнулись").count() > 0
            
            if success:
                logger.info("✅ Отклик успешно отправлен!")
                return {"success": True, "status": "SENT"}
            else:
                logger.warning("️ Не удалось подтвердить отправку")
                screenshot_path = "debug_respond_unknown.png"
                await self.page.screenshot(path=screenshot_path)
                return {"success": False, "status": "UNKNOWN", "reason": "Нет подтверждения"}
                
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")
            screenshot_path = "debug_respond_critical.png"
            await self.page.screenshot(path=screenshot_path)
            return {"success": False, "status": "ERROR", "reason": str(e)}

    async def _auto_fill_questions(self):
        logger.info("🤖 Запуск автоматического заполнения вопросов...")
        
        text_fields = await self.page.locator('input[type="text"], input[type="number"]').all()
        logger.info(f"Найдено текстовых полей: {len(text_fields)}")
        
        for i, field in enumerate(text_fields):
            try:
                val = await field.input_value()
                if val and len(val) > 50:
                    continue
                
                question_text = ""
                try:
                    parent = field.locator('..')
                    label_elem = parent.locator('label').first
                    question_text = await label_elem.text_content(timeout=1500)
                    question_text = question_text.strip().lower() if question_text else ""
                except:
                    pass
                
                logger.info(f"Поле #{i}: вопрос = '{question_text[:50]}...'")
                await field.scroll_into_view_if_needed()
                
                if "доход" in question_text or "зарплат" in question_text or "уровень дохода" in question_text:
                    await field.fill("150000")
                    logger.info(f"✅ Заполнено как зарплата: 150000")
                elif question_text:
                    answer = await self._generate_answer(question_text)
                    await field.fill(answer)
                    logger.info(f"✅ Заполнено через LLM: {answer[:60]}...")
                else:
                    safe_answer = "Мой опыт и навыки подробно описаны в резюме."
                    await field.fill(safe_answer)
                    logger.info(f"✅ Заполнено универсальным ответом")
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка заполнения текстового поля #{i}: {e}")
        
        radio_inputs = await self.page.locator('input[type="radio"]').all()
        logger.info(f"Найдено radio-кнопок: {len(radio_inputs)}")
        
        radio_groups = {}
        for radio in radio_inputs:
            name = await radio.get_attribute('name')
            if name:
                if name not in radio_groups:
                    radio_groups[name] = []
                radio_groups[name].append(radio)
        
        for name, radios in radio_groups.items():
            try:
                logger.info(f"Обрабатываю группу radio: {name}")
                selected = False
                
                for radio in radios:
                    try:
                        label_text = ""
                        try:
                            val = await radio.get_attribute('value')
                            lbl = self.page.locator(f'label:has(input[name="{name}"][value="{val}"])').first
                            label_text = await lbl.text_content(timeout=1000)
                            label_text = label_text.strip().lower() if label_text else ""
                        except:
                            pass
                        
                        if "нет" in label_text or "свой вариант" in label_text:
                            await radio.click()
                            logger.info(f"✅ Выбран вариант: '{label_text}'")
                            selected = True
                            break
                        elif "да" in label_text or "есть" in label_text or "готов" in label_text:
                            await radio.click()
                            logger.info(f"✅ Выбран вариант: '{label_text}'")
                            selected = True
                            break
                    except:
                        continue
                
                if not selected and radios:
                    await radios[0].click()
                    logger.info("✅ Выбран первый доступный вариант")
                
                await self.human_delay(0.3, 0.8)
                
            except Exception as e:
                logger.warning(f"⚠️ Ошибка заполнения radio группы {name}: {e}")
        
        logger.info("✅ Автоматическое заполнение вопросов завершено")

    async def _generate_answer(self, question: str) -> str:
        try:
            prompt = f"""
Ты — помощник соискателя. Ответь на вопрос работодателя ОЧЕНЬ КРАТКО (1-2 предложения), основываясь на резюме.
Не используй маркдаун, пиши обычным текстом.

Резюме: {USER_RESUME}
Вопрос: {question}
Ответ:
"""
            # Модель из .env (LLM_PROVIDER), а не жёстко заданная qwen2.5:7b; асинхронно — не блокирует бота
            return await generate_short_answer(prompt, fallback="Готов подробно обсудить этот вопрос на собеседовании.")
        except Exception as e:
            logger.error(f" Ошибка LLM: {e}")
            return "Готов подробно обсудить этот вопрос на собеседовании."

    async def close(self):
        logger.info("Закрытие браузера...")
        if self.context:
            await self.context.close()
        if self.playwright:
            await self.playwright.stop()
        logger.info("✅ Сессия корректно завершена и сохранена.")