"""
Сбор ВСЕХ вакансий с hh.ru без ИИ и без Telegram.

Отдельный, полностью самостоятельный скрипт: не трогает основной конвейер
(нет вызовов Ollama, нет отправки в Telegram, нет записи в hh_agent.db).
Использует те же фильтры поиска, что заданы в .env (SEARCH_QUERY, SEARCH_AREA,
SEARCH_WORK_FORMAT и т.д.), но с более широким окном свежести (по умолчанию
30 дней вместо 3 — можно поменять ниже константой FRESHNESS_DAYS или флагом --days).

Каждый запуск создаёт НОВЫЙ CSV-файл рядом со скриптом:
    vacancies_collected_<дата_время>.csv
со столбцами: vacancy_id, title, company, url (без текста описания вакансии).

Вакансии, на которые уже отправлен отклик (hh.ru -> "Отклики и приглашения"),
в файл не попадают.

ВАЖНО: использует тот же браузерный профиль, что и основной бот hh.ru
(./browser_profile), чтобы видеть, на что вы уже откликались. Chromium не
даёт открыть один и тот же профиль из двух процессов одновременно — если
"Запустить бота HH.bat" сейчас запущен, закройте его перед запуском этого
скрипта (Habr-боту и его отдельному профилю это не мешает).
"""
import argparse
import asyncio
import csv
import os
from datetime import datetime

from browser.worker import BrowserWorker

# Было 3 дня в основном боте — тут сознательно шире, т.к. это разовый
# широкий сбор без ИИ-фильтра релевантности.
FRESHNESS_DAYS = 30
MAX_PAGES = 40  # потолок на число страниц выдачи (по 100 вакансий each) — защита от зависания

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


async def get_responded_vacancy_ids(worker: BrowserWorker) -> set:
    """ID вакансий, на которые уже отправлен отклик (hh.ru -> Отклики и приглашения)."""
    responded: set[str] = set()
    page_num = 0
    while True:
        url = "https://hh.ru/applicant/negotiations" + (f"?page={page_num}" if page_num else "")
        try:
            await worker.page.goto(url, wait_until="domcontentloaded")
            await worker.human_delay(1.5, 3.0)
        except Exception as e:
            print(f"⚠️ Не удалось открыть список откликов: {e}")
            break

        if "login" in worker.page.url.lower():
            print("⚠️ Похоже, в этом браузерном профиле нет входа в hh.ru — пропускаю "
                  "проверку уже отправленных откликов (войдите в аккаунт в этом же "
                  "профиле браузера и запустите скрипт снова, если нужна эта проверка).")
            break

        links = await worker.page.locator('a[href*="/vacancy/"]').all()
        found_on_page = 0
        for link in links:
            try:
                href = await link.get_attribute("href")
            except Exception:
                continue
            if not href or "/vacancy/" not in href:
                continue
            vacancy_id = href.split("/vacancy/")[-1].split("?")[0]
            if vacancy_id not in responded:
                responded.add(vacancy_id)
                found_on_page += 1

        if found_on_page == 0:
            break
        page_num += 1
        if page_num > 50:  # разумный предел на случай сотен откликов
            break

    return responded


async def _goto_with_captcha_check(worker: BrowserWorker, url: str):
    """Открывает URL; если hh.ru подсунул капчу/проверку — ждёт, пока пользователь пройдёт её вручную."""
    await worker.page.goto(url, wait_until="domcontentloaded")
    await worker.human_delay(2.0, 4.0)
    if "captcha" in worker.page.url.lower() or "check" in worker.page.url.lower():
        print("⚠️ Обнаружена капча! Пройдите её вручную в открывшемся окне браузера "
              "и нажмите Enter здесь...")
        input()
        await worker.page.goto(url, wait_until="domcontentloaded")
        await worker.human_delay(2.0, 4.0)


async def collect_all_vacancies(worker: BrowserWorker, freshness_days: int, max_pages: int) -> dict:
    """Обходит выдачу hh.ru СТРАНИЦАМИ (hh.ru не подгружает вакансии бесконечным скроллом —
    внизу выдачи обычная нумерация 1, 2, 3..., и именно по ней мы переключаемся: page=0,1,2...
    в ссылке). Останавливаемся, когда очередная страница пришла пустой — это и есть конец выдачи.
    """
    collected: dict[str, dict] = {}

    for page_num in range(max_pages):
        url = worker.build_search_url(page=page_num, freshness_days=freshness_days)
        # По 100 карточек на страницу вместо дефолтных ~20 — меньше переходов между страницами
        if "items_on_page=" not in url:
            url += "&items_on_page=100"

        print(f"Страница {page_num + 1}: {url}")
        await _goto_with_captcha_check(worker, url)

        # Часть карточек на странице подгружается лениво — лёгкий скролл, чтобы они точно отрисовались
        await worker.human_scroll(max_scrolls=3)
        await worker.human_delay(1.0, 2.0)

        cards = await worker.page.locator(
            '[data-qa="serp-item__title"], a[data-qa="serp-item__title"]'
        ).all()

        if not cards:
            print("Страница пустая — это конец выдачи, останавливаюсь.")
            break

        new_on_page = 0
        for card in cards:
            try:
                title = await card.text_content()
                href = await card.get_attribute("href")
                if not title or not href:
                    continue
                if href.startswith("/"):
                    href = f"https://hh.ru{href}"
                if "/vacancy/" not in href:
                    continue

                vacancy_id = href.split("/vacancy/")[-1].split("?")[0]
                if vacancy_id in collected:
                    continue

                # Название компании лежит рядом с заголовком в карточке; если верстка
                # немного другая — просто оставляем пустую строку, это не критично.
                company = ""
                try:
                    container = card.locator(
                        "xpath=ancestor::*[contains(@data-qa,'serp-item') "
                        "or contains(@class,'serp-item') or contains(@class,'vacancy-serp-item')][1]"
                    )
                    company_el = container.locator('[data-qa="vacancy-serp__vacancy-employer"]').first
                    if await company_el.count() > 0:
                        company = (await company_el.text_content() or "").strip()
                except Exception:
                    company = ""

                collected[vacancy_id] = {
                    "vacancy_id": vacancy_id,
                    "title": title.strip(),
                    "company": company,
                    "url": href,
                }
                new_on_page += 1
            except Exception as e:
                print(f"Ошибка парсинга карточки: {e}")
                continue

        print(f"Страница {page_num + 1}: карточек {len(cards)}, новых {new_on_page}, "
              f"всего собрано уникальных {len(collected)}")

        await worker.human_delay(1.5, 3.0)
    else:
        print(f"⚠️ Достигнут потолок в {max_pages} страниц — возможно, выдача ещё не закончилась. "
              f"Запустите с --max-pages побольше, если нужно собрать всё.")

    return collected


async def main(freshness_days: int, max_pages: int):
    output_path = os.path.join(
        SCRIPT_DIR, f"vacancies_collected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    print("=" * 60)
    print("Сбор ВСЕХ вакансий hh.ru без ИИ-анализа и без Telegram")
    print(f"Глубина поиска: {freshness_days} дн.")
    print("=" * 60)

    worker = BrowserWorker(user_data_dir="./browser_profile")
    try:
        await worker.start()
    except Exception as e:
        print(f"\n❌ Не удалось запустить браузер: {e}")
        print("Если сейчас открыт 'Запустить бота HH.bat' — закройте его и попробуйте снова:")
        print("Chromium не даёт открыть один и тот же профиль браузера из двух процессов.")
        return

    try:
        print("\nПроверяю список уже отправленных откликов на hh.ru...")
        responded_ids = await get_responded_vacancy_ids(worker)
        print(f"Уже откликались на {len(responded_ids)} вакансий — исключу их из файла.")

        print("\nЗапускаю поиск вакансий по фильтрам из .env...")
        collected = await collect_all_vacancies(worker, freshness_days=freshness_days, max_pages=max_pages)

        rows = []
        skipped = 0
        for vacancy_id, data in collected.items():
            if vacancy_id in responded_ids:
                skipped += 1
                continue
            rows.append(data)

        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["vacancy_id", "title", "company", "url"])
            writer.writeheader()
            writer.writerows(rows)

        print("\n" + "=" * 60)
        print(f"✅ Готово!")
        print(f"Всего найдено вакансий: {len(collected)}")
        print(f"Исключено (уже откликались): {skipped}")
        print(f"Записано в файл: {len(rows)}")
        print(f"Файл: {output_path}")
        print("=" * 60)
    finally:
        await worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сбор всех вакансий hh.ru без ИИ и без Telegram")
    parser.add_argument("--days", type=int, default=FRESHNESS_DAYS,
                         help=f"Глубина поиска в днях (по умолчанию {FRESHNESS_DAYS}, 0 = без ограничения)")
    parser.add_argument("--max-pages", type=int, default=MAX_PAGES,
                         help=f"Максимум страниц выдачи, по 100 вакансий на странице (по умолчанию {MAX_PAGES})")
    args = parser.parse_args()

    asyncio.run(main(freshness_days=args.days, max_pages=args.max_pages))
