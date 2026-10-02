"""Парсер вакансий Habr Career (https://career.habr.com/vacancies).

Использует ту же страницу браузера (Playwright), что и BrowserWorker для hh.ru,
поэтому отдельный браузер/профиль не нужен. Вход в аккаунт Habr не требуется.

Идентификаторы вакансий хранятся в БД с префиксом "habr_" (habr_1000123456),
чтобы не пересекаться с id hh.ru и не менять схему таблицы Vacancy.
"""
import re
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from config import settings
from database import is_vacancy_known

logger = logging.getLogger('HabrScraper')

HABR_BASE = "https://career.habr.com"
HABR_ID_PREFIX = "habr_"
SOURCE_NAME = "Habr Career"

# Достаём карточки одним вызовом в браузере — так меньше round-trip'ов, чем locator.all()
_CARDS_JS = """
() => Array.from(document.querySelectorAll('.vacancy-card')).map(c => {
    const a = c.querySelector('a.vacancy-card__title-link');
    const t = c.querySelector('.vacancy-card__date time');
    const co = c.querySelector('.vacancy-card__company a.link-comp')
            || c.querySelector('.vacancy-card__company');
    return {
        href: a ? a.getAttribute('href') : null,
        title: a ? a.textContent.trim() : '',
        dt: t ? t.getAttribute('datetime') : null,
        company: co ? co.textContent.trim() : ''
    };
})
"""


def is_habr_id(vacancy_id: str) -> bool:
    return str(vacancy_id).startswith(HABR_ID_PREFIX)


def parse_queries(raw: str) -> list[str]:
    """'Python OR "QA Engineer"' / 'Python, QA' / 'Python|QA' -> ['Python', 'QA Engineer'].

    Habr не понимает hh-синтаксис «A OR B», поэтому запросы гоняем по одному.
    """
    parts = re.split(r'\s+OR\s+|[,;|\n]', raw or "", flags=re.IGNORECASE)
    result = []
    for p in parts:
        p = p.strip().strip('"\'«»').strip()
        if p and p not in result:
            result.append(p)
    return result


def get_specializations() -> list[str]:
    """ID специализаций Habr из HABR_SPECIALIZATIONS ('12,10,13,87' -> ['12','10','13','87']).

    ID видно в ссылке при выборе фильтра «Специализация» на career.habr.com/vacancies (параметр s[]).
    """
    return re.findall(r'\d+', settings.habr_specializations or "")


def get_queries() -> list[str]:
    """Список текстовых запросов; пустая строка = поиск только по фильтру специализаций.

    1) HABR_QUERIES — если задан, идёт каждый запрос отдельно (вместе с фильтром специализаций, если он есть);
    2) иначе, если задан HABR_SPECIALIZATIONS — один поиск только по специализациям, без текста;
    3) иначе — те же должности, что в SEARCH_QUERY для hh (старое поведение).
    """
    queries = parse_queries(settings.habr_queries)
    if queries:
        return queries
    if get_specializations():
        return [""]
    return parse_queries(settings.search_query)


def describe_search() -> str:
    """Короткое описание поиска для лога."""
    specs = get_specializations()
    parts = []
    if specs:
        parts.append(f"специализации s[]={','.join(specs)}")
    queries = [q for q in get_queries() if q]
    if queries:
        parts.append(f"текстовые запросы {queries}")
    return "; ".join(parts) or "без фильтров"


def build_search_url(query: str, page: int = 1) -> str:
    params = []
    for spec_id in get_specializations():
        params.append(("s[]", spec_id))
    if query:
        params.append(("q", query))
    params += [("type", "all"), ("sort", "date")]
    if settings.habr_remote_only:
        params.append(("remote", "true"))
    if page > 1:
        params.append(("page", page))
    return f"{HABR_BASE}/vacancies?{urlencode(params)}"


def parse_card_date(value):
    """ISO-дата из <time datetime="2026-09-17T16:00:18+03:00"> -> aware datetime или None."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone(timedelta(hours=3)))  # Habr отдаёт московское время
    return dt


class HabrScraper:
    def __init__(self, worker):
        # worker — BrowserWorker: берём у него page и human_delay/human_scroll
        self.worker = worker

    @property
    def page(self):
        return self.worker.page

    async def search_vacancies(self) -> list:
        """Возвращает только НОВЫЕ вакансии: [{"vacancy_id","title","url","source"}]."""
        queries = get_queries()
        limit = settings.max_vacancies_per_cycle
        max_pages = max(1, settings.habr_max_pages)
        cutoff = None
        if settings.freshness_days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=settings.freshness_days)

        logger.info(f"🔍 Habr Career: {describe_search()}; лимит новых: {limit}, страниц на запрос: {max_pages}")

        new_vacancies: list[dict] = []
        seen: set[str] = set()

        for query in queries:
            if len(new_vacancies) >= limit:
                break
            for page_no in range(1, max_pages + 1):
                if len(new_vacancies) >= limit:
                    break

                url = build_search_url(query, page_no)
                logger.info(f"URL поиска Habr: {url}")
                try:
                    await self.page.goto(url, wait_until="domcontentloaded")
                    await self.page.wait_for_selector('.vacancy-card', timeout=8000)
                except Exception:
                    logger.info(f"Habr: по запросу «{query or '(фильтр специализаций)'}» на стр. {page_no} карточек нет")
                    break
                await self.worker.human_delay(1.0, 2.5)

                cards = await self.page.evaluate(_CARDS_JS)
                logger.info(f"Habr: карточек на странице: {len(cards)}")
                if not cards:
                    break

                fresh_on_page = 0
                for card in cards:
                    href = card.get("href") or ""
                    m = re.search(r'/vacancies/(\d+)', href)
                    if not m:
                        continue
                    vacancy_id = f"{HABR_ID_PREFIX}{m.group(1)}"
                    if vacancy_id in seen:
                        continue
                    seen.add(vacancy_id)

                    posted = parse_card_date(card.get("dt"))
                    if cutoff and posted and posted < cutoff:
                        continue
                    fresh_on_page += 1

                    title = card.get("title") or "Без названия"
                    if await is_vacancy_known(vacancy_id):
                        logger.info(f"⏭️ Habr-вакансия {vacancy_id} уже известна: {title}")
                        continue

                    logger.info(f"✅ НОВАЯ Habr-вакансия {vacancy_id}: {title}")
                    new_vacancies.append({
                        "vacancy_id": vacancy_id,
                        "title": title,
                        "url": f"{HABR_BASE}/vacancies/{m.group(1)}",
                        "source": SOURCE_NAME,
                    })
                    if len(new_vacancies) >= limit:
                        logger.info(f"🎯 Habr: достигнут лимит {limit} новых вакансий")
                        break

                # Выдача отсортирована по дате: если на странице не осталось свежих — дальше нет смысла
                if cutoff and fresh_on_page == 0:
                    logger.info("Habr: на странице нет свежих вакансий, перехожу к следующему запросу")
                    break

                await self.worker.human_delay(1.5, 3.0)

        logger.info(f"✅ Habr Career: найдено НОВЫХ вакансий: {len(new_vacancies)}")
        return new_vacancies

    async def extract_vacancy_data(self, url: str) -> dict:
        logger.info(f"📖 Habr: извлечение данных: {url}")
        await self.page.goto(url, wait_until="domcontentloaded")
        await self.worker.human_delay(1.5, 3.0)

        async def first_text(selector: str, default: str = "") -> str:
            try:
                loc = self.page.locator(selector).first
                if await loc.count() == 0:
                    return default
                return (await loc.text_content() or "").strip() or default
            except Exception:
                return default

        try:
            title = await first_text('h1.page-title__title, h1', "Неизвестная должность")
            company = await first_text('.company_name a, .company_name', "Неизвестная компания")

            description = ""
            try:
                desc = self.page.locator('.vacancy-description__text').first
                await desc.wait_for(state="visible", timeout=5000)
                # inner_text сохраняет переносы строк между абзацами/пунктами — LLM читает лучше
                description = (await desc.inner_text()).strip()
            except Exception:
                description = await first_text('main', "") or await first_text('body', "")

            description = re.sub(r'\n{3,}', '\n\n', description)[:3000] or "Описание отсутствует"

            await self.worker.human_scroll(max_scrolls=3)
            await self.worker.human_delay(1.5, 3.0)

            return {"title": title, "company": company, "description": description, "source": SOURCE_NAME}
        except Exception as e:
            logger.error(f"❌ Habr: ошибка парсинга страницы: {e}")
            return {"title": "Ошибка парсинга", "company": "N/A", "description": "", "source": SOURCE_NAME}
