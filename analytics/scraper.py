"""Сборщик данных для анализа рынка: обход ВСЕХ страниц выдачи и полный разбор вакансий.

В отличие от BrowserWorker.search_vacancies():
  * не останавливается на max_vacancies_per_cycle и листает выдачу постранично (?page=N);
  * не пропускает вакансии, которые уже есть в таблице vacancies (это нужно боту, а тут
    интересна вся выдача целиком);
  * забирает теги «Ключевые навыки» и полный текст описания (без обрезки до 3000 символов),
    включая вакансии с кастомным оформлением работодателя.
"""
import asyncio
import logging
import re
from typing import List, Optional

from browser.worker import BrowserWorker

logger = logging.getLogger('MarketScraper')

_VACANCY_ID_RE = re.compile(r"/vacancy/(\d+)")

# Ссылки на вакансии на странице выдачи.
_JS_COLLECT_LINKS = """
() => {
  let links = [...document.querySelectorAll('a[data-qa="serp-item__title"]')];
  if (!links.length) {
    links = [...document.querySelectorAll('a[href*="/vacancy/"]')];
  }
  const found = document.querySelector('[data-qa="vacancies-search-header"], h1');
  return {
    hrefs: links.map(a => a.href),
    header: found ? found.innerText.trim().slice(0, 120) : ''
  };
}
"""

# Данные со страницы вакансии. Выполняется в браузере.
_JS_EXTRACT_VACANCY = """
() => {
  const txt = el => (el && el.innerText ? el.innerText : '').trim();
  const q = s => document.querySelector(s);

  const title = txt(q('h1[data-qa="vacancy-title"]')) || txt(q('h1'));
  const company = txt(q('[data-qa="vacancy-company-name"]'))
               || txt(q('a[data-qa="company-name"]'))
               || txt(q('[data-qa="vacancy-company"]'));

  // --- Ключевые навыки (теги-«печеньки») ---
  let skills = [...document.querySelectorAll('[data-qa="skills-element"]')].map(txt).filter(Boolean);
  if (!skills.length) {
    // Запасной путь: найти заголовок «Ключевые навыки» и собрать «листовые» элементы рядом.
    const heading = [...document.querySelectorAll('h1,h2,h3,h4,div,span,p')]
      .find(e => e.children.length === 0 && e.textContent.trim().toLowerCase() === 'ключевые навыки');
    if (heading) {
      let box = heading.parentElement;
      for (let i = 0; i < 3 && box && box.innerText.trim().length <= heading.innerText.trim().length + 2; i++) {
        box = box.parentElement;
      }
      if (box) {
        skills = [...box.querySelectorAll('*')]
          .filter(e => e.children.length === 0 && e !== heading)
          .map(txt)
          .filter(t => t && t.toLowerCase() !== 'ключевые навыки');
      }
    }
  }
  skills = [...new Set(skills)];

  // --- Описание: стандартное и кастомное (брендированное) оформление ---
  const selectors = [
    '[data-qa="vacancy-description"]',
    '.vacancy-branded-user-content',
    '.vacancy-description',
    '.g-user-content',
  ];
  const parts = [];
  for (const s of selectors) {
    for (const el of document.querySelectorAll(s)) {
      const t = txt(el);
      if (t) parts.push({ selector: s, text: t });
    }
  }
  const main = txt(q('main')) || txt(document.body);
  return { title, company, skills, parts, main };
}
"""


class MarketScraper(BrowserWorker):
    """Расширение BrowserWorker для сбора данных по всей выдаче."""

    async def _goto(self, url: str, retries: int = 2) -> bool:
        for attempt in range(retries + 1):
            try:
                await self.page.goto(url, wait_until="domcontentloaded", timeout=45000)
                return True
            except Exception as e:
                logger.warning(f"Не удалось открыть {url} (попытка {attempt + 1}/{retries + 1}): {e}")
                await self.human_delay(3.0, 6.0)
        return False

    async def _pause_if_blocked(self):
        """Если hh.ru показал капчу / проверку — ждём, пока пользователь пройдёт её вручную."""
        url = self.page.url.lower()
        if "captcha" in url or "/account/login" in url:
            logger.warning("⚠️ hh.ru просит пройти проверку или войти в аккаунт.")
            print("\n⚠️  Пройдите проверку (капчу) в открытом окне браузера,")
            print("    дождитесь загрузки страницы и нажмите ENTER здесь.")
            await asyncio.to_thread(input)

    async def collect_vacancy_ids(self, max_pages: int = 100, query: Optional[str] = None,
                                  freshness_days: Optional[int] = None) -> List[str]:
        """Пройти по всем страницам выдачи и вернуть ID всех найденных вакансий.

        hh.ru отдаёт максимум ~2000 результатов на один поисковый запрос (100 страниц по 20),
        поэтому max_pages по умолчанию 100.
        """
        ids: List[str] = []
        seen = set()

        for page_num in range(max_pages):
            url = self.build_search_url(page=page_num, query=query, freshness_days=freshness_days)
            if page_num == 0:
                logger.info(f"URL поиска: {url}")
            if not await self._goto(url):
                logger.error(f"Страница {page_num} не открылась — останавливаю обход выдачи.")
                break
            await self.human_delay(1.5, 3.0)
            await self._pause_if_blocked()

            try:
                await self.page.wait_for_selector('a[href*="/vacancy/"]', timeout=10000)
            except Exception:
                pass  # пустая страница — обработаем ниже

            data = await self.page.evaluate(_JS_COLLECT_LINKS)
            if page_num == 0 and data.get("header"):
                logger.info(f"Заголовок выдачи: {data['header']}")

            new_on_page = 0
            for href in data.get("hrefs", []):
                m = _VACANCY_ID_RE.search(href)
                if not m:
                    continue
                vid = m.group(1)
                if vid not in seen:
                    seen.add(vid)
                    ids.append(vid)
                    new_on_page += 1

            logger.info(f"Страница {page_num + 1}: новых ссылок {new_on_page}, всего собрано {len(ids)}")
            if new_on_page == 0:
                logger.info("Выдача закончилась.")
                break

            await self.human_delay(1.0, 2.5)

        return ids

    async def scrape_vacancy(self, vacancy_id: str) -> Optional[dict]:
        """Открыть вакансию и вернуть dict(title, company, url, skills, description) или None."""
        url = f"https://hh.ru/vacancy/{vacancy_id}"
        if not await self._goto(url):
            return None
        await self._pause_if_blocked()

        try:
            await self.page.wait_for_selector("h1", timeout=15000)
        except Exception:
            logger.warning(f"Вакансия {vacancy_id}: заголовок не появился (закрыта / удалена / проверка).")
            return None
        await self.human_delay(0.8, 1.6)

        data = await self.page.evaluate(_JS_EXTRACT_VACANCY)
        description = self._pick_description(data)

        # Описание может лежать в iframe (часть брендированных страниц).
        if len(description) < 200:
            for frame in self.page.frames[1:]:
                try:
                    body = await frame.evaluate("() => (document.body ? document.body.innerText : '').trim()")
                except Exception:
                    continue
                if len(body) > len(description):
                    description = body

        if not description:
            logger.warning(f"Вакансия {vacancy_id}: описание не найдено.")
            return None

        return {
            "vacancy_id": vacancy_id,
            "title": (data.get("title") or "").strip(),
            "company": (data.get("company") or "").strip(),
            "url": url,
            "skills": data.get("skills") or [],
            "description": description,
        }

    @staticmethod
    def _pick_description(data: dict) -> str:
        """Выбрать текст описания из найденных кандидатов.

        Кастомные страницы могут содержать описание сразу в нескольких блоках (вложенных друг в друга),
        поэтому берём самый длинный из специфичных блоков. Если они пустые — весь <main>.
        """
        parts = [p["text"] for p in data.get("parts", []) if p.get("text")]
        best = max(parts, key=len) if parts else ""
        if len(best) >= 200:
            return best.strip()
        main = (data.get("main") or "").strip()
        return best.strip() if len(best) >= len(main) else main
