"""Анализ рынка: что работодатели хотят видеть в вакансиях по вашему фильтру.

Скрипт:
  1. Проходит по ВСЕМ страницам выдачи hh.ru по текущим фильтрам из .env
     (SEARCH_QUERY, SEARCH_AREA, SEARCH_EXPERIENCE, ...). Уже известные боту вакансии не пропускаются.
  2. Открывает каждую вакансию и сохраняет теги «Ключевые навыки» и полный текст описания
     в таблицу vacancy_raw (hh_agent.db). Повторный запуск продолжает с того места, где остановились.
  3. Выделяет ключевики: по умолчанию их выписывает ИИ (Ollama, та же модель, что оценивает вакансии;
     либо облачный GigaChat — флаг --llm-backend gigachat), который читает ПОЛНЫЙ текст вакансии;
     к ним добавляются теги hh и точный поиск по словарю (страховка: модель иногда пропускает очевидное). Ответы ИИ кэшируются в hh_agent.db
     (таблица vacancy_keywords). Режим --engine rules — только словарь и правила, без ИИ.
  4. Считает, в скольких вакансиях встретился каждый ключевик, печатает топ и пишет CSV.

Запуск (из папки hh_agent, основной бот при этом должен быть остановлен —
у них общий профиль браузера):

    venv\\Scripts\\python.exe analyze_market.py
    venv\\Scripts\\python.exe analyze_market.py --limit 30          # пробный прогон на 30 вакансиях
    venv\\Scripts\\python.exe analyze_market.py --query "QA Engineer" --period 0
    venv\\Scripts\\python.exe analyze_market.py --report-only       # без похода на hh.ru: досчитать ИИ и построить топ
    venv\\Scripts\\python.exe analyze_market.py --engine rules      # без ИИ, только словарь и правила
    venv\\Scripts\\python.exe analyze_market.py --refresh-llm       # заново прогнать вакансии через ИИ
    venv\\Scripts\\python.exe analyze_market.py --llm-backend gigachat --llm-limit 30   # проба GigaChat на 30 вакансиях
    venv\\Scripts\\python.exe analyze_market.py --report-only --llm-backend gigachat    # GigaChat по всем собранным
"""
import argparse
import asyncio
import csv
import logging
import os
import random
import sys

# Работаем из папки скрипта, чтобы БД, профиль браузера и отчёты лежали рядом с ним.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
sys.path.insert(0, BASE_DIR)

from analytics import keywords as kw  # noqa: E402  (не тянет БД и браузер)

logger = logging.getLogger('AnalyzeMarket')


def parse_args():
    p = argparse.ArgumentParser(description="Сбор и подсчёт ключевиков по вакансиям hh.ru")
    p.add_argument("--query", help="Поисковый запрос (по умолчанию SEARCH_QUERY из .env)")
    p.add_argument("--period", type=int, default=None,
                   help="За сколько последних дней брать вакансии; 0 — без ограничения. "
                        "По умолчанию FRESHNESS_DAYS из .env")
    p.add_argument("--max-pages", type=int, default=100, help="Максимум страниц выдачи (по 20 вакансий), по умолчанию 100")
    p.add_argument("--limit", type=int, default=0, help="Обработать не больше N новых вакансий (0 — все)")
    p.add_argument("--refresh", action="store_true", help="Заново собрать вакансии, уже сохранённые ранее")
    p.add_argument("--report-only", action="store_true", help="Не ходить на hh.ru, только посчитать по сохранённым данным")
    p.add_argument("--top", type=int, default=50, help="Сколько строк рейтинга печатать в консоль")
    p.add_argument("--min-df", type=int, default=3, help="Минимум вакансий, в которых ключевик должен встретиться")
    p.add_argument("--engine", choices=["llm", "rules"], default="llm",
                   help="Чем выделять ключевики: llm — ИИ (Ollama) читает полный текст вакансии (по умолчанию); "
                        "rules — словарь и правила, без ИИ")
    p.add_argument("--llm-backend", choices=["ollama", "gigachat"], default="ollama",
                   help="Какой ИИ выделяет ключевики: ollama — локальная модель (по умолчанию); "
                        "gigachat — облачный GigaChat (нужны GIGACHAT_CREDENTIALS в .env и пакет gigachat)")
    p.add_argument("--llm-limit", type=int, default=0,
                   help="Прогнать через ИИ не больше N ещё не обработанных вакансий (0 — все). "
                        "Удобно для пробы и чтобы не расходовать токены GigaChat разом")
    p.add_argument("--refresh-llm", action="store_true",
                   help="Заново прогнать через ИИ вакансии, для которых ключевики уже сохранены")
    p.add_argument("--no-dictionary", action="store_true",
                   help="(только --engine llm) не подстраховывать ИИ поиском по словарю: считать только то, "
                        "что назвал ИИ, и теги hh")
    p.add_argument("--no-phrases", action="store_true", help="(только --engine rules) не искать частые фразы")
    p.add_argument("--out", default="keywords_report.csv", help="Файл отчёта CSV")
    return p.parse_args()


def print_ranking(ranking: list, total: int, top: int):
    print(f"  {'#':>3}  {'ключевик':<46}{'вакансий':>8}{'%':>8}")
    print("  " + "-" * 68)
    for i, (keyword, n) in enumerate(ranking[:top], 1):
        pct = round(100.0 * n / total, 1) if total else 0.0
        print(f"  {i:>3}  {keyword[:45]:<46}{n:>8}{pct:>7}%")


def write_csv(path: str, ranking: list):
    """Простой отчёт: два столбца — ключевик и число вакансий, от большего к меньшему."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:  # utf-8-sig — чтобы Excel не ломал кириллицу
        w = csv.writer(f, delimiter=";")
        w.writerow(["ключевик", "вакансий"])
        for keyword, n in ranking:
            w.writerow([keyword, n])


def write_details_llm_csv(path: str, rows: list):
    """Подробный отчёт для режима ИИ: проценты и откуда ключевик (теги hh / ИИ)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["ключевик", "вакансий", "процент", "в тегах hh", "назвал ИИ", "нашёл словарь"])
        for r in rows:
            w.writerow([r["keyword"], r["vacancies"], r["percent"], r["in_tags"], r["in_llm"], r["in_dict"]])


def write_details_csv(path: str, result: dict):
    """Подробный отчёт: тип ключевика, проценты и откуда он взят (теги / текст)."""
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["тип", "ключевик", "вакансий", "процент", "в тегах", "в тексте"])
        for kind, key in (("навык", "skills"), ("латинский термин", "terms"), ("фраза", "phrases")):
            for r in result[key]:
                w.writerow([kind, r["keyword"], r["vacancies"], r["percent"], r["in_tags"], r["in_text"]])


async def collect(args) -> set:
    """Собрать вакансии с hh.ru. Возвращает ID вакансий текущей выдачи."""
    from database import save_vacancy_raw, get_raw_vacancy_ids
    from analytics.scraper import MarketScraper

    scraper = MarketScraper(user_data_dir="./browser_profile")
    await scraper.start()
    try:
        logger.info("Обхожу выдачу...")
        ids = await scraper.collect_vacancy_ids(
            max_pages=args.max_pages, query=args.query, freshness_days=args.period)
        logger.info(f"Всего вакансий в выдаче: {len(ids)}")
        if len(ids) >= 1900:
            logger.warning("hh.ru отдаёт не больше ~2000 вакансий на запрос. Если вакансий в выдаче больше — "
                           "сузьте фильтр (регион, опыт, период) и запустите по частям.")

        already = await get_raw_vacancy_ids()
        todo = ids if args.refresh else [i for i in ids if i not in already]
        logger.info(f"Уже сохранено ранее: {len(ids) - len(todo)}, к сбору: {len(todo)}")
        if args.limit:
            todo = todo[:args.limit]
            logger.info(f"Ограничение --limit: беру {len(todo)}")

        failed = []
        for n, vid in enumerate(todo, 1):
            data = await scraper.scrape_vacancy(vid)
            if data is None:
                failed.append(vid)
                logger.warning(f"[{n}/{len(todo)}] {vid}: пропущена")
            else:
                await save_vacancy_raw(**data)
                logger.info(f"[{n}/{len(todo)}] {data['title'][:60]} | навыков: {len(data['skills'])}, "
                            f"текст: {len(data['description'])} симв.")
            await scraper.human_delay(2.0, 5.0)
            if n % 25 == 0:  # изредка — длинная пауза, чтобы вести себя как человек
                pause = random.uniform(20, 45)
                logger.info(f"Пауза {pause:.0f} сек...")
                await asyncio.sleep(pause)

        if failed:
            logger.warning(f"Не удалось собрать {len(failed)} вакансий (закрыты, удалены или проверка hh.ru). "
                           f"Запустите скрипт ещё раз — он попробует их снова.")
        return set(ids)
    finally:
        await scraper.close()


async def extract_with_llm(vacancies: list, refresh: bool, gigachat=None, limit: int = 0) -> dict:
    """Прогнать вакансии через ИИ. Уже обработанные берутся из базы (кэш).

    gigachat — готовый GigaChatBackend (None — локальная Ollama). limit — сколько новых вакансий
    обработать за этот запуск (0 — все). Возвращает {vacancy_id: [ключевики]}.
    """
    import time
    from config import settings
    from database import get_llm_keywords, save_llm_keywords
    from analytics.llm_keywords import extract_keywords

    # Ответы GigaChat и Ollama в один отчёт не мешаем: при GigaChat кэш берём только от GigaChat.
    cached = await get_llm_keywords("GigaChat" if gigachat else None)
    todo = [v for v in vacancies if refresh or v["vacancy_id"] not in cached]
    result = {v["vacancy_id"]: cached[v["vacancy_id"]] for v in vacancies
              if v["vacancy_id"] in cached and not refresh}
    label = f"GigaChat, {gigachat.current_model}" if gigachat else settings.ollama_model
    logger.info(f"ИИ ({label}): в кэше {len(result)}, нужно обработать {len(todo)}")
    if limit and len(todo) > limit:
        todo = todo[:limit]
        logger.info(f"Ограничение --llm-limit: обрабатываю {len(todo)}")
    if not todo:
        return result

    started = time.time()
    consecutive_errors = 0
    for n, v in enumerate(todo, 1):
        t0 = time.time()
        try:
            if gigachat:
                keywords = await asyncio.to_thread(
                    extract_keywords, v["title"], v["description"], v["skills"], gigachat, gigachat.current_model)
                model_used = gigachat.used_model
            else:
                keywords = await asyncio.to_thread(
                    extract_keywords, v["title"], v["description"], v["skills"], None, settings.ollama_model)
                model_used = settings.ollama_model
            consecutive_errors = 0
        except Exception as e:
            kind = getattr(e, "kind", None)  # GigaChatFailure
            if kind in ("no_tokens", "auth", "ssl"):
                hints = {"no_tokens": "Токены GigaChat закончились на всех моделях каскада.",
                         "auth": "Ключ GigaChat недействителен или истёк — проверьте GIGACHAT_CREDENTIALS.",
                         "ssl": "Проблема с сертификатом — проверьте GIGACHAT_CA_BUNDLE_FILE."}
                logger.error(f"{hints[kind]} {e.detail}\n"
                             f"Уже обработанное сохранено — после исправления запустите скрипт ещё раз, он продолжит.")
                break
            consecutive_errors += 1
            logger.warning(f"[{n}/{len(todo)}] {v['title'][:50]}: ошибка ИИ: {e}")
            if consecutive_errors >= 3:
                if gigachat:
                    logger.error("GigaChat не отвечает 3 раза подряд. Проверьте интернет и ключ. "
                                 "Уже обработанное сохранено — запустите скрипт ещё раз.")
                else:
                    logger.error("ИИ не отвечает 3 раза подряд. Запущен ли Ollama и скачана ли модель "
                                 f"{settings.ollama_model}? Запустите Ollama и повторите: уже обработанное сохранено.")
                break
            continue

        if not keywords:
            logger.warning(f"[{n}/{len(todo)}] {v['title'][:50]}: ИИ не вернул ключевиков — пропускаю")
            continue

        await save_llm_keywords(v["vacancy_id"], keywords, model_used)
        result[v["vacancy_id"]] = keywords
        avg = (time.time() - started) / n
        eta_min = avg * (len(todo) - n) / 60
        logger.info(f"[{n}/{len(todo)}] {v['title'][:50]} -> {len(keywords)} ключевиков "
                    f"({time.time() - t0:.0f} c, осталось ~{eta_min:.0f} мин)")
    return result


async def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Настройки GigaChat проверяем сразу, до многочасового сбора вакансий, чтобы не узнать об ошибке в конце.
    gigachat = None
    if args.engine == "llm" and args.llm_backend == "gigachat":
        from analytics.llm_gigachat import GigaChatBackend
        try:
            gigachat = GigaChatBackend.from_env(BASE_DIR)
        except ValueError as e:
            sys.exit(f"\nGigaChat: {e}")
        logger.info(f"GigaChat: модели по порядку {', '.join(gigachat.models)}")

    from database import init_db, get_raw_vacancies
    await init_db()

    # Если сбор прервать (Ctrl+C), уже сохранённое не пропадёт: запустите скрипт снова — он продолжит,
    # или добавьте --report-only, чтобы посчитать по тому, что успели собрать.
    ids = None
    if not args.report_only:
        ids = await collect(args)

    n_extra = kw.load_extra_terms(os.path.join(BASE_DIR, "keywords_extra.txt"))
    if n_extra:
        logger.info(f"Подгружено своих терминов из keywords_extra.txt: {n_extra}")

    vacancies = await get_raw_vacancies(ids)
    if not vacancies:
        print("\nНет сохранённых вакансий для анализа.")
        return
    if ids is None:
        logger.info("Отчёт по всем сохранённым вакансиям.")

    if args.engine == "llm":
        try:
            llm_map = await extract_with_llm(vacancies, args.refresh_llm, gigachat, args.llm_limit)
        finally:
            if gigachat:
                gigachat.close()
        result = kw.analyze_llm(vacancies, llm_map, min_df=args.min_df, use_dictionary=not args.no_dictionary)
        total = result["total"]
        if total == 0:
            print("\nИИ не выделил ключевики ни в одной вакансии (см. сообщения выше).")
            return
        if total < len(vacancies):
            logger.warning(f"ИИ обработал {total} из {len(vacancies)} вакансий — отчёт по обработанным. "
                           f"Запустите скрипт ещё раз, чтобы дообработать остальные.")
        ranking = [(r["keyword"], r["vacancies"]) for r in result["rows"]]
        details_writer = lambda path: write_details_llm_csv(path, result["rows"])
        details_note = "проценты, откуда ключевик: теги hh / ИИ / словарь"
    else:
        result = kw.analyze(vacancies, min_df=args.min_df, phrases=not args.no_phrases)
        total = result["total"]
        ranking = kw.merge_ranking(result)
        details_writer = lambda path: write_details_csv(path, result)
        details_note = "тип, проценты, теги/текст"

    print("\n" + "=" * 78)
    ai = "GigaChat" if gigachat else "ИИ"
    how = (ai + " + теги hh" + ("" if args.no_dictionary else " + словарь")) if args.engine == "llm" else "словарь и правила"
    print(f"ТОП КЛЮЧЕВИКОВ ПО {total} ВАКАНСИЯМ ({how})")
    print("=" * 78)
    print_ranking(ranking, total, args.top)

    details_path = os.path.splitext(args.out)[0] + "_details.csv"
    write_csv(args.out, ranking)
    details_writer(details_path)
    print(f"\nОтчёт (ключевик; число вакансий): {os.path.join(BASE_DIR, args.out)}")
    print(f"Подробности ({details_note}): {os.path.join(BASE_DIR, details_path)}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nОстановлено.")
