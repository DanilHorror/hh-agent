"""Извлечение и подсчёт ключевых слов из вакансий.

Модуль не зависит ни от БД, ни от браузера — только чистые функции, поэтому
его можно тестировать отдельно (см. test/test_keywords.py).

Откуда берутся ключевики:
  1. Теги из блока «Ключевые навыки» («печеньки» на странице вакансии).
  2. Словарь известных технологий и терминов (TERMS ниже) — ищется в тексте описания
     по нескольким написаниям («CI/CD», «CICD», «ci-cd» → один ключевик).
  3. Латинские слова, которых нет в словаре (например «Zabbix»), — «неизвестные термины».
  4. Частые фразы из 2–3 слов («автоматизации тестирования», «разработка стратегий
     тестирования») — для требований, которые написаны обычным текстом.

Считаем количество ВАКАНСИЙ, где ключевик встретился хотя бы раз (а не число
упоминаний): вакансия с пятью «Python» даёт вклад 1.
"""
import functools
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Словарь терминов: «канонический ключевик» -> список регулярных выражений.
# Регистр не важен. Границы слова добавляются автоматически.
# Дополнять можно прямо здесь или файлом keywords_extra.txt (см. load_extra_terms).
# ---------------------------------------------------------------------------
TERMS: Dict[str, List[str]] = {
    # --- языки и базы ---
    "python": [r"python", r"питон"],
    "java": [r"java"],
    "javascript": [r"javascript"],
    "typescript": [r"typescript"],
    "kotlin": [r"kotlin"],
    "golang": [r"golang"],
    "c++": [r"c\+\+"],
    "c#": [r"c#", r"c\s?sharp"],
    "bash": [r"bash", r"shell"],
    "powershell": [r"powershell"],
    "sql": [r"sql"],
    "postgresql": [r"postgres(?:ql)?"],
    "mysql": [r"mysql"],
    "oracle": [r"oracle"],
    "mongodb": [r"mongo(?:db)?"],
    "redis": [r"redis"],
    "kafka": [r"kafka"],
    "rabbitmq": [r"rabbit(?:mq)?"],
    "clickhouse": [r"clickhouse"],
    # --- тестирование: инструменты ---
    "pytest": [r"py\.?test"],
    "unittest": [r"unittest"],
    "robot framework": [r"robot\s*framework"],
    "selenium": [r"selenium"],
    "playwright": [r"playwright"],
    "appium": [r"appium"],
    "cypress": [r"cypress"],
    "postman": [r"postman"],
    "soapui": [r"soap\s*ui"],
    "jmeter": [r"jmeter"],
    "locust": [r"locust"],
    "allure": [r"allure"],
    "testng": [r"testng"],
    "junit": [r"junit"],
    "cucumber": [r"cucumber"],
    "bdd": [r"bdd"],
    "rest api": [r"rest(?:ful)?(?:\s*api)?"],
    "api": [r"api"],
    "soap": [r"soap"],
    "graphql": [r"graphql"],
    "jira": [r"jira"],
    "confluence": [r"confluence"],
    "testrail": [r"testrail"],
    "qase": [r"qase"],
    # --- тестирование: практики (по стемам, чтобы ловить падежи) ---
    "автоматизация тестирования": [
        r"автоматизаци\w*\s+тестирован\w*",
        r"автоматизирован\w*\s+тест\w*",
        r"автотест\w*",
        r"автоматизац\w*\s+тест\w*",
        r"автоматическ\w*\s+тест\w*",
        r"test\s*automation",
    ],
    "функциональное тестирование": [r"функциональн\w*(?=[^.\n;]{0,60}?тестирован)"],
    "регрессионное тестирование": [r"регрессионн\w*"],
    "интеграционное тестирование": [r"интеграционн\w*(?=[^.\n;]{0,60}?тестирован)"],
    "нагрузочное тестирование": [r"нагрузочн\w*", r"load\s*testing", r"performance\s*testing"],
    "ручное тестирование": [r"ручн\w*(?=[^.\n;]{0,40}?тестирован)", r"manual\s*testing"],
    "тест-кейсы": [r"тест[\s-]?кейс\w*", r"test[\s-]?case\w*"],
    "тест-планы": [r"тест[\s-]?план\w*", r"test[\s-]?plan\w*"],
    "чек-листы": [r"чек[\s-]?лист\w*", r"check[\s-]?list\w*"],
    "баг-репорты": [r"баг[\s-]?репорт\w*", r"bug[\s-]?report\w*"],
    "тестовая документация": [r"тестов\w*\s+документаци\w*"],
    # --- DevOps ---
    "ci/cd": [r"ci\s*/\s*cd", r"ci[\s-]?cd", r"cicd", r"ci\s+и\s+cd"],
    "docker": [r"docker"],
    "kubernetes": [r"kubernetes", r"k8s"],
    "helm": [r"helm"],
    "terraform": [r"terraform"],
    "ansible": [r"ansible"],
    "jenkins": [r"jenkins"],
    "gitlab ci": [r"gitlab[\s-]*ci"],
    "github actions": [r"github[\s-]*actions"],
    "git": [r"git"],
    "linux": [r"linux", r"линукс"],
    "windows server": [r"windows\s*server"],
    "active directory": [r"active\s*directory"],
    "azure": [r"azure"],
    "aws": [r"aws", r"amazon\s*web\s*services"],
    "gcp": [r"gcp", r"google\s*cloud"],
    "zabbix": [r"zabbix"],
    "grafana": [r"grafana"],
    "prometheus": [r"prometheus"],
    "elk": [r"elk", r"elasticsearch"],
    # --- сети ---
    "tcp/ip": [r"tcp\s*/\s*ip", r"tcp[\s-]?ip"],
    "osi": [r"osi"],
    "bgp": [r"bgp"],
    "ospf": [r"ospf"],
    "is-is": [r"is[\s-]?is"],
    "mpls": [r"mpls"],
    "vxlan": [r"vxlan"],
    "evpn": [r"evpn"],
    "vlan": [r"vlan"],
    "stp": [r"stp"],
    "dns": [r"dns"],
    "dhcp": [r"dhcp"],
    "nat": [r"nat"],
    "vpn": [r"vpn", r"впн"],
    "ipsec": [r"ipsec"],
    "l2/l3": [r"l2\s*/\s*l3", r"l3\s*/\s*l2"],
    "wireshark": [r"wireshark"],
    "tcpdump": [r"tcpdump"],
    "scapy": [r"scapy"],
    "gns3": [r"gns3"],
    "eve-ng": [r"eve[\s-]?ng"],
    "cisco": [r"cisco"],
    "juniper": [r"juniper"],
    "huawei": [r"huawei"],
    "mikrotik": [r"mikrotik"],
    # --- embedded ---
    "embedded": [r"embedded", r"встраиваем\w*"],
    "uart": [r"uart"],
    "i2c": [r"i2c"],
    "spi": [r"spi"],
    "ble": [r"ble"],
    "осциллограф": [r"осциллограф\w*"],
    # --- общее ---
    "ооп": [r"ооп", r"oop"],
    "pep8": [r"pep\s*-?\s*8"],
    "agile/scrum": [r"scrum", r"agile", r"kanban"],
    "английский язык": [r"английск\w*\s+язык\w*", r"английск\w*", r"english"],
}

# Границы «слова» для латиницы, цифр и + # (чтобы "git" не находился в "github", "c" в "c++").
_L = r"(?<![\w+#])"
_R = r"(?![\w+#])"

_COMPILED_SEARCH: Dict[str, "re.Pattern"] = {}
_COMPILED_FULL: Dict[str, "re.Pattern"] = {}


def _compile(canon: str, patterns: List[str]) -> None:
    alt = "|".join(f"(?:{p})" for p in patterns)
    _COMPILED_SEARCH[canon] = re.compile(f"{_L}(?:{alt}){_R}", re.IGNORECASE)
    _COMPILED_FULL[canon] = re.compile(f"(?:{alt})", re.IGNORECASE)


for _canon, _pats in TERMS.items():
    _compile(_canon, _pats)


def load_extra_terms(path: str) -> int:
    """Подгрузить свои термины из текстового файла.

    Формат строки:  ключевик: написание1, написание2      (написания — обычный текст)
    или просто:     ключевик
    Строки с # в начале игнорируются. Возвращает число добавленных терминов.
    """
    import os
    if not os.path.exists(path):
        return 0
    added = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            canon, _, aliases = line.partition(":")
            canon = canon.strip().lower()
            variants = [a.strip() for a in aliases.split(",") if a.strip()] or [canon]
            patterns = [re.escape(v.lower()) for v in variants]
            if canon not in TERMS:
                TERMS[canon] = []
            TERMS[canon].extend(patterns)
            _compile(canon, TERMS[canon])
            added += 1
    return added


# ---------------------------------------------------------------------------
# Теги и словарные термины
# ---------------------------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(" ", " ")).strip()


def normalize_tag(tag: str) -> str:
    """Привести тег из «Ключевых навыков» к каноническому виду."""
    t = _clean(tag).lower()
    for canon, rx in _COMPILED_FULL.items():
        if rx.fullmatch(t):
            return canon
    return t


def find_dictionary_terms(text: str) -> set:
    """Найти в тексте все словарные термины (каждый — один раз)."""
    found = set()
    for canon, rx in _COMPILED_SEARCH.items():
        if rx.search(text):
            found.add(canon)
    return found


def is_known_term(text: str) -> bool:
    t = _clean(text).lower()
    return t in TERMS or any(rx.fullmatch(t) for rx in _COMPILED_FULL.values())


# ---------------------------------------------------------------------------
# Частые фразы и неизвестные латинские термины
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[a-zа-яё][a-zа-яё0-9+#]*(?:-[a-zа-яё0-9+#]+)*", re.IGNORECASE)
# Разделители фраз: перенос строки, знаки препинания, скобки, кавычки, маркеры списков, тире.
_SPLIT_RE = re.compile(r"[\n\r.;:!?()\[\]{}«»\"“”„•·▪●■◦—–,/|\\]+")

# Служебные слова — ломают фразу (по стему/форме слова, нижний регистр).
STOP_WORDS = set("""
и в во на с со по к ко от до из за для о об обо при как что это то а но или не да же бы ли
также еще ещё уже быть будет был была были есть либо чтобы если так там тут где когда
который которая которое которые которых которым том эти этот эта этих
из-за над под про через между без у мы вы ты они он она оно наш ваш свой
the and for with you our are will of to in on a an or be as is at by from that this
""".split())

# Слишком общие слова — фразы с ними не интересны («опыт работы», «знание языка»).
# Пишем обычные слова; сравнение идёт по стему, так что падежи учитываются.
_GENERIC_WORDS = """
опыт работа работать знание знать умение уметь навык уверенный хороший понимание понимать
владение владеть обязанность требование условие задача готовый желательно наличие участие
компания команда проект сотрудник лет год месяц плюс преимущество являться должен нужный
необходимый возможность оформление график зарплата офис полный рабочий деятельность
различный другой любой вакансия кандидат ищем приглашать предлагать тк рф высший
образование средний специалист самостоятельный ответственный коммуникабельный внимательный
быстрый качество уровень весь наш ваш свой себя основной новый коллега заниматься
предстоять видеть чем каким
работодатель связаться свяжется вопрос оплата плата труд карта москва центр собеседование
испытательный реферальный квалификация ипотека скидка мероприятие компенсация трудоустройство
атмосфера корпоративный помощь материальный льготный дружественный формат удаленный жизнь
день делать важно желание развиваться развитие рост профессиональный современный технология
библиотека этап получить получит свяжется место располагаться заработный обеспечение официальный
проведение написание изучение понятный первый дня
""".split()

_EN_STOP = set("""
the and for with you our are will of to in on an or be as is at by from that this we your it its
have has can not but all any more must should etc
engineer engineers skills skill soft middle junior senior lead team tester developer manager
test testing web
""".split())

_RU_ENDINGS = sorted("""
ениями ениям ениях ением ению ении ения ение
ость остью ости остей
ием иям иях ию ией ии ия ие ий
ами ями ого его ому ему ыми ими ую юю ым им
ых их ый ой ая яя ое ее ые ов ев ей ом ем ам ям ах ях ью ть ет ют ут ит ат ят
а я у ю ы и е о ь
""".split(), key=len, reverse=True)


def stem(word: str) -> str:
    """Очень простой стеммер: отрезает типичные русские окончания.

    Нужен только чтобы «тестирования / тестирование / тестированию» считались одним словом.
    Английские слова возвращаются как есть.
    """
    w = word.lower().replace("ё", "е")
    if not re.search(r"[а-я]", w):
        return w
    for end in _RU_ENDINGS:
        if w.endswith(end) and len(w) - len(end) >= 4:
            return w[: -len(end)]
    return w


GENERIC_STEMS = {stem(w) for w in _GENERIC_WORDS} | set(_GENERIC_WORDS)


def _is_generic(word: str) -> bool:
    return stem(word) in GENERIC_STEMS or word.lower() in GENERIC_STEMS


def _is_stop(word: str) -> bool:
    w = word.lower()
    return w in STOP_WORDS or w in _EN_STOP or len(w) < 2


def _segments(text: str) -> List[List[str]]:
    """Разбить текст на «цепочки» подряд идущих значимых слов.

    Служебные и слишком общие слова разрывают цепочку, поэтому фразы не «перепрыгивают»
    через них: «опыт работы с pytest» даёт только слово pytest.
    """
    chains: List[List[str]] = []
    for part in _SPLIT_RE.split(text):
        chain: List[str] = []
        for m in _WORD_RE.finditer(part):
            w = m.group(0).lower()
            if _is_stop(w) or _is_generic(w):
                if chain:
                    chains.append(chain)
                chain = []
            else:
                chain.append(w)
        if chain:
            chains.append(chain)
    return chains


def extract_phrases(text: str, sizes=(2, 3)) -> Tuple[set, Dict[tuple, Counter]]:
    """Вернуть (множество ключей-фраз этой вакансии, поверхностные формы).

    Ключ фразы — кортеж стемов. Для показа запоминаем реальную форму слов.
    """
    keys = set()
    surfaces: Dict[tuple, Counter] = defaultdict(Counter)
    for chain in _segments(text):
        for n in sizes:
            for i in range(len(chain) - n + 1):
                gram = chain[i:i + n]
                key = tuple(stem(w) for w in gram)
                keys.add(key)
                surfaces[key][" ".join(gram)] += 1
    return keys, surfaces


def extract_unknown_latin_terms(text: str) -> set:
    """Латинские слова (в основном названия технологий), которых нет в словаре."""
    found = set()
    for m in _WORD_RE.finditer(text):
        w = m.group(0).lower()
        if not re.fullmatch(r"[a-z][a-z0-9+#.\-]*", w) or len(w) < 3:
            continue
        if w in _EN_STOP or w in STOP_WORDS:
            continue
        if is_known_term(w):
            continue
        found.add(w)
    return found


# ---------------------------------------------------------------------------
# Очистка текста от служебных блоков hh.ru
# ---------------------------------------------------------------------------

# Всё, что идёт после этих подписей — интерфейс сайта, а не текст вакансии
# («Задайте вопрос работодателю», «Какая оплата труда?», «Где располагается место работы» ...).
_BOILERPLATE_MARKERS = [
    "задайте вопрос работодателю", "какая оплата труда", "работодатель получит",
    "где располагается", "вакансия опубликована", "похожие вакансии", "другие вакансии компании",
]


def clean_description(text: str) -> str:
    """Обрезать текст вакансии там, где начинаются служебные блоки hh.ru."""
    low = text.lower()
    cut = len(text)
    for marker in _BOILERPLATE_MARKERS:
        i = low.find(marker)
        if 0 < i < cut:
            cut = i
    return text[:cut]


def is_noise_phrase(phrase: str) -> bool:
    """Фраза состоит из служебных/общих слов (условия работы, интерфейс сайта)."""
    words = [m.group(0).lower() for m in _WORD_RE.finditer(phrase)]
    return not words or any(_is_stop(w) or _is_generic(w) for w in words)


# ---------------------------------------------------------------------------
# Сводный подсчёт
# ---------------------------------------------------------------------------

def analyze(vacancies: Iterable[dict], min_df: int = 3, phrases: bool = True) -> dict:
    """Посчитать ключевики по списку вакансий.

    Каждая вакансия — dict с ключами skills (список тегов) и description (текст).
    Возвращает {"total": N, "skills": [...], "terms": [...], "phrases": [...]}, где элементы —
    dict(keyword, vacancies, percent, in_tags, in_text), отсортированные по убыванию.
    Ключевики, встретившиеся меньше чем в min_df вакансиях, отбрасываются во всех списках.
    skills — теги + словарные термины; terms — неизвестные латинские слова; phrases — фразы.
    """
    vacancies = list(vacancies)
    total = len(vacancies)

    skill_tags: Counter = Counter()     # вакансий, где ключевик есть среди тегов
    skill_text: Counter = Counter()     # вакансий, где ключевик найден в тексте
    skill_any: Counter = Counter()      # где есть хоть где-то

    term_df: Counter = Counter()
    phrase_df: Counter = Counter()
    phrase_surface: Dict[tuple, Counter] = defaultdict(Counter)

    for v in vacancies:
        text = clean_description(v.get("description") or "")
        tags = {normalize_tag(t) for t in (v.get("skills") or []) if _clean(t)}
        in_text = find_dictionary_terms(text)

        for k in tags:
            skill_tags[k] += 1
        for k in in_text:
            skill_text[k] += 1
        for k in tags | in_text:
            skill_any[k] += 1

        # Неизвестные латинские термины — только те, что не попали в теги/словарь.
        for w in extract_unknown_latin_terms(text) - tags:
            term_df[w] += 1

        if phrases:
            keys, surfaces = extract_phrases(text)
            for k in keys:
                phrase_df[k] += 1
            for k, c in surfaces.items():
                phrase_surface[k].update(c)

    def pct(n: int) -> float:
        return round(100.0 * n / total, 1) if total else 0.0

    skills = [
        {"keyword": k, "vacancies": n, "percent": pct(n),
         "in_tags": skill_tags.get(k, 0), "in_text": skill_text.get(k, 0)}
        for k, n in skill_any.items() if n >= min_df
    ]
    skills.sort(key=lambda r: (-r["vacancies"], r["keyword"]))

    terms = [
        {"keyword": k, "vacancies": n, "percent": pct(n), "in_tags": 0, "in_text": n}
        for k, n in term_df.items() if n >= min_df
    ]
    terms.sort(key=lambda r: (-r["vacancies"], r["keyword"]))

    phrase_rows: List[dict] = []
    if phrases:
        # Убираем короткую фразу, если она почти всегда встречается как часть более длинной.
        kept = {k: n for k, n in phrase_df.items() if n >= min_df}
        by_len = defaultdict(list)
        for k in kept:
            by_len[len(k)].append(k)
        for k, n in list(kept.items()):
            for longer in by_len.get(len(k) + 1, []):
                if _contains(longer, k) and kept[longer] >= 0.8 * n:
                    kept.pop(k, None)
                    break
        for k, n in kept.items():
            shown = phrase_surface[k].most_common(1)[0][0]
            if is_known_term(shown):
                continue  # уже учтено как словарный термин
            phrase_rows.append({"keyword": shown, "vacancies": n, "percent": pct(n),
                                "in_tags": 0, "in_text": n})
        phrase_rows.sort(key=lambda r: (-r["vacancies"], r["keyword"]))

    return {"total": total, "skills": skills, "terms": terms, "phrases": phrase_rows}


def _contains(longer: tuple, shorter: tuple) -> bool:
    n = len(shorter)
    return any(longer[i:i + n] == shorter for i in range(len(longer) - n + 1))


def merge_ranking(result: dict, min_df: int = 1) -> List[Tuple[str, int]]:
    """Свести навыки, термины и фразы в один рейтинг: [(ключевик, число вакансий), ...].

    Если один и тот же ключевик найден несколькими способами, берётся наибольшее число.
    Сортировка: от большего к меньшему, при равенстве — по алфавиту.
    """
    best: Dict[str, int] = {}
    for group in ("skills", "terms", "phrases"):
        for row in result.get(group, []):
            key = row["keyword"].strip().lower()
            if row["vacancies"] >= min_df:
                best[key] = max(best.get(key, 0), row["vacancies"])
    return sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))


# ---------------------------------------------------------------------------
# Ключевики, полученные от ИИ (см. analytics/llm_keywords.py)
# ---------------------------------------------------------------------------

# Слова, которые не меняют смысл ключевика, когда он и так узнаётся по словарю:
# «разработка тест-кейсов» = «тест-кейсы», «английский B1 средний» = «английский язык».
_LIGHT_WORDS = set("""
разработка создание составление ведение использование применение подготовка оформление
b1 b2 a1 a2 c1 c2 средний базовый продвинутый свободный intermediate upper advanced
""".split())

# Слишком общие ключевики: встречаются почти везде и ничего не говорят о требованиях.
NOISE_KEYWORDS = {
    "тестирование", "тест", "тесты", "тестировщик", "qa", "qa engineer", "qa-инженер", "quality assurance",
    "testing", "test", "инженер", "инженер по тестированию", "проведение тестирований", "тестирований",
}

# Латинские «двойники» кириллических букв: модель иногда пишет «ooп» (латинская o) вместо «ооп».
_LATIN_TO_CYR = str.maketrans("aceopxyk", "асеорхук")


def _fix_homoglyphs(text: str) -> str:
    def fix(m):
        w = m.group(0)
        if re.search(r"[а-яё]", w, re.IGNORECASE) and re.search(r"[a-z]", w, re.IGNORECASE):
            return w.translate(_LATIN_TO_CYR)
        return w
    return re.sub(r"[A-Za-zА-Яа-яЁё]+", fix, text)


def _prune_nested(hits: list) -> list:
    """Из совпадений словаря убрать те, что целиком лежат внутри более длинного («api» внутри «rest api»)."""
    return [h for h in hits
            if not any(o is not h and o[0] <= h[0] and h[1] <= o[1] and (o[1] - o[0]) > (h[1] - h[0]) for o in hits)]


@functools.lru_cache(maxsize=50000)
def _canonicalize(k: str) -> tuple:
    """Если ключевик по сути и есть словарный термин (с необязательными «лёгкими» словами) —
    вернуть канонические названия; иначе — сам ключевик."""
    hits = []
    for canon, rx in _COMPILED_SEARCH.items():
        for m in rx.finditer(k):
            hits.append((m.start(), m.end(), canon))
    if not hits:
        return (k,)
    hits = _prune_nested(hits)
    for m in _WORD_RE.finditer(k):
        w = m.group(0)
        covered = any(s <= m.start() and m.end() <= e for s, e, _ in hits)
        if not covered and not (_is_stop(w) or _is_generic(w) or w in _LIGHT_WORDS):
            return (k,)
    return tuple(sorted({c for _, _, c in hits}))


def keyword_variants(text: str) -> List[str]:
    """Привести ключевик (от ИИ или тег hh) к каноническому виду.

    Возвращает список (обычно из одного элемента, пустой — если это мусор).
    «разработка тест-кейсов» -> ["тест-кейсы"], «test case» -> ["тест-кейсы"],
    «тестирование API» останется как есть.
    """
    t = _fix_homoglyphs(_clean(text)).strip(" .,;:-–—•*\"'«»()[]").lower()
    if not t or len(t) > 60 or re.fullmatch(r"[\d\W]+", t) or t in NOISE_KEYWORDS:
        return []
    words = [m.group(0) for m in _WORD_RE.finditer(t)]
    if len(words) == 1 and (_is_stop(words[0]) or _is_generic(words[0])):
        return []
    return [c for c in _canonicalize(t) if c not in NOISE_KEYWORDS]


def normalize_keyword(text: str) -> str:
    """Первый канонический вариант ключевика или пустая строка (для совместимости)."""
    v = keyword_variants(text)
    return v[0] if v else ""


def _group_key(keyword: str) -> str:
    """Ключ склейки вариантов: «тестирование api», «тестирования api» и «api-тестирование» — одно и то же
    (падеж и порядок слов не важны)."""
    words = [m.group(0) for m in _WORD_RE.finditer(keyword)]
    return " ".join(sorted(stem(w) for w in words)) if words else keyword


def analyze_llm(vacancies: Iterable[dict], llm_keywords: Dict[str, List[str]], min_df: int = 1,
                use_dictionary: bool = True) -> dict:
    """Посчитать ключевики: то, что назвал ИИ, + теги «Ключевые навыки» hh + (по умолчанию) словарь.

    Словарь по полному тексту подстраховывает ИИ: модель бывает невнимательна и пропускает
    очевидные вещи («тест-кейсы», «чек-листы»), а точный поиск по словарю их не пропускает.
    vacancies — dict с vacancy_id, skills, description; llm_keywords — {vacancy_id: [ключевики]}.
    В подсчёт попадают только вакансии, для которых ИИ уже отработал.
    Возвращает {"total": N, "rows": [dict(keyword, vacancies, percent, in_tags, in_llm, in_dict)]}.
    """
    total = 0
    df: Counter = Counter()
    from_tags: Counter = Counter()
    from_llm: Counter = Counter()
    from_dict: Counter = Counter()
    surfaces: Dict[str, Counter] = defaultdict(Counter)

    def groups_of(keywords) -> set:
        out = set()
        for t in keywords:
            for k in keyword_variants(t):
                g = _group_key(k)
                out.add(g)
                surfaces[g][k] += 1
        return out

    for v in vacancies:
        vid = v.get("vacancy_id")
        if vid not in llm_keywords:
            continue
        total += 1

        tag_groups = groups_of(v.get("skills") or [])
        llm_groups = groups_of(llm_keywords[vid])
        dict_groups = groups_of(find_dictionary_terms(clean_description(v.get("description") or ""))) \
            if use_dictionary else set()

        for g in tag_groups | llm_groups | dict_groups:
            df[g] += 1
        for g in tag_groups:
            from_tags[g] += 1
        for g in llm_groups:
            from_llm[g] += 1
        for g in dict_groups:
            from_dict[g] += 1

    rows = []
    for g, n in df.items():
        if n < min_df:
            continue
        shown = surfaces[g].most_common(1)[0][0]
        rows.append({"keyword": shown, "vacancies": n,
                     "percent": round(100.0 * n / total, 1) if total else 0.0,
                     "in_tags": from_tags.get(g, 0), "in_llm": from_llm.get(g, 0),
                     "in_dict": from_dict.get(g, 0)})
    rows.sort(key=lambda r: (-r["vacancies"], r["keyword"]))
    return {"total": total, "rows": rows}
