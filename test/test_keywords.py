"""Оффлайн-тест подсчёта ключевиков (без браузера и БД).

Запуск из папки hh_agent:   venv\\Scripts\\python.exe test\\test_keywords.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analytics.keywords import (
    analyze, normalize_tag, find_dictionary_terms, extract_unknown_latin_terms, stem,
)

V1 = {
    "skills": ["Сетевые протоколы", "сетевое оборудование", "Robot Framework", "Jenkins", "TCP/IP", "OSPF"],
    "description": """Обязанности
— Мы ищем разработчика тестов, автоматизации тестирования, который специализируется на сетевых технологиях. Основная задача — проектирование, разработка и поддержка фреймворков автоматизации тестирования сетевого ПО.
— Разработка фреймворков: Проектирование и написание масштабируемых инструментов и библиотек автоматизации на Python.
— Автоматизация тестирования: Создание и поддержка тест-кейсов (функциональное, регрессионное, нагрузочное тестирование).
— Тестирование сетевой функциональности: Проверка реализации сетевых протоколов (L2/L3, MPLS, безопасность).
— Работа с оборудованием/симуляторами: Настройка физических сетевых устройств или виртуальных сред (GNS3, виртуальный эмулятор).
— Анализ результатов: Диагностика причин сбоев (troubleshooting), анализ логов, дампов трафика (Wireshark/Tcpdump).
— CI/CD: Интеграция автотестов в процессы CI/CD
— Язык: Python 3.x
— Тестирование: Pytest, Scapy, Robot Framework
— Сети: BGP, OSPF, MPLS, VXLAN, EVPN
— Инструменты: Docker, Jenkins, Git, Wireshark, Ansible
— Окружение: Linux (Ubuntu/CentOS), виртуальные лаборатории (EVE-NG/GNS3)
Требования
— Опыт тестирования сетевых устройств (маршрутизаторы, коммутаторы и др.)
— Python: Уверенное владение. Знание ООП, понимание структур данных, умение писать чистый, поддерживаемый код (PEP8).
— Сетевые знания: понимание модели OSI, стека TCP/IP, принципов маршрутизации (OSPF, BGP, IS-IS) и коммутации (VLAN, STP).
— Инструментарий тестирования: Опыт работы с pytest (или unittest)""",
}
V2 = {
    "skills": [],
    "description": """ЧЕМ ПРЕДСТОИТ ЗАНИМАТЬСЯ
Формирование и развитие группы тестирования электроники и встраиваемого ПО.
Организация полного цикла тестирования электронных устройств и embedded ПО.
Организация функционального, интеграционного, регрессионного и системного тестирования.
Разработка и внедрение стендов, оснастки и средств автоматизации тестирования.
Ведение базы дефектов, результатов испытаний.
КАКИМ МЫ ВИДИМ НОВОГО КОЛЛЕГУ
Опыт тестирования электронных устройств и/или embedded ПО от 3 лет.
Уверенная работа с осциллографом, мультиметром, логическим анализатором.
Знание основных аппаратных интерфейсов: UART, I2C, SPI, CAN, USB; опыт работы с BLE.
Опыт функционального, интеграционного и регрессионного тестирования.
Опыт составления тест-кейсов, чек-листов, тест-планов и отчетов об испытаниях.""",
}
V3 = {
    "skills": ["Python", "Docker", "Git", "CI/CD", "Zabbix"],
    "description": "Разработка стратегий тестирования. Автоматизация тестирования на Python, pytest. Zabbix, Grafana. Опыт работы с Docker.",
}
V4 = {
    "skills": ["python", "GitHub Actions", "Zabbix"],
    "description": "Требуется разработка стратегии тестирования, автоматизации тестирования. Знание Zabbix, Linux, ci-cd.",
}


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        check.failed = True


check.failed = False

# --- теги ---
check(normalize_tag("TCP/IP") == "tcp/ip", "тег TCP/IP -> tcp/ip")
check(normalize_tag("  Robot   Framework ") == "robot framework", "тег Robot Framework нормализуется")
check(normalize_tag("Python 3") == "python 3", "неизвестный тег остаётся как есть")
check(normalize_tag("GitHub Actions") == "github actions", "тег GitHub Actions")
check(normalize_tag("CI-CD") == "ci/cd", "тег CI-CD -> ci/cd")

# --- словарь по тексту ---
found = find_dictionary_terms(V1["description"])
for k in ["python", "pytest", "unittest", "scapy", "robot framework", "bgp", "ospf", "mpls", "vxlan", "evpn",
          "docker", "jenkins", "git", "wireshark", "tcpdump", "ansible", "linux", "eve-ng", "gns3", "ci/cd",
          "автоматизация тестирования", "нагрузочное тестирование", "регрессионное тестирование",
          "функциональное тестирование", "тест-кейсы", "ооп", "pep8", "osi", "tcp/ip", "is-is", "vlan", "stp",
          "l2/l3"]:
    check(k in found, f"V1: найден «{k}»")
check("github actions" not in found and "gitlab ci" not in found, "V1: git не путается с github/gitlab")

found2 = find_dictionary_terms(V2["description"])
for k in ["embedded", "uart", "i2c", "spi", "ble", "осциллограф", "чек-листы", "тест-планы", "тест-кейсы",
          "интеграционное тестирование", "регрессионное тестирование", "автоматизация тестирования"]:
    check(k in found2, f"V2: найден «{k}»")
check("git" not in found2 and "java" not in found2, "V2: нет ложных срабатываний git/java")
check("javascript" not in find_dictionary_terms("Опыт Java, JavaScript") or "java" in find_dictionary_terms("Опыт Java, JavaScript"),
      "java и javascript различаются")
check("java" not in find_dictionary_terms("знание JavaScript"), "JavaScript не считается как Java")
check("c++" in find_dictionary_terms("C++, C#") and "c#" in find_dictionary_terms("C++, C#"), "C++ и C# находятся")

# --- стеммер ---
check(stem("тестирования") == stem("тестирование") == stem("тестированию"), "стемы «тестирование» совпадают")
check(stem("стратегий") == stem("стратегия") == stem("стратегии"), "стемы «стратегия» совпадают")

# --- неизвестные латинские термины ---
unk = extract_unknown_latin_terms(V3["description"])
check("zabbix" in unk or "zabbix" in find_dictionary_terms(V3["description"]), "zabbix распознан")
check("troubleshooting" in extract_unknown_latin_terms(V1["description"]), "V1: troubleshooting — неизвестный термин")

# --- сводка ---
res = analyze([V1, V2, V3, V4], min_df=2)
by = {r["keyword"]: r for r in res["skills"]}
check(res["total"] == 4, "всего 4 вакансии")
check(by["python"]["vacancies"] == 3, f"python в 3 вакансиях (получено {by['python']['vacancies']})")
check(by["python"]["in_tags"] == 2, "python в тегах у 2")
check(by["zabbix"]["vacancies"] == 2 and by["zabbix"]["in_tags"] == 2, "zabbix: 2 вакансии, оба раза в тегах")
check(by["автоматизация тестирования"]["vacancies"] == 4, "автоматизация тестирования в 4 вакансиях")
check(by["ci/cd"]["vacancies"] == 3, "ci/cd (тег, текст, «ci-cd»)")
check(by["ci/cd"]["percent"] == 75.0, "ci/cd = 75%")
check(res["skills"][0]["vacancies"] >= res["skills"][-1]["vacancies"], "отсортировано по убыванию")

phr = {r["keyword"]: r["vacancies"] for r in res["phrases"]}
print("Фразы:", phr)
check(any("стратег" in k and "тестирован" in k for k in phr), "фраза про стратегии тестирования найдена")
check(not any(k.startswith("опыт") for k in phr), "нет мусорных фраз «опыт ...»")

print("\nТоп навыков:")
for r in res["skills"][:12]:
    print(f"  {r['keyword']:<32}{r['vacancies']:>3}  {r['percent']:>5}%  теги={r['in_tags']} текст={r['in_text']}")

print("\nРЕЗУЛЬТАТ:", "ЕСТЬ ОШИБКИ" if check.failed else "ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
sys.exit(1 if check.failed else 0)
