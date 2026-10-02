"""Демо-страница hh-агента на Streamlit. Ходит в API (не в базу напрямую).

    venv\\Scripts\\python.exe -m streamlit run demo\\app.py
    Адрес API — переменная HH_API_URL (по умолчанию http://localhost:8000)
"""
import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("HH_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(300.0, connect=5.0)

st.set_page_config(page_title="hh-agent", page_icon="🔎", layout="wide")


def api_get(path: str, **params):
    response = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict):
    response = httpx.post(f"{API_URL}{path}", json=payload, timeout=TIMEOUT)
    if response.status_code >= 400:
        detail = response.json().get("detail", response.text)
        raise RuntimeError(detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False))
    return response.json()


def show_error(e: Exception):
    if isinstance(e, httpx.ConnectError):
        st.error(f"API не отвечает по адресу {API_URL}. Запустите «Запустить API.bat».")
    else:
        st.error(str(e))


# --- Боковая панель: состояние и цифры -------------------------------------------------
with st.sidebar:
    st.header("hh-agent")
    try:
        health = api_get("/health")
        icon = "🟢" if health["status"] == "ok" else "🟠"
        st.write(f"{icon} База: {health['database']}  \nМодель ({health['llm_provider']}): {health['llm']}")
        stats = api_get("/stats")
        st.metric("Полных текстов вакансий", stats["vacancy_texts"])
        if stats.get("rag_chunks") is not None:
            st.metric("Фрагментов в поиске", stats["rag_chunks"])
        agent = stats["agent"]
        if agent.get("decided"):
            st.metric("Решений по черновикам", agent["decided"])
            if agent.get("agreement") is not None:
                st.metric("Согласие с моделью", f"{agent['agreement']:.0%}")
    except Exception as e:  # noqa: BLE001
        show_error(e)
    st.caption(f"API: {API_URL} · [документация]({API_URL}/docs)")

tab_ask, tab_search, tab_eval = st.tabs(["💬 Спросить о рынке", "🔎 Поиск вакансий", "🧠 Оценить вакансию"])

# --- Вопрос о рынке: потоковый ответ со ссылками ----------------------------------------
with tab_ask:
    st.write("Ответ строится только по найденным фрагментам вакансий и ссылается на них.")
    question = st.text_input("Вопрос", "В каких вакансиях нужно тестировать AI-агентов или LLM?")
    if st.button("Спросить", type="primary") and question.strip():
        sources = []

        def tokens():
            with httpx.stream("POST", f"{API_URL}/ask/stream", json={"question": question},
                              timeout=TIMEOUT) as response:
                if response.status_code >= 400:
                    response.read()
                    raise RuntimeError(response.json().get("detail", response.text))
                for line in response.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event["type"] == "sources":
                        sources.extend(event["sources"])
                    elif event["type"] == "token":
                        yield event["text"]
                    elif event["type"] == "error":
                        raise RuntimeError(event["detail"])

        try:
            with st.spinner("Ищу по базе вакансий…"):
                st.write_stream(tokens())
            if sources:
                st.markdown("**Источники:**")
                for s in sources:
                    st.markdown(f"[{s['n']}] [{s['title']} — {s['company']}]({s['url']}) · _{s['section']}_")
        except Exception as e:  # noqa: BLE001
            show_error(e)

# --- Поиск ------------------------------------------------------------------------------
with tab_search:
    col_q, col_mode, col_k = st.columns([4, 2, 1])
    query = col_q.text_input("Что искать", "автотесты на Playwright и TypeScript")
    modes = {"Гибрид + реранкер": "hybrid_rerank", "Гибрид": "hybrid", "Только векторы": "vector",
             "Только слова": "text"}
    mode_label = col_mode.selectbox("Режим", list(modes))
    k = col_k.number_input("Сколько", 1, 20, 5)
    if st.button("Найти") and query.strip():
        try:
            result = api_get("/search", q=query, k=k, mode=modes[mode_label])
            st.caption(f"{len(result['hits'])} фрагментов за {result['took_ms']} мс")
            for hit in result["hits"]:
                with st.container(border=True):
                    st.markdown(f"**[{hit['title']} — {hit['company']}]({hit['url']})** · "
                                f"_{hit['section']}_ · оценка {hit['score']:.3f}")
                    st.write(hit["snippet"].split("\n", 1)[-1][:500])
        except Exception as e:  # noqa: BLE001
            show_error(e)

# --- Оценка вакансии --------------------------------------------------------------------
with tab_eval:
    st.write("Вставьте текст вакансии — модель оценит соответствие резюме и напишет черновик письма.")
    title = st.text_input("Должность", "QA Automation Engineer (Python)")
    company = st.text_input("Компания", "")
    description = st.text_area("Текст вакансии", height=250,
                               placeholder="Обязанности, требования, условия — целиком, как на hh.ru")
    if st.button("Оценить"):
        try:
            with st.spinner("Модель читает вакансию…"):
                result = api_post("/evaluate", {"title": title, "company": company, "description": description})
            st.metric("Релевантность", f"{result['relevance_score']}/100")
            st.write(result["reasoning"])
            st.text_area("Черновик письма", result["cover_letter"], height=250)
        except Exception as e:  # noqa: BLE001
            show_error(e)
