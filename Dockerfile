# Образ сервисов hh-агента: HTTP API (FastAPI), демо-страница (Streamlit), MCP-сервер.
# Бот, который откликается на hh.ru, сюда не входит: ему нужен браузер с профилем, где выполнен вход.
#
# Сборка и запуск — через docker-compose.yml (профиль services):
#   docker compose --profile services up -d --build
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_DEVICE=cpu

WORKDIR /app

# torch без CUDA: в контейнере модели поиска работают на процессоре — образ меньше на ~4 ГБ,
# а видеокарта остаётся Ollama на хосте
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# Код. Личные файлы (.env, resume.txt, criteria.txt, сертификат) в образ не попадают — см. .dockerignore;
# они подключаются при запуске томами из docker-compose.yml
COPY . .

# Сервисы не работают от root
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app

EXPOSE 8000 8501 8765
CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
