@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo Демо-страница: откройте в браузере http://localhost:8501  (API должен быть запущен: "Запустить API.bat")
venv\Scripts\python.exe -m streamlit run demo\app.py --server.port 8501 --browser.gatherUsageStats false --server.headless true
pause
