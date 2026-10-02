@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo API hh-агента: http://localhost:8000/docs
venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
pause
