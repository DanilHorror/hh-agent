@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv not found next to this file.
    echo Make sure this file sits in the hh_agent folder, next to venv and collect_vacancies.py.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   Collecting ALL hh.ru vacancies (no AI, no Telegram)
echo ============================================================
echo.
echo If the HH bot window is currently running - close it first:
echo the same browser profile cannot be opened by two processes at once.
echo.
pause

venv\Scripts\python.exe collect_vacancies.py

echo.
echo ============================================================
echo   Done
echo ============================================================
pause
