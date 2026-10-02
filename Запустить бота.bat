@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
    echo ERROR: venv not found next to this file.
    echo Make sure this file sits in the hh_agent folder, next to venv and main.py.
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo   Starting hh.ru job search automation agent
echo ============================================================
echo.

venv\Scripts\python.exe main.py

echo.
echo ============================================================
echo   Agent finished
echo ============================================================
pause
