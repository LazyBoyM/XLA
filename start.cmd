@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Hay cai dat theo README.md truoc khi mo cong cu.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 pause
