@echo off
setlocal
cd /d "%~dp0"

set "PYTHONW=%~dp0.venv_qt\Scripts\pythonw.exe"
if not exist "%PYTHONW%" (
    echo Qt environment was not found.
    echo Run: python -m venv .venv_qt
    echo Then install: .venv_qt\Scripts\python -m pip install -r requirements.txt
    pause
    exit /b 1
)

start "" /D "%~dp0" "%PYTHONW%" "%~dp0main_qt.py"
exit /b 0
