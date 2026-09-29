@echo off
setlocal
cd /d "%~dp0"

set "PYTHONW=%~dp0.venv_qt\Scripts\pythonw.exe"
if exist "%PYTHONW%" (
    start "" /D "%~dp0" "%PYTHONW%" "%~dp0main.py"
    exit /b 0
)

where pythonw >nul 2>nul
if %errorlevel% equ 0 (
    start "" /D "%~dp0" pythonw "%~dp0main.py"
    exit /b 0
)

echo Python was not found. Install Python or create .venv_qt first.
pause
exit /b 1
