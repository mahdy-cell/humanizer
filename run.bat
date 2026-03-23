@echo off
REM =============================================================================
REM  run.bat — Launcher for the Academic Text Humanizer (Windows)
REM  Usage:  run.bat              → Launch GUI
REM          run.bat --cli file.txt → CLI mode
REM          run.bat --test         → Run tests
REM =============================================================================

setlocal enabledelayedexpansion
title Academic Text Humanizer

cd /d "%~dp0"

echo.
echo =====================================================
echo   Academic Text Humanizer - 100-Method Pipeline
echo =====================================================
echo.

REM ── Find Python ────────────────────────────────────────────────────────
set PYTHON=
for %%p in (python python3 py) do (
    if "!PYTHON!"=="" (
        %%p --version >nul 2>&1 && set PYTHON=%%p
    )
)

if "%PYTHON%"=="" (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    echo         Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('!PYTHON! --version 2^>^&1') do echo [INFO]  Using !PYTHON! - %%v

REM ── Virtual environment ────────────────────────────────────────────────
set VENV_DIR=%~dp0.venv

if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO]  Creating virtual environment at .venv\ ...
    !PYTHON! -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat" 2>nul
if errorlevel 1 (
    echo [WARN]  Could not activate venv; using system Python.
) else (
    echo [INFO]  Virtual environment activated.
)

REM ── Install dependencies ───────────────────────────────────────────────
if not "%1"=="--skip-install" (
    echo [INFO]  Checking dependencies ...
    pip install --quiet --upgrade pip >nul 2>&1

    REM Install core lightweight dependencies
    pip install --quiet "pyyaml>=6.0.1" >nul 2>&1
    pip install --quiet "python-docx>=1.0.1" >nul 2>&1
    pip install --quiet "reportlab>=4.0.4" >nul 2>&1
    pip install --quiet "requests>=2.31.0" >nul 2>&1
    pip install --quiet "beautifulsoup4>=4.12.2" >nul 2>&1
    pip install --quiet "unidecode>=1.3.7" >nul 2>&1

    if exist "%~dp0requirements.txt" (
        pip install --quiet -r "%~dp0requirements.txt" --no-deps >nul 2>&1
        echo [INFO]  Dependencies checked.
    )
)

REM ── Launch application ─────────────────────────────────────────────────
echo [INFO]  Starting Academic Text Humanizer ...
echo.

!PYTHON! "%~dp0main.py" %*

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with an error.
    pause
)
endlocal
