@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHONW=%PROJECT_DIR%.venv\Scripts\pythonw.exe"
set "GUI_SCRIPT=%PROJECT_DIR%experimental\gui005.py"

if not exist "%PYTHONW%" (
    echo Could not find the project Python environment:
    echo %PYTHONW%
    echo.
    echo Create or restore .venv before launching GUI005.
    pause
    exit /b 1
)

if not exist "%GUI_SCRIPT%" (
    echo Could not find GUI005:
    echo %GUI_SCRIPT%
    pause
    exit /b 1
)

start "GUI005 - First Paper Closed Loop" /D "%PROJECT_DIR%experimental" "%PYTHONW%" "%GUI_SCRIPT%"

