@echo off
setlocal
set "PROJECT_DIR=%~dp0"
if not exist "%PROJECT_DIR%.venv\Scripts\python.exe" (
    echo Could not find the project Python environment.
    pause
    exit /b 1
)
pushd "%PROJECT_DIR%"
"%PROJECT_DIR%.venv\Scripts\python.exe" "%PROJECT_DIR%experimental\gui005_agent.py"
if errorlevel 1 pause
popd
