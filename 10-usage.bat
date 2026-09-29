@echo off
setlocal

set "PROJECT_DIR=%~dp0"
set "PYTHON=%PROJECT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    echo Could not find the project Python environment:
    echo %PYTHON%
    echo.
    echo Create or restore .venv before running the launcher.
    pause
    exit /b 1
)

pushd "%PROJECT_DIR%"
"%PYTHON%" "%PROJECT_DIR%usage_launcher.py"
set "LAUNCHER_EXIT_CODE=%ERRORLEVEL%"
popd

echo.
if not "%LAUNCHER_EXIT_CODE%"=="0" (
    echo Launcher finished with error code %LAUNCHER_EXIT_CODE%.
) else (
    echo Finished successfully.
)
pause
exit /b %LAUNCHER_EXIT_CODE%
