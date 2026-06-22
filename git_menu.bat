@echo off
cd /d "%~dp0"

:menu
cls
echo ==============================
echo        GIT ACTIONS
echo ==============================
call :show_project_info
echo ==============================
echo 1^) GitHub
echo 2^) VPS
echo 3^) Setup
echo 4^) Status
echo 5^) Exit
echo ==============================
echo Help:
echo  GitHub: stage, commit, push, pull, or reset GitHub code.
echo  VPS: stage, commit, or push main to the VPS remote.
echo  Setup: configure remotes or run first-time VPS setup.
echo  Status: show current branch and changed files.
echo ==============================
set /p opt=Choose option: 

if "%opt%"=="1" goto github_menu
if "%opt%"=="2" goto vps_menu
if "%opt%"=="3" goto setup_menu
if "%opt%"=="4" goto status
if "%opt%"=="5" goto end
goto menu

:show_project_info
for %%d in ("%CD%") do echo Folder: %%~nxd
echo Path: %CD%

if defined VIRTUAL_ENV (
    echo Venv: active - %VIRTUAL_ENV%
) else if exist ".venv\Scripts\python.exe" (
    echo Venv: .venv
) else if exist "venv\Scripts\python.exe" (
    echo Venv: venv
) else if exist "env\Scripts\python.exe" (
    echo Venv: env
) else (
    echo Venv: none found
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo Git: none found
    exit /b
)

set "git_branch="
set "git_remote="
set "git_worktree_status="
set "git_index_status="

for /f "delims=" %%b in ('git branch --show-current 2^>nul') do set "git_branch=%%b"
if not defined git_branch set "git_branch=detached HEAD"

for /f "delims=" %%r in ('git remote get-url origin 2^>nul') do set "git_remote=%%r"
if not defined git_remote set "git_remote=no origin remote"

git diff --quiet --ignore-submodules -- 2>nul
set "git_worktree_status=clean"
if errorlevel 1 set "git_worktree_status=modified"

git diff --cached --quiet --ignore-submodules -- 2>nul
set "git_index_status=clean"
if errorlevel 1 set "git_index_status=staged"

echo Git: %git_branch% ^| %git_worktree_status% ^| %git_index_status%
echo Remote: %git_remote%
exit /b

:github_menu
cls
echo ==============================
echo          GITHUB
echo ==============================
echo 1^) Stage + Commit
echo 2^) Push
echo 3^) Push To Main
echo 4^) Pull
echo 5^) Fetch + Reset
echo 6^) Exit
echo ==============================
echo Help:
echo  Stage + Commit: runs git add --all and commits with message "update".
echo  Push To Main: pushes main branch to the remote named github.
echo  Pull: pulls the latest code from the default remote.
echo  Fetch + Reset: WARNING, deletes local changes and resets to origin/main.
echo  Exit: return to the main menu.
echo ==============================
set /p opt=Choose option: 

if "%opt%"=="1" goto stage_commit_github
if "%opt%"=="2" goto push_github
if "%opt%"=="3" goto push_github_main
if "%opt%"=="4" goto pull
if "%opt%"=="5" goto reset
if "%opt%"=="6" goto menu
goto github_menu

:vps_menu
cls
echo ==============================
echo            VPS
echo ==============================
echo 1^) Stage + Commit
echo 2^) Push To Main
echo 3^) Exit
echo ==============================
echo Help:
echo  Stage + Commit: runs git add --all and commits with message "update".
echo  Push To Main: pushes main branch to the remote named vps.
echo  Exit: return to the main menu.
echo ==============================
set /p opt=Choose option: 

if "%opt%"=="1" goto stage_commit_vps
if "%opt%"=="2" goto push_vps
if "%opt%"=="3" goto menu
goto vps_menu

:setup_menu
cls
echo ==============================
echo           SETUP
echo ==============================
echo 1^) Define Gits
echo 2^) First Deploy
echo 3^) Exit
echo ==============================
echo Help:
echo  Define Gits: configures GitHub and VPS git remotes from infra/.env.
echo  First Deploy: runs the first-time VPS setup and deploy script.
echo  Exit: return to the main menu.
echo ==============================
set /p opt=Choose option: 

if "%opt%"=="1" goto define_gits
if "%opt%"=="2" goto first_deploy
if "%opt%"=="3" goto menu
goto setup_menu

:stage_commit_github
git add --all
git commit -m "update"
echo.
pause
goto github_menu

:pull
git pull
echo.
pause
goto github_menu

:reset
echo WARNING: this will delete local changes.
set /p conf=Continue? (y/n): 
if /i not "%conf%"=="y" goto github_menu
git fetch origin
git reset --hard origin/main
echo.
pause
goto github_menu

:push_github
git push
echo.
pause
goto github_menu

:push_github_main
git push github main
echo.
pause
goto github_menu

:stage_commit_vps
git add --all
git commit -m "update"
echo.
pause
goto vps_menu

:push_vps
git push vps main
echo.
pause
goto vps_menu

:define_gits
bash infra/local/define-gits.sh
echo.
pause
goto setup_menu

:first_deploy
bash infra/local/first-setup-vps.sh
echo.
pause
goto setup_menu

:status
git status
echo.
pause
goto menu

:end
exit
