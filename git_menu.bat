@echo off
cd /d "%~dp0"

:menu
cls
echo ==============================
echo        GIT ACTIONS
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

:github_menu
cls
echo ==============================
echo          GITHUB
echo ==============================
echo 1^) Stage + Commit
echo 2^) Push To Main
echo 3^) Pull
echo 4^) Fetch + Reset
echo 5^) Exit
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
if "%opt%"=="3" goto pull
if "%opt%"=="4" goto reset
if "%opt%"=="5" goto menu
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
