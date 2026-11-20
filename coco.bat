@echo off
rem ============================================================================
rem  Double-click to start Coco.
rem    1. Checks Python, Node.js (for the notes MCP server) and the Python packages
rem    2. Starts llama-server in its own minimized window (unless it's already running)
rem    3. Waits for the model to load, then runs Coco in this window
rem    4. When you quit Coco, stops the llama-server it started (frees GPU memory)
rem ============================================================================
setlocal EnableExtensions
title Coco

rem ---- Settings. PORT must match BASE_URL in src\loop\agent.py ----
set "MODEL=unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M"
set "PORT=8080"
set "CONTEXT=16384"
set "MAX_WAIT_SECONDS=600"

rem Run from the repo root, so notes\, memory.md and logs\ are the ones here
cd /d "%~dp0"
set "PYTHONUTF8=1"

rem ---- 1. Requirements ----
where python >nul 2>nul || (echo Python was not found on PATH. Install Python 3.11+ and try again. & goto :fail)
where npx >nul 2>nul || (echo Node.js was not found on PATH. Install it from https://nodejs.org and try again. & goto :fail)
python -c "import openai, mcp" >nul 2>nul
if errorlevel 1 (
    echo Installing Python requirements...
    python -m pip install -r src\loop\requirements.txt || goto :fail
)

rem ---- 2. llama-server ----
set "LLAMA_PID="
curl -s -f -o nul "http://localhost:%PORT%/health" && (echo llama-server is already running on port %PORT%. & goto :run)

set "LLAMA_SERVER="
for /d %%D in ("%~dp0llama-*") do if exist "%%D\llama-server.exe" set "LLAMA_SERVER=%%D\llama-server.exe"
if not defined LLAMA_SERVER (echo Could not find llama-server.exe. Put the llama.cpp release folder ^(llama-...-bin-win-...^) next to coco.bat. & goto :fail)

echo Starting llama-server: %MODEL%
for /f %%P in ('powershell -NoProfile -Command "(Start-Process -FilePath '%LLAMA_SERVER%' -ArgumentList '-hf %MODEL% --jinja -c %CONTEXT% --port %PORT%' -WindowStyle Minimized -PassThru).Id"') do set "LLAMA_PID=%%P"
if not defined LLAMA_PID (echo Could not start llama-server. & goto :fail)

rem ---- 3. Wait until the model is loaded (/health answers 200 only when ready) ----
echo Waiting for the model to load (a first run downloads it, which can take a while)...
set /a WAITED=0
:wait
curl -s -f -o nul "http://localhost:%PORT%/health" && goto :run
tasklist /fi "pid eq %LLAMA_PID%" | find "%LLAMA_PID%" >nul || (echo llama-server stopped while loading. Run it by hand to see the error: & echo   "%LLAMA_SERVER%" -hf %MODEL% --jinja -c %CONTEXT% --port %PORT% & goto :fail)
rem 2-second pause (ping works even when "timeout" can't, e.g. with redirected input)
ping -n 3 127.0.0.1 >nul
set /a WAITED+=2
if %WAITED% geq %MAX_WAIT_SECONDS% (echo llama-server was not ready after %MAX_WAIT_SECONDS% seconds. & goto :fail)
goto :wait

rem ---- 4. Coco ----
:run
echo.
python src\loop\agent.py
set "COCO_EXIT=%errorlevel%"
call :stop_llama
if not "%COCO_EXIT%"=="0" (echo. & echo Coco exited with an error. & pause)
exit /b %COCO_EXIT%

:fail
call :stop_llama
echo.
pause
exit /b 1

:stop_llama
rem Only stops a llama-server this script started, never one you started yourself
if defined LLAMA_PID (
    echo Stopping llama-server...
    taskkill /pid %LLAMA_PID% /t /f >nul 2>nul
)
exit /b 0
