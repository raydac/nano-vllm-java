@echo off
REM Download gte-small Q2_K GGUF (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-gte-small-gguf.ps1"
exit /b %ERRORLEVEL%
