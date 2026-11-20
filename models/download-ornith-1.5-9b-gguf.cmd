@echo off
REM Download the smallest Ornith-1.5-9B GGUF (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-ornith-1.5-9b-gguf.ps1"
exit /b %ERRORLEVEL%
