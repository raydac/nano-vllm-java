@echo off
REM Download Gemma3-270M into models\Gemma3-270M (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-gemma3-270m.ps1"
exit /b %ERRORLEVEL%
