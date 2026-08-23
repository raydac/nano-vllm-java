@echo off
REM Download openai/whisper-base into models\whisper-base (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-whisper-base.ps1"
exit /b %ERRORLEVEL%
