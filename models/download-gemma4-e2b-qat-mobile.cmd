@echo off
REM Download Gemma4-E2B-IT-QAT-Mobile into models\Gemma4-E2B-IT-QAT-Mobile (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-gemma4-e2b-qat-mobile.ps1"
exit /b %ERRORLEVEL%
