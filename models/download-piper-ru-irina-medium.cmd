@echo off
REM Download Piper Russian Irina medium into models\piper-ru-irina-medium (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-piper-ru-irina-medium.ps1"
exit /b %ERRORLEVEL%
