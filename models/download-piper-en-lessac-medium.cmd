@echo off
REM Download Piper US English Lessac medium into models\piper-en-lessac-medium (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-piper-en-lessac-medium.ps1"
exit /b %ERRORLEVEL%
