@echo off
REM Download Meta fastText lid.176.ftz into models\fasttext-lid-176 (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-fasttext-lid-176.ps1"
exit /b %ERRORLEVEL%
