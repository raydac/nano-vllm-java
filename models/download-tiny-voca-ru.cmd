@echo off
REM Download Tiny Voca Russian into models\tiny-voca-ru (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-tiny-voca-ru.ps1"
exit /b %ERRORLEVEL%
