@echo off
REM Download multilingual-e5-small ONNX into models\multilingual-e5-small (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-multilingual-e5-small.ps1"
exit /b %ERRORLEVEL%
