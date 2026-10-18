@echo off
REM Download Jeff-Qwen3.5-0.8B into models\Jeff-Qwen3.5-0.8B (Windows wrapper for the PowerShell script).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-jeff-qwen3.5-0.8b.ps1"
exit /b %ERRORLEVEL%
