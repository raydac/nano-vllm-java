@echo off
REM Download Qwen3-0.6B Q8_0 GGUF (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-qwen3-0.6b-gguf.ps1"
exit /b %ERRORLEVEL%
