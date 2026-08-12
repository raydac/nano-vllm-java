@echo off
REM Download LFM2.5-2.6B Q4_K_M GGUF (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-lfm2.5-2.6b-gguf.ps1"
exit /b %ERRORLEVEL%
