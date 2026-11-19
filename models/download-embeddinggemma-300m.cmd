@echo off
REM Download EmbeddingGemma-300M into models\EmbeddingGemma-300M (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-embeddinggemma-300m.ps1"
exit /b %ERRORLEVEL%
