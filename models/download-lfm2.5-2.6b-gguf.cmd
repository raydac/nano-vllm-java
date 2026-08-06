@echo off
REM Download LFM2.5-2.6B Q4_K_M GGUF (Windows wrapper).
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-lfm2.5-2.6b-gguf.ps1"
exit /b %ERRORLEVEL%
