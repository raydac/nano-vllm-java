@echo off
REM Download gte-small Q2_K GGUF (Windows wrapper).
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-gte-small-gguf.ps1"
exit /b %ERRORLEVEL%
