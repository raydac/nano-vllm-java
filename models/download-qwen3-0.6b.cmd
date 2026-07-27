@echo off
REM Download Qwen3-0.6B into models\Qwen3-0.6B (Windows wrapper for the PowerShell script).
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-qwen3-0.6b.ps1"
exit /b %ERRORLEVEL%
