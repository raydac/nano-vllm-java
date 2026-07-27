@echo off
REM Download Gemma3-270M into models\Gemma3-270M (Windows wrapper).
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-gemma3-270m.ps1"
exit /b %ERRORLEVEL%
