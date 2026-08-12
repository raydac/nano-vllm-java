@echo off
REM Download Tiny-LLM-ONNX into models\Tiny-LLM-ONNX (Windows wrapper for the PowerShell script).
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-tiny-llm-onnx.ps1"
exit /b %ERRORLEVEL%
