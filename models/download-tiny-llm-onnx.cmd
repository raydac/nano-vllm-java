@echo off
REM Download Tiny-LLM-ONNX into models\Tiny-LLM-ONNX (Windows wrapper for the PowerShell script).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-tiny-llm-onnx.ps1"
exit /b %ERRORLEVEL%
