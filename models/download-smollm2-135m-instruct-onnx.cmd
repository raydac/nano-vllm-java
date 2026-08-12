@echo off
REM Download SmolLM2-135M-Instruct-ONNX into models\SmolLM2-135M-Instruct-ONNX.
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-smollm2-135m-instruct-onnx.ps1"
exit /b %ERRORLEVEL%
