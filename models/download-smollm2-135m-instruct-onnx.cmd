@echo off
REM Download SmolLM2-135M-Instruct-ONNX into models\SmolLM2-135M-Instruct-ONNX.
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-smollm2-135m-instruct-onnx.ps1"
exit /b %ERRORLEVEL%
