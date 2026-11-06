@echo off
REM Export YOLO26 nano ONNX files into models\YOLO26n (Windows wrapper).
setlocal
where powershell.exe >nul 2>&1
if errorlevel 1 (
  echo powershell.exe not found on PATH.
  exit /b 1
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download-yolo26n.ps1"
exit /b %ERRORLEVEL%
