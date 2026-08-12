# Download onnx-community Tiny-LLM-ONNX (Llama, ~10M) + tokenizer from arnir0/Tiny-LLM.
# Usage (PowerShell):  .\models\download-tiny-llm-onnx.ps1
# Or double-click / run: models\download-tiny-llm-onnx.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest = Join-Path $ModelsRoot 'Tiny-LLM-ONNX'
$OnnxDir = Join-Path $Dest 'onnx'
$OnnxBase = 'https://huggingface.co/onnx-community/Tiny-LLM-ONNX/resolve/main'
$TokBase = 'https://huggingface.co/arnir0/Tiny-LLM/resolve/main'

New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null
Set-Location $Dest

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl

Write-Host 'Downloading config / generation_config ...'
& $Curl -L --fail --retry 3 -C - -o 'config.json' "$OnnxBase/config.json"
if ($LASTEXITCODE -ne 0) { throw "Download failed for config.json (exit $LASTEXITCODE)" }
& $Curl -L --fail --retry 3 -C - -o 'generation_config.json' "$OnnxBase/generation_config.json"
if ($LASTEXITCODE -ne 0) { throw "Download failed for generation_config.json (exit $LASTEXITCODE)" }

Write-Host 'Downloading tokenizer from arnir0/Tiny-LLM ...'
foreach ($f in @(
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer.model'
)) {
    & $Curl -L --fail --retry 3 -C - -o $f "$TokBase/$f"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Optional tokenizer file skipped: $f"
    }
}

Write-Host 'Downloading onnx\model.onnx (fp32) ...'
$OnnxFile = Join-Path $OnnxDir 'model.onnx'
& $Curl -L --fail --retry 3 -C - -o $OnnxFile "$OnnxBase/onnx/model.onnx"
if ($LASTEXITCODE -ne 0) { throw "Download failed for model.onnx (exit $LASTEXITCODE)" }

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Get-ChildItem $OnnxDir | Format-Table Name, Length, LastWriteTime
