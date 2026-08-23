# Download onnx-community SmolLM2-135M-Instruct-ONNX (Llama ChatML, ~135M).
# Prefer fp16 ONNX (~270 MiB) - loader converts to float32.
# Usage (PowerShell):  .\models\download-smollm2-135m-instruct-onnx.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'SmolLM2-135M-Instruct-ONNX'
$OnnxDir = Join-Path $Dest 'onnx'
$Base = 'https://huggingface.co/onnx-community/SmolLM2-135M-Instruct-ONNX/resolve/main'

New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

Write-Host 'Downloading config / generation_config ...'
foreach ($f in @('config.json', 'generation_config.json')) {
    $out = Join-Path $Dest $f
    Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$f"
}

Write-Host 'Downloading tokenizer sidecars ...'
foreach ($f in @(
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'vocab.json',
    'merges.txt'
)) {
    $out = Join-Path $Dest $f
    Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$f"
}

Write-Host 'Downloading onnx\model_fp16.onnx (~270 MiB) ...'
$OnnxFile = Join-Path $OnnxDir 'model_fp16.onnx'
Invoke-CurlResume -Curl $Curl -OutFile $OnnxFile -Url "$Base/onnx/model_fp16.onnx"

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Get-ChildItem $OnnxDir | Format-Table Name, Length, LastWriteTime
