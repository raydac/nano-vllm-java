# Download intfloat/multilingual-e5-small as an ONNX embedding folder.
# BERT graph + XLM-RoBERTa Unigram tokenizer - not a causal chat model.
# Source: https://huggingface.co/intfloat/multilingual-e5-small
# Usage (PowerShell):  .\models\download-multilingual-e5-small.ps1
# Or double-click / run: models\download-multilingual-e5-small.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'multilingual-e5-small'
$OnnxDir = Join-Path $Dest 'onnx'
$Base = 'https://huggingface.co/intfloat/multilingual-e5-small/resolve/main'

New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl

Write-Host 'Downloading config / tokenizer sidecars ...'
foreach ($f in @(
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json'
)) {
    $out = Join-Path $Dest $f
    & $Curl -L --fail --retry 3 -C - -o $out "$Base/$f"
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $f (exit $LASTEXITCODE)" }
}

Write-Host 'Downloading onnx\model.onnx (~470MB fp32) ...'
$OnnxFile = Join-Path $OnnxDir 'model.onnx'
& $Curl -L --fail --retry 3 -C - -o $OnnxFile "$Base/onnx/model.onnx"
if ($LASTEXITCODE -ne 0) { throw "Download failed for model.onnx (exit $LASTEXITCODE)" }

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Get-ChildItem $OnnxDir | Format-Table Name, Length, LastWriteTime
Write-Host 'Note: embedding model (BERT / multilingual E5), context up to 512 tokens - not for LLM chat samples.'
Write-Host 'E5 expects prefixes: query: ...  and  passage: ...'
