# Download FacebookAI/xlm-roberta-base as an ONNX embedding folder.
# XLM-RoBERTa encoder (fill-mask checkpoint, mean-pooled embeddings) - not a causal chat model.
# Source: https://huggingface.co/FacebookAI/xlm-roberta-base
# Saves Hub model.onnx as onnx\model.onnx so the loader sees the expected name.
# Do not copy model.safetensors - safetensors would win and HF BERT-family safetensors is not loaded.
# Usage (PowerShell):  .\models\download-xlm-roberta-base.ps1
# Or double-click / run: models\download-xlm-roberta-base.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'xlm-roberta-base'
$OnnxDir = Join-Path $Dest 'onnx'
$Base = 'https://huggingface.co/FacebookAI/xlm-roberta-base/resolve/main'

New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

function Get-HfAuthHeaders {
    if ($env:HF_TOKEN) {
        return @('-H', "Authorization: Bearer $($env:HF_TOKEN)")
    }
    if ($env:HF_HOME) {
        $tokenFile = Join-Path $env:HF_HOME 'token'
        if (Test-Path -LiteralPath $tokenFile) {
            $tok = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
            if ($tok) {
                return @('-H', "Authorization: Bearer $tok")
            }
        }
    }
    $tokenFile = Join-Path $env:USERPROFILE '.cache\huggingface\token'
    if (Test-Path -LiteralPath $tokenFile) {
        $tok = (Get-Content -LiteralPath $tokenFile -Raw).Trim()
        if ($tok) {
            return @('-H', "Authorization: Bearer $tok")
        }
    }
    return @()
}

$Curl = Get-Curl
$Auth = Get-HfAuthHeaders

function Download-HfFile([string]$DestRel, [string]$SrcRel) {
    Write-Host "Downloading $SrcRel -> $DestRel ..."
    $out = Join-Path $Dest $DestRel
    & $Curl -L --fail --retry 3 -C - @Auth -o $out "$Base/$SrcRel"
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Download failed. Retry, or set HF_TOKEN / huggingface-cli login if rate-limited.' -ForegroundColor Red
        Write-Host 'https://huggingface.co/FacebookAI/xlm-roberta-base' -ForegroundColor Red
        throw "Download failed for $SrcRel (exit $LASTEXITCODE)"
    }
}

Write-Host 'Downloading config / tokenizer sidecars ...'
foreach ($f in @(
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json'
)) {
    Download-HfFile $f $f
}

Write-Host 'Downloading onnx\model.onnx (~1.9GB fp32) from Hugging Face model.onnx ...'
Download-HfFile (Join-Path 'onnx' 'model.onnx') 'model.onnx'

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Get-ChildItem $OnnxDir | Format-Table Name, Length, LastWriteTime
Write-Host 'Note: embedding encoder (XLM-RoBERTa / BERT graph), context up to 512 tokens - not for LLM chat samples.'
