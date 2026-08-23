# Download Qwen3-0.6B weights into this models/ directory (not into src/).
# Usage (PowerShell):  .\models\download-qwen3-0.6b.ps1
# Or double-click / run: models\download-qwen3-0.6b.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'Qwen3-0.6B'
$Base = 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

$SmallFiles = @(
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'merges.txt',
    'vocab.json'
)

foreach ($f in $SmallFiles) {
    Write-Host "Downloading $f ..."
    $out = Join-Path $Dest $f
    Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$f"
}

Write-Host 'Downloading model.safetensors (~1.4GB) ...'
$Weights = Join-Path $Dest 'model.safetensors'
Invoke-CurlResume -Curl $Curl -OutFile $Weights -Url "$Base/model.safetensors"

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
$size = (Get-ChildItem -LiteralPath $Dest -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($size / 1MB))
