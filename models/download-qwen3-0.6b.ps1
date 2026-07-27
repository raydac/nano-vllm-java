# Download Qwen3-0.6B weights into this models/ directory (not into src/).
# Usage (PowerShell):  .\models\download-qwen3-0.6b.ps1
# Or double-click / run: models\download-qwen3-0.6b.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest = Join-Path $ModelsRoot 'Qwen3-0.6B'
$Base = 'https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Set-Location $Dest

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl

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
    & $Curl -L --fail --retry 3 -C - -o $f "$Base/$f"
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed for $f (exit $LASTEXITCODE)"
    }
}

Write-Host 'Downloading model.safetensors (~1.4GB) ...'
& $Curl -L --fail --retry 3 -C - -o model.safetensors "$Base/model.safetensors"
if ($LASTEXITCODE -ne 0) {
    throw "Download failed for model.safetensors (exit $LASTEXITCODE)"
}

Write-Host "Installed to $Dest"
Get-ChildItem | Format-Table Name, Length, LastWriteTime
$size = (Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($size / 1MB))
