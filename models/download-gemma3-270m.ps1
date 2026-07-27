# Download Gemma 3 270M IT into models/Gemma3-270M.
# Requires Gemma license acceptance on Hugging Face + HF_TOKEN (or huggingface-cli login).
# Usage:  .\models\download-gemma3-270m.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Dest = Join-Path $ModelsRoot 'Gemma3-270M'
$Base = 'https://huggingface.co/google/gemma-3-270m-it/resolve/main'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
Set-Location $Dest

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
    $tokenFile = Join-Path $env:USERPROFILE '.cache\huggingface\token'
    if (Test-Path $tokenFile) {
        $tok = (Get-Content -Raw $tokenFile).Trim()
        if ($tok) {
            return @('-H', "Authorization: Bearer $tok")
        }
    }
    return @()
}

$Curl = Get-Curl
$Auth = Get-HfAuthHeaders

function Download-File([string]$Name) {
    Write-Host "Downloading $Name ..."
    & $Curl -L --fail --retry 3 -C - @Auth -o $Name "$Base/$Name"
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Download failed. Gemma is gated: accept the license at' -ForegroundColor Red
        Write-Host 'https://huggingface.co/google/gemma-3-270m-it' -ForegroundColor Red
        Write-Host 'Then: huggingface-cli login  (or set HF_TOKEN)' -ForegroundColor Red
        throw "Download failed for $Name (exit $LASTEXITCODE)"
    }
}

$SmallFiles = @(
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'tokenizer.model',
    'special_tokens_map.json',
    'added_tokens.json'
)

foreach ($f in $SmallFiles) {
    Download-File $f
}

Write-Host 'Downloading model.safetensors (~0.5GB) ...'
Download-File 'model.safetensors'

Write-Host "Installed to $Dest"
Get-ChildItem | Format-Table Name, Length, LastWriteTime
$size = (Get-ChildItem -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($size / 1MB))
