# Download openai/whisper-base as a Hugging Face Whisper safetensors folder.
# Speech-to-text (audio -> text). Not a chat model. Not CTranslate2 model.bin.
# Source: https://huggingface.co/openai/whisper-base
# Usage (PowerShell):  .\models\download-whisper-base.ps1
# Or double-click / run: models\download-whisper-base.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'whisper-base'
$Base = 'https://huggingface.co/openai/whisper-base/resolve/main'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

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
. (Join-Path $ModelsRoot '_curl_resume.ps1')
$Auth = Get-HfAuthHeaders

function Download-HfFile([string]$Name) {
    Write-Host "Downloading $Name ..."
    $out = Join-Path $Dest $Name
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$Name" -ExtraArgs $Auth
    } catch {
        Write-Host 'Download failed. Retry, or set HF_TOKEN / huggingface-cli login if rate-limited.' -ForegroundColor Red
        Write-Host 'https://huggingface.co/openai/whisper-base' -ForegroundColor Red
        throw
    }
}

Write-Host 'Downloading config / tokenizer sidecars ...'
foreach ($f in @(
    'config.json',
    'tokenizer.json',
    'tokenizer_config.json'
)) {
    Download-HfFile $f
}

Write-Host 'Downloading model.safetensors (~290MB) ...'
Download-HfFile 'model.safetensors'

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Write-Host 'Note: Whisper speech-to-text (HF safetensors). Do not use a faster-whisper model.bin folder.'
