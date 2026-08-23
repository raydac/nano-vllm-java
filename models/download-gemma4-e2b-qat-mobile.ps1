# Download Gemma 4 E2B IT QAT mobile into models/Gemma4-E2B-IT-QAT-Mobile.
# Apache 2.0, ungated. Optional HF_TOKEN if Hugging Face rate-limits you.
# Usage:  .\models\download-gemma4-e2b-qat-mobile.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'Gemma4-E2B-IT-QAT-Mobile'
$Base = 'https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers/resolve/main'

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

function Download-File([string]$Name) {
    Write-Host "Downloading $Name ..."
    $out = Join-Path $Dest $Name
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$Name" -ExtraArgs $Auth
    } catch {
        Write-Host 'Download failed. Retry, or set HF_TOKEN / huggingface-cli login if rate-limited.' -ForegroundColor Red
        Write-Host 'https://huggingface.co/google/gemma-4-E2B-it-qat-mobile-transformers' -ForegroundColor Red
        throw
    }
}

$SmallFiles = @(
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'chat_template.jinja',
    'preprocessor_config.json',
    'processor_config.json'
)

foreach ($f in $SmallFiles) {
    Download-File $f
}

Write-Host 'Downloading model.safetensors (~2.3GB) ...'
Download-File 'model.safetensors'

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
$size = (Get-ChildItem -LiteralPath $Dest -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($size / 1MB))
