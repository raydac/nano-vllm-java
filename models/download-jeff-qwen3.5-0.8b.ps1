# Download Jeff-Qwen3.5-0.8B (text) into models/Jeff-Qwen3.5-0.8B.
# Apache 2.0, ungated. Vision files and the Jeff readout head are not fetched.
# Optional HF_TOKEN if Hugging Face rate-limits you.
# Usage:  .\models\download-jeff-qwen3.5-0.8b.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'Jeff-Qwen3.5-0.8B'
$Base = 'https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B/resolve/main'

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
        Write-Host 'https://huggingface.co/jeff-legacy/Jeff-Qwen3.5-0.8B' -ForegroundColor Red
        throw
    }
}

foreach ($f in @('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')) {
    Download-File $f
}

Write-Host 'Downloading model.safetensors (~1.7GB) ...'
Download-File 'model.safetensors'

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
$size = (Get-ChildItem -LiteralPath $Dest -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("Total: {0:N1} MB" -f ($size / 1MB))
