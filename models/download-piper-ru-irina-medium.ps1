# Download Piper Russian Irina medium (ONNX + sidecar) and espeak-ng-data.
# Text-to-speech (text -> WAV). Not a chat model. Not ONNX Runtime.
# Voice: https://huggingface.co/rhasspy/piper-voices
# Usage (PowerShell):  .\models\download-piper-ru-irina-medium.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'piper-ru-irina-medium'
$Base = 'https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/irina/medium'

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
. (Join-Path $ModelsRoot '_espeak_ng_data.ps1')
$Auth = Get-HfAuthHeaders

function Download-HfFile([string]$Name) {
    Write-Host "Downloading $Name ..."
    $out = Join-Path $Dest $Name
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$Name" -ExtraArgs $Auth
    } catch {
        Write-Host 'Download failed. Retry, or set HF_TOKEN / huggingface-cli login if rate-limited.' -ForegroundColor Red
        Write-Host 'https://huggingface.co/rhasspy/piper-voices' -ForegroundColor Red
        throw
    }
}

Write-Host 'Downloading Piper Irina medium ONNX + sidecar ...'
Download-HfFile 'ru_RU-irina-medium.onnx'
Download-HfFile 'ru_RU-irina-medium.onnx.json'

Install-EspeakNgData -ModelDir $Dest -Curl $Curl

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Write-Host 'Load: LlmModelFactory.open(path).optionalData(LlmOptionalData.ESPEAK_DATA, espeakDir).make()'
