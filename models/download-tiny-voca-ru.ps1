# Download Tiny Voca Russian (config + safetensors).
# Text-to-speech (text -> WAV). Not a chat model.
# https://huggingface.co/egnch/tiny-voca-ru
# Usage:  .\models\download-tiny-voca-ru.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'tiny-voca-ru'
$Base = 'https://huggingface.co/egnch/tiny-voca-ru/resolve/main'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$curl = Get-Curl
$headers = @()
if ($env:HF_TOKEN) { $headers = @('-H', "Authorization: Bearer $($env:HF_TOKEN)") }

function Get-File($remote, $dest) {
    Write-Host "Downloading $remote ..."
    & $curl -L --fail --retry 3 --retry-delay 2 @headers -o $dest "$Base/$remote"
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $remote" }
}

Get-File 'config.json' (Join-Path $Dest 'config.json')
Get-File 'model.safetensors' (Join-Path $Dest 'model.safetensors')
Write-Host "Installed to $Dest"
