# Download Tiny Voca English (config, f16 GGUF, pronunciation lexicon).
# Text-to-speech (text -> WAV). Not a chat model.
# https://huggingface.co/n4ze3m/tiny-voca
# Usage:  .\models\download-tiny-voca.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'tiny-voca'
$Base = 'https://huggingface.co/n4ze3m/tiny-voca/resolve/main'
New-Item -ItemType Directory -Force -Path (Join-Path $Dest 'phonemizer') | Out-Null

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
Get-File 'gguf/tiny-voca-f16.gguf' (Join-Path $Dest 'tiny-voca-f16.gguf')
Get-File 'phonemizer/lexicon.tsv' (Join-Path $Dest 'phonemizer/lexicon.tsv')
Write-Host "Installed to $Dest"
