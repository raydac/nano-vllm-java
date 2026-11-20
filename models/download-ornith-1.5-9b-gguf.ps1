# Download the smallest Ornith-1.5-9B GGUF into this models/ directory.
# AtomicChat/Ornith-1.5-9B-GGUF, MIT, not gated. general.architecture=qwen35.
# AD-IQ2_XXS-IQ1_M is the smallest file (~2.8GB).
# Usage:  .\models\download-ornith-1.5-9b-gguf.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$File = 'Ornith-1.5-9B-AD-IQ2_XXS-IQ1_M.gguf'
$Dest = Join-Path $ModelsRoot $File
$Base = 'https://huggingface.co/AtomicChat/Ornith-1.5-9B-GGUF/resolve/main'

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

Write-Host "Downloading $File (~2.8GB) into $ModelsRoot ..."
Invoke-CurlResume -Curl $Curl -OutFile $Dest -Url "$Base/$File"

Write-Host "Installed to $Dest"
Get-Item -LiteralPath $Dest | Format-Table Name, Length, LastWriteTime
Write-Host "This file is general.architecture=qwen35. Load it with LlmModelFactory.make as Qwen3.5 text chat."
