# Download ChristianAzinn gte-small Q2_K GGUF (smallest quant, ~25MB) into this models/ directory.
# Embedding / feature-extraction BERT GGUF — not a causal chat model for Example/HelloWorld.
# Source: https://huggingface.co/ChristianAzinn/gte-small-gguf
# Usage:  .\models\download-gte-small-gguf.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$File = 'gte-small.Q2_K.gguf'
$Dest = Join-Path $ModelsRoot $File
$Base = 'https://huggingface.co/ChristianAzinn/gte-small-gguf/resolve/main'

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl

Write-Host "Downloading $File (~25MB, smallest GTE-small GGUF) into $ModelsRoot ..."
& $Curl -L --fail --retry 3 -C - -o $Dest "$Base/$File"
if ($LASTEXITCODE -ne 0) {
    throw "Download failed for $File (exit $LASTEXITCODE)"
}

Write-Host "Installed to $Dest"
Get-Item $Dest | Format-Table Name, Length, LastWriteTime
Write-Host "Note: embedding model (BERT), context up to 512 tokens — not for LLM chat samples."
