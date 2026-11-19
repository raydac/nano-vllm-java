# Download EmbeddingGemma-300M into models/EmbeddingGemma-300M.
# Requires the Gemma license on Hugging Face + HF_TOKEN (or huggingface-cli login).
# Usage:  .\models\download-embeddinggemma-300m.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'EmbeddingGemma-300M'
$Base = 'https://huggingface.co/google/embeddinggemma-300m/resolve/main'

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
    $out = Join-Path $Dest ($Name -replace '/', '\')
    $parent = Split-Path -Parent $out
    if ($parent) {
        New-Item -ItemType Directory -Force -Path $parent | Out-Null
    }
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$Name" -ExtraArgs $Auth
    } catch {
        Write-Host 'Download failed. EmbeddingGemma is gated: accept the license at' -ForegroundColor Red
        Write-Host 'https://huggingface.co/google/embeddinggemma-300m' -ForegroundColor Red
        Write-Host 'Then: huggingface-cli login  (or set HF_TOKEN)' -ForegroundColor Red
        throw
    }
}

$Files = @(
    'config.json',
    'generation_config.json',
    'tokenizer.json',
    'tokenizer_config.json',
    'tokenizer.model',
    'special_tokens_map.json',
    'added_tokens.json',
    'modules.json',
    'config_sentence_transformers.json',
    'sentence_bert_config.json',
    '1_Pooling/config.json',
    '2_Dense/config.json',
    '2_Dense/model.safetensors',
    '3_Dense/config.json',
    '3_Dense/model.safetensors',
    'model.safetensors'
)

foreach ($f in $Files) {
    Download-File $f
}

Write-Host "Installed to $Dest"
Get-ChildItem -LiteralPath $Dest -Recurse -File | Format-Table FullName, Length
$size = (Get-ChildItem -LiteralPath $Dest -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ("{0:N1} GB" -f ($size / 1GB))
