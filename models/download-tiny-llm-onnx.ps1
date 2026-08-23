# Download onnx-community Tiny-LLM-ONNX (Llama, ~10M) + tokenizer from arnir0/Tiny-LLM.
# Usage (PowerShell):  .\models\download-tiny-llm-onnx.ps1
# Or double-click / run: models\download-tiny-llm-onnx.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'Tiny-LLM-ONNX'
$OnnxDir = Join-Path $Dest 'onnx'
$OnnxBase = 'https://huggingface.co/onnx-community/Tiny-LLM-ONNX/resolve/main'
$TokBase = 'https://huggingface.co/arnir0/Tiny-LLM/resolve/main'

New-Item -ItemType Directory -Force -Path $OnnxDir | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

Write-Host 'Downloading config / generation_config ...'
foreach ($pair in @(
    @{ Name = 'config.json'; Url = "$OnnxBase/config.json" },
    @{ Name = 'generation_config.json'; Url = "$OnnxBase/generation_config.json" }
)) {
    $out = Join-Path $Dest $pair.Name
    Invoke-CurlResume -Curl $Curl -OutFile $out -Url $pair.Url
}

Write-Host 'Downloading tokenizer from arnir0/Tiny-LLM ...'
foreach ($f in @(
    'tokenizer.json',
    'tokenizer_config.json',
    'special_tokens_map.json',
    'tokenizer.model'
)) {
    $out = Join-Path $Dest $f
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$TokBase/$f"
    } catch {
        Write-Host "Optional tokenizer file skipped: $f"
        if (Test-Path -LiteralPath $out) {
            Remove-Item -LiteralPath $out -Force -ErrorAction SilentlyContinue
        }
    }
}

Write-Host 'Downloading onnx\model.onnx (fp32) ...'
$OnnxFile = Join-Path $OnnxDir 'model.onnx'
Invoke-CurlResume -Curl $Curl -OutFile $OnnxFile -Url "$OnnxBase/onnx/model.onnx"

Write-Host "Installed to $Dest"
Get-ChildItem $Dest | Format-Table Name, Length, LastWriteTime
Get-ChildItem $OnnxDir | Format-Table Name, Length, LastWriteTime
