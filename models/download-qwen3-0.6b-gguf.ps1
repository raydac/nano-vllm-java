# Download Qwen3-0.6B Q8_0 GGUF (general.architecture=qwen3) into this models/ directory.
# Apache 2.0, not gated. The safetensors folder is models/download-qwen3-0.6b.ps1.
# Usage:  .\models\download-qwen3-0.6b-gguf.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$File = 'Qwen3-0.6B-Q8_0.gguf'
$Dest = Join-Path $ModelsRoot $File
$Base = 'https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main'

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

Write-Host "Downloading $File (~570MB) into $ModelsRoot ..."
Invoke-CurlResume -Curl $Curl -OutFile $Dest -Url "$Base/$File"

Write-Host "Installed to $Dest"
Get-Item -LiteralPath $Dest | Format-Table Name, Length, LastWriteTime
Write-Host "Hint: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Example `"-Dexec.args=$Dest`""
