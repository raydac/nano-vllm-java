# Download LiquidAI LFM2.5-2.6B Q4_K_M GGUF into this models/ directory.
# Dequantizes to ~10GB float32 - plan on -Xmx16g (default in .mvn/jvm.config).
# Usage:  .\models\download-lfm2.5-2.6b-gguf.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$File = 'LFM2.5-2.6B-Q4_K_M.gguf'
$Dest = Join-Path $ModelsRoot $File
$Base = 'https://huggingface.co/LiquidAI/LFM2.5-2.6B-GGUF/resolve/main'

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl

Write-Host "Downloading $File (~1.67GB) into $ModelsRoot ..."
& $Curl -L --fail --retry 3 -C - -o $Dest "$Base/$File"
if ($LASTEXITCODE -ne 0) {
    throw "Download failed for $File (exit $LASTEXITCODE)"
}

Write-Host "Installed to $Dest"
Get-Item -LiteralPath $Dest | Format-Table Name, Length, LastWriteTime
Write-Host "Hint: mvn -pl nano-vllm-java-samples -q exec:java -Dexec.mainClass=com.igormaznitsa.nanollvm.samples.Example `"-Dexec.args=$Dest`""
