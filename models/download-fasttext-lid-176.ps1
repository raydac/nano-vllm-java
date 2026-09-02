# Download Meta fastText lid.176.bin (language identification, 176 languages).
# Denser than lid.176.ftz — slightly more accurate (and faster) per Meta.
# Text classification (text -> labels). Not a chat / embedding / Whisper / Piper model.
# Docs: https://fasttext.cc/docs/en/language-identification.html
# Usage (PowerShell):  .\models\download-fasttext-lid-176.ps1
# Or double-click / run: models\download-fasttext-lid-176.cmd

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'fasttext-lid-176'
$Url = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin'
$File = 'lid.176.bin'

New-Item -ItemType Directory -Force -Path $Dest | Out-Null

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

Write-Host "Downloading $File (~126MB) ..."
$out = Join-Path $Dest $File
try {
    Invoke-CurlResume -Curl $Curl -OutFile $out -Url $Url
}
catch {
    Write-Host 'Download failed. See https://fasttext.cc/docs/en/language-identification.html' -ForegroundColor Red
    throw
}

Write-Host "Installed to $Dest"
Get-ChildItem -LiteralPath $Dest | Format-Table Name, Length
Write-Host 'Load: LlmModelFactory.make(Path.of("models/fasttext-lid-176"));'
Write-Host '      llm.generate(LlmInText.of("Bonjour"), LlmModality.LABELS);'
Write-Host 'Try: mvn -pl nano-vllm-java-samples -q exec:java "-Dexec.mainClass=com.igormaznitsa.nanollvm.samples.LanguageIdHelloWorld"'
