# Download the seven YOLO26 nano ONNX files into models\YOLO26n\<task>.
# Image models, not chat. Source: https://github.com/ultralytics/assets/releases/tag/v8.4.0
# Usage: .\models\download-yolo26n.ps1

$ErrorActionPreference = 'Stop'

$ModelsRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$Dest = Join-Path $ModelsRoot 'YOLO26n'
$Base = 'https://github.com/ultralytics/assets/releases/download/v8.4.0'

function Get-Curl {
    $cmd = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }
    throw 'curl.exe not found. Install curl or use Windows 10+ (curl is included).'
}

$Curl = Get-Curl
. (Join-Path $ModelsRoot '_curl_resume.ps1')

function Download-Onnx([string]$Folder, [string]$File) {
    $dir = Join-Path $Dest $Folder
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    Write-Host "Downloading $File ..."
    $out = Join-Path $dir $File
    try {
        Invoke-CurlResume -Curl $Curl -OutFile $out -Url "$Base/$File"
    } catch {
        Write-Host 'Download failed.' -ForegroundColor Red
        Write-Host 'https://github.com/ultralytics/assets/releases/tag/v8.4.0' -ForegroundColor Red
        throw
    }
}

Download-Onnx 'detect' 'yolo26n.onnx'
Download-Onnx 'seg' 'yolo26n-seg.onnx'
Download-Onnx 'sem' 'yolo26n-sem.onnx'
Download-Onnx 'depth' 'yolo26n-depth.onnx'
Download-Onnx 'cls' 'yolo26n-cls.onnx'
Download-Onnx 'pose' 'yolo26n-pose.onnx'
Download-Onnx 'obb' 'yolo26n-obb.onnx'

function Require-Classes([string]$Folder, [int]$Count) {
    $sidecar = Join-Path (Join-Path $Dest $Folder) 'classes.txt'
    if (-not (Test-Path -LiteralPath $sidecar -PathType Leaf)) {
        Write-Host "Missing $sidecar ($Count names, one per line)." -ForegroundColor Red
        Write-Host 'Those lists are in the git tree. This script downloads only the ONNX file.' -ForegroundColor Red
        exit 1
    }
}

Require-Classes 'detect' 80
Require-Classes 'seg' 80
Require-Classes 'sem' 19
Require-Classes 'cls' 1000
Require-Classes 'pose' 1
Require-Classes 'obb' 15

Write-Host "Installed to $Dest"
Get-ChildItem $Dest -Recurse -Filter *.onnx | Format-Table FullName, Length
Write-Host 'Class names are classes.txt in the same folder as the ONNX file. Depth has none.'
Write-Host 'Those lists ship with the repository. Replace classes.txt when a head was trained on another list.'
