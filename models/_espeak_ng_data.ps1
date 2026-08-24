# Copy espeak-ng lang/voices plus dictsource (*_list / *_rules) next to a Piper voice.
# GPL data, not shipped in the library JAR.
# Usage: Install-EspeakNgData -ModelDir $Dest -Curl $Curl

function Install-EspeakNgData {
    param(
        [Parameter(Mandatory = $true)][string]$ModelDir,
        [Parameter(Mandatory = $true)][string]$Curl
    )
    $dest = Join-Path $ModelDir 'espeak-ng-data'
    $lang = Join-Path $dest 'lang'
    $dictsource = Join-Path $dest 'dictsource'
    if ((Test-Path -LiteralPath $lang) -and (Test-Path -LiteralPath $dictsource)) {
        Write-Host 'espeak-ng-data already has lang/ and dictsource/'
        Copy-CompiledEspeakFromSystem -Dest $dest
        return
    }
    $url = 'https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.1.tar.gz'
    Write-Host 'Downloading espeak-ng-data (GPL data, not shipped in the library JAR) ...'
    $tmp = Join-Path $env:TEMP ('espeak-ng-' + [guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Force -Path $tmp | Out-Null
    try {
        $archive = Join-Path $tmp 'espeak-ng.tar.gz'
        Invoke-CurlResume -Curl $Curl -OutFile $archive -Url $url -ExtraArgs @()
        tar -xzf $archive -C $tmp
        $srcData = Get-ChildItem -Path $tmp -Directory -Recurse -Filter 'espeak-ng-data' | Select-Object -First 1
        $srcDict = Get-ChildItem -Path $tmp -Directory -Recurse -Filter 'dictsource' |
            Where-Object { $_.FullName -notmatch '[\\/]espeak-ng-data[\\/]' } |
            Select-Object -First 1
        if (-not $srcData) {
            throw 'espeak-ng archive did not contain espeak-ng-data'
        }
        if (-not $srcDict) {
            throw 'espeak-ng archive did not contain dictsource'
        }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null
        if (-not (Test-Path -LiteralPath $lang)) {
            Copy-Item -Path (Join-Path $srcData.FullName '*') -Destination $dest -Recurse -Force
        }
        if (-not (Test-Path -LiteralPath $dictsource)) {
            Copy-Item -LiteralPath $srcDict.FullName -Destination $dictsource -Recurse
        }
    } finally {
        Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
    }
    Copy-CompiledEspeakFromSystem -Dest $dest
}

function Copy-CompiledEspeakFromSystem {
    param([Parameter(Mandatory = $true)][string]$Dest)
    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $candidates = @(
        (Join-Path ${env:ProgramFiles} 'eSpeak NG\espeak-ng-data'),
        (Join-Path ${env:ProgramFiles(x86)} 'eSpeak NG\espeak-ng-data')
    )
    foreach ($src in $candidates) {
        if (-not (Test-Path -LiteralPath (Join-Path $src 'phontab'))) {
            continue
        }
        foreach ($file in @('phontab', 'phondata', 'phonindex')) {
            $from = Join-Path $src $file
            $to = Join-Path $Dest $file
            if ((Test-Path -LiteralPath $from) -and -not (Test-Path -LiteralPath $to)) {
                Copy-Item -LiteralPath $from -Destination $to
            }
        }
        Get-ChildItem -LiteralPath $src -Filter '*_dict' | ForEach-Object {
            $to = Join-Path $Dest $_.Name
            if (-not (Test-Path -LiteralPath $to)) {
                Copy-Item -LiteralPath $_.FullName -Destination $to
            }
        }
        Write-Host "Copied compiled espeak-ng dictionaries from $src"
        return
    }
}
