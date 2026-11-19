# Hugging Face returns HTTP 416 when a resume starts past EOF (the local file is already complete).
# --fail maps that to exit 22, which looks like a missing URL.
# A 401/403 body must not stay in the destination: the next resume would append the real file
# after the error text.

function Test-HfErrorBody {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $buf = New-Object byte[] 180
        $read = $stream.Read($buf, 0, $buf.Length)
        $text = [System.Text.Encoding]::UTF8.GetString($buf, 0, $read)
    } finally {
        $stream.Dispose()
    }
    return $text.StartsWith('Access to model ') `
        -or $text.Contains('Please log in') `
        -or $text.Contains('Entry not found') `
        -or $text.StartsWith('<!DOCTYPE') `
        -or $text.StartsWith('<html') `
        -or $text.StartsWith('{"error":')
}

function Invoke-CurlResume {
    param(
        [Parameter(Mandatory = $true)][string]$Curl,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][string]$Url,
        [string[]]$ExtraArgs = @()
    )
    if (Test-HfErrorBody $OutFile) {
        Remove-Item -LiteralPath $OutFile -Force
    }
    $offset = 0L
    if (Test-Path -LiteralPath $OutFile) {
        $offset = (Get-Item -LiteralPath $OutFile).Length
    }
    $body = "$OutFile.part.$([guid]::NewGuid().ToString('N').Substring(0, 8))"
    try {
        $curlArgs = @('-L', '--retry', '3', '-o', $body, '-w', '%{http_code}')
        if ($offset -gt 0) {
            $curlArgs += @('-C', "$offset")
        }
        if ($ExtraArgs.Count -gt 0) {
            $curlArgs += $ExtraArgs
        }
        $curlArgs += $Url
        $httpCode = (& $Curl @curlArgs | Select-Object -Last 1)
        $curlStatus = $LASTEXITCODE
        if ($null -ne $httpCode) {
            $httpCode = $httpCode.ToString().Trim()
        }
        if ($httpCode -eq '416') {
            return
        }
        if ($httpCode -eq '200' -or $httpCode -eq '206') {
            if (Test-HfErrorBody $body) {
                throw ("Download failed for {0} (gated or error page)" -f $OutFile)
            }
            if ($httpCode -eq '200' -and $curlStatus -eq 0) {
                Move-Item -LiteralPath $body -Destination $OutFile -Force
                return
            }
            if ($httpCode -eq '206') {
                $inputStream = [System.IO.File]::OpenRead($body)
                $outputStream = [System.IO.File]::Open($OutFile, [System.IO.FileMode]::Append)
                try {
                    $inputStream.CopyTo($outputStream)
                } finally {
                    $inputStream.Dispose()
                    $outputStream.Dispose()
                }
                if ($curlStatus -eq 0) {
                    return
                }
                throw ("Download failed for {0} (HTTP 206, incomplete)" -f $OutFile)
            }
            if ($offset -eq 0) {
                Move-Item -LiteralPath $body -Destination $OutFile -Force
            }
            throw ("Download failed for {0} (HTTP 200, incomplete)" -f $OutFile)
        }
        throw ("Download failed for {0} (HTTP {1})" -f $OutFile, $httpCode)
    } finally {
        if (Test-Path -LiteralPath $body) {
            Remove-Item -LiteralPath $body -Force
        }
    }
}
