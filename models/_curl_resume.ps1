# Hugging Face returns HTTP 416 when curl -C - resumes past EOF (local file already complete).
# --fail maps that to exit 22, which looks like a missing URL.
function Invoke-CurlResume {
    param(
        [Parameter(Mandatory = $true)][string]$Curl,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [Parameter(Mandatory = $true)][string]$Url,
        [string[]]$ExtraArgs = @()
    )
    $httpCode = (& $Curl -L --retry 3 -C - -o $OutFile -w '%{http_code}' @ExtraArgs $Url | Select-Object -Last 1)
    if ($null -ne $httpCode) {
        $httpCode = $httpCode.ToString().Trim()
    }
    if ($httpCode -eq '200' -or $httpCode -eq '206' -or $httpCode -eq '416') {
        return
    }
    throw ("Download failed for {0} (HTTP {1})" -f $OutFile, $httpCode)
}
