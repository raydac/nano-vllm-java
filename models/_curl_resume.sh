# Hugging Face returns HTTP 416 when curl -C - resumes past EOF (local file already complete).
# --fail maps that to exit 22, which looks like a missing URL. Call curl_resume dest url [curl args...].
curl_resume() {
  local dest="$1"
  local url="$2"
  shift 2
  local http_code
  http_code="$(curl -L --retry 3 -C - -o "$dest" -w "%{http_code}" "$@" "$url" || true)"
  case "$http_code" in
    200|206|416) return 0 ;;
  esac
  echo "Download failed: $dest (HTTP ${http_code:-000})" >&2
  return 1
}
