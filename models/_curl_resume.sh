# Hugging Face returns HTTP 416 when a resume starts past EOF (the local file is already complete).
# --fail maps that to exit 22, which looks like a missing URL.
# A 401/403 body must not stay in the destination: the next resume would append the real file
# after the error text. Call curl_resume dest url [curl args...].

is_hf_error_body() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  local prefix
  prefix="$(head -c 180 "$file" 2>/dev/null || true)"
  case "$prefix" in
    "Access to model "*|*"Please log in"*|*"Entry not found"*|"<!DOCTYPE"*|"<html"*|'{"error":'*)
      return 0
      ;;
  esac
  return 1
}

curl_resume() {
  local dest="$1"
  local url="$2"
  shift 2
  if is_hf_error_body "$dest"; then
    rm -f "$dest"
  fi

  local offset=0
  if [[ -f "$dest" ]]; then
    offset="$(wc -c < "$dest" | tr -d ' ')"
  fi

  local body
  body="$(mktemp "${dest}.part.XXXXXX")"
  local http_code=""
  local curl_status=0
  if [[ "$offset" -gt 0 ]]; then
    http_code="$(curl -L --retry 3 -C "$offset" -o "$body" -w "%{http_code}" "$@" "$url")" || curl_status=$?
  else
    http_code="$(curl -L --retry 3 -o "$body" -w "%{http_code}" "$@" "$url")" || curl_status=$?
  fi

  case "$http_code" in
    416)
      rm -f "$body"
      return 0
      ;;
    200)
      if is_hf_error_body "$body"; then
        rm -f "$body"
        echo "Download failed: $dest (gated or error page)" >&2
        return 1
      fi
      if [[ "$curl_status" -eq 0 ]]; then
        mv "$body" "$dest"
        return 0
      fi
      if [[ "$offset" -eq 0 ]]; then
        mv "$body" "$dest"
      else
        rm -f "$body"
      fi
      echo "Download failed: $dest (HTTP 200, incomplete)" >&2
      return 1
      ;;
    206)
      if is_hf_error_body "$body"; then
        rm -f "$body"
        echo "Download failed: $dest (gated or error page)" >&2
        return 1
      fi
      cat "$body" >> "$dest"
      rm -f "$body"
      if [[ "$curl_status" -eq 0 ]]; then
        return 0
      fi
      echo "Download failed: $dest (HTTP 206, incomplete)" >&2
      return 1
      ;;
  esac

  rm -f "$body"
  echo "Download failed: $dest (HTTP ${http_code:-000})" >&2
  return 1
}
