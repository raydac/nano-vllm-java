# Copy espeak-ng lang/voices plus dictsource (*_list / *_rules) next to a Piper voice.
# Also copy compiled phontab/phondata/*_dict from a system espeak-ng-data if present
# (Russian lexical stress lives in ru_dict). GPL data, not shipped in the library JAR.
# Call after sourcing _curl_resume.sh.
# Usage: install_espeak_ng_data MODEL_DIR
ESPEAK_ARCHIVE_URL="${ESPEAK_ARCHIVE_URL:-https://github.com/espeak-ng/espeak-ng/archive/refs/tags/1.51.1.tar.gz}"

espeak_ng_data_complete() {
  local dest="$1/espeak-ng-data"
  [[ -d "$dest/lang" && -d "$dest/dictsource" ]]
}

copy_compiled_espeak_from_system() {
  local dest="$1/espeak-ng-data"
  mkdir -p "$dest"
  local src
  for src in \
    /usr/lib/x86_64-linux-gnu/espeak-ng-data \
    /usr/lib/espeak-ng-data \
    /usr/share/espeak-ng-data
  do
    if [[ -f "$src/phontab" ]]; then
      local file
      for file in phontab phondata phonindex; do
        if [[ -f "$src/$file" && ! -f "$dest/$file" ]]; then
          cp "$src/$file" "$dest/"
        fi
      done
      local dict
      for dict in "$src"/*_dict; do
        [[ -f "$dict" ]] || continue
        if [[ ! -f "$dest/$(basename "$dict")" ]]; then
          cp "$dict" "$dest/"
        fi
      done
      echo "Copied compiled espeak-ng dictionaries from $src"
      return 0
    fi
  done
  return 0
}

install_espeak_ng_data() {
  local model_dir="$1"
  local dest="$model_dir/espeak-ng-data"
  if espeak_ng_data_complete "$model_dir"; then
    echo "espeak-ng-data already has lang/ and dictsource/"
    copy_compiled_espeak_from_system "$model_dir"
    return 0
  fi
  echo "Downloading espeak-ng-data (GPL data, not shipped in the library JAR) ..."
  local tmp
  tmp="$(mktemp -d)"
  if ! curl_resume "$tmp/espeak-ng.tar.gz" "$ESPEAK_ARCHIVE_URL"; then
    echo "Failed to download espeak-ng-data from $ESPEAK_ARCHIVE_URL" >&2
    rm -rf "$tmp"
    copy_compiled_espeak_from_system "$model_dir"
    return 1
  fi
  tar -xzf "$tmp/espeak-ng.tar.gz" -C "$tmp"
  local src_data src_dict
  src_data="$(find "$tmp" -type d -name espeak-ng-data | head -n 1)"
  src_dict="$(find "$tmp" -type d -name dictsource ! -path '*/espeak-ng-data/*' | head -n 1)"
  if [[ -z "$src_data" ]]; then
    echo "espeak-ng archive did not contain espeak-ng-data" >&2
    rm -rf "$tmp"
    return 1
  fi
  if [[ -z "$src_dict" ]]; then
    echo "espeak-ng archive did not contain dictsource" >&2
    rm -rf "$tmp"
    return 1
  fi
  mkdir -p "$dest"
  if [[ ! -d "$dest/lang" ]]; then
    cp -a "$src_data"/. "$dest"/
  fi
  if [[ ! -d "$dest/dictsource" ]]; then
    cp -a "$src_dict" "$dest/dictsource"
  fi
  rm -rf "$tmp"
  copy_compiled_espeak_from_system "$model_dir"
}
