#!/usr/bin/env bash
set -euo pipefail

# NOTE on Seediq/Truku:
# Both Seediq and Truku use ISO 639-3 code 'trv'. To avoid collisions
# (same download dir/output names), this script runs TRV ONCE.
# If you need separate labeling later, we can post-process or adjust the tools.

# Pick a Python executable
if command -v python >/dev/null 2>&1; then PY=python
elif command -v python3 >/dev/null 2>&1; then PY=python3
else
  echo "Python not found on PATH." >&2
  exit 1
fi

# Name:Code pairs (for nicer logs). Truku omitted to prevent duplicate 'trv' runs.
LANGS=(
  "Amis:ami"
  "Atayal:tay"
  "Bunun:bnn"
  "Kanakanavu:xnb"
  "Kavalan:ckv"
  "Paiwan:pwn"
  "Puyuma:pyu"
  "Rukai:dru"
  "Saaroa:sxr"
  "Saisiyat:xsy"
  "Sakizaya:szy"
  "Seediq:trv"
  "Thao:ssf"
  "Tsou:tsu"
  "Yami/Tao:tao"
)

run_for_lang() {
  local name="$1"
  local code="$2"

  echo "============================================================"
  echo ">> ${name} (${code})"
  echo "------------------------------------------------------------"

  # 1) fetch
  echo "[1/4] Fetching XML for ${code}..."
  "$PY" scripts/local/fetch_xml.py --src-lang "$code"

  # 2) clean
  echo "[2/4] Cleaning XML for ${code}..."
  "$PY" scripts/local/clean_xml.py --src-lang "$code"

  # 3) make corpus (target: chinese)
  echo "[3/4] Building Chinese corpus for ${code}..."
  "$PY" scripts/local/make_corpus.py --xml-dir "downloaded_${code}" --target chinese --out "${code}_zh.csv"

  # 4) make corpus (target: english)
  echo "[4/4] Building English corpus for ${code}..."
  "$PY" scripts/local/make_corpus.py --xml-dir "downloaded_${code}" --target english --out "${code}_en.csv"

  echo "✔ Done: ${name} (${code})"
  echo
}

main() {
  echo "Starting corpus build…"
  echo "Working directory: $(pwd)"
  echo

  for entry in "${LANGS[@]}"; do
    IFS=":" read -r NAME CODE <<<"$entry"
    run_for_lang "$NAME" "$CODE"
  done

  echo "All corpora complete."
  echo "NOTE: Truku shares code 'trv' with Seediq and was not run separately."
}

main "$@"
