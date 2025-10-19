#!/usr/bin/env bash
set -euo pipefail

# BIG HARNESS THAT RUNS THROUGH ALL LANGUAGES
# 1. FETCH XML
# 2. CLEAN XML
# 3. MAKE CORPUS (TARGET: CHINESE)
# 4. MAKE CORPUS (TARGET: ENGLISH)
# 5. FILTER AND SPLIT CORPUS (FORMOSAN <-> CHINESE)
# 6. FILTER AND SPLIT CORPUS (FORMOSAN <-> ENGLISH)

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
  "$PY" scripts/local/fetch_xml.py --src-lang "$code" --public

  # 2) clean
  echo "[2/4] Cleaning XML for ${code}..."
  "$PY" scripts/local/clean_xml.py --src-lang "$code"

  # 3) make corpus (target: chinese)
  echo "[3/4] Building Chinese corpus for ${code}..."
  "$PY" scripts/local/make_corpus.py --xml-dir "downloaded_${code}" --target chinese --out "raw_corpora/${code}_zh.csv"

  # 4) make corpus (target: english)
  echo "[4/4] Building English corpus for ${code}..."
  "$PY" scripts/local/make_corpus.py --xml-dir "downloaded_${code}" --target english --out "raw_corpora/${code}_en.csv"

  echo "✔ Done: ${name} (${code})"
  echo

  # 5) filter and split corpus for MT training (formosan <-> chinese)
  echo "[5/6] Filtering and splitting Chinese corpus for MT training for ${code}..."
  "$PY" scripts/local/filter_split_corpus.py --input "raw_corpora/${code}_zh.csv" --output "processed_corpora/${code}_zh_processed.csv" --workers 32

  # 6) filter and split corpus for MT training (formosan <-> english)
  echo "[6/6] Filtering and splitting English corpus for MT training for ${code}..."
  "$PY" scripts/local/filter_split_corpus.py --input "raw_corpora/${code}_en.csv" --output "processed_corpora/${code}_en_processed.csv" --workers 32
}

main() {
  echo "Starting corpus build…"
  echo "Working directory: $(pwd)"
  echo

  # Create output directories
  mkdir -p raw_corpora processed_corpora

  for entry in "${LANGS[@]}"; do
    IFS=":" read -r NAME CODE <<<"$entry"
    run_for_lang "$NAME" "$CODE"
  done

  echo "All corpora complete."
  echo "NOTE: Truku shares code 'trv' with Seediq and was not run separately."
}

main "$@"
