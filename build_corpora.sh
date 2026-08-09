#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the reproducible Python build pipeline.
#
# Examples:
#   ./build_corpora.sh --corpus-name public_no_bible --public --with-pivot --exclude-bible
#   ./build_corpora.sh --corpus-name public_no_bible --public --languages ami,tay

if [[ $# -eq 0 ]]; then
  echo "Choose an isolated corpus build explicitly." >&2
  echo "Public build: ./build_corpora.sh --corpus-name public_no_bible --public --with-pivot --exclude-bible" >&2
  exit 2
fi

if [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "Python not found on PATH." >&2
  exit 1
fi

exec "$PY" scripts/local/build_mt_corpus.py "$@"
