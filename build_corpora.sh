#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper for the reproducible Python build pipeline.
#
# Examples:
#   ./build_corpora.sh
#   ./build_corpora.sh --languages ami,tay --skip-fetch --with-pivot --pivot-skip-translation

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
