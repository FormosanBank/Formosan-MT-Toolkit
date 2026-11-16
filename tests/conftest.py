"""Make the pipeline script directories importable from every test module."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for directory in ("scripts/local", "scripts/shared", "formosan_mt_experiments/scripts"):
    sys.path.insert(0, str(ROOT / directory))
