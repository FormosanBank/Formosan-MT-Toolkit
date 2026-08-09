"""Checksum-bound Parquet companions for large canonical corpus CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from experiment_config import sha256_file


def cache_paths(csv_path: Path) -> tuple[Path, Path]:
    return csv_path.with_suffix(".parquet"), csv_path.with_suffix(".parquet.json")


def write_columnar_cache(frame: pd.DataFrame, csv_path: Path) -> Path:
    parquet_path, manifest_path = cache_paths(csv_path)
    temporary = parquet_path.with_suffix(".parquet.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        raise SystemExit("Columnar corpus caching requires pyarrow; install requirements.txt") from exc
    temporary.replace(parquet_path)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "canonical_csv": str(csv_path.resolve()),
        "canonical_csv_sha256": sha256_file(csv_path),
        "parquet": str(parquet_path.resolve()),
        "parquet_sha256": sha256_file(parquet_path),
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return parquet_path


def read_csv_or_columnar(csv_path: Path, **csv_options) -> pd.DataFrame:
    parquet_path, manifest_path = cache_paths(csv_path)
    if not parquet_path.exists() and not manifest_path.exists():
        return pd.read_csv(csv_path, **csv_options)
    if not parquet_path.is_file() or not manifest_path.is_file():
        raise SystemExit(f"Incomplete columnar cache beside {csv_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Malformed columnar cache manifest {manifest_path}: {exc}") from exc
    if (
        manifest.get("complete") is not True
        or manifest.get("canonical_csv_sha256") != sha256_file(csv_path)
        or manifest.get("parquet_sha256") != sha256_file(parquet_path)
    ):
        raise SystemExit(f"Stale or corrupt columnar cache beside {csv_path}")
    try:
        frame = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise SystemExit("Reading the columnar corpus cache requires pyarrow") from exc
    if len(frame) != manifest.get("rows") or list(frame.columns) != manifest.get("columns"):
        raise SystemExit(f"Columnar cache schema mismatch beside {csv_path}")
    return frame
