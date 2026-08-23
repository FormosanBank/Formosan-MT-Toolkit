"""Checksum-bound Parquet companions for large canonical corpus CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from experiment_config import sha256_file

CACHE_SCHEMA_VERSION = 2


def cache_paths(csv_path: Path) -> tuple[Path, Path]:
    return csv_path.with_suffix(".parquet"), csv_path.with_suffix(".parquet.json")


def artifact_record(path: Path, digest: str | None = None) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": digest or sha256_file(path),
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def verified_artifact(path: Path, record: object) -> dict[str, object] | None:
    if not path.is_file() or not isinstance(record, dict):
        return None
    expected_hash = record.get("sha256")
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        return None
    stat = path.stat()
    identity = {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }
    expected_identity = {name: record.get(name) for name in identity}
    if identity == expected_identity:
        return record
    if sha256_file(path) != expected_hash:
        return None
    return artifact_record(path, expected_hash)


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_columnar_cache(frame: pd.DataFrame, csv_path: Path) -> Path:
    parquet_path, manifest_path = cache_paths(csv_path)
    temporary = parquet_path.with_suffix(".parquet.tmp")
    try:
        frame.to_parquet(temporary, index=False, compression="zstd")
    except ImportError as exc:
        raise SystemExit("Columnar corpus caching requires pyarrow; install requirements.txt") from exc
    temporary.replace(parquet_path)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": True,
        "canonical_csv": artifact_record(csv_path),
        "parquet": artifact_record(parquet_path),
        "rows": len(frame),
        "columns": list(frame.columns),
    }
    write_manifest(manifest_path, manifest)
    return parquet_path


def upgrade_manifest(
    csv_path: Path,
    parquet_path: Path,
    manifest_path: Path,
    manifest: dict[str, object],
) -> dict[str, object]:
    if manifest.get("schema_version") != 1:
        return manifest
    csv_hash = sha256_file(csv_path)
    parquet_hash = sha256_file(parquet_path)
    if manifest.get("canonical_csv_sha256") != csv_hash or manifest.get("parquet_sha256") != parquet_hash:
        raise SystemExit(f"Stale or corrupt columnar cache beside {csv_path}")
    upgraded = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "complete": manifest.get("complete"),
        "canonical_csv": artifact_record(csv_path, csv_hash),
        "parquet": artifact_record(parquet_path, parquet_hash),
        "rows": manifest.get("rows"),
        "columns": manifest.get("columns"),
    }
    write_manifest(manifest_path, upgraded)
    return upgraded


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
    manifest = upgrade_manifest(csv_path, parquet_path, manifest_path, manifest)
    csv_record = verified_artifact(csv_path, manifest.get("canonical_csv"))
    parquet_record = verified_artifact(parquet_path, manifest.get("parquet"))
    if (
        manifest.get("schema_version") != CACHE_SCHEMA_VERSION
        or manifest.get("complete") is not True
        or csv_record is None
        or parquet_record is None
    ):
        raise SystemExit(f"Stale or corrupt columnar cache beside {csv_path}")
    if csv_record != manifest["canonical_csv"] or parquet_record != manifest["parquet"]:
        manifest["canonical_csv"] = csv_record
        manifest["parquet"] = parquet_record
        write_manifest(manifest_path, manifest)
    try:
        frame = pd.read_parquet(parquet_path)
    except ImportError as exc:
        raise SystemExit("Reading the columnar corpus cache requires pyarrow") from exc
    if len(frame) != manifest.get("rows") or list(frame.columns) != manifest.get("columns"):
        raise SystemExit(f"Columnar cache schema mismatch beside {csv_path}")
    return frame
