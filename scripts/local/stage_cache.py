"""Content-addressed stage cache for local corpus builds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipeline_common import atomic_write_json, sha256_file, stable_json_hash, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def stage_key(name: str, payload: dict[str, object], scripts: list[Path]) -> str:
    return stable_json_hash(
        {
            "stage": name,
            "inputs": payload,
            "scripts": {
                str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
                for path in scripts
            },
            "runtime": {
                "python": list(sys.version_info[:3]),
                "requirements_sha256": sha256_file(PROJECT_ROOT / "requirements.txt"),
            },
        }
    )


def load_stage_cache(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"schema_version": 1, "stages": {}}
    if payload.get("schema_version") != 1 or not isinstance(payload.get("stages"), dict):
        return {"schema_version": 1, "stages": {}}
    return payload


def cached_stage_valid(
    root: Path,
    cache: dict[str, object],
    name: str,
    key: str,
) -> bool:
    stages = cache.get("stages")
    record = stages.get(name, {}) if isinstance(stages, dict) else {}
    if not isinstance(record, dict) or record.get("key") != key:
        return False
    outputs = record.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        return False
    for relative, expected_hash in outputs.items():
        path = root / str(relative)
        if not path.is_file() or sha256_file(path) != expected_hash:
            return False
    return True


def record_cached_stage(
    root: Path,
    cache_path: Path,
    cache: dict[str, object],
    name: str,
    key: str,
    outputs: list[Path],
    label: str,
) -> None:
    records = {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(set(outputs))
        if path.is_file()
    }
    if not records:
        raise SystemExit(f"Cannot cache {label} {name}; it produced no outputs")
    stages = cache.setdefault("stages", {})
    if not isinstance(stages, dict):
        raise SystemExit(f"Invalid stage cache structure at {cache_path}")
    stages[name] = {
        "key": key,
        "outputs": records,
        "created_at": utc_now(),
    }
    atomic_write_json(cache_path, cache)


def file_inventory(paths: list[Path], root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(set(paths))
        if path.is_file()
    ]
