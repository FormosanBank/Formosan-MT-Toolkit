"""Content-addressed stage cache for local corpus builds."""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

from pipeline_common import atomic_write_json, sha256_file, stable_json_hash, utc_now

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_SCHEMA_VERSION = 2
HASH_INDEX_SCHEMA_VERSION = 1
_HASH_INDEX_LOCK = threading.RLock()


def _empty_cache() -> dict[str, object]:
    return {"schema_version": CACHE_SCHEMA_VERSION, "stages": {}}


def _file_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
    }


def _hash_index_path(root: Path) -> Path:
    return root / ".stage_cache" / "artifacts.json"


def _load_hash_index(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {"schema_version": HASH_INDEX_SCHEMA_VERSION, "files": {}}
    if (
        value.get("schema_version") != HASH_INDEX_SCHEMA_VERSION
        or not isinstance(value.get("files"), dict)
    ):
        return {"schema_version": HASH_INDEX_SCHEMA_VERSION, "files": {}}
    return value


def _upgrade_cache(path: Path, payload: dict[str, object]) -> dict[str, object]:
    """Verify and upgrade v1 cache records without rerunning valid stages."""
    if payload.get("schema_version") != 1 or not isinstance(payload.get("stages"), dict):
        return _empty_cache()
    root = path.parent.parent
    for stage in payload["stages"].values():
        if not isinstance(stage, dict) or not isinstance(stage.get("outputs"), dict):
            continue
        upgraded: dict[str, object] = {}
        for relative, expected_hash in stage["outputs"].items():
            output = root / str(relative)
            if not output.is_file() or not isinstance(expected_hash, str):
                upgraded[str(relative)] = {"sha256": expected_hash}
                continue
            current_hash = cached_sha256(output, root)
            if current_hash == expected_hash:
                upgraded[str(relative)] = {
                    **_file_stat(output),
                    "sha256": current_hash,
                }
            else:
                upgraded[str(relative)] = {"sha256": expected_hash}
        stage["outputs"] = upgraded
    payload["schema_version"] = CACHE_SCHEMA_VERSION
    atomic_write_json(path, payload)
    return payload


def _path_key(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path.resolve())


def cached_sha256(path: Path, root: Path) -> str:
    """Hash a file once while its identity and metadata remain unchanged."""
    index_path = _hash_index_path(root)
    key = _path_key(path, root)
    with _HASH_INDEX_LOCK:
        index = _load_hash_index(index_path)
        files = index["files"]
        assert isinstance(files, dict)
        before = _file_stat(path)
        record = files.get(key)
        if isinstance(record, dict) and all(record.get(name) == value for name, value in before.items()):
            digest = record.get("sha256")
            if isinstance(digest, str) and len(digest) == 64:
                return digest

        digest = sha256_file(path)
        after = _file_stat(path)
        if before != after:
            raise RuntimeError(f"File changed while hashing: {path}")
        files[key] = {**after, "sha256": digest}
        atomic_write_json(index_path, index)
        return digest


def artifact_record(path: Path, root: Path) -> dict[str, object]:
    return {**_file_stat(path), "sha256": cached_sha256(path, root)}


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
        return _empty_cache()
    if payload.get("schema_version") == 1:
        return _upgrade_cache(path, payload)
    if payload.get("schema_version") != CACHE_SCHEMA_VERSION or not isinstance(payload.get("stages"), dict):
        return _empty_cache()
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
    for relative, expected in outputs.items():
        path = root / str(relative)
        if not path.is_file() or not isinstance(expected, dict):
            return False
        expected_hash = expected.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            return False
        current = _file_stat(path)
        expected_stat = {name: expected.get(name) for name in ("bytes", "mtime_ns", "inode")}
        if current == expected_stat:
            continue
        if cached_sha256(path, root) != expected_hash:
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
        str(path.relative_to(root)): artifact_record(path, root)
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
            "sha256": cached_sha256(path, root),
        }
        for path in sorted(set(paths))
        if path.is_file()
    ]
