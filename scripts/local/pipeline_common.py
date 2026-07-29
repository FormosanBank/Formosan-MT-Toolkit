"""Shared, dependency-light helpers for the corpus build pipeline."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "corpus_pipeline.json"


def load_pipeline_config() -> dict[str, Any]:
    try:
        value = json.loads(PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load pipeline config {PIPELINE_CONFIG_PATH}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 2:
        raise RuntimeError(f"Unsupported corpus pipeline config: {PIPELINE_CONFIG_PATH}")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        Path(tmp_name).replace(path)
    except Exception:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def git_state(repo: Path = PROJECT_ROOT) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo), *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty_lines = git("status", "--porcelain").splitlines()
        remote = git("config", "--get", "remote.origin.url")
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    return {
        "available": True,
        "commit": commit,
        "dirty": bool(dirty_lines),
        "dirty_paths": [line[3:] for line in dirty_lines],
        "remote": remote,
    }


def content_row_id(*parts: object) -> str:
    payload = "\u241f".join(str(part or "").strip() for part in parts)
    return sha256_bytes(payload.encode("utf-8"))[:24]
