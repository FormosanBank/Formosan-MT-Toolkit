"""Dependency-light hashing, environment, and atomic JSON helpers."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

PINNED_REQUIREMENT_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;]+)$")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(payload)


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def pinned_requirements(path: Path) -> dict[str, str]:
    """Return the exact pins in a requirements file, following ``-r`` includes."""
    pins: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r"):
            pins.update(pinned_requirements(path.parent / line[2:].strip()))
            continue
        match = PINNED_REQUIREMENT_RE.match(line)
        if match is None:
            raise ValueError(f"Requirement is not an exact pin in {path.name}: {raw_line!r}")
        pins[match.group(1).lower()] = match.group(2)
    return pins


def environment_report(requirements: Path) -> dict[str, Any]:
    """Compare the installed distributions with a pinned requirements file.

    ``drift`` names every package that is missing, at another version, or an
    editable checkout. An editable install can differ from the release that
    carries the same version number, so it is not the pinned artifact.
    """
    packages: dict[str, dict[str, Any]] = {}
    for name, pinned in sorted(pinned_requirements(requirements).items()):
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = {"pinned": pinned, "installed": None, "editable": False}
            continue
        direct_url = distribution.read_text("direct_url.json")
        packages[name] = {
            "pinned": pinned,
            "installed": distribution.version,
            "editable": bool(direct_url and json.loads(direct_url).get("dir_info", {}).get("editable")),
        }
    return {
        "requirements": requirements.name,
        "requirements_sha256": sha256_file(requirements),
        "packages": packages,
        "drift": [
            name
            for name, record in packages.items()
            if record["installed"] != record["pinned"] or record["editable"]
        ],
    }
