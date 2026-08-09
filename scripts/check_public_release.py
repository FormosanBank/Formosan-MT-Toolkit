#!/usr/bin/env python3
"""Fail when tracked files cross the repository's public-data boundary."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_BYTES = 5 * 1024 * 1024
FORBIDDEN_PATHS = {".env"}
FORBIDDEN_PREFIXES = (
    "Corpora/",
    "corpus_builds/",
    "downloaded_",
    "formosan_mt_experiments/manifests/",
    "processed_corpora/",
    "protected_corpora/",
    "pivot_corpora_final/",
    "raw_corpora/",
)
PRIVATE_MACHINE_MARKERS = (
    "/Users/hunterschep",
    "/home/scheppat",
    "/scratch/scheppat",
    "/projects/prudlab",
)
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"(?m)^\s*[A-Z][A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD)\s*=\s*"
        r"[\"']?[A-Za-z0-9_:/+.-]{16,}"
    ),
}


def tracked_paths() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [Path(value) for value in output.split("\0") if value]


def inspect(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for relative in paths:
        name = relative.as_posix()
        path = ROOT / relative
        if not path.is_file():
            continue
        if name in FORBIDDEN_PATHS or name.startswith(FORBIDDEN_PREFIXES):
            errors.append(f"forbidden tracked path: {name}")
            continue
        size = path.stat().st_size
        if size > MAX_TRACKED_BYTES:
            errors.append(f"tracked file exceeds 5 MiB: {name} ({size} bytes)")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if relative == Path(__file__).resolve().relative_to(ROOT):
            continue
        for marker in PRIVATE_MACHINE_MARKERS:
            if marker in content:
                errors.append(f"private machine path in {name}: {marker}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(content):
                errors.append(f"possible {label} in {name}")
    return errors


def main() -> int:
    errors = inspect(tracked_paths())
    if errors:
        print("Public release check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("Public release check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
