#!/usr/bin/env python3
"""Validate a tracked experiment snapshot and optionally hash its local corpora."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def manifest_errors(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    corpora = payload.get("corpora", {})
    if not corpora:
        errors.append("corpora must not be empty")
    for name, corpus in corpora.items():
        rows = int(corpus.get("rows", -1))
        splits = corpus.get("splits", {})
        if sum(int(value) for value in splits.values()) != rows:
            errors.append(f"{name}: split counts do not sum to rows")
        if rows > 0:
            if int(splits.get("test", 0)) / rows < 0.075:
                errors.append(f"{name}: global test ratio is below 0.075")
            if int(splits.get("validate", 0)) / rows < 0.025:
                errors.append(f"{name}: global validate ratio is below 0.025")
        if len(str(corpus.get("sha256", ""))) != 64:
            errors.append(f"{name}: invalid SHA-256")

    job_ids: list[int] = []
    for scope in payload.get("jobs", {}).values():
        for value in scope.values():
            job_ids.extend(value if isinstance(value, list) else [value])
    if job_ids and len(job_ids) != len(set(job_ids)):
        errors.append("Slurm job IDs are not unique")
    return errors


def hash_and_count(path: Path) -> tuple[str, int, dict[str, int]]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    splits: Counter[str] = Counter()
    rows = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "split" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no split column")
        for row in reader:
            rows += 1
            splits[row["split"]] += 1
    return digest.hexdigest(), rows, dict(splits)


def verify_files(payload: dict[str, Any], repo_root: Path) -> list[str]:
    errors: list[str] = []
    for name, corpus in payload["corpora"].items():
        path = repo_root / corpus["relative_path"]
        if not path.is_file():
            errors.append(f"{name}: missing {path}")
            continue
        digest, rows, splits = hash_and_count(path)
        if digest != corpus["sha256"]:
            errors.append(f"{name}: SHA-256 mismatch")
        if rows != int(corpus["rows"]):
            errors.append(f"{name}: row-count mismatch")
        expected_splits = {key: int(value) for key, value in corpus["splits"].items()}
        if splits != expected_splits:
            errors.append(f"{name}: split-count mismatch: {splits}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = manifest_errors(payload)
    if args.check_files:
        errors.extend(verify_files(payload, args.repo_root.resolve()))
    if errors:
        raise SystemExit("Experiment manifest validation failed:\n- " + "\n- ".join(errors))
    mode = "manifest and corpus files" if args.check_files else "manifest"
    print(f"Validated {mode}: {args.manifest}")


if __name__ == "__main__":
    main()
