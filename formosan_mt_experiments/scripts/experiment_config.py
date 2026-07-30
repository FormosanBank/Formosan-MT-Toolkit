#!/usr/bin/env python3
"""Small reproducibility helpers shared by experiment entry points."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = EXPERIMENT_ROOT / "configs" / "default_experiment.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_profile(path: Path = DEFAULT_PROFILE) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load experiment profile {path}: {exc}") from exc
    if profile.get("schema_version") != 2:
        raise SystemExit(f"Unsupported experiment profile schema: {path}")
    model_family = str(profile.get("model_family") or "nllb").strip().lower()
    profile["model_family"] = model_family
    tokenizer = profile.get("tokenizer", {})
    if model_family == "nllb":
        if tokenizer.get("mode") != "spm":
            raise SystemExit("The NLLB recipe requires tokenizer.mode=spm")
        if tokenizer.get("default_spm_vocab") != 8192:
            raise SystemExit(
                "The supported NLLB recipe requires an 8192-piece auxiliary SPM"
            )
    elif model_family == "madlad400":
        if tokenizer.get("mode") != "native":
            raise SystemExit(
                "The MADLAD-400 recipe requires tokenizer.mode=native"
            )
        expected_prefixes = {
            "english": "<2en>",
            "chinese": "<2zh_Hant>",
        }
        if tokenizer.get("target_prefixes") != expected_prefixes:
            raise SystemExit(
                "MADLAD target prefixes must be <2en> and <2zh_Hant>"
            )
        if tokenizer.get("formosan_target_template") != "<2{lang_code}>":
            raise SystemExit(
                "MADLAD Formosan target template must be <2{lang_code}>"
            )
    else:
        raise SystemExit(f"Unsupported model family in {path}: {model_family}")
    revision = str(profile.get("base_model", {}).get("revision") or "")
    if len(revision) != 40:
        raise SystemExit("Experiment profile must pin a full base-model revision")
    if profile.get("splits", {}).get("tiers") != ["in_domain_hard"]:
        raise SystemExit("Experiment profile must use only the in_domain_hard tier")
    if tokenizer.get("setup_splits") != ["train"]:
        raise SystemExit("Tokenizer setup must use training rows only")
    if tokenizer.get("training_columns") != [
        "formosan_sentence"
    ]:
        raise SystemExit(
            "Auxiliary SPM must be trained only on Formosan training text"
        )
    return profile


def profile_record(path: Path) -> dict[str, str]:
    profile = load_profile(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "recipe_id": str(profile["recipe_id"]),
    }


def git_record(root: Path = EXPERIMENT_ROOT.parent) -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return {"available": False}
    return {"available": True, "commit": commit, "dirty": dirty}


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "sacrebleu",
        "sentencepiece",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def manifest_contains_hash(value: object, expected: str) -> bool:
    if isinstance(value, dict):
        return any(
            manifest_contains_hash(child, expected)
            for child in value.values()
        )
    if isinstance(value, list):
        return any(manifest_contains_hash(child, expected) for child in value)
    return isinstance(value, str) and value == expected
