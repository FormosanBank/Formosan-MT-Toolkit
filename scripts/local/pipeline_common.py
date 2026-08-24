"""Shared, dependency-light helpers for the corpus build pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SHARED_SCRIPTS = Path(__file__).resolve().parents[1] / "shared"
sys.path.insert(0, str(SHARED_SCRIPTS))
from reproducibility import (  # noqa: E402,F401
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    stable_json_hash,
)
from split_policy import (  # noqa: E402,F401
    target_split_ratios,
    validate_target_split_ratios,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG_PATH = PROJECT_ROOT / "config" / "corpus_pipeline.json"


def load_pipeline_config() -> dict[str, Any]:
    try:
        value = json.loads(PIPELINE_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load pipeline config {PIPELINE_CONFIG_PATH}: {exc}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 3:
        raise RuntimeError(f"Unsupported corpus pipeline config: {PIPELINE_CONFIG_PATH}")
    mt_standard = value.get("mt_standardization", {})
    if (
        mt_standard.get("profile_id") != "formosan-mt-standard-v3"
        or mt_standard.get("namespace") != "formosan-mt"
        or mt_standard.get("source_xml_immutable") is not True
    ):
        raise RuntimeError("Corpus pipeline has an invalid MT-standard contract")
    cleaning = value.get("cleaning", {})
    if (
        not isinstance(cleaning.get("max_training_units_per_side"), int)
        or cleaning["max_training_units_per_side"] < 1
    ):
        raise RuntimeError("Corpus pipeline has an invalid training-length policy")
    exposure = value.get("exposure_audit", {})
    if exposure.get("tool") != "tame-mt" or exposure.get("version") != "0.2.2":
        raise RuntimeError("Corpus pipeline must pin tame-mt==0.2.2")
    pivot = value.get("pivot", {})
    if (
        pivot.get("require_complete") is not True
        or pivot.get("eligible_row_types") != ["sentence"]
        or pivot.get("require_mt_eval_eligible") is not True
        or not isinstance(pivot.get("min_formosan_tokens"), int)
        or pivot["min_formosan_tokens"] < 1
        or not isinstance(pivot.get("min_source_units"), int)
        or pivot["min_source_units"] < 1
    ):
        raise RuntimeError("Corpus pipeline has an invalid sentence-only pivot policy")
    splits = value.get("splits", {})
    try:
        validate_target_split_ratios(splits)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Corpus pipeline has invalid target split ratios") from exc
    if (
        splits.get("lexical_eval") is not False
        or splits.get("synthetic_eval") is not False
        or splits.get("synthetic_eval_policy")
        != "train_only_after_human_split"
        or splits.get("ratio_basis")
        != "deduplicated_human_pairs_by_language_and_source"
        or not isinstance(splits.get("source_ratio_tolerance"), (int, float))
        or not 0 <= splits["source_ratio_tolerance"] <= 1
        or any(
            not isinstance(splits.get(field), int) or splits[field] < 1
            for field in (
                "min_formosan_tokens",
                "min_target_tokens",
                "min_combined_tokens",
                "min_punctuated_combined_tokens",
                "max_eval_units_per_side",
            )
        )
        or splits["min_punctuated_combined_tokens"]
        > splits["min_combined_tokens"]
        or splits["max_eval_units_per_side"] < 1
        or splits["max_eval_units_per_side"]
        > cleaning["max_training_units_per_side"]
        or splits.get("headline_tier") != "in_domain_hard"
    ):
        raise RuntimeError("Corpus pipeline has an invalid hard-split policy")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_csv_or_columnar(path: Path, **csv_options: Any) -> Any:
    from columnar_io import read_csv_or_columnar as read_table

    return read_table(path, **csv_options)


def write_csv_atomic(frame: Any, path: Path) -> None:
    from columnar_io import write_csv_atomic as write_table

    write_table(frame, path)


def write_columnar_cache(frame: Any, csv_path: Path) -> Path:
    from columnar_io import write_columnar_cache as write_cache

    return write_cache(frame, csv_path)


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
