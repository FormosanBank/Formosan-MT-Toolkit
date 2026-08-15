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
PROJECT_ROOT = EXPERIMENT_ROOT.parent
DEFAULT_PROFILE = EXPERIMENT_ROOT / "configs" / "default_experiment.json"
MILMMT_PROFILE = EXPERIMENT_ROOT / "configs" / "milmmt_1b_experiment.json"
CORPUS_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "corpus_pipeline.json"
MODEL_FAMILIES = {"nllb", "milmmt"}

SHARED_SPLIT_FIELDS = (
    "train_ratio",
    "validate_ratio",
    "test_ratio",
    "min_test_rows",
    "min_validate_rows",
    "min_formosan_tokens",
    "min_target_tokens",
    "min_combined_tokens",
    "min_punctuated_combined_tokens",
    "source_ratio_tolerance",
    "ratio_basis",
    "synthetic_eval_policy",
    "character_ngram_jaccard_threshold",
    "headline_tier",
)


def load_corpus_pipeline_config() -> dict[str, Any]:
    try:
        pipeline = json.loads(CORPUS_PIPELINE_CONFIG.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load corpus pipeline config {CORPUS_PIPELINE_CONFIG}: {exc}") from exc
    if pipeline.get("schema_version") != 3 or pipeline.get("pipeline_version") != "formosan-mt-corpus-v3":
        raise SystemExit("Unsupported corpus pipeline configuration")
    return pipeline


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
    if profile.get("schema_version") != 3:
        raise SystemExit(f"Unsupported experiment profile schema: {path}")
    mt_standard = profile.get("mt_standardization", {})
    if (
        mt_standard.get("id") != "formosan-mt-standard-v3"
        or mt_standard.get("namespace") != "formosan-mt"
        or not isinstance(mt_standard.get("sha256"), str)
        or len(mt_standard["sha256"]) != 64
    ):
        raise SystemExit("Experiment profile has an invalid MT-standard contract")
    if profile.get("corpus_pipeline_version") != "formosan-mt-corpus-v3":
        raise SystemExit("Experiment profile must use corpus pipeline V3")
    model_family = profile.get("model_family")
    if model_family not in MODEL_FAMILIES:
        raise SystemExit(f"Experiment profile has unsupported model_family={model_family!r}: {path}")
    tokenizer = profile.get("tokenizer", {})
    if model_family == "nllb":
        if tokenizer.get("mode") != "spm":
            raise SystemExit("The NLLB recipe requires tokenizer.mode=spm")
        if tokenizer.get("default_spm_vocab") != 8192:
            raise SystemExit("The supported NLLB recipe requires an 8192-piece auxiliary SPM")
    elif tokenizer.get("mode") != "native":
        raise SystemExit("The MiLMMT recipe requires tokenizer.mode=native")
    revision = str(profile.get("base_model", {}).get("revision") or "")
    if len(revision) != 40:
        raise SystemExit("Experiment profile must pin a full base-model revision")
    splits = profile.get("splits", {})
    if splits.get("tiers") != ["in_domain_hard"]:
        raise SystemExit("Experiment profile must use only the in_domain_hard tier")
    pipeline = load_corpus_pipeline_config()
    pipeline_splits = pipeline.get("splits", {})
    mismatches = {
        field: {
            "profile": splits.get(field),
            "corpus_pipeline": pipeline_splits.get(field),
        }
        for field in SHARED_SPLIT_FIELDS
        if splits.get(field) != pipeline_splits.get(field)
    }
    if mismatches:
        raise SystemExit("Experiment and corpus split policies differ: " + json.dumps(mismatches, sort_keys=True))
    if model_family == "nllb":
        if tokenizer.get("setup_splits") != ["train"]:
            raise SystemExit("Tokenizer setup must use training rows only")
        if tokenizer.get("training_columns") != ["formosan_sentence"]:
            raise SystemExit("Auxiliary SPM must be trained only on Formosan training text")
    else:
        prompt = profile.get("prompt", {})
        if prompt.get("format") != "xiaomi_translation_v1":
            raise SystemExit("MiLMMT requires the Xiaomi translation prompt")
    training = profile.get("training_defaults", {})
    for name in ("domain_tag_dropout", "dialect_tag_dropout"):
        value = training.get(name)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise SystemExit(f"Experiment profile has invalid {name}")
    if training.get("validation_metadata_mode") not in {"default", "oracle"}:
        raise SystemExit("Experiment profile has invalid validation_metadata_mode")
    if training.get("effective_batch_size") != (training.get("batch_size", 0) * training.get("grad_accum_steps", 0)):
        raise SystemExit("Experiment profile effective batch size is inconsistent")
    if model_family == "milmmt":
        if training.get("optimizer") != "adamw":
            raise SystemExit("MiLMMT requires AdamW")
        if training.get("lr_scheduler") != "inverse_sqrt":
            raise SystemExit("MiLMMT requires inverse-sqrt learning-rate scheduling")
        if profile.get("generation_defaults", {}).get("beam") != 1:
            raise SystemExit("MiLMMT headline generation must use greedy decoding")
    if "default" not in profile.get("generation_defaults", {}).get("metadata_modes", []):
        raise SystemExit("Headline generation must include default metadata")
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
        return any(manifest_contains_hash(child, expected) for child in value.values())
    if isinstance(value, list):
        return any(manifest_contains_hash(child, expected) for child in value)
    return isinstance(value, str) and value == expected
