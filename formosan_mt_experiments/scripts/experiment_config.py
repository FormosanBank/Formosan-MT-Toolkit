#!/usr/bin/env python3
"""Small reproducibility helpers shared by experiment entry points."""

from __future__ import annotations

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
SHARED_SCRIPTS = PROJECT_ROOT / "scripts" / "shared"
sys.path.insert(0, str(SHARED_SCRIPTS))
from columnar_io import read_csv_or_columnar, write_columnar_cache  # noqa: E402,F401,I001
from reproducibility import sha256_file, stable_json_hash  # noqa: E402
from split_policy import (  # noqa: E402,F401
    target_split_ratios,
    validate_target_split_ratios,
)

DEFAULT_PROFILE = EXPERIMENT_ROOT / "configs" / "default_experiment.json"
NLLB_1_3B_PROFILE = EXPERIMENT_ROOT / "configs" / "nllb_1_3b_experiment.json"
NLLB_3_3B_PROFILE = EXPERIMENT_ROOT / "configs" / "nllb_3_3b_experiment.json"
MILMMT_PROFILE = EXPERIMENT_ROOT / "configs" / "milmmt_1b_experiment.json"
CORPUS_PIPELINE_CONFIG = PROJECT_ROOT / "config" / "corpus_pipeline.json"
MODEL_FAMILIES = {"nllb", "milmmt"}
MODEL_VARIANTS = {
    "nllb-600m": {
        "model_family": "nllb",
        "base_model": "facebook/nllb-200-distilled-600M",
    },
    "nllb-1.3b": {
        "model_family": "nllb",
        "base_model": "facebook/nllb-200-1.3B",
    },
    "nllb-3.3b": {
        "model_family": "nllb",
        "base_model": "facebook/nllb-200-3.3B",
    },
    "milmmt-1b": {
        "model_family": "milmmt",
        "base_model": "xiaomi-research/MiLMMT-46-1B-v1.0",
    },
}

SHARED_SPLIT_FIELDS = (
    "ratios_by_target",
    "min_test_rows",
    "min_validate_rows",
    "min_formosan_tokens",
    "min_target_tokens",
    "min_combined_tokens",
    "min_punctuated_combined_tokens",
    "max_eval_units_per_side",
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
    try:
        validate_target_split_ratios(pipeline.get("splits", {}))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("Corpus pipeline has invalid target split ratios") from exc
    return pipeline


def stable_hash(value: Any) -> str:
    return stable_json_hash(value)


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
    model_variant = profile.get("model_variant")
    variant = MODEL_VARIANTS.get(model_variant)
    if variant is None:
        raise SystemExit(f"Experiment profile has unsupported model_variant={model_variant!r}: {path}")
    if variant["model_family"] != model_family or variant["base_model"] != profile.get("base_model", {}).get("name"):
        raise SystemExit("Experiment profile model variant, family, and base model disagree")
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
    for name in ("language_sampling_alpha", "dialect_tag_dropout"):
        value = training.get(name)
        if not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 1.0:
            raise SystemExit(f"Experiment profile has invalid {name}")
    lexical_weight = training.get("lexical_row_sampling_weight")
    if not isinstance(lexical_weight, (int, float)) or not 0.0 < float(lexical_weight) <= 1.0:
        raise SystemExit("Experiment profile has invalid lexical_row_sampling_weight")
    if training.get("validation_metadata_mode") not in {"default", "oracle"}:
        raise SystemExit("Experiment profile has invalid validation_metadata_mode")
    if training.get("effective_batch_size") != (training.get("batch_size", 0) * training.get("grad_accum_steps", 0)):
        raise SystemExit("Experiment profile effective batch size is inconsistent")
    sampling_chunk_size = training.get("language_sampling_chunk_size")
    if (
        not isinstance(sampling_chunk_size, int)
        or sampling_chunk_size <= 0
        or training.get("batch_size", 0) % sampling_chunk_size
    ):
        raise SystemExit("Experiment profile has an invalid language sampling chunk size")
    if model_family == "milmmt":
        if bool(profile.get("prompt", {}).get("use_metadata")) != bool(training.get("use_tags")):
            raise SystemExit("MiLMMT prompt metadata and use_tags must agree")
        if training.get("optimizer") != "adamw":
            raise SystemExit("MiLMMT requires AdamW")
        if training.get("lr_scheduler") != "inverse_sqrt":
            raise SystemExit("MiLMMT requires inverse-sqrt learning-rate scheduling")
        comparison = profile.get("comparison", {})
        try:
            baseline = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot load comparison baseline {DEFAULT_PROFILE}: {exc}") from exc
        if comparison.get("baseline_recipe_id") != baseline.get("recipe_id"):
            raise SystemExit("MiLMMT comparison baseline does not match the NLLB recipe")
        if comparison.get("budget_basis") != "sample_presentations":
            raise SystemExit("MiLMMT comparison must use a sample-presentation budget")
        if comparison.get("claim_scope") != "matched_data_exposure_not_parameter_or_flop_matched":
            raise SystemExit("MiLMMT comparison has an invalid claim scope")
        presentations = training.get("steps", 0) * training.get("effective_batch_size", 0)
        if comparison.get("sample_presentations") != presentations:
            raise SystemExit("MiLMMT comparison sample-presentation budget is inconsistent")
        for section, field_key in (
            ("training_defaults", "matched_training_fields"),
            ("generation_defaults", "matched_generation_fields"),
        ):
            fields = comparison.get(field_key)
            if not isinstance(fields, list) or not fields:
                raise SystemExit(f"MiLMMT comparison has no {field_key}")
            mismatched = [
                field
                for field in fields
                if field not in profile.get(section, {})
                or field not in baseline.get(section, {})
                or profile[section][field] != baseline[section][field]
            ]
            if mismatched:
                raise SystemExit(f"MiLMMT comparison fields differ from NLLB in {section}: {mismatched}")
    if "default" not in profile.get("generation_defaults", {}).get("metadata_modes", []):
        raise SystemExit("Headline generation must include default metadata")
    return profile


def profile_record(path: Path) -> dict[str, str]:
    profile = load_profile(path)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "recipe_id": str(profile["recipe_id"]),
        "model_variant": str(profile["model_variant"]),
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
