"""Checkpoint state and immutable run contracts for MT training."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from experiment_config import (
    CORPUS_PIPELINE_CONFIG,
    dependency_versions,
    git_record,
    manifest_contains_hash,
    profile_record,
    sha256_file,
    stable_hash,
)
from mt_common import write_json


def metric_value(metrics: dict, name: str) -> float:
    if name == "mean_token_loss":
        return float(metrics[name])
    if name.startswith("macro_"):
        metric = name.removeprefix("macro_")
        values = [
            float(language_metrics[metric])
            for language_metrics in metrics["generation"]["by_language"].values()
        ]
        if not values:
            raise ValueError(f"Cannot compute {name} without per-language metrics")
        return float(np.mean(values))
    return float(metrics["generation"]["global"][name])


def metric_improved(current: float, best: float | None, name: str, min_delta: float) -> bool:
    if best is None:
        return True
    if name == "mean_token_loss" or name.endswith("TER"):
        return current < best - min_delta
    return current > best + min_delta


def save_checkpoint(model, tokenizer, path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    training_use_cache = getattr(model.config, "use_cache", None)
    if training_use_cache is not None:
        model.config.use_cache = True
    model.save_pretrained(
        path,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    if training_use_cache is not None:
        model.config.use_cache = training_use_cache
    tokenizer.save_pretrained(path)
    write_json(path / "experiment_metadata.json", metadata)


def save_resume_checkpoint(model, tokenizer, optimizer, scheduler, scaler, path: Path, state: dict) -> None:
    """Overwrite one restart checkpoint so preemption costs at most one eval interval."""
    tmp = path.with_name(f"{path.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(model, tokenizer, tmp, {"step": state["step"], "kind": "resume"})
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "trainer_state.pt",
    )
    shutil.rmtree(path, ignore_errors=True)
    tmp.replace(path)


def restore_training_state(path: Path, optimizer, scheduler, scaler) -> dict:
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    scaler.load_state_dict(state.pop("scaler"))
    random.setstate(state.pop("python_rng"))
    np.random.set_state(state.pop("numpy_rng"))
    torch.set_rng_state(state.pop("torch_rng"))
    cuda_rng = state.pop("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    return state


def read_complete_manifest(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed {label} {path}: {exc}") from exc
    if value.get("complete") is not True:
        raise SystemExit(f"Incomplete {label}: {path}")
    return value


def verify_setup_artifacts(
    setup_manifest: dict,
    tokenizer_dir: Path,
    model_dir: Path,
) -> None:
    for section, directory in (
        ("tokenizer", tokenizer_dir),
        ("model", model_dir),
    ):
        records = setup_manifest.get(section, {}).get("files", [])
        if not records:
            raise SystemExit(f"Setup manifest has no {section} artifacts")
        for record in records:
            source_name = Path(str(record.get("path") or "")).name
            path = directory / source_name
            if not path.is_file():
                raise SystemExit(f"Missing setup {section} artifact: {path}")
            actual = sha256_file(path)
            if actual != record.get("sha256"):
                raise SystemExit(f"Setup {section} checksum mismatch for {path}")


def build_run_contract(args, profile: dict) -> dict:
    input_hash = sha256_file(args.input)
    expected_profile = profile_record(args.profile)
    corpus_manifest = read_complete_manifest(
        args.corpus_manifest,
        "corpus build manifest",
    )
    if corpus_manifest.get("pipeline_version") != profile["corpus_pipeline_version"]:
        raise SystemExit("Corpus pipeline version does not match the recipe")
    expected_mt_standard = profile["mt_standardization"]
    if corpus_manifest.get("mt_standardization") != expected_mt_standard:
        raise SystemExit("Corpus manifest does not match the recipe MT-standard profile")
    if corpus_manifest.get("pipeline_config", {}).get("sha256") != sha256_file(CORPUS_PIPELINE_CONFIG):
        raise SystemExit("Corpus manifest does not match the current pipeline configuration")
    if not manifest_contains_hash(corpus_manifest, input_hash):
        raise SystemExit("Training CSV checksum is absent from the corpus build manifest")
    validation = read_complete_manifest(
        args.validation_report,
        "corpus validation report",
    )
    if validation.get("input_sha256") != input_hash:
        raise SystemExit("Corpus validation report does not match the training CSV")
    split_validation = validation.get("split_validation", {})
    expected_split_policy = profile["splits"]
    expected_minimums = {
        "test": expected_split_policy["test_ratio"],
        "validate": expected_split_policy["validate_ratio"],
        "min_test_rows": expected_split_policy["min_test_rows"],
        "min_validate_rows": expected_split_policy["min_validate_rows"],
    }
    if (
        validation.get("profile", {}).get("sha256") != expected_profile["sha256"]
        or split_validation.get("minimum_ratios") != expected_minimums
        or split_validation.get("ngram_jaccard_threshold")
        != expected_split_policy["character_ngram_jaccard_threshold"]
        or split_validation.get("source_ratio_tolerance")
        != expected_split_policy["source_ratio_tolerance"]
        or split_validation.get("ratio_basis") != expected_split_policy["ratio_basis"]
        or split_validation.get("synthetic_eval_rows") != 0
        or split_validation.get("synthetic_eval_allowed") is not False
    ):
        raise SystemExit("Corpus validation did not use the current experiment split policy")
    if validation.get("provenance_validation", {}).get("mt_standardization") != {
        key: expected_mt_standard[key] for key in ("id", "sha256")
    }:
        raise SystemExit("Corpus validation does not match the recipe MT-standard profile")
    setup = read_complete_manifest(
        args.setup_manifest,
        "model setup manifest",
    )
    setup_mismatch = (
        setup.get("recipe_id") != profile["recipe_id"]
        or setup.get("model_family") != profile["model_family"]
        or setup.get("profile", {}).get("sha256") != expected_profile["sha256"]
        or setup.get("mt_standardization") != expected_mt_standard
    )
    if setup_mismatch:
        raise SystemExit("Model setup manifest does not match the experiment profile")
    if profile["model_family"] == "nllb":
        setup_inputs = setup.get("inputs")
        if not isinstance(setup_inputs, list):
            setup_inputs = [setup.get("input", {})]
        if not any(
            record.get("sha256") == input_hash
            for record in setup_inputs
            if isinstance(record, dict)
        ):
            raise SystemExit("NLLB setup manifest does not match the training corpus")
    elif setup.get("base_model") != profile["base_model"]:
        raise SystemExit("MiLMMT setup manifest has the wrong base model")
    verify_setup_artifacts(
        setup,
        args.tokenizer,
        args.model,
    )
    hyperparameters = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "output_dir",
            "resume_from",
            "corpus_manifest",
            "validation_report",
            "setup_manifest",
            "profile",
        }
    }
    return {
        "schema_version": 3,
        "complete": True,
        "recipe_id": profile["recipe_id"],
        "model_family": profile["model_family"],
        "mt_standardization": expected_mt_standard,
        "profile": expected_profile,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": input_hash,
        },
        "corpus_manifest": {
            "path": str(args.corpus_manifest.resolve()),
            "sha256": sha256_file(args.corpus_manifest),
        },
        "validation_report": {
            "path": str(args.validation_report.resolve()),
            "sha256": sha256_file(args.validation_report),
        },
        "setup_manifest": {
            "path": str(args.setup_manifest.resolve()),
            "sha256": sha256_file(args.setup_manifest),
        },
        "repository": git_record(),
        "dependencies": dependency_versions(),
        "hyperparameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in hyperparameters.items()
        },
    }


def write_or_verify_run_contract(
    output_dir: Path,
    contract: dict,
) -> str:
    path = output_dir / "run_contract.json"
    digest = stable_hash(contract)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_hash(existing) != digest:
            raise SystemExit(f"Existing run contract does not match this invocation: {path}")
    else:
        write_json(path, contract)
    return digest
