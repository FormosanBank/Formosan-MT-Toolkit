#!/usr/bin/env python3
"""Inventory the versioned code artifacts used by the production MT flight."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

ACTIVE_EXPERIMENT_FILES = (
    "configs/default_experiment.json",
    "scripts/build_experiment_splits.py",
    "scripts/evaluate_directional.py",
    "scripts/experiment_config.py",
    "scripts/mt_common.py",
    "scripts/mt_metrics.py",
    "scripts/setup_formosan_nllb200.py",
    "scripts/setup_tokenizer_sweep.py",
    "scripts/tokenizer_audit.py",
    "scripts/train_directional_nllb.py",
    "scripts/training_code_inventory.py",
    "scripts/validate_experiment.py",
    "scripts/write_submission_manifest.py",
    "slurm/evaluate_directional.sl",
    "slurm/setup_spm_sweep.sl",
    "slurm/submit_v1_spm8k_directional.sh",
    "slurm/train_directional.sl",
    "slurm/validate_corpus.sl",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(deployment_path: Path, repository_path: str) -> dict[str, str]:
    if not deployment_path.is_file():
        raise FileNotFoundError(f"Missing training code artifact: {deployment_path}")
    return {
        "repository_path": repository_path,
        "deployment_path": str(deployment_path.resolve()),
        "sha256": sha256_file(deployment_path),
    }


def build_code_inventory(
    experiment_root: Path,
) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    artifacts = [
        artifact_record(
            experiment_root / relative_path,
            f"formosan_mt_experiments/{relative_path}",
        )
        for relative_path in ACTIVE_EXPERIMENT_FILES
    ]
    artifacts.sort(key=lambda row: row["repository_path"])
    return {
        "experiment_root": str(experiment_root),
        "artifacts": artifacts,
    }
