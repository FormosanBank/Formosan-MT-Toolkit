#!/usr/bin/env python3
"""Inventory the versioned code artifacts used by the production MT flight."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARED_SCRIPTS = PROJECT_ROOT / "scripts" / "shared"
sys.path.insert(0, str(SHARED_SCRIPTS))
from reproducibility import sha256_file  # noqa: E402

ACTIVE_EXPERIMENT_FILES = (
    "configs/default_experiment.json",
    "configs/milmmt_1b_experiment.json",
    "scripts/bootstrap_predictions.py",
    "scripts/build_experiment_splits.py",
    "scripts/evaluate_directional.py",
    "scripts/experiment_config.py",
    "scripts/formosan_mt_inference.py",
    "scripts/milmmt_runtime.py",
    "scripts/nllb_runtime.py",
    "scripts/mt_common.py",
    "scripts/mt_metrics.py",
    "scripts/publish_huggingface_models.py",
    "scripts/setup_formosan_nllb200.py",
    "scripts/setup_milmmt.py",
    "scripts/setup_tokenizer_sweep.py",
    "scripts/split_allocation.py",
    "scripts/split_similarity.py",
    "scripts/tokenizer_audit.py",
    "scripts/train_directional.py",
    "scripts/training_code_inventory.py",
    "scripts/validation_similarity.py",
    "scripts/validate_experiment.py",
    "scripts/write_submission_manifest.py",
    "slurm/bootstrap_metrics.sl",
    "slurm/evaluate_directional.sl",
    "slurm/setup_base_model.sl",
    "slurm/setup_spm_sweep.sl",
    "slurm/submit_directional_experiment.sh",
    "slurm/train_directional.sl",
    "slurm/validate_corpus.sl",
)

ACTIVE_REPOSITORY_FILES = (
    "config/corpus_pipeline.json",
    "config/mt_standardization.json",
    "scripts/local/corpus_quality.py",
    "scripts/local/mt_standardization.py",
    "scripts/local/standardize_mt_corpus.py",
    "scripts/shared/columnar_io.py",
    "scripts/shared/reproducibility.py",
)


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
    project_root = experiment_root.parent
    artifacts.extend(
        artifact_record(
            project_root / relative_path,
            relative_path,
        )
        for relative_path in ACTIVE_REPOSITORY_FILES
    )
    artifacts.sort(key=lambda row: row["repository_path"])
    return {
        "experiment_root": str(experiment_root),
        "artifacts": artifacts,
    }
