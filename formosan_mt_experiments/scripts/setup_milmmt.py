#!/usr/bin/env python3
"""Download and verify the pinned MiLMMT base model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from experiment_config import (
    MILMMT_PROFILE,
    dependency_versions,
    git_record,
    load_profile,
    profile_record,
    sha256_file,
)
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=MILMMT_PROFILE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    if profile["model_family"] != "milmmt":
        raise SystemExit("setup_milmmt.py requires a MiLMMT profile")

    output_dir = args.output_dir.resolve()
    model_dir = output_dir / "model"
    manifest_path = output_dir / "setup_manifest.json"
    temporary_dir = output_dir / "model.incomplete"
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_dir, ignore_errors=True)

    base_model = profile["base_model"]
    snapshot_download(
        repo_id=base_model["name"],
        revision=base_model["revision"],
        local_dir=temporary_dir,
    )
    config = AutoConfig.from_pretrained(temporary_dir)
    tokenizer = AutoTokenizer.from_pretrained(temporary_dir, use_fast=True)
    if config.model_type != "gemma3_text":
        raise SystemExit(f"Expected Gemma 3 text configuration, found {config.model_type!r}")
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise SystemExit("MiLMMT tokenizer is missing pad or EOS tokens")
    weight_files = sorted(temporary_dir.glob("*.safetensors"))
    if not weight_files:
        raise SystemExit("MiLMMT snapshot contains no safetensors weights")

    shutil.rmtree(temporary_dir / ".cache", ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    temporary_dir.replace(model_dir)
    tokenizer_names = {
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "special_tokens_map.json",
    }
    tokenizer_files = [
        artifact_record(path) for path in sorted(model_dir.iterdir()) if path.is_file() and path.name in tokenizer_names
    ]
    model_files = [
        artifact_record(path)
        for path in sorted(model_dir.iterdir())
        if path.is_file() and (path.name in {"config.json", "generation_config.json"} or path.suffix == ".safetensors")
    ]
    if not tokenizer_files or not model_files:
        raise SystemExit("MiLMMT snapshot is missing required model artifacts")

    manifest = {
        "schema_version": 3,
        "complete": True,
        "stage": "base_model_ready",
        "recipe_id": profile["recipe_id"],
        "model_family": profile["model_family"],
        "profile": profile_record(args.profile),
        "mt_standardization": profile["mt_standardization"],
        "base_model": base_model,
        "upstream_code": profile["upstream_code"],
        "repository": git_record(),
        "runtime_dependencies": dependency_versions(),
        "tokenizer": {
            "path": str(model_dir),
            "vocab_size": len(tokenizer),
            "files": tokenizer_files,
        },
        "model": {
            "path": str(model_dir),
            "architecture": config.architectures,
            "model_type": config.model_type,
            "files": model_files,
        },
    }
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    print(f"model/tokenizer: {model_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
