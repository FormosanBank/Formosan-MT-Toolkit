#!/usr/bin/env python3
"""Build and audit the one supported train-only SPM8k tokenizer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch
from experiment_config import (
    DEFAULT_PROFILE,
    load_profile,
    profile_record,
    sha256_file,
)
from mt_common import (
    DEFAULT_SETUP_SCRIPT,
    normalize_target_language,
    read_parallel_csv,
    special_tokens_from_corpus,
    target_col_for,
    write_json,
)
from tokenizer_audit import audit_tokenizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


def artifact_inventory(directory: Path) -> list[dict[str, object]]:
    return [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(directory.iterdir())
        if path.is_file()
    ]


def training_rows(path: Path, *, target_col: str) -> pd.DataFrame:
    frame = read_parallel_csv(path, target_col=target_col)
    if "split" not in frame:
        raise SystemExit("Tokenizer input must contain split assignments")
    train = frame[
        frame["split"].astype(str).str.strip().str.lower().eq("train")
    ].copy()
    if train.empty:
        raise SystemExit("Tokenizer input has no training rows")
    if "kindOf" not in train or not train["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Tokenizer input must contain only kindOf=standard rows")
    return train


def add_experiment_special_tokens(
    tokenizer_dir: Path,
    model_dir: Path,
    train: pd.DataFrame,
    *,
    max_dialect_tags: int,
    min_dialect_frequency: int,
) -> dict[str, object]:
    desired_tokens = special_tokens_from_corpus(
        train,
        max_dialect_tags=max_dialect_tags,
        min_dialect_frequency=min_dialect_frequency,
    )
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    new_tokens = [
        token
        for token in desired_tokens
        if token not in tokenizer.get_vocab()
    ]
    old_size = len(tokenizer)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens},
        replace_additional_special_tokens=False,
    )
    if len(tokenizer) != old_size + added:
        raise SystemExit("Tokenizer special-token accounting failed")

    if added:
        old_embeddings = (
            model.get_input_embeddings().weight.detach().clone()
        )
        model.resize_token_embeddings(len(tokenizer))
        with torch.no_grad():
            embeddings = model.get_input_embeddings().weight
            mean = old_embeddings.mean(dim=0)
            standard_deviation = (
                old_embeddings.std(dim=0).clamp_min(1e-6) * 0.01
            )
            embeddings[old_size:] = mean + (
                torch.randn_like(embeddings[old_size:])
                * standard_deviation
            )
        model.tie_weights()

    model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = (
            tokenizer.eos_token_id
        )
    bad = [
        token
        for token in desired_tokens
        if (
            tokenizer.convert_tokens_to_ids(token)
            == tokenizer.unk_token_id
            or tokenizer.convert_ids_to_tokens(
                tokenizer.convert_tokens_to_ids(token)
            )
            != token
        )
    ]
    if bad:
        raise SystemExit(
            "Tokenizer failed to preserve control tags as single tokens: "
            f"{bad[:20]}"
        )
    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir, safe_serialization=True)
    return {
        "source_split": "train",
        "desired_special_tokens": len(desired_tokens),
        "new_special_tokens": int(added),
        "vocab_size": len(tokenizer),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the pinned Formosan SPM8k tokenizer/model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--target-lang",
        choices=["english", "chinese"],
        required=True,
    )
    parser.add_argument("--target-col")
    parser.add_argument(
        "--setup-script",
        type=Path,
        default=DEFAULT_SETUP_SCRIPT,
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--base-model")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spm-vocabs", default="8192")
    parser.add_argument("--min-char-frequency", type=int)
    parser.add_argument("--max-dialect-tags", type=int)
    parser.add_argument("--min-dialect-frequency", type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    target_lang = normalize_target_language(
        args.target_lang,
        args.target_col,
    )
    target_col = args.target_col or target_col_for(target_lang)
    expected_vocab = int(profile["tokenizer"]["default_spm_vocab"])
    vocabs = [
        int(value.strip())
        for value in args.spm_vocabs.split(",")
        if value.strip()
    ]
    if vocabs != [expected_vocab]:
        raise SystemExit(
            f"The supported recipe requires --spm-vocabs {expected_vocab}"
        )
    train = training_rows(args.input, target_col=target_col)
    base_model = args.base_model or profile["base_model"]["name"]
    base_revision = (
        args.base_model_revision
        or profile["base_model"]["revision"]
    )
    minimum_character_frequency = (
        args.min_char_frequency
        if args.min_char_frequency is not None
        else int(profile["tokenizer"]["min_char_frequency"])
    )
    max_dialect_tags = (
        args.max_dialect_tags
        if args.max_dialect_tags is not None
        else int(profile["tokenizer"]["max_dialect_tags"])
    )
    min_dialect_frequency = (
        args.min_dialect_frequency
        if args.min_dialect_frequency is not None
        else int(profile["tokenizer"]["min_dialect_frequency"])
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        args.output_dir
        / f"formosan_multilingual_nllb_spm{expected_vocab}"
    )
    tokenizer_dir = Path(f"{prefix}_tokenizer")
    model_dir = Path(f"{prefix}_model")
    setup_manifest = Path(f"{prefix}_setup_manifest.json")
    command = [
        args.python,
        str(args.setup_script),
        "--input",
        str(args.input),
        "--target-col",
        target_col,
        "--output-prefix",
        str(prefix),
        "--profile",
        str(args.profile),
        "--base-model",
        base_model,
        "--base-model-revision",
        base_revision,
        "--spm-vocab",
        str(expected_vocab),
        "--min-char-frequency",
        str(minimum_character_frequency),
    ]
    print(" ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    if not setup_manifest.is_file():
        raise SystemExit(
            f"Setup implementation did not write {setup_manifest}"
        )
    manifest = json.loads(setup_manifest.read_text(encoding="utf-8"))
    tag_report = add_experiment_special_tokens(
        tokenizer_dir,
        model_dir,
        train,
        max_dialect_tags=max_dialect_tags,
        min_dialect_frequency=min_dialect_frequency,
    )
    audit_json = (
        args.output_dir
        / f"tokenizer_audit_spm{expected_vocab}.json"
    )
    audit_csv = (
        args.output_dir
        / f"tokenizer_audit_spm{expected_vocab}.csv"
    )
    audit_report = audit_tokenizer(
        tokenizer_dir=tokenizer_dir,
        input_csv=args.input,
        output_json=audit_json,
        output_csv=audit_csv,
        max_rows_per_lang=20_000,
        target_col=target_col,
    )
    manifest["profile"] = profile_record(args.profile)
    manifest["control_tags"] = tag_report
    manifest["tokenizer"]["files"] = artifact_inventory(tokenizer_dir)
    manifest["model"]["files"] = artifact_inventory(model_dir)
    manifest["audit"] = {
        "json": str(audit_json),
        "json_sha256": sha256_file(audit_json),
        "csv": str(audit_csv),
        "csv_sha256": sha256_file(audit_csv),
        "macro_avg_pieces_per_word": audit_report[
            "macro_avg_pieces_per_word"
        ],
        "macro_avg_pct_words_ge_5_pieces": audit_report[
            "macro_avg_pct_words_ge_5_pieces"
        ],
    }
    manifest["complete"] = True
    write_json(setup_manifest, manifest)
    write_json(
        args.output_dir / "tokenizer_sweep_report.json",
        {
            "schema_version": 2,
            "complete": True,
            "recipe_id": profile["recipe_id"],
            "profile": profile_record(args.profile),
            "input": {
                "path": str(args.input.resolve()),
                "sha256": sha256_file(args.input),
                "training_rows": len(train),
                "setup_split": "train",
            },
            "setup_manifest": {
                "path": str(setup_manifest),
                "sha256": sha256_file(setup_manifest),
            },
        },
    )
    print(f"setup manifest: {setup_manifest}")


if __name__ == "__main__":
    main()
