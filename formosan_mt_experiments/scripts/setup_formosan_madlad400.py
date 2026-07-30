#!/usr/bin/env python3
"""Build the pinned MADLAD-400 3B base with Formosan control tokens."""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import pandas as pd
import torch
from experiment_config import (
    dependency_versions,
    git_record,
    load_profile,
    profile_record,
    sha256_file,
)
from model_backends import (
    MADLAD_TARGET_TOKENS,
    madlad_formosan_target_tokens,
    token_id,
)
from mt_common import (
    FORMOSAN_CODES,
    LANG_CODE_TO_CANON,
    read_parallel_csv,
    special_tokens_from_corpus,
)
from tokenizer_audit import audit_tokenizer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

AUSTRONESIAN_TARGET_ANCHORS = (
    "<2mi>",
    "<2ms>",
    "<2id>",
    "<2haw>",
    "<2sm>",
    "<2to>",
)


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
    required = {"split", "kindOf", "row_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"MADLAD setup input {path} is missing: {missing}")
    if not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit(f"MADLAD setup input contains non-standard rows: {path}")
    train = frame[
        frame["split"].astype(str).str.strip().str.lower().eq("train")
    ].copy()
    if train.empty:
        raise SystemExit(f"MADLAD setup input has no training rows: {path}")
    return train


def setup_input_record(
    path: Path,
    frame: pd.DataFrame,
    *,
    target_lang: str,
    target_col: str,
) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "rows": len(frame),
        "split": "train",
        "target_lang": target_lang,
        "target_column": target_col,
    }


def token_seed_text(token: str) -> str:
    if token.startswith("<2") and token.endswith(">"):
        code = token[2:-1]
        return LANG_CODE_TO_CANON.get(code, code)
    surface = token.strip("<>")
    surface = re.sub(r"^(to|src|dom|dialect)_", "", surface)
    return surface.replace("_", " ")


def base_piece_ids(tokenizer, text: str, old_size: int) -> list[int]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    ignored = {
        tokenizer.unk_token_id,
        tokenizer.pad_token_id,
        tokenizer.eos_token_id,
    }
    return [
        int(value)
        for value in encoded
        if int(value) < old_size and int(value) not in ignored
    ]


def resize_and_initialize(
    model,
    tokenizer,
    new_tokens: list[str],
    *,
    old_size: int,
) -> dict[str, object]:
    old_input = model.get_input_embeddings().weight.detach().clone()
    old_output = model.get_output_embeddings().weight.detach().clone()
    anchor_ids = [
        token_id(tokenizer, token)
        for token in AUSTRONESIAN_TARGET_ANCHORS
        if token in tokenizer.get_vocab()
    ]
    if not anchor_ids:
        raise SystemExit("MADLAD tokenizer has no Austronesian target anchors")
    anchor_ids = [value for value in anchor_ids if value < old_size]
    model.resize_token_embeddings(
        len(tokenizer),
        mean_resizing=False,
    )
    input_embeddings = model.get_input_embeddings().weight
    output_embeddings = model.get_output_embeddings().weight
    output_mean = old_output.float().mean(dim=0).to(old_output.dtype)
    initialization: dict[str, dict[str, object]] = {}
    with torch.no_grad():
        for token in new_tokens:
            destination = token_id(tokenizer, token)
            if token.startswith("<2") and token.endswith(">"):
                seeds = anchor_ids
                method = "austronesian_target_prefix_mean"
            else:
                seeds = base_piece_ids(
                    tokenizer,
                    token_seed_text(token),
                    old_size,
                )
                method = "surface_piece_mean"
                if not seeds:
                    seeds = anchor_ids
                    method = "austronesian_target_prefix_mean_fallback"
            input_embeddings[destination] = old_input[seeds].mean(dim=0)
            output_embeddings[destination] = output_mean
            initialization[token] = {
                "id": destination,
                "input_method": method,
                "input_seed_ids": seeds,
                "output_method": "base_output_row_mean",
            }
    if not torch.equal(
        model.get_input_embeddings().weight[:old_size],
        old_input,
    ):
        raise SystemExit("MADLAD resize changed existing input embeddings")
    if not torch.equal(
        model.get_output_embeddings().weight[:old_size],
        old_output,
    ):
        raise SystemExit("MADLAD resize changed existing output embeddings")
    return {
        "old_vocab_size": old_size,
        "new_vocab_size": len(tokenizer),
        "added_tokens": len(new_tokens),
        "initialization": initialization,
    }


def validate_model_contract(model, tokenizer) -> dict[str, int]:
    expected = {
        "decoder_start_token_id": 0,
        "pad_token_id": 1,
        "eos_token_id": 2,
    }
    for field, value in expected.items():
        actual = int(getattr(model.config, field))
        if actual != value:
            raise SystemExit(f"MADLAD {field} changed: expected {value}, found {actual}")
        if getattr(model, "generation_config", None) is not None:
            setattr(model.generation_config, field, value)
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        raise SystemExit("MADLAD input embeddings do not match tokenizer")
    if model.get_output_embeddings().out_features != len(tokenizer):
        raise SystemExit("MADLAD untied output head does not match tokenizer")
    return expected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-en", type=Path, required=True)
    parser.add_argument("--input-zh", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--max-dialect-tags", type=int)
    parser.add_argument("--min-dialect-frequency", type=int)
    parser.add_argument("--audit-rows-per-language", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    if profile["model_family"] != "madlad400":
        raise SystemExit("MADLAD setup requires a madlad400 experiment profile")
    base_model = args.base_model or profile["base_model"]["name"]
    base_revision = (
        args.base_model_revision
        or profile["base_model"]["revision"]
    )
    if (
        base_model != profile["base_model"]["name"]
        or base_revision != profile["base_model"]["revision"]
    ):
        raise SystemExit("MADLAD base model and revision must match the profile")

    english = training_rows(
        args.input_en,
        target_col="english_sentence",
    )
    chinese = training_rows(
        args.input_zh,
        target_col="chinese_sentence",
    )
    combined = pd.concat([english, chinese], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=[
            column
            for column in (
                "row_id",
                "lang_code",
                "source",
                "dialect",
                "source_bucket",
            )
            if column in combined
        ]
    )
    languages = sorted(set(combined["lang_code"]) & set(FORMOSAN_CODES))
    if languages != list(FORMOSAN_CODES):
        missing = sorted(set(FORMOSAN_CODES) - set(languages))
        raise SystemExit(f"MADLAD setup is missing Formosan languages: {missing}")

    tokenizer_settings = profile["tokenizer"]
    maximum_dialects = (
        args.max_dialect_tags
        if args.max_dialect_tags is not None
        else int(tokenizer_settings["max_dialect_tags"])
    )
    minimum_dialect_frequency = (
        args.min_dialect_frequency
        if args.min_dialect_frequency is not None
        else int(tokenizer_settings["min_dialect_frequency"])
    )
    metadata_tokens = special_tokens_from_corpus(
        combined,
        max_dialect_tags=maximum_dialects,
        min_dialect_frequency=minimum_dialect_frequency,
    )
    desired_tokens = [
        *MADLAD_TARGET_TOKENS.values(),
        *madlad_formosan_target_tokens(),
        *metadata_tokens,
    ]
    desired_tokens = list(dict.fromkeys(desired_tokens))

    output_dir = args.output_dir.resolve()
    tokenizer_dir = output_dir / "tokenizer"
    model_dir = output_dir / "model"
    manifest_path = output_dir / "setup_manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("complete") is True:
            raise SystemExit(f"MADLAD setup is already complete: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        revision=base_revision,
        use_fast=False,
    )
    old_size = len(tokenizer)
    existing_target_ids = {
        token: token_id(tokenizer, token)
        for token in MADLAD_TARGET_TOKENS.values()
    }
    new_tokens = [
        token
        for token in desired_tokens
        if token not in tokenizer.get_vocab()
    ]
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": new_tokens},
        replace_additional_special_tokens=False,
    )
    if added != len(new_tokens) or len(tokenizer) != old_size + len(new_tokens):
        raise SystemExit("MADLAD special-token accounting failed")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model,
        revision=base_revision,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    resize_report = resize_and_initialize(
        model,
        tokenizer,
        new_tokens,
        old_size=old_size,
    )
    special_ids = validate_model_contract(model, tokenizer)
    for token, expected_id in existing_target_ids.items():
        if token_id(tokenizer, token) != expected_id:
            raise SystemExit(f"MADLAD base target-token ID changed for {token}")
    for token in desired_tokens:
        token_id(tokenizer, token)

    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(
        model_dir,
        safe_serialization=True,
        max_shard_size="5GB",
    )
    reloaded = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
    for token in desired_tokens:
        if token_id(reloaded, token) != token_id(tokenizer, token):
            raise SystemExit(f"MADLAD token ID changed after save/reload: {token}")

    audits: dict[str, dict[str, object]] = {}
    for short, path, target_col in (
        ("en", args.input_en, "english_sentence"),
        ("zh", args.input_zh, "chinese_sentence"),
    ):
        audit_json = output_dir / f"tokenizer_audit_{short}.json"
        audit_csv = output_dir / f"tokenizer_audit_{short}.csv"
        report = audit_tokenizer(
            tokenizer_dir=tokenizer_dir,
            input_csv=path,
            output_json=audit_json,
            output_csv=audit_csv,
            max_rows_per_lang=args.audit_rows_per_language,
            target_col=target_col,
            split="train",
        )
        audits[short] = {
            "json": str(audit_json),
            "json_sha256": sha256_file(audit_json),
            "csv": str(audit_csv),
            "csv_sha256": sha256_file(audit_csv),
            "macro_avg_pieces_per_word": report[
                "macro_avg_pieces_per_word"
            ],
            "macro_avg_pct_words_ge_5_pieces": report[
                "macro_avg_pct_words_ge_5_pieces"
            ],
        }

    manifest = {
        "schema_version": 2,
        "complete": True,
        "stage": "madlad_native_ready",
        "recipe_id": profile["recipe_id"],
        "model_family": profile["model_family"],
        "profile": profile_record(args.profile),
        "repository": git_record(),
        "runtime_dependencies": dependency_versions(),
        "inputs": [
            setup_input_record(
                args.input_en,
                english,
                target_lang="english",
                target_col="english_sentence",
            ),
            setup_input_record(
                args.input_zh,
                chinese,
                target_lang="chinese",
                target_col="chinese_sentence",
            ),
        ],
        "base_model": {
            "name": base_model,
            "revision": base_revision,
            "load_dtype": "bfloat16",
        },
        "tokenizer": {
            "path": str(tokenizer_dir),
            "files": artifact_inventory(tokenizer_dir),
            "languages": languages,
            "desired_special_tokens": desired_tokens,
            "new_special_tokens": new_tokens,
            "token_ids": {
                token: token_id(tokenizer, token)
                for token in desired_tokens
            },
            **resize_report,
        },
        "model": {
            "path": str(model_dir),
            "files": artifact_inventory(model_dir),
            "config_special_ids": special_ids,
            "input_embeddings": list(
                model.get_input_embeddings().weight.shape
            ),
            "output_embeddings": list(
                model.get_output_embeddings().weight.shape
            ),
        },
        "audits": audits,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(f"tokenizer: {tokenizer_dir}")
    print(f"model: {model_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
