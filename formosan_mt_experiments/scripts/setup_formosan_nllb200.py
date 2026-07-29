#!/usr/bin/env python3
"""Build the pinned train-only NLLB-200 plus Formosan SPM8k base."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
import sentencepiece as spm
import torch
from experiment_config import (
    DEFAULT_PROFILE,
    dependency_versions,
    git_record,
    load_profile,
    profile_record,
    sha256_file,
)
from mt_common import CODE_TO_LID, FORMOSAN_CODES, read_parallel_csv
from sentencepiece import sentencepiece_model_pb2
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


def clean_training_text(value: object) -> str:
    return " ".join(
        unicodedata.normalize("NFC", str(value or "")).split()
    )


def load_training_rows(
    path: Path,
    *,
    target_col: str,
) -> pd.DataFrame:
    frame = read_parallel_csv(path, target_col=target_col)
    required = {"split", "kindOf", "row_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Tokenizer input is missing columns: {missing}")
    if not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Tokenizer input contains non-standard Formosan rows")
    train = frame[
        frame["split"].astype(str).str.strip().str.lower().eq("train")
    ].copy()
    if train.empty:
        raise SystemExit("Tokenizer input has no training rows")
    return train


def write_training_text(
    frame: pd.DataFrame,
    *,
    path: Path,
) -> Counter[str]:
    character_counts: Counter[str] = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for value in frame["formosan_sentence"].tolist():
            text = clean_training_text(value)
            if not text:
                continue
            handle.write(text + "\n")
            character_counts.update(
                character
                for character in text
                if not character.isspace()
            )
    return character_counts


def train_auxiliary_spm(
    input_text: Path,
    *,
    output_model: Path,
    vocab_size: int,
    required_chars: str,
) -> None:
    prefix = output_model.with_suffix("")
    spm.SentencePieceTrainer.train(
        input=str(input_text),
        model_prefix=str(prefix),
        vocab_size=vocab_size,
        character_coverage=1.0,
        num_threads=max(1, torch.get_num_threads()),
        add_dummy_prefix=False,
        max_sentencepiece_length=128,
        max_sentence_length=16_768,
        hard_vocab_limit=False,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        required_chars=required_chars,
    )
    generated = prefix.with_suffix(".model")
    if generated != output_model:
        generated.replace(output_model)
    prefix.with_suffix(".vocab").unlink(missing_ok=True)


def merge_sentencepiece(
    tokenizer: NllbTokenizer,
    auxiliary_model: Path,
    output_model: Path,
) -> int:
    base = sentencepiece_model_pb2.ModelProto()
    base.ParseFromString(tokenizer.sp_model.serialized_model_proto())
    auxiliary = sentencepiece_model_pb2.ModelProto()
    auxiliary.ParseFromString(auxiliary_model.read_bytes())
    existing = {piece.piece for piece in base.pieces}
    minimum_score = min(piece.score for piece in base.pieces)
    added = 0
    for piece in auxiliary.pieces:
        if getattr(piece, "type", 1) != 1 or piece.piece in existing:
            continue
        new_piece = base.pieces.add()
        new_piece.piece = piece.piece
        new_piece.score = minimum_score + piece.score
        new_piece.type = piece.type
        existing.add(piece.piece)
        added += 1
    output_model.write_bytes(base.SerializeToString())
    return added


def rebuild_tokenizer(
    base: NllbTokenizer,
    *,
    merged_spm: Path,
    language_codes: list[str],
) -> NllbTokenizer:
    with tempfile.TemporaryDirectory(prefix="formosan_nllb_tokenizer_") as tmp:
        directory = Path(tmp)
        base.save_pretrained(directory)
        shutil.copy2(merged_spm, directory / "sentencepiece.bpe.model")
        tokenizer = NllbTokenizer.from_pretrained(directory, use_fast=False)
    missing = [
        code
        for code in language_codes
        if code not in tokenizer.get_vocab()
    ]
    if missing:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,
        )
    return tokenizer


def source_piece_ids(
    tokenizer: NllbTokenizer,
    token: str,
) -> list[int]:
    surface = token.replace("▁", " ").strip() or token
    values = tokenizer(
        surface,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [
        int(value)
        for value in values
        if int(value) != tokenizer.unk_token_id
    ] or [int(tokenizer.unk_token_id)]


def realign_embeddings(
    model,
    old_tokenizer: NllbTokenizer,
    new_tokenizer: NllbTokenizer,
    language_codes: set[str],
) -> dict[str, int]:
    old_vocab = old_tokenizer.get_vocab()
    new_vocab = new_tokenizer.get_vocab()
    old_embeddings = model.get_input_embeddings().weight.detach().clone()
    model.resize_token_embeddings(len(new_tokenizer))
    embeddings = model.get_input_embeddings().weight
    copied = 0
    initialized = 0
    seeded_languages = 0
    english_id = old_vocab["eng_Latn"]
    with torch.no_grad():
        for token, new_id in new_vocab.items():
            if token in old_vocab:
                embeddings[new_id] = old_embeddings[old_vocab[token]]
                copied += 1
            elif token in language_codes:
                embeddings[new_id] = old_embeddings[english_id]
                seeded_languages += 1
            else:
                ids = source_piece_ids(old_tokenizer, token)
                embeddings[new_id] = old_embeddings[ids].mean(dim=0)
                initialized += 1
    model.tie_weights()
    return {
        "shared_tokens_realigned": copied,
        "new_piece_rows_initialized": initialized,
        "formosan_language_rows_seeded_from_english": seeded_languages,
    }


def artifact_record(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the supported Formosan NLLB SPM8k base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--base-model")
    parser.add_argument("--base-model-revision")
    parser.add_argument("--spm-vocab", type=int, default=8192)
    parser.add_argument("--min-char-frequency", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    expected_vocab = int(profile["tokenizer"]["default_spm_vocab"])
    if args.spm_vocab != expected_vocab:
        raise SystemExit(
            f"Supported recipe requires --spm-vocab {expected_vocab}"
        )
    base_model = args.base_model or profile["base_model"]["name"]
    base_revision = (
        args.base_model_revision
        or profile["base_model"]["revision"]
    )
    if (
        base_model != profile["base_model"]["name"]
        or base_revision != profile["base_model"]["revision"]
    ):
        raise SystemExit(
            "Base model and revision must match the experiment profile"
        )

    train = load_training_rows(args.input, target_col=args.target_col)
    languages = sorted(set(train["lang_code"]) & set(FORMOSAN_CODES))
    if not languages:
        raise SystemExit("No supported Formosan languages in training rows")
    language_codes = [CODE_TO_LID[language] for language in languages]
    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = Path(f"{output_prefix}_tokenizer")
    model_dir = Path(f"{output_prefix}_model")
    manifest_path = Path(f"{output_prefix}_setup_manifest.json")

    with tempfile.TemporaryDirectory(prefix="formosan_spm8k_") as temporary:
        temporary_dir = Path(temporary)
        training_text = temporary_dir / "training.txt"
        character_counts = write_training_text(
            train,
            path=training_text,
        )
        required_chars = "".join(
            sorted(
                character
                for character, count in character_counts.items()
                if count >= args.min_char_frequency
            )
        )
        auxiliary = temporary_dir / "auxiliary.model"
        train_auxiliary_spm(
            training_text,
            output_model=auxiliary,
            vocab_size=args.spm_vocab,
            required_chars=required_chars,
        )

        base_tokenizer = NllbTokenizer.from_pretrained(
            base_model,
            revision=base_revision,
            use_fast=False,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model,
            revision=base_revision,
        )
        merged = temporary_dir / "merged.model"
        added_pieces = merge_sentencepiece(
            base_tokenizer,
            auxiliary,
            merged,
        )
        tokenizer = rebuild_tokenizer(
            base_tokenizer,
            merged_spm=merged,
            language_codes=language_codes,
        )
        embedding_report = realign_embeddings(
            model,
            base_tokenizer,
            tokenizer,
            set(language_codes),
        )

    model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = (
            tokenizer.eos_token_id
        )
    model.to(torch.device(args.device))
    shutil.rmtree(tokenizer_dir, ignore_errors=True)
    shutil.rmtree(model_dir, ignore_errors=True)
    tokenizer.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir, safe_serialization=True)

    tokenizer_files = [
        artifact_record(path)
        for path in sorted(tokenizer_dir.iterdir())
        if path.is_file()
    ]
    model_files = [
        artifact_record(path)
        for path in sorted(model_dir.iterdir())
        if path.is_file()
    ]
    manifest = {
        "schema_version": 2,
        "complete": False,
        "stage": "base_spm_ready",
        "recipe_id": profile["recipe_id"],
        "profile": profile_record(args.profile),
        "repository": git_record(),
        "runtime_dependencies": dependency_versions(),
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": len(train),
            "split": "train",
            "target_column": args.target_col,
            "tokenizer_training_columns": ["formosan_sentence"],
        },
        "base_model": {
            "name": base_model,
            "revision": base_revision,
        },
        "tokenizer": {
            "vocab_size": len(tokenizer),
            "auxiliary_spm_vocab": args.spm_vocab,
            "added_spm_pieces": added_pieces,
            "minimum_character_frequency": args.min_char_frequency,
            "languages": languages,
            "language_codes": language_codes,
            **embedding_report,
            "path": str(tokenizer_dir),
            "files": tokenizer_files,
        },
        "model": {
            "path": str(model_dir),
            "files": model_files,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"tokenizer: {tokenizer_dir}")
    print(f"model: {model_dir}")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
