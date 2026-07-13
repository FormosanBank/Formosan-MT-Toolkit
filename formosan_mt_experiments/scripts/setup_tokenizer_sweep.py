#!/usr/bin/env python3
"""Run SPM tokenizer/model setup sweeps and add experiment control tags."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
from mt_common import (
    DEFAULT_INPUT,
    DEFAULT_SETUP_SCRIPT,
    normalize_target_language,
    read_parallel_csv,
    special_tokens_from_corpus,
    target_col_for,
    write_json,
)
from tokenizer_audit import audit_tokenizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def parse_split_list(value: str) -> set[str]:
    return {v.strip().lower() for v in value.split(",") if v.strip()}


def add_experiment_special_tokens(
    tokenizer_dir: Path,
    model_dir: Path,
    input_csv: Path,
    target_col: str,
    max_dialect_tags: int,
    min_dialect_frequency: int,
) -> dict:
    df = read_parallel_csv(input_csv, target_col=target_col)
    desired_tokens = special_tokens_from_corpus(
        df,
        max_dialect_tags=max_dialect_tags,
        min_dialect_frequency=min_dialect_frequency,
    )

    tok = NllbTokenizer.from_pretrained(tokenizer_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    existing_vocab = tok.get_vocab()
    new_tokens = [t for t in desired_tokens if t not in existing_vocab]
    old_size = len(tok)
    added = 0
    if new_tokens:
        try:
            added = tok.add_special_tokens(
                {"additional_special_tokens": new_tokens},
                replace_additional_special_tokens=False,
            )
        except TypeError:
            tok.add_special_tokens(
                {"additional_special_tokens": list(tok.additional_special_tokens) + new_tokens}
            )
            added = len(tok) - old_size

    if len(tok) != model.get_input_embeddings().num_embeddings:
        old_embeddings = model.get_input_embeddings().weight.detach().clone()
        model.resize_token_embeddings(len(tok))
        with torch.no_grad():
            emb = model.get_input_embeddings().weight
            if len(tok) > old_embeddings.shape[0]:
                mean = old_embeddings.mean(dim=0)
                noise = old_embeddings.std(dim=0).clamp_min(1e-6) * 0.01
                emb[old_embeddings.shape[0] :] = mean + torch.randn_like(emb[old_embeddings.shape[0] :]) * noise

    model.config.decoder_start_token_id = tok.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tok.eos_token_id

    bad = []
    for token in desired_tokens:
        tid = tok.convert_tokens_to_ids(token)
        if tid == tok.unk_token_id or tok.convert_ids_to_tokens(tid) != token:
            bad.append(token)
    if bad:
        raise SystemExit(f"Tokenizer failed to preserve experiment tags as single tokens: {bad[:20]}")

    tok.save_pretrained(tokenizer_dir)
    model.save_pretrained(model_dir)
    return {
        "tokenizer": str(tokenizer_dir),
        "model": str(model_dir),
        "desired_special_tokens": len(desired_tokens),
        "new_special_tokens": int(added),
        "vocab_size": int(len(tok)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--setup-script", type=Path, default=DEFAULT_SETUP_SCRIPT)
    parser.add_argument("--base-model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("formosan_mt_experiments/data/tokenizer_sweep"),
    )
    parser.add_argument("--spm-vocabs", default="8192,16384,32768")
    parser.add_argument("--min-char-frequency", type=int, default=3)
    parser.add_argument("--max-dialect-tags", type=int, default=200)
    parser.add_argument("--min-dialect-frequency", type=int, default=3)
    parser.add_argument(
        "--setup-splits",
        default="train,validate,valid,val",
        help=(
            "Comma-separated split values used to train/extend the tokenizer. "
            "Default avoids test-set text. Use 'all' to include every row."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-smoke", action="store_true")
    parser.add_argument("--samples-per-lang", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    target_lang = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_lang)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = {}

    setup_input = args.input
    temp_dir_obj = None
    setup_df = None
    probe = pd.read_csv(args.input, nrows=5, low_memory=False)
    setup_splits = parse_split_list(args.setup_splits)
    if setup_splits and setup_splits != {"all"}:
        full_df = pd.read_csv(args.input, low_memory=False)
        if "split" not in full_df.columns:
            raise SystemExit("--setup-splits was set but input has no split column.")
        setup_df = full_df[full_df["split"].astype(str).str.lower().isin(setup_splits)].copy()
        if setup_df.empty:
            raise SystemExit(f"No rows matched --setup-splits={args.setup_splits!r}.")
    if "chinese_sentence" not in probe.columns or setup_df is not None:
        # The legacy setup script still requires this column even for English
        # setup runs. Stage a compatibility copy rather than modifying corpus.
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="formosan_mt_setup_")
        setup_input = Path(temp_dir_obj.name) / "setup_input.csv"
        df = setup_df if setup_df is not None else pd.read_csv(args.input, low_memory=False)
        if "chinese_sentence" not in df.columns:
            df["chinese_sentence"] = ""
        df.to_csv(setup_input, index=False)

    for vocab in parse_int_list(args.spm_vocabs):
        prefix = args.output_dir / f"formosan_multilingual_nllb_spm{vocab}"
        tokenizer_dir = Path(f"{prefix}_tokenizer")
        model_dir = Path(f"{prefix}_model")
        cmd = [
            args.python,
            str(args.setup_script),
            "--input",
            str(setup_input),
            "--output-prefix",
            str(prefix),
            "--base-model",
            args.base_model,
            "--add-mode",
            "spm",
            "--spm-vocab",
            str(vocab),
            "--min-char-frequency",
            str(args.min_char_frequency),
        ]
        if args.run_smoke:
            cmd.extend(["--run-eval", "--samples-per-lang", str(args.samples_per_lang), "--also-eng"])

        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)
            tag_report = add_experiment_special_tokens(
                tokenizer_dir=tokenizer_dir,
                model_dir=model_dir,
                input_csv=args.input,
                target_col=target_col,
                max_dialect_tags=args.max_dialect_tags,
                min_dialect_frequency=args.min_dialect_frequency,
            )
            audit_json = args.output_dir / f"tokenizer_audit_spm{vocab}.json"
            audit_csv = args.output_dir / f"tokenizer_audit_spm{vocab}.csv"
            audit_report = audit_tokenizer(
                tokenizer_dir=tokenizer_dir,
                input_csv=args.input,
                output_json=audit_json,
                output_csv=audit_csv,
                max_rows_per_lang=20000,
                target_col=target_col,
            )
            reports[str(vocab)] = {
                "setup_prefix": str(prefix),
                "tokenizer": str(tokenizer_dir),
                "model": str(model_dir),
                "tags": tag_report,
                "audit": {
                    "json": str(audit_json),
                    "csv": str(audit_csv),
                    "macro_avg_pieces_per_word": audit_report["macro_avg_pieces_per_word"],
                    "macro_avg_pct_words_ge_5_pieces": audit_report["macro_avg_pct_words_ge_5_pieces"],
                },
            }

    if not args.dry_run:
        write_json(args.output_dir / "tokenizer_sweep_report.json", reports)
        print(f"report: {args.output_dir / 'tokenizer_sweep_report.json'}")
    if temp_dir_obj is not None:
        temp_dir_obj.cleanup()


if __name__ == "__main__":
    main()
