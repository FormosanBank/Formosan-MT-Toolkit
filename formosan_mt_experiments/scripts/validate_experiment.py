#!/usr/bin/env python3
"""Validate experiment CSV leakage and tokenizer control tags."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import NllbTokenizer

from mt_common import (
    add_normalized_columns,
    build_prefix,
    direction_choices,
    normalize_target_language,
    read_parallel_csv,
    source_bucket,
    target_col_for,
    write_json,
)


def validate_splits(df: pd.DataFrame, target_col: str, target_lang: str) -> dict:
    keyed = add_normalized_columns(df, target_col=target_col, target_lang=target_lang)
    split = keyed["split"].astype(str).str.lower()
    train = keyed[split.eq("train")]
    eval_df = keyed[split.isin(["validate", "valid", "val", "test"])]
    lexeme_eval_rows = int(
        (
            keyed["row_type"].astype(str).str.lower().isin({"lexeme", "morpheme"})
            & split.isin(["validate", "valid", "val", "test"])
        ).sum()
    )
    report = {}
    for name, col in (("formosan", "_formosan_key"), ("target", "_target_key"), ("pair", "_pair_key")):
        overlap = set(train[col]) & set(eval_df[col])
        report[name] = {
            "train_unique": int(train[col].nunique()),
            "eval_unique": int(eval_df[col].nunique()),
            "overlap_unique": int(len(overlap)),
        }
    skeleton_report = {}
    for name, col in (
        ("formosan", "_formosan_skeleton"),
        ("target", "_target_skeleton"),
        ("pair", "_pair_skeleton"),
    ):
        overlap = set(train[col]) & set(eval_df[col])
        skeleton_report[name] = {
            "train_unique": int(train[col].nunique()),
            "eval_unique": int(eval_df[col].nunique()),
            "overlap_unique": int(len(overlap)),
        }
    failed = {k: v["overlap_unique"] for k, v in report.items() if v["overlap_unique"]}
    skeleton_failed = {
        k: v["overlap_unique"] for k, v in skeleton_report.items() if v["overlap_unique"]
    }
    return {
        "ok": not failed and not skeleton_failed and lexeme_eval_rows == 0,
        "overlaps": report,
        "skeleton_overlaps": skeleton_report,
        "failures": failed,
        "skeleton_failures": skeleton_failed,
        "lexeme_eval_rows": lexeme_eval_rows,
    }


def validate_tags(df: pd.DataFrame, tokenizer_dir: Path, direction: str, target_lang: str) -> dict:
    tok = NllbTokenizer.from_pretrained(tokenizer_dir)
    work = df.copy()
    if "source_bucket" not in work.columns:
        work["source_bucket"] = work["source"].map(source_bucket)
    tags = set()
    for _, row in work.iterrows():
        tags.update(build_prefix(row, direction, target_lang=target_lang).split())
    bad = []
    for token in sorted(tags):
        tid = tok.convert_tokens_to_ids(token)
        if tid == tok.unk_token_id or tok.convert_ids_to_tokens(tid) != token:
            bad.append(token)
    return {
        "ok": not bad,
        "direction": direction,
        "checked_tags": int(len(tags)),
        "bad_tags": bad,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--direction", choices=direction_choices() + ["dae"], default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    target_lang = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_lang)
    df = read_parallel_csv(args.input, target_col=target_col)
    if "split" not in df.columns:
        raise SystemExit("Input must have split column.")
    split_report = validate_splits(df, target_col=target_col, target_lang=target_lang)
    report = {"input": str(args.input), "split_validation": split_report}
    if args.tokenizer and args.direction:
        report["tag_validation"] = validate_tags(df, args.tokenizer, args.direction, target_lang=target_lang)
    if args.output_json:
        write_json(args.output_json, report)
    print(report)
    if not split_report["ok"] or ("tag_validation" in report and not report["tag_validation"]["ok"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
