#!/usr/bin/env python3
"""Summarize prediction CSVs from the production experiment evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mt_common import source_bucket, write_json

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
    _SACREBLEU_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
    BLEU = CHRF = TER = None  # type: ignore
    _SACREBLEU_ERROR = exc


def infer_columns(df: pd.DataFrame, direction: str) -> tuple[str, str]:
    if {"ref", "hyp"}.issubset(df.columns):
        return "ref", "hyp"
    if direction == "f2en" and {"ref_en", "hyp_en"}.issubset(df.columns):
        return "ref_en", "hyp_en"
    if direction == "en2f" and {"src_formosan", "hyp_formosan"}.issubset(df.columns):
        return "src_formosan", "hyp_formosan"
    raise SystemExit(f"Could not infer ref/hyp columns for direction={direction}. Columns={list(df.columns)}")


def score(hyp: list[str], ref: list[str]) -> dict:
    if _SACREBLEU_ERROR is not None:
        raise SystemExit(f"sacrebleu is required for prediction analysis: {_SACREBLEU_ERROR}")
    return {
        "BLEU": float(BLEU(tokenize="13a", effective_order=True).corpus_score(hyp, [ref]).score),
        "chrF2": float(CHRF().corpus_score(hyp, [ref]).score),
        "TER": float(TER().corpus_score(hyp, [ref]).score),
    }


def grouped(df: pd.DataFrame, ref_col: str, hyp_col: str, col: str) -> dict:
    if col not in df.columns:
        return {}
    out = {}
    for key, sub in df.groupby(col, dropna=False):
        out[str(key)] = {"samples": int(len(sub))} | score(sub[hyp_col].astype(str).tolist(), sub[ref_col].astype(str).tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--direction", choices=["f2en", "en2f"], required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, low_memory=False)
    if "source_bucket" not in df.columns and "source" in df.columns:
        df["source_bucket"] = df["source"].map(source_bucket)
    if "src_tokens" not in df.columns:
        src_col = "src" if "src" in df.columns else ("src_formosan" if args.direction == "f2en" else "ref_en")
        if src_col in df.columns:
            df["src_tokens"] = df[src_col].fillna("").astype(str).map(lambda s: len(s.split()))
            df["length_bin"] = pd.cut(
                df["src_tokens"],
                bins=[-1, 3, 8, 16, 32, 10**9],
                labels=["001_003", "004_008", "009_016", "017_032", "033_plus"],
            ).astype(str)

    ref_col, hyp_col = infer_columns(df, args.direction)
    summary = {
        "predictions": str(args.predictions),
        "direction": args.direction,
        "samples": int(len(df)),
        "global": {"samples": int(len(df))} | score(df[hyp_col].astype(str).tolist(), df[ref_col].astype(str).tolist()),
        "by_language": grouped(df, ref_col, hyp_col, "lang_code"),
        "by_source_bucket": grouped(df, ref_col, hyp_col, "source_bucket"),
        "by_length_bin": grouped(df, ref_col, hyp_col, "length_bin"),
    }
    write_json(args.output_json, summary)
    if args.output_csv:
        rows = []
        for section, values in summary.items():
            if isinstance(values, dict) and section.startswith("by_"):
                for key, metrics in values.items():
                    rows.append({"section": section, "key": key, **metrics})
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
    print(summary["global"])


if __name__ == "__main__":
    main()
