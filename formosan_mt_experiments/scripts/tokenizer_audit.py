#!/usr/bin/env python3
"""Audit Formosan tokenization fragmentation for an NLLB tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import NllbTokenizer

from mt_common import DEFAULT_INPUT, FORMOSAN_CODES, read_parallel_csv, write_json


def audit_tokenizer(
    tokenizer_dir: Path,
    input_csv: Path,
    output_json: Path,
    output_csv: Path | None,
    max_rows_per_lang: int,
) -> dict:
    tok = NllbTokenizer.from_pretrained(tokenizer_dir)
    df = read_parallel_csv(input_csv)
    rows: list[dict] = []

    for lang in FORMOSAN_CODES:
        sub = df[df["lang_code"].eq(lang)]
        if max_rows_per_lang > 0 and len(sub) > max_rows_per_lang:
            sub = sub.sample(max_rows_per_lang, random_state=13)

        word_count = 0
        piece_count = 0
        long_word_count = 0
        unk_piece_count = 0
        total_sentences = 0
        total_chars = 0

        for sentence in sub["formosan_sentence"].fillna("").astype(str):
            total_sentences += 1
            total_chars += len(sentence)
            for word in sentence.split():
                if not word:
                    continue
                pieces = tok.tokenize(word)
                ids = tok.convert_tokens_to_ids(pieces)
                word_count += 1
                piece_count += len(pieces)
                long_word_count += int(len(pieces) >= 5)
                unk_piece_count += sum(1 for tid in ids if tid == tok.unk_token_id)

        rows.append(
            {
                "lang_code": lang,
                "sentences": int(total_sentences),
                "words": int(word_count),
                "chars": int(total_chars),
                "pieces": int(piece_count),
                "pieces_per_word": float(piece_count / max(word_count, 1)),
                "pct_words_ge_5_pieces": float(100.0 * long_word_count / max(word_count, 1)),
                "unk_pieces": int(unk_piece_count),
                "unk_pieces_per_10k_words": float(10000.0 * unk_piece_count / max(word_count, 1)),
            }
        )

    table = pd.DataFrame(rows).sort_values("pieces_per_word", ascending=False)
    summary = {
        "tokenizer": str(tokenizer_dir),
        "input": str(input_csv),
        "max_rows_per_lang": max_rows_per_lang,
        "languages": rows,
        "macro_avg_pieces_per_word": float(table["pieces_per_word"].mean()),
        "macro_avg_pct_words_ge_5_pieces": float(table["pct_words_ge_5_pieces"].mean()),
        "worst_languages_by_pieces_per_word": table.head(5).to_dict(orient="records"),
    }
    write_json(output_json, summary)
    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(output_csv, index=False)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--max-rows-per-lang",
        type=int,
        default=20000,
        help="Use 0 for all rows. Sampling keeps audits fast during sweeps.",
    )
    args = parser.parse_args()

    summary = audit_tokenizer(
        tokenizer_dir=args.tokenizer,
        input_csv=args.input,
        output_json=args.output_json,
        output_csv=args.output_csv,
        max_rows_per_lang=args.max_rows_per_lang,
    )
    print(f"macro pieces/word: {summary['macro_avg_pieces_per_word']:.3f}")
    print(f"report: {args.output_json}")


if __name__ == "__main__":
    main()
