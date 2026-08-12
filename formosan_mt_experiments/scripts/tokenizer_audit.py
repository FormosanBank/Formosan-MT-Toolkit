#!/usr/bin/env python3
"""Audit Formosan tokenization and training-sequence lengths."""

from __future__ import annotations

import argparse
from pathlib import Path

import milmmt_runtime as milmmt
import pandas as pd
from mt_common import (
    FORMOSAN_CODES,
    is_formosan_to_target,
    is_target_to_formosan,
    normalize_target_language,
    read_parallel_csv,
    target_col_for,
    write_json,
)
from transformers import AutoTokenizer


def token_ids(tokenizer, text: object) -> list[int]:
    return list(
        tokenizer(
            str(text),
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )["input_ids"]
    )


def distribution(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"median": 0.0, "p95": 0.0, "maximum": 0}
    series = pd.Series(values, dtype="int64")
    return {
        "median": float(series.quantile(0.5)),
        "p95": float(series.quantile(0.95)),
        "maximum": int(series.max()),
    }


def audit_tokenizer(
    tokenizer_dir: Path,
    input_csv: Path,
    output_json: Path,
    output_csv: Path | None,
    max_rows_per_lang: int,
    target_col: str = "english_sentence",
    split: str | None = None,
    model_family: str | None = None,
    direction: str | None = None,
    max_length: int = 0,
) -> dict:
    if bool(model_family) != bool(direction):
        raise ValueError("model_family and direction must be supplied together")
    if model_family not in {None, "nllb", "milmmt"}:
        raise ValueError(f"Unsupported model family: {model_family}")
    tok = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        use_fast=model_family == "milmmt",
    )
    df = read_parallel_csv(input_csv, target_col=target_col)
    if split is not None:
        if "split" not in df:
            raise SystemExit(f"Tokenizer audit input has no split column: {input_csv}")
        df = df[
            df["split"].astype(str).str.strip().str.lower().eq(split)
        ].copy()
    rows: list[dict] = []
    sequence_records: list[dict[str, object]] = []

    for lang in FORMOSAN_CODES:
        sub = df[df["lang_code"].eq(lang)]
        if max_rows_per_lang > 0 and len(sub) > max_rows_per_lang:
            sub = sub.sample(max_rows_per_lang, random_state=13)

        word_count = 0
        long_word_count = 0
        unk_piece_count = 0
        unk_sentence_count = 0
        total_sentences = 0
        total_chars = 0
        sentence_piece_counts: list[int] = []
        target_piece_counts: list[int] = []
        training_sequence_counts: list[int] = []

        for _, record in sub.iterrows():
            sentence = str(record["formosan_sentence"])
            sentence_ids = token_ids(tok, sentence)
            target = str(record[target_col])
            target_ids = token_ids(tok, target)
            total_sentences += 1
            total_chars += len(sentence)
            sentence_piece_counts.append(len(sentence_ids))
            target_piece_counts.append(len(target_ids))
            unk_piece_count += sum(1 for token_id in sentence_ids if token_id == tok.unk_token_id)
            unk_sentence_count += int(tok.unk_token_id in sentence_ids)
            for word in sentence.split():
                if not word:
                    continue
                pieces = tok.tokenize(word)
                word_count += 1
                long_word_count += int(len(pieces) >= 5)
            if direction:
                if is_formosan_to_target(direction):
                    source_text, target_text = sentence, target
                elif is_target_to_formosan(direction):
                    source_text, target_text = target, sentence
                else:  # pragma: no cover - argparse and experiment profiles constrain this
                    raise ValueError(f"Unsupported direction: {direction}")
                if model_family == "milmmt":
                    target_lang = "chinese" if target_col == "chinese_sentence" else "english"
                    prompt = milmmt.format_source(
                        record,
                        source_text,
                        direction,
                        target_lang=target_lang,
                        use_tags=False,
                    )
                    sequence_length = len(token_ids(tok, prompt)) + len(token_ids(tok, target_text)) + 1
                else:
                    sequence_length = max(
                        len(token_ids(tok, source_text)),
                        len(token_ids(tok, target_text)),
                    )
                training_sequence_counts.append(sequence_length)
                sequence_records.append(
                    {
                        "lang_code": lang,
                        "row_id": str(record.get("row_id", "")),
                        "source": str(record.get("source", "")),
                        "tokens": sequence_length,
                    }
                )

        piece_count = sum(sentence_piece_counts)
        sentence_distribution = distribution(sentence_piece_counts)
        training_distribution = distribution(training_sequence_counts)
        over_max_length = (
            sum(length > max_length for length in training_sequence_counts)
            if max_length > 0
            else 0
        )

        rows.append(
            {
                "lang_code": lang,
                "sentences": int(total_sentences),
                "words": int(word_count),
                "chars": int(total_chars),
                "pieces": int(piece_count),
                "pieces_per_word": float(piece_count / max(word_count, 1)),
                "pieces_per_sentence": float(piece_count / max(total_sentences, 1)),
                "pieces_per_character": float(piece_count / max(total_chars, 1)),
                "sentence_pieces_median": sentence_distribution["median"],
                "sentence_pieces_p95": sentence_distribution["p95"],
                "sentence_pieces_max": sentence_distribution["maximum"],
                "formosan_to_target_piece_ratio": float(
                    piece_count / max(sum(target_piece_counts), 1)
                ),
                "pct_words_ge_5_pieces": float(100.0 * long_word_count / max(word_count, 1)),
                "unk_pieces": int(unk_piece_count),
                "pct_sentences_with_unk": float(
                    100.0 * unk_sentence_count / max(total_sentences, 1)
                ),
                "unk_pieces_per_10k_words": float(10000.0 * unk_piece_count / max(word_count, 1)),
                "training_sequence_tokens_median": training_distribution["median"],
                "training_sequence_tokens_p95": training_distribution["p95"],
                "training_sequence_tokens_max": training_distribution["maximum"],
                "training_examples_over_max_length": int(over_max_length),
                "pct_training_examples_over_max_length": float(
                    100.0 * over_max_length / max(len(training_sequence_counts), 1)
                ),
            }
        )

    table = pd.DataFrame(rows).sort_values("pieces_per_word", ascending=False)
    longest_examples = sorted(
        sequence_records,
        key=lambda record: int(record["tokens"]),
        reverse=True,
    )
    summary = {
        "tokenizer": str(tokenizer_dir),
        "input": str(input_csv),
        "max_rows_per_lang": max_rows_per_lang,
        "split": split,
        "model_family": model_family,
        "direction": direction,
        "max_length": max_length,
        "languages": rows,
        "longest_training_examples": longest_examples[:20],
        "training_examples_over_max_length": (
            [record for record in longest_examples if int(record["tokens"]) > max_length][
                :100
            ]
            if max_length > 0
            else []
        ),
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
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument(
        "--max-rows-per-lang",
        type=int,
        default=20000,
        help="Use 0 for all rows. Sampling keeps audits fast during sweeps.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validate", "test"],
        default=None,
    )
    parser.add_argument("--model-family", choices=["nllb", "milmmt"], default=None)
    parser.add_argument("--direction", choices=["f2en", "en2f", "f2zh", "zh2f"], default=None)
    parser.add_argument(
        "--max-length",
        type=int,
        default=0,
        help="Report training examples exceeding this length; 0 disables the count.",
    )
    args = parser.parse_args()
    target_lang = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_lang)

    summary = audit_tokenizer(
        tokenizer_dir=args.tokenizer,
        input_csv=args.input,
        output_json=args.output_json,
        output_csv=args.output_csv,
        max_rows_per_lang=args.max_rows_per_lang,
        target_col=target_col,
        split=args.split,
        model_family=args.model_family,
        direction=args.direction,
        max_length=args.max_length,
    )
    print(f"macro pieces/word: {summary['macro_avg_pieces_per_word']:.3f}")
    print(f"report: {args.output_json}")


if __name__ == "__main__":
    main()
