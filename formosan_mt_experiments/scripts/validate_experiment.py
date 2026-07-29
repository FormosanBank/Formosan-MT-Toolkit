#!/usr/bin/env python3
"""Independently validate a release MT corpus and its control tags."""

from __future__ import annotations

import argparse
import math
import unicodedata
from collections import Counter
from itertools import chain
from pathlib import Path

import pandas as pd
from experiment_config import (
    DEFAULT_PROFILE,
    load_profile,
    profile_record,
    sha256_file,
)
from mt_common import (
    build_prefix,
    direction_choices,
    normalize_target_language,
    read_parallel_csv,
    source_bucket,
    target_col_for,
    write_json,
)

EVAL_SPLITS = {"validate", "valid", "val", "test"}
REQUIRED_PROVENANCE = {
    "row_id",
    "source_record_id",
    "repository",
    "repository_commit",
    "xml_path",
    "xml_id",
    "xml_element_index",
    "kindOf",
    "standard_origin",
    "standard_after_qc_sha256",
    "qc_transform_id",
    "qc_revision",
    "row_type",
    "source",
    "dialect",
}


def normalize(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(text.split())


def skeleton(value: object) -> str:
    return "".join(
        character
        for character in normalize(value)
        if unicodedata.category(character)[0] in {"L", "N", "M"}
    )


def add_validation_keys(
    frame: pd.DataFrame,
    *,
    target_col: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["_formosan_key"] = output["formosan_sentence"].map(normalize)
    output["_target_key"] = output[target_col].map(normalize)
    output["_pair_key"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_formosan_key"]
        + "\u241f"
        + output["_target_key"]
    )
    output["_formosan_skeleton"] = output["formosan_sentence"].map(skeleton)
    output["_target_skeleton"] = output[target_col].map(skeleton)
    output["_pair_skeleton"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_formosan_skeleton"]
        + "\u241f"
        + output["_target_skeleton"]
    )
    output["_document_key"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["source"].astype(str)
    )
    return output


def overlap_count(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> int:
    return len(set(left[column].astype(str)) & set(right[column].astype(str)))


def deletion_keys(value: str) -> set[str]:
    return {
        value[:position] + value[position + 1 :]
        for position in range(len(value))
    }


def one_edit_conflict_count(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
) -> int:
    """Count candidate rows at Levenshtein distance at most one."""
    if reference.empty or candidates.empty:
        return 0
    conflicts: set[int] = set()
    reference_groups = (
        reference.groupby("lang_code", sort=False)
        if by_language
        else [("_global", reference)]
    )
    candidate_groups = (
        {
            str(language): group
            for language, group in candidates.groupby("lang_code", sort=False)
        }
        if by_language
        else {"_global": candidates}
    )
    for language, reference_group in reference_groups:
        candidate_group = candidate_groups.get(str(language))
        if candidate_group is None:
            continue
        exact: set[str] = set()
        deleted: set[str] = set()
        for value in reference_group[column].astype(str):
            if not value:
                continue
            exact.add(value)
            deleted.update(deletion_keys(value))
        for index, value in candidate_group[column].astype(str).items():
            if not value:
                continue
            if value in exact or value in deleted:
                conflicts.add(int(index))
                continue
            if deletion_keys(value) & (exact | deleted):
                conflicts.add(int(index))
    return len(conflicts)


def char_ngrams(value: str, size: int = 4) -> frozenset[str]:
    if not value:
        return frozenset()
    if len(value) <= size:
        return frozenset({value})
    return frozenset(
        value[position : position + size]
        for position in range(len(value) - size + 1)
    )


def jaccard_prefix(
    grams: frozenset[str],
    frequency: Counter[str],
    threshold: float,
) -> tuple[str, ...]:
    prefix_length = len(grams) - math.ceil(threshold * len(grams)) + 1
    return tuple(
        sorted(grams, key=lambda gram: (frequency[gram], gram))[
            : max(1, prefix_length)
        ]
    )


def ngram_conflict_count(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
    threshold: float,
) -> int:
    """Find high character n-gram similarity with an exact prefix join."""
    if reference.empty or candidates.empty:
        return 0
    conflicts: set[int] = set()
    reference_groups = (
        reference.groupby("lang_code", sort=False)
        if by_language
        else [("_global", reference)]
    )
    candidate_groups = (
        {
            str(language): group
            for language, group in candidates.groupby("lang_code", sort=False)
        }
        if by_language
        else {"_global": candidates}
    )
    for language, reference_group in reference_groups:
        candidate_group = candidate_groups.get(str(language))
        if candidate_group is None:
            continue
        candidate_grams = {
            int(index): char_ngrams(value)
            for index, value in candidate_group[column].astype(str).items()
            if len(value) >= 8
        }
        reference_grams = {
            value: char_ngrams(value)
            for value in set(reference_group[column].astype(str))
            if len(value) >= 8
        }
        frequency: Counter[str] = Counter()
        for grams in chain(
            candidate_grams.values(),
            reference_grams.values(),
        ):
            frequency.update(grams)
        prefix_index: dict[str, list[int]] = {}
        for index, grams in candidate_grams.items():
            for gram in jaccard_prefix(grams, frequency, threshold):
                prefix_index.setdefault(gram, []).append(index)
        for grams in reference_grams.values():
            possible: set[int] = set()
            for gram in jaccard_prefix(grams, frequency, threshold):
                possible.update(prefix_index.get(gram, ()))
            for index in possible:
                other = candidate_grams[index]
                if not (
                    threshold * len(grams)
                    <= len(other)
                    <= len(grams) / threshold
                ):
                    continue
                union = len(grams | other)
                if union and len(grams & other) / union >= threshold:
                    conflicts.add(index)
    return len(conflicts)


def pairwise_leakage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    ngram_threshold: float,
) -> dict[str, object]:
    exact = {
        name: overlap_count(left, right, column)
        for name, column in (
            ("formosan", "_formosan_key"),
            ("target", "_target_key"),
            ("pair", "_pair_key"),
        )
    }
    skeleton_overlap = {
        name: overlap_count(left, right, column)
        for name, column in (
            ("formosan", "_formosan_skeleton"),
            ("target", "_target_skeleton"),
            ("pair", "_pair_skeleton"),
        )
    }
    one_edit = {
        "formosan": one_edit_conflict_count(
            left,
            right,
            "_formosan_skeleton",
            by_language=True,
        ),
        "target": one_edit_conflict_count(
            left,
            right,
            "_target_skeleton",
            by_language=False,
        ),
    }
    character_ngram = {
        "formosan": ngram_conflict_count(
            left,
            right,
            "_formosan_skeleton",
            by_language=True,
            threshold=ngram_threshold,
        ),
        "target": ngram_conflict_count(
            left,
            right,
            "_target_skeleton",
            by_language=False,
            threshold=ngram_threshold,
        ),
    }
    return {
        "exact_overlap": exact,
        "skeleton_overlap": skeleton_overlap,
        "one_edit_conflicting_rows": one_edit,
        "character_ngram_conflicting_rows": character_ngram,
        "document_overlap": overlap_count(left, right, "_document_key"),
        "ok": not any(
            value
            for family in (exact, skeleton_overlap, one_edit, character_ngram)
            for value in family.values()
        )
        and overlap_count(left, right, "_document_key") == 0,
    }


def validate_provenance(frame: pd.DataFrame) -> dict[str, object]:
    missing_columns = sorted(REQUIRED_PROVENANCE - set(frame.columns))
    empty_counts = {
        column: int(frame[column].astype(str).str.strip().eq("").sum())
        for column in sorted(REQUIRED_PROVENANCE & set(frame.columns))
        if column not in {"xml_id", "dialect"}
    }
    empty_counts = {
        column: count
        for column, count in empty_counts.items()
        if count
    }
    duplicate_row_ids = (
        int(frame["row_id"].astype(str).duplicated().sum())
        if "row_id" in frame
        else len(frame)
    )
    non_standard = (
        int(
            (
                ~frame["kindOf"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("standard")
            ).sum()
        )
        if "kindOf" in frame
        else len(frame)
    )
    return {
        "missing_columns": missing_columns,
        "empty_required_values": empty_counts,
        "duplicate_row_ids": duplicate_row_ids,
        "non_standard_rows": non_standard,
        "ok": (
            not missing_columns
            and not empty_counts
            and duplicate_row_ids == 0
            and non_standard == 0
        ),
    }


def validate_splits(
    frame: pd.DataFrame,
    *,
    target_col: str,
    min_test_ratio: float,
    min_validate_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
    ngram_threshold: float,
) -> dict[str, object]:
    keyed = add_validation_keys(frame, target_col=target_col)
    split = keyed["split"].astype(str).str.strip().str.lower()
    unknown_splits = sorted(set(split) - {"train", "validate", "test"})
    train = keyed[split.eq("train")]
    test = keyed[split.eq("test")]
    validate = keyed[split.eq("validate")]
    evaluation = keyed[split.isin(EVAL_SPLITS)]

    pivot_origin = keyed.get(
        "pivot_origin",
        pd.Series("original", index=keyed.index),
    ).astype(str)
    synthetic_eval_rows = int(
        (pivot_origin.eq("synthetic") & split.isin(EVAL_SPLITS)).sum()
    )
    lexical_eval_rows = int(
        (
            keyed["row_type"]
            .astype(str)
            .str.lower()
            .isin({"lexeme", "morpheme"})
            & split.isin(EVAL_SPLITS)
        ).sum()
    )
    non_sentence_eval_rows = int(
        (
            ~keyed["row_type"].astype(str).str.lower().eq("sentence")
            & split.isin(EVAL_SPLITS)
        ).sum()
    )

    ratios: dict[str, dict[str, object]] = {}
    ratio_failures: dict[str, dict[str, object]] = {}
    for language, group in keyed.groupby("lang_code", sort=True):
        group_split = group["split"].astype(str).str.lower()
        total = len(group)
        test_rows = int(group_split.eq("test").sum())
        validate_rows = int(group_split.eq("validate").sum())
        required_test = max(math.ceil(total * min_test_ratio), min_test_rows)
        required_validate = max(
            math.ceil(total * min_validate_ratio),
            min_validate_rows,
        )
        values = {
            "rows": total,
            "train": int(group_split.eq("train").sum()),
            "test": test_rows,
            "validate": validate_rows,
            "test_ratio": test_rows / total,
            "validate_ratio": validate_rows / total,
            "required_test": required_test,
            "required_validate": required_validate,
        }
        ratios[str(language)] = values
        if test_rows < required_test or validate_rows < required_validate:
            ratio_failures[str(language)] = values

    train_eval = pairwise_leakage(
        train,
        evaluation,
        ngram_threshold=ngram_threshold,
    )
    validate_test = pairwise_leakage(
        test,
        validate,
        ngram_threshold=ngram_threshold,
    )
    duplicate_pairs = int(keyed["_pair_key"].duplicated().sum())
    ok = (
        not unknown_splits
        and not ratio_failures
        and synthetic_eval_rows == 0
        and lexical_eval_rows == 0
        and non_sentence_eval_rows == 0
        and duplicate_pairs == 0
        and bool(train_eval["ok"])
        and bool(validate_test["ok"])
    )
    return {
        "ok": ok,
        "unknown_splits": unknown_splits,
        "duplicate_pairs": duplicate_pairs,
        "synthetic_eval_rows": synthetic_eval_rows,
        "lexical_eval_rows": lexical_eval_rows,
        "non_sentence_eval_rows": non_sentence_eval_rows,
        "train_evaluation": train_eval,
        "validate_test": validate_test,
        "minimum_ratios": {
            "test": min_test_ratio,
            "validate": min_validate_ratio,
            "min_test_rows": min_test_rows,
            "min_validate_rows": min_validate_rows,
        },
        "ratios_by_language": ratios,
        "ratio_failures": ratio_failures,
        "ngram_jaccard_threshold": ngram_threshold,
    }


def validate_tags(
    frame: pd.DataFrame,
    tokenizer_dir: Path,
    direction: str,
    target_lang: str,
) -> dict[str, object]:
    from transformers import NllbTokenizer

    tokenizer = NllbTokenizer.from_pretrained(tokenizer_dir)
    work = frame.copy()
    if "source_bucket" not in work.columns:
        work["source_bucket"] = work["source"].map(source_bucket)
    tags: set[str] = set()
    for _, row in work.iterrows():
        tags.update(
            build_prefix(
                row,
                direction,
                target_lang=target_lang,
            ).split()
        )
    bad = []
    for token in sorted(tags):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id == tokenizer.unk_token_id
            or tokenizer.convert_ids_to_tokens(token_id) != token
        ):
            bad.append(token)
    return {
        "ok": not bad,
        "direction": direction,
        "checked_tags": len(tags),
        "bad_tags": bad,
    }


def parse_args() -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
    )
    known, _ = preliminary.parse_known_args()
    profile = load_profile(known.profile)
    split_defaults = profile["splits"]
    parser = argparse.ArgumentParser(
        parents=[preliminary],
        description="Independently validate a release Formosan MT corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--target-lang",
        choices=["english", "chinese"],
        default=None,
    )
    parser.add_argument("--target-col")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--direction", choices=direction_choices())
    parser.add_argument(
        "--min-test-ratio",
        type=float,
        default=split_defaults["test_ratio"],
    )
    parser.add_argument(
        "--min-validate-ratio",
        type=float,
        default=split_defaults["validate_ratio"],
    )
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=split_defaults["min_test_rows"],
    )
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=split_defaults["min_validate_rows"],
    )
    parser.add_argument(
        "--ngram-jaccard-threshold",
        type=float,
        default=split_defaults[
            "character_ngram_jaccard_threshold"
        ],
    )
    parser.add_argument("--report", "--output-json", dest="report", type=Path)
    parser.add_argument(
        "--require-human-eval",
        action="store_true",
        help="Compatibility flag; human-only evaluation is always required.",
    )
    parser.add_argument(
        "--require-document-holdout-report",
        action="store_true",
        help="Compatibility flag; document disjointness is always required.",
    )
    args = parser.parse_args()
    args.profile = known.profile
    return args


def main() -> None:
    args = parse_args()
    target_lang = normalize_target_language(
        args.target_lang,
        args.target_col,
    )
    target_col = args.target_col or target_col_for(target_lang)
    frame = read_parallel_csv(args.input, target_col=target_col)
    if "split" not in frame:
        raise SystemExit("Input must have a split column")

    provenance = validate_provenance(frame)
    split_validation = validate_splits(
        frame,
        target_col=target_col,
        min_test_ratio=args.min_test_ratio,
        min_validate_ratio=args.min_validate_ratio,
        min_test_rows=args.min_test_rows,
        min_validate_rows=args.min_validate_rows,
        ngram_threshold=args.ngram_jaccard_threshold,
    )
    report: dict[str, object] = {
        "schema_version": 2,
        "complete": bool(provenance["ok"] and split_validation["ok"]),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "target_language": target_lang,
        "target_column": target_col,
        "profile": profile_record(args.profile),
        "rows": len(frame),
        "provenance_validation": provenance,
        "split_validation": split_validation,
    }
    if args.tokenizer or args.direction:
        if not args.tokenizer or not args.direction:
            raise SystemExit("--tokenizer and --direction must be provided together")
        report["tag_validation"] = validate_tags(
            frame,
            args.tokenizer,
            args.direction,
            target_lang,
        )
        report["complete"] = bool(
            report["complete"]
            and report["tag_validation"]["ok"]
        )
    if args.report:
        write_json(args.report, report)
    print(report)
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
