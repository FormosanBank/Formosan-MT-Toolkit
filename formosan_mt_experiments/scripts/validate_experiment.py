#!/usr/bin/env python3
"""Independently validate a release MT corpus and its control tags."""

from __future__ import annotations

import argparse
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd
from experiment_config import (
    DEFAULT_PROFILE,
    load_corpus_pipeline_config,
    load_profile,
    profile_record,
    sha256_file,
)
from model_backends import get_backend
from mt_common import (
    add_normalized_columns,
    bool_series,
    direction_choices,
    evaluation_candidate_mask,
    mt_standard_contract,
    normalize_target_language,
    read_parallel_csv,
    source_bucket,
    source_corpus,
    target_col_for,
    write_json,
)

SPLIT_DEFAULTS = load_corpus_pipeline_config()["splits"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "local"))
from corpus_quality import has_annotation_gloss_structure  # noqa: E402

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
    "standard_namespace",
    "standard_origin",
    "standard_after_qc_sha256",
    "qc_transform_id",
    "qc_revision",
    "row_type",
    "formosan_original_raw",
    "formosan_source_standard",
    "formosan_mt_standard",
    "mt_standard_sha256",
    "mt_normalization_status",
    "mt_normalization_confidence",
    "mt_eval_eligible",
    "mt_standard_profile",
    "mt_standard_profile_sha256",
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
    output["_formosan_task_key"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_formosan_key"]
    )
    output["_target_task_key"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_target_key"]
    )
    output["_formosan_task_skeleton"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_formosan_skeleton"]
    )
    output["_target_task_skeleton"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_target_skeleton"]
    )
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
        reference_neighborhood = exact | deleted
        for index, value in candidate_group[column].astype(str).items():
            if not value:
                continue
            if value in reference_neighborhood:
                conflicts.add(int(index))
                continue
            if not deletion_keys(value).isdisjoint(reference_neighborhood):
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
    threshold: float,
) -> tuple[str, ...]:
    prefix_length = len(grams) - math.ceil(threshold * len(grams)) + 1
    return tuple(sorted(grams)[: max(1, prefix_length)])


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
        if len(candidate_group) <= len(reference_group):
            indexed = {
                int(index): char_ngrams(value)
                for index, value in candidate_group[column].astype(str).items()
                if len(value) >= 8
            }
            prefix_index: dict[str, list[int]] = {}
            for index, grams in indexed.items():
                for gram in jaccard_prefix(grams, threshold):
                    prefix_index.setdefault(gram, []).append(index)
            for value in set(reference_group[column].astype(str)):
                if len(value) < 8:
                    continue
                grams = char_ngrams(value)
                possible: set[int] = set()
                for gram in jaccard_prefix(grams, threshold):
                    possible.update(prefix_index.get(gram, ()))
                for index in possible:
                    other = indexed[index]
                    if not (
                        threshold * len(grams)
                        <= len(other)
                        <= len(grams) / threshold
                    ):
                        continue
                    union = len(grams | other)
                    if union and len(grams & other) / union >= threshold:
                        conflicts.add(index)
        else:
            indexed = {
                position: char_ngrams(value)
                for position, value in enumerate(
                    set(reference_group[column].astype(str))
                )
                if len(value) >= 8
            }
            prefix_index = {}
            for position, grams in indexed.items():
                for gram in jaccard_prefix(grams, threshold):
                    prefix_index.setdefault(gram, []).append(position)
            for index, value in candidate_group[column].astype(str).items():
                if len(value) < 8:
                    continue
                grams = char_ngrams(value)
                possible: set[int] = set()
                for gram in jaccard_prefix(grams, threshold):
                    possible.update(prefix_index.get(gram, ()))
                for position in possible:
                    other = indexed[position]
                    if not (
                        threshold * len(grams)
                        <= len(other)
                        <= len(grams) / threshold
                    ):
                        continue
                    union = len(grams | other)
                    if union and len(grams & other) / union >= threshold:
                        conflicts.add(int(index))
                        break
    return len(conflicts)


def pairwise_leakage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    ngram_threshold: float,
    formosan_by_language: bool = True,
    target_by_language: bool = False,
) -> dict[str, object]:
    formosan_key = (
        "_formosan_task_key"
        if formosan_by_language
        else "_formosan_key"
    )
    target_key = (
        "_target_task_key"
        if target_by_language
        else "_target_key"
    )
    formosan_skeleton = (
        "_formosan_task_skeleton"
        if formosan_by_language
        else "_formosan_skeleton"
    )
    target_skeleton = (
        "_target_task_skeleton"
        if target_by_language
        else "_target_skeleton"
    )
    exact = {
        name: overlap_count(left, right, column)
        for name, column in (
            ("formosan", formosan_key),
            ("target", target_key),
            ("pair", "_pair_key"),
        )
    }
    skeleton_overlap = {
        name: overlap_count(left, right, column)
        for name, column in (
            ("formosan", formosan_skeleton),
            ("target", target_skeleton),
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
            by_language=target_by_language,
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
            by_language=target_by_language,
            threshold=ngram_threshold,
        ),
    }
    return {
        "exact_overlap": exact,
        "skeleton_overlap": skeleton_overlap,
        "one_edit_conflicting_rows": one_edit,
        "character_ngram_conflicting_rows": character_ngram,
        "document_overlap": overlap_count(left, right, "_document_key"),
        "formosan_by_language": formosan_by_language,
        "target_by_language": target_by_language,
        "ok": not any(
            value
            for family in (exact, skeleton_overlap, one_edit, character_ngram)
            for value in family.values()
        ),
        "document_disjoint": (
            overlap_count(left, right, "_document_key") == 0
        ),
    }


def validate_provenance(frame: pd.DataFrame) -> dict[str, object]:
    missing_columns = sorted(REQUIRED_PROVENANCE - set(frame.columns))
    empty_counts = {
        column: int(frame[column].astype(str).str.strip().eq("").sum())
        for column in sorted(REQUIRED_PROVENANCE & set(frame.columns))
        if column not in {"xml_id", "dialect", "formosan_original_raw"}
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
    try:
        mt_profile = mt_standard_contract(frame, context="corpus validation input")
        mt_contract_error = ""
    except SystemExit as exc:
        mt_profile = {}
        mt_contract_error = str(exc)
    return {
        "missing_columns": missing_columns,
        "empty_required_values": empty_counts,
        "duplicate_row_ids": duplicate_row_ids,
        "non_standard_rows": non_standard,
        "mt_standardization": mt_profile,
        "mt_standard_contract_error": mt_contract_error,
        "ok": (
            not missing_columns
            and not empty_counts
            and duplicate_row_ids == 0
            and non_standard == 0
            and not mt_contract_error
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
    min_formosan_tokens: int = SPLIT_DEFAULTS["min_formosan_tokens"],
    min_target_tokens: int = SPLIT_DEFAULTS["min_target_tokens"],
    source_ratio_tolerance: float = SPLIT_DEFAULTS["source_ratio_tolerance"],
    require_human_eval: bool = False,
    require_document_holdout: bool = False,
) -> dict[str, object]:
    target_language = (
        "chinese" if target_col == "chinese_sentence" else "english"
    )
    normalized = add_normalized_columns(
        frame,
        target_col=target_col,
        target_lang=target_language,
    )
    keyed = add_validation_keys(normalized, target_col=target_col)
    if "source_corpus" in keyed.columns:
        keyed["_source_corpus"] = keyed["source_corpus"].astype(str)
    else:
        keyed["_source_corpus"] = keyed["source"].map(source_corpus)
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
    mt_eval_eligible = bool_series(
        keyed["mt_eval_eligible"],
        context="corpus validation input:mt_eval_eligible",
    )
    mt_ineligible_eval_rows = int((~mt_eval_eligible & split.isin(EVAL_SPLITS)).sum())
    ambiguous_normalization_eval_rows = int(
        (
            keyed["mt_normalization_confidence"].astype(str).eq("ambiguous")
            & split.isin(EVAL_SPLITS)
        ).sum()
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
    translation_kind = keyed.get(
        "translation_kind",
        pd.Series("", index=keyed.index),
    ).astype(str).str.strip().str.casefold().str.replace(
        r"[\s_]+", "-", regex=True
    )
    gloss_translation_rows = int(
        translation_kind.isin({"gloss", "interlinear-gloss"}).sum()
    )
    annotation_gloss_rows = int(
        keyed[target_col].astype(str).map(has_annotation_gloss_structure).sum()
        if target_language == "english"
        else 0
    )
    candidate = evaluation_candidate_mask(
        keyed,
        min_formosan_tokens=min_formosan_tokens,
        min_target_tokens=min_target_tokens,
    )
    lexical_like_eval_rows = int(
        (split.isin(EVAL_SPLITS) & ~candidate).sum()
    )

    ratios: dict[str, dict[str, object]] = {}
    ratio_failures: dict[str, dict[str, object]] = {}
    for language, group in keyed.groupby("lang_code", sort=True):
        group_candidate = candidate.loc[group.index]
        eligible_group = group[group_candidate]
        group_split = eligible_group["split"].astype(str).str.lower()
        total = len(eligible_group)
        test_rows = int(group_split.eq("test").sum())
        validate_rows = int(group_split.eq("validate").sum())
        required_test = max(math.ceil(total * min_test_ratio), min_test_rows)
        required_validate = max(
            math.ceil(total * min_validate_ratio),
            min_validate_rows,
        )
        values = {
            "rows": total,
            "all_rows": len(group),
            "train": int(group_split.eq("train").sum()),
            "test": test_rows,
            "validate": validate_rows,
            "test_ratio": test_rows / max(total, 1),
            "validate_ratio": validate_rows / max(total, 1),
            "required_test": required_test,
            "required_validate": required_validate,
        }
        ratios[str(language)] = values
        if test_rows < required_test or validate_rows < required_validate:
            ratio_failures[str(language)] = values

    source_ratios: dict[str, dict[str, dict[str, object]]] = {}
    source_ratio_failures: dict[str, dict[str, dict[str, object]]] = {}
    source_distribution_tvd: dict[str, dict[str, float]] = {}
    for language, language_frame in keyed.groupby("lang_code", sort=True):
        language_key = str(language)
        language_candidate = language_frame[
            candidate.loc[language_frame.index]
        ]
        eligible_distribution = Counter(language_candidate["_source_corpus"])
        source_ratios[language_key] = {}
        source_ratio_failures[language_key] = {}
        source_distribution_tvd[language_key] = {}
        for split_name in ("test", "validate"):
            distribution = Counter(
                language_candidate[
                    language_candidate["split"].astype(str).str.lower().eq(split_name)
                ]["_source_corpus"]
            )
            total = sum(distribution.values())
            eligible_total = sum(eligible_distribution.values())
            source_distribution_tvd[language_key][split_name] = 0.5 * sum(
                abs(
                    eligible_distribution[bucket] / max(eligible_total, 1)
                    - distribution[bucket] / max(total, 1)
                )
                for bucket in set(eligible_distribution) | set(distribution)
            )
        for bucket, source_frame in language_candidate.groupby(
            "_source_corpus", sort=True
        ):
            bucket_key = str(bucket)
            source_split = source_frame["split"].astype(str).str.lower()
            total = len(source_frame)
            test_rows = int(source_split.eq("test").sum())
            validate_rows = int(source_split.eq("validate").sum())
            row_tolerance = max(1, math.ceil(total * source_ratio_tolerance))
            required_test = max(0, math.floor(total * min_test_ratio) - row_tolerance)
            required_validate = max(
                0,
                math.floor(total * min_validate_ratio) - row_tolerance,
            )
            allowed_test = math.ceil(total * min_test_ratio) + row_tolerance
            allowed_validate = (
                math.ceil(total * min_validate_ratio) + row_tolerance
            )
            values = {
                "eligible_sentence_rows": total,
                "train": int(source_split.eq("train").sum()),
                "test": test_rows,
                "validate": validate_rows,
                "test_ratio": test_rows / total,
                "validate_ratio": validate_rows / total,
                "test_bounds": [required_test, allowed_test],
                "validate_bounds": [required_validate, allowed_validate],
            }
            source_ratios[language_key][bucket_key] = values
            if not (
                required_test <= test_rows <= allowed_test
                and required_validate <= validate_rows <= allowed_validate
            ):
                source_ratio_failures[language_key][bucket_key] = values
        if not source_ratio_failures[language_key]:
            source_ratio_failures.pop(language_key)

    train_eval = pairwise_leakage(
        train,
        evaluation,
        ngram_threshold=ngram_threshold,
    )
    validate_test = pairwise_leakage(
        test,
        validate,
        ngram_threshold=ngram_threshold,
        target_by_language=True,
    )
    validate_test_cross_language_diagnostic = pairwise_leakage(
        test,
        validate,
        ngram_threshold=ngram_threshold,
    )
    duplicate_pairs = int(keyed["_pair_key"].duplicated().sum())
    ok = (
        not unknown_splits
        and not ratio_failures
        and not source_ratio_failures
        and (not require_human_eval or synthetic_eval_rows == 0)
        and mt_ineligible_eval_rows == 0
        and ambiguous_normalization_eval_rows == 0
        and lexical_eval_rows == 0
        and non_sentence_eval_rows == 0
        and lexical_like_eval_rows == 0
        and gloss_translation_rows == 0
        and annotation_gloss_rows == 0
        and duplicate_pairs == 0
        and bool(train_eval["ok"])
        and bool(validate_test["ok"])
        and (
            not require_document_holdout
            or (
                bool(train_eval["document_disjoint"])
                and bool(validate_test["document_disjoint"])
            )
        )
    )
    return {
        "ok": ok,
        "unknown_splits": unknown_splits,
        "duplicate_pairs": duplicate_pairs,
        "synthetic_eval_rows": synthetic_eval_rows,
        "synthetic_eval_allowed": not require_human_eval,
        "mt_ineligible_eval_rows": mt_ineligible_eval_rows,
        "ambiguous_normalization_eval_rows": ambiguous_normalization_eval_rows,
        "lexical_eval_rows": lexical_eval_rows,
        "non_sentence_eval_rows": non_sentence_eval_rows,
        "lexical_like_eval_rows": lexical_like_eval_rows,
        "gloss_translation_rows": gloss_translation_rows,
        "annotation_gloss_rows": annotation_gloss_rows,
        "train_evaluation": train_eval,
        "validate_test": validate_test,
        "validate_test_cross_language_diagnostic": (
            validate_test_cross_language_diagnostic
        ),
        "minimum_ratios": {
            "test": min_test_ratio,
            "validate": min_validate_ratio,
            "min_test_rows": min_test_rows,
            "min_validate_rows": min_validate_rows,
        },
        "ratios_by_language": ratios,
        "ratio_failures": ratio_failures,
        "ratios_by_language_and_source": source_ratios,
        "source_ratio_failures": source_ratio_failures,
        "source_distribution_total_variation": source_distribution_tvd,
        "source_ratio_tolerance": source_ratio_tolerance,
        "ngram_jaccard_threshold": ngram_threshold,
    }


def validate_tags(
    frame: pd.DataFrame,
    tokenizer_dir: Path,
    direction: str,
    target_lang: str,
    profile: dict,
) -> dict[str, object]:
    backend = get_backend(profile)
    tokenizer = backend.load_tokenizer(tokenizer_dir)
    work = frame.copy()
    work["source_bucket"] = work["source"].map(source_bucket)
    tags: set[str] = set()
    for _, row in work.iterrows():
        tags.update(
            backend.source_prefix(
                row,
                direction,
                target_lang=target_lang,
                use_tags=True,
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
        "model_family": backend.family,
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
    load_profile(known.profile)
    split_defaults = SPLIT_DEFAULTS
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
    parser.add_argument(
        "--min-formosan-tokens",
        type=int,
        default=split_defaults["min_formosan_tokens"],
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=split_defaults["min_target_tokens"],
    )
    parser.add_argument(
        "--source-ratio-tolerance",
        type=float,
        default=split_defaults["source_ratio_tolerance"],
        help="Allowed per-source split deviation in addition to one row.",
    )
    parser.add_argument("--report", "--output-json", dest="report", type=Path)
    parser.add_argument(
        "--require-human-eval",
        action="store_true",
        help="Reject synthetic pivot references in evaluation.",
    )
    parser.add_argument(
        "--require-document-holdout-report",
        action="store_true",
        help="Require source XML documents to be disjoint across splits.",
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
        min_formosan_tokens=args.min_formosan_tokens,
        min_target_tokens=args.min_target_tokens,
        source_ratio_tolerance=args.source_ratio_tolerance,
        require_human_eval=args.require_human_eval,
        require_document_holdout=args.require_document_holdout_report,
    )
    report: dict[str, object] = {
        "schema_version": 3,
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
            load_profile(args.profile),
        )
        report["complete"] = bool(
            report["complete"]
            and report["tag_validation"]["ok"]
        )
    if args.report:
        write_json(args.report, report)
    leakage = split_validation["train_evaluation"]
    print(
        f"Corpus validation: {'PASS' if report['complete'] else 'FAIL'} "
        f"({len(frame):,} rows, {target_lang})"
    )
    print(
        "  eval: "
        f"synthetic={split_validation['synthetic_eval_rows']:,}, "
        f"lexical-like={split_validation['lexical_like_eval_rows']:,}, "
        f"gloss={split_validation['gloss_translation_rows'] + split_validation['annotation_gloss_rows']:,}"
    )
    print(
        "  train/eval conflicts: "
        f"exact={sum(leakage['exact_overlap'].values()):,}, "
        f"skeleton={sum(leakage['skeleton_overlap'].values()):,}, "
        f"one-edit={sum(leakage['one_edit_conflicting_rows'].values()):,}, "
        f"char-ngram={sum(leakage['character_ngram_conflicting_rows'].values()):,}"
    )
    print(
        "  ratio failures: "
        f"languages={len(split_validation['ratio_failures']):,}, "
        "sources="
        f"{sum(len(rows) for rows in split_validation['source_ratio_failures'].values()):,}"
    )
    if args.report:
        print(f"  report: {args.report}")
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
