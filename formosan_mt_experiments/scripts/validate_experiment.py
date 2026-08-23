#!/usr/bin/env python3
"""Independently validate a release MT corpus and its control tags."""

from __future__ import annotations

import argparse
import json
import math
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import nllb_runtime as nllb
import pandas as pd
from experiment_config import (
    DEFAULT_PROFILE,
    load_corpus_pipeline_config,
    load_profile,
    profile_record,
    sha256_file,
)
from mt_common import (
    add_normalized_columns,
    bool_series,
    direction_choices,
    evaluation_candidate_mask,
    mt_standard_contract,
    normalize_target_language,
    read_parallel_csv,
    source_corpus,
    target_col_for,
    weighted_apportioned_counts,
    write_json,
)
from validation_similarity import ValidationNgramIndex

PIPELINE_DEFAULTS = load_corpus_pipeline_config()
SPLIT_DEFAULTS = PIPELINE_DEFAULTS["splits"]
MAX_TRAINING_UNITS_PER_SIDE = PIPELINE_DEFAULTS["cleaning"][
    "max_training_units_per_side"
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "local"))
from corpus_quality import (  # noqa: E402
    english_language_quality,
    has_malformed_escaping,
    has_unbalanced_target_delimiters,
    lexical_quality_reason,
    target_gloss_reason,
    target_metadata_reason,
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
    "standard_namespace",
    "standard_origin",
    "standard_after_qc_sha256",
    "qc_transform_id",
    "qc_revision",
    "row_type",
    "xml_unit_context",
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


def pairwise_leakage(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    formosan_ngram_index: ValidationNgramIndex,
    target_ngram_index: ValidationNgramIndex,
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
        "formosan": len(
            formosan_ngram_index.conflicts(left.index, right.index)
        ),
        "target": len(
            target_ngram_index.conflicts(
                left.index,
                right.index,
                same_language=target_by_language,
            )
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
    min_combined_tokens: int = SPLIT_DEFAULTS["min_combined_tokens"],
    min_punctuated_combined_tokens: int = SPLIT_DEFAULTS[
        "min_punctuated_combined_tokens"
    ],
    max_eval_units_per_side: int = SPLIT_DEFAULTS["max_eval_units_per_side"],
    source_ratio_tolerance: float = SPLIT_DEFAULTS["source_ratio_tolerance"],
    require_human_eval: bool = True,
    require_document_holdout: bool = False,
    split_report: dict[str, object] | None = None,
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
    formosan_ngram_index = ValidationNgramIndex(
        keyed,
        "_formosan_key",
        by_language=True,
        threshold=ngram_threshold,
    )
    target_ngram_index = ValidationNgramIndex(
        keyed,
        "_target_key",
        by_language=False,
        threshold=ngram_threshold,
    )

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
        keyed[target_col]
        .astype(str)
        .map(
            lambda target: bool(
                target_gloss_reason(
                    target,
                    translation_kind="",
                    target_language=target_language,
                )
            )
        )
        .sum()
    )
    target_metadata_rows = sum(
        bool(target_metadata_reason(target, source=source))
        for source, target in zip(
            keyed["formosan_sentence"].astype(str),
            keyed[target_col].astype(str),
            strict=True,
        )
    )
    malformed_escaping_rows = int(
        keyed["formosan_sentence"].astype(str).map(has_malformed_escaping).sum()
        + keyed[target_col].astype(str).map(has_malformed_escaping).sum()
    )
    target_language_mismatch_rows = 0
    uncertain_target_language_eval_rows = 0
    if target_language == "english":
        english_quality = keyed[target_col].astype(str).map(
            english_language_quality
        )
        target_language_mismatch_rows = int(
            english_quality.map(lambda result: bool(result[0])).sum()
        )
        uncertain_target_language_eval_rows = int(
            (
                split.isin(EVAL_SPLITS)
                & english_quality.map(
                    lambda result: "english_language_uncertain" in result[1]
                )
            ).sum()
        )
    unbalanced_target_eval_rows = int(
        (
            split.isin(EVAL_SPLITS)
            & keyed[target_col].astype(str).map(has_unbalanced_target_delimiters)
        ).sum()
    )
    unit_context = keyed.get(
        "xml_unit_context",
        pd.Series("", index=keyed.index),
    )
    lexical_mask = keyed["row_type"].astype(str).str.casefold().isin(
        {"lexeme", "morpheme"}
    )
    lexical_rows = keyed[lexical_mask]
    lexical_quality_rows = sum(
        bool(
            lexical_quality_reason(
                target,
                row_type=row_type,
                xml_unit_context=context,
                target_language=target_language,
            )
        )
        for target, row_type, context in zip(
            lexical_rows[target_col].astype(str),
            lexical_rows["row_type"],
            unit_context.loc[lexical_rows.index],
            strict=True,
        )
    )
    candidate = evaluation_candidate_mask(
        keyed,
        min_formosan_tokens=min_formosan_tokens,
        min_target_tokens=min_target_tokens,
        min_combined_tokens=min_combined_tokens,
        min_punctuated_combined_tokens=min_punctuated_combined_tokens,
        max_eval_units_per_side=max_eval_units_per_side,
    )
    lexical_like_eval_rows = int(
        (split.isin(EVAL_SPLITS) & ~candidate).sum()
    )
    model_length_overflow_rows = int(
        (
            keyed["_formosan_tokens"].gt(MAX_TRAINING_UNITS_PER_SIDE)
            | keyed["_target_tokens"].gt(MAX_TRAINING_UNITS_PER_SIDE)
        ).sum()
    )

    ratios: dict[str, dict[str, object]] = {}
    ratio_failures: dict[str, dict[str, object]] = {}
    split_report_errors: list[str] = []
    report_languages = (
        split_report.get("languages", {})
        if isinstance(split_report, dict)
        else {}
    )
    if split_report is not None:
        if split_report.get("complete") is not True:
            split_report_errors.append("split report is incomplete")
        if split_report.get("ratio_basis") != SPLIT_DEFAULTS["ratio_basis"]:
            split_report_errors.append("split report ratio basis does not match policy")
        if (
            split_report.get("synthetic_eval_policy")
            != SPLIT_DEFAULTS["synthetic_eval_policy"]
        ):
            split_report_errors.append(
                "split report synthetic evaluation policy does not match policy"
            )
        expected_length_policy = {
            "min_formosan_tokens": min_formosan_tokens,
            "min_target_tokens": min_target_tokens,
            "min_combined_tokens": min_combined_tokens,
            "min_punctuated_combined_tokens": min_punctuated_combined_tokens,
            "max_eval_units_per_side": max_eval_units_per_side,
        }
        if split_report.get("evaluation_length_policy") != expected_length_policy:
            split_report_errors.append(
                "split report evaluation length policy does not match validation"
            )
    for language, group in keyed.groupby("lang_code", sort=True):
        language_key = str(language)
        group_candidate = candidate.loc[group.index]
        eligible_group = group[group_candidate]
        group_split = group["split"].astype(str).str.lower()
        report_language = report_languages.get(language_key, {})
        total = int(report_language.get("rows_total", len(group)))
        test_rows = int(group_split.eq("test").sum())
        validate_rows = int(group_split.eq("validate").sum())
        required_test = int(
            report_language.get(
                "target_test_rows",
                max(math.ceil(total * min_test_ratio), min_test_rows),
            )
        )
        required_validate = int(
            report_language.get(
                "target_validate_rows",
                max(math.ceil(total * min_validate_ratio), min_validate_rows),
            )
        )
        values = {
            "rows": total,
            "output_rows": len(group),
            "eligible_sentence_rows": len(eligible_group),
            "train": int(group_split.eq("train").sum()),
            "test": test_rows,
            "validate": validate_rows,
            "test_ratio": test_rows / max(total, 1),
            "validate_ratio": validate_rows / max(total, 1),
            "required_test": required_test,
            "required_validate": required_validate,
        }
        ratios[language_key] = values
        ratio_mismatch = (
            test_rows != required_test or validate_rows != required_validate
            if split_report is not None
            else test_rows < required_test or validate_rows < required_validate
        )
        if ratio_mismatch:
            ratio_failures[language_key] = values
    missing_report_languages = sorted(set(report_languages) - set(ratios))
    if missing_report_languages:
        split_report_errors.append(
            "split report languages absent from corpus: "
            + ", ".join(missing_report_languages)
        )

    source_ratios: dict[str, dict[str, dict[str, object]]] = {}
    source_ratio_failures: dict[str, dict[str, dict[str, object]]] = {}
    source_distribution_tvd: dict[str, dict[str, float]] = {}
    for language, language_frame in keyed.groupby("lang_code", sort=True):
        language_key = str(language)
        language_synthetic = pivot_origin.loc[language_frame.index].eq("synthetic")
        human_language = language_frame[~language_synthetic]
        eligible_language = language_frame[
            candidate.loc[language_frame.index] & ~language_synthetic
        ]
        report_sources = (
            split_report.get("source_strata", {}).get(language_key, {})
            if isinstance(split_report, dict)
            else {}
        )
        if report_sources:
            all_distribution = Counter(
                {
                    str(source): int(
                        values.get("human_input_rows", values["input_rows"])
                    )
                    for source, values in report_sources.items()
                }
            )
            evaluation_targets = {
                str(source): int(values["target_test_rows"])
                + int(values["target_validate_rows"])
                for source, values in report_sources.items()
            }
            validate_targets = {
                str(source): int(values["target_validate_rows"])
                for source, values in report_sources.items()
            }
        else:
            all_distribution = Counter(human_language["_source_corpus"])
            eligible_distribution = Counter(eligible_language["_source_corpus"])
            language_test = max(
                math.ceil(len(language_frame) * min_test_ratio),
                min_test_rows,
            )
            language_validate = max(
                math.ceil(len(language_frame) * min_validate_ratio),
                min_validate_rows,
            )
            evaluation_targets = weighted_apportioned_counts(
                all_distribution,
                eligible_distribution,
                language_test + language_validate,
            )
            validate_targets = weighted_apportioned_counts(
                all_distribution,
                evaluation_targets,
                language_validate,
            )
        source_ratios[language_key] = {}
        source_ratio_failures[language_key] = {}
        source_distribution_tvd[language_key] = {}
        for split_name in ("test", "validate"):
            distribution = Counter(
                eligible_language[
                    eligible_language["split"].astype(str).str.lower().eq(split_name)
                ]["_source_corpus"]
            )
            total = sum(distribution.values())
            all_total = sum(all_distribution.values())
            source_distribution_tvd[language_key][split_name] = 0.5 * sum(
                abs(
                    all_distribution[source] / max(all_total, 1)
                    - distribution[source] / max(total, 1)
                )
                for source in set(all_distribution) | set(distribution)
            )
        output_sources = {
            str(source): source_frame
            for source, source_frame in language_frame.groupby(
                "_source_corpus", sort=True
            )
        }
        for source in sorted(set(output_sources) | set(all_distribution)):
            source_frame = output_sources.get(
                source,
                language_frame.iloc[0:0],
            )
            source_key = str(source)
            source_split = source_frame["split"].astype(str).str.lower()
            total = int(all_distribution.get(source_key, len(source_frame)))
            source_human = ~pivot_origin.loc[source_frame.index].eq("synthetic")
            eligible_rows = int(
                (candidate.loc[source_frame.index] & source_human).sum()
            )
            test_rows = int(source_split.eq("test").sum())
            validate_rows = int(source_split.eq("validate").sum())
            target_validate = validate_targets.get(source_key, 0)
            target_test = evaluation_targets.get(source_key, 0) - target_validate
            row_tolerance = int(
                report_sources.get(source_key, {}).get(
                    "assignment_tolerance_rows",
                    max(1, math.ceil(total * source_ratio_tolerance)),
                )
            )
            required_test = max(0, target_test - row_tolerance)
            required_validate = max(0, target_validate - row_tolerance)
            allowed_test = target_test + row_tolerance
            allowed_validate = target_validate + row_tolerance
            values = {
                "rows": total,
                "eligible_sentence_rows": eligible_rows,
                "train": int(source_split.eq("train").sum()),
                "test": test_rows,
                "validate": validate_rows,
                "test_ratio": test_rows / max(total, 1),
                "validate_ratio": validate_rows / max(total, 1),
                "target_test": target_test,
                "target_validate": target_validate,
                "test_bounds": [required_test, allowed_test],
                "validate_bounds": [required_validate, allowed_validate],
            }
            source_ratios[language_key][source_key] = values
            if not (
                required_test <= test_rows <= allowed_test
                and required_validate <= validate_rows <= allowed_validate
            ):
                source_ratio_failures[language_key][source_key] = values
        if not source_ratio_failures[language_key]:
            source_ratio_failures.pop(language_key)

    train_eval = pairwise_leakage(
        train,
        evaluation,
        target_by_language=True,
        formosan_ngram_index=formosan_ngram_index,
        target_ngram_index=target_ngram_index,
    )
    validate_test = pairwise_leakage(
        test,
        validate,
        target_by_language=True,
        formosan_ngram_index=formosan_ngram_index,
        target_ngram_index=target_ngram_index,
    )
    validate_test_cross_language_diagnostic = pairwise_leakage(
        test,
        validate,
        formosan_ngram_index=formosan_ngram_index,
        target_ngram_index=target_ngram_index,
    )
    duplicate_pairs = int(keyed["_pair_key"].duplicated().sum())
    ok = (
        not unknown_splits
        and not ratio_failures
        and not source_ratio_failures
        and not split_report_errors
        and (not require_human_eval or synthetic_eval_rows == 0)
        and mt_ineligible_eval_rows == 0
        and ambiguous_normalization_eval_rows == 0
        and lexical_eval_rows == 0
        and non_sentence_eval_rows == 0
        and lexical_like_eval_rows == 0
        and model_length_overflow_rows == 0
        and gloss_translation_rows == 0
        and annotation_gloss_rows == 0
        and target_metadata_rows == 0
        and malformed_escaping_rows == 0
        and target_language_mismatch_rows == 0
        and uncertain_target_language_eval_rows == 0
        and unbalanced_target_eval_rows == 0
        and lexical_quality_rows == 0
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
        "model_length_overflow_rows": model_length_overflow_rows,
        "gloss_translation_rows": gloss_translation_rows,
        "annotation_gloss_rows": annotation_gloss_rows,
        "target_metadata_rows": target_metadata_rows,
        "malformed_escaping_rows": malformed_escaping_rows,
        "target_language_mismatch_rows": target_language_mismatch_rows,
        "uncertain_target_language_eval_rows": uncertain_target_language_eval_rows,
        "unbalanced_target_eval_rows": unbalanced_target_eval_rows,
        "lexical_quality_rows": lexical_quality_rows,
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
        "ratio_basis": SPLIT_DEFAULTS["ratio_basis"],
        "split_report_errors": split_report_errors,
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
) -> dict[str, object]:
    tokenizer = nllb.load_tokenizer(tokenizer_dir)
    tags: set[str] = set()
    for _, row in frame.iterrows():
        tags.update(
            nllb.source_prefix(
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
        "model_family": nllb.MODEL_FAMILY,
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
    parser.add_argument(
        "--split-report",
        type=Path,
        help="Original hard-split report used to validate all-pair targets.",
    )
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
        "--min-combined-tokens",
        type=int,
        default=split_defaults["min_combined_tokens"],
    )
    parser.add_argument(
        "--min-punctuated-combined-tokens",
        type=int,
        default=split_defaults["min_punctuated_combined_tokens"],
    )
    parser.add_argument(
        "--max-eval-units-per-side",
        type=int,
        default=split_defaults["max_eval_units_per_side"],
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
        default=not split_defaults["synthetic_eval"],
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
    split_report = None
    if args.split_report:
        try:
            split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read split report {args.split_report}: {exc}") from exc
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
        min_combined_tokens=args.min_combined_tokens,
        min_punctuated_combined_tokens=args.min_punctuated_combined_tokens,
        max_eval_units_per_side=args.max_eval_units_per_side,
        source_ratio_tolerance=args.source_ratio_tolerance,
        require_human_eval=args.require_human_eval,
        require_document_holdout=args.require_document_holdout_report,
        split_report=split_report,
    )
    report: dict[str, object] = {
        "schema_version": 3,
        "complete": bool(provenance["ok"] and split_validation["ok"]),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "target_language": target_lang,
        "target_column": target_col,
        "profile": profile_record(args.profile),
        "split_report": (
            {
                "path": str(args.split_report),
                "sha256": sha256_file(args.split_report),
            }
            if args.split_report
            else None
        ),
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
    leakage = split_validation["train_evaluation"]
    print(
        f"Corpus validation: {'PASS' if report['complete'] else 'FAIL'} "
        f"({len(frame):,} rows, {target_lang})"
    )
    print(
        "  eval: "
        f"synthetic={split_validation['synthetic_eval_rows']:,}, "
        f"lexical-like={split_validation['lexical_like_eval_rows']:,}, "
        f"gloss={split_validation['gloss_translation_rows'] + split_validation['annotation_gloss_rows']:,}, "
        f"malformed={split_validation['malformed_escaping_rows']:,}, "
        f"language-mismatch={split_validation['target_language_mismatch_rows']:,}, "
        f"language-uncertain-eval={split_validation['uncertain_target_language_eval_rows']:,}, "
        f"unbalanced-eval={split_validation['unbalanced_target_eval_rows']:,}, "
        f"lexical-quality={split_validation['lexical_quality_rows']:,}"
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
