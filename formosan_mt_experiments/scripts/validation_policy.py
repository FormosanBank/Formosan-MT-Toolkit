"""Independent provenance, quality, ratio, and leakage validation."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from experiment_config import (
    load_corpus_pipeline_config,
)
from mt_common import (
    add_normalized_columns,
    bool_series,
    evaluation_candidate_mask,
    mt_standard_contract,
    source_corpus,
    weighted_apportioned_counts,
)
from validation_similarity import (
    ValidationNgramIndex,
    add_validation_keys,
    pairwise_leakage,
)

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
