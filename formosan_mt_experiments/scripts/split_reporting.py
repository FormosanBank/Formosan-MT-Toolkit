"""Accounting and release reports for hard MT splits."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from experiment_config import load_corpus_pipeline_config
from mt_common import (
    bool_series,
    evaluation_candidate_mask,
    split_counts,
    split_counts_by_language,
)
from split_allocation import source_assignment_tolerance
from split_similarity import overlap_summary

SPLIT_DEFAULTS = load_corpus_pipeline_config()["splits"]
TIER = SPLIT_DEFAULTS["headline_tier"]


@dataclass(frozen=True)
class SplitReportContext:
    frame: pd.DataFrame
    output: pd.DataFrame
    excluded: pd.DataFrame
    duplicate_rows: pd.DataFrame
    candidate_mask: pd.Series
    effective_candidate_mask: pd.Series
    group_ids: pd.Series
    targets: dict[str, tuple[int, int]]
    source_targets: dict[tuple[str, str], tuple[int, int]]
    language_reports: dict[str, dict]
    heldout_group_non_eval: pd.Series
    validate_test_blocked_indexes: set[int]
    ngram_train_blocked_indexes: set[int]
    train_eval_blocked_indexes: set[int]
    final_near: set[int]
    validate_test_near: set[int]
    validate_test_near_global: set[int]
    near_iterations: list[dict[str, int]]
    ngram_threshold: float
    test_ratio: float
    val_ratio: float
    target_language: str
    min_formosan_tokens: int
    min_target_tokens: int
    min_combined_tokens: int
    min_punctuated_combined_tokens: int
    max_eval_units_per_side: int
    mt_profile: dict
    registry_in: Path | None
    registry_stats: dict


def build_split_report(context: SplitReportContext) -> dict:
    frame = context.frame
    output = context.output
    train = output[output["split"].eq("train")]
    evaluation = output[output["split"].isin({"validate", "test"})]
    test = output[output["split"].eq("test")]
    validate = output[output["split"].eq("validate")]

    ratio_shortfalls: dict[str, dict] = {}
    for language, language_frame in output.groupby("lang_code", sort=True):
        language_key = str(language)
        counts = Counter(language_frame["split"])
        synthetic = language_frame.get(
            "pivot_origin",
            pd.Series("original", index=language_frame.index),
        ).astype(str).eq("synthetic")
        human_frame = language_frame[~synthetic]
        human_counts = Counter(human_frame["split"])
        human_rows = len(human_frame)
        target_test, target_validate = context.targets[language_key]
        context.language_reports[language_key].update(
            {
                "output_rows": len(language_frame),
                "train_rows": counts["train"],
                "test_rows": counts["test"],
                "validate_rows": counts["validate"],
                "human_train_rows": human_counts["train"],
                "human_test_rows": human_counts["test"],
                "human_validate_rows": human_counts["validate"],
                "human_train_fraction": human_counts["train"] / max(human_rows, 1),
                "human_test_fraction": human_counts["test"] / max(human_rows, 1),
                "human_validate_fraction": (
                    human_counts["validate"] / max(human_rows, 1)
                ),
                "test_fraction_of_eligible_sentences": (
                    counts["test"]
                    / max(context.language_reports[language_key]["eligible_sentence_rows"], 1)
                ),
                "validate_fraction_of_eligible_sentences": (
                    counts["validate"]
                    / max(context.language_reports[language_key]["eligible_sentence_rows"], 1)
                ),
                "test_fraction_of_all_input_rows": (
                    counts["test"] / max(context.language_reports[language_key]["rows_total"], 1)
                ),
                "validate_fraction_of_all_input_rows": (
                    counts["validate"] / max(context.language_reports[language_key]["rows_total"], 1)
                ),
                "synthetic_eval_rows": int(
                    (
                        language_frame.get(
                            "pivot_origin",
                            pd.Series("original", index=language_frame.index),
                        ).eq("synthetic")
                        & language_frame["split"].isin({"test", "validate"})
                    ).sum()
                ),
                "human_eval_rows": int(
                    (
                        ~language_frame.get(
                            "pivot_origin",
                            pd.Series("original", index=language_frame.index),
                        ).eq("synthetic")
                        & language_frame["split"].isin({"test", "validate"})
                    ).sum()
                ),
            }
        )
        if counts["test"] != target_test or counts["validate"] != target_validate:
            ratio_shortfalls[language_key] = {
                "test": counts["test"],
                "target_test": target_test,
                "validate": counts["validate"],
                "target_validate": target_validate,
            }

    source_reports: dict[str, dict[str, dict[str, object]]] = {}
    source_shortfalls: dict[str, dict[str, dict[str, int]]] = {}
    source_distribution_tvd: dict[str, dict[str, float]] = {}
    for language, language_frame in frame.groupby("lang_code", sort=True):
        language_key = str(language)
        source_reports[language_key] = {}
        source_shortfalls[language_key] = {}
        language_synthetic = (
            language_frame.get(
                "pivot_origin",
                pd.Series("original", index=language_frame.index),
            )
            .astype(str)
            .eq("synthetic")
        )
        human_language = language_frame[~language_synthetic]
        eligible_language = language_frame[context.candidate_mask.loc[language_frame.index]]
        all_distribution = Counter(human_language["_source_corpus"])
        split_distributions = {
            split_name: Counter(
                eligible_language[eligible_language["split"].eq(split_name)]["_source_corpus"]
            )
            for split_name in ("test", "validate")
        }
        source_distribution_tvd[language_key] = {}
        for split_name, distribution in split_distributions.items():
            total = sum(distribution.values())
            all_total = sum(all_distribution.values())
            source_distribution_tvd[language_key][split_name] = 0.5 * sum(
                abs(
                    all_distribution[source] / max(all_total, 1)
                    - distribution[source] / max(total, 1)
                )
                for source in set(all_distribution) | set(distribution)
            )
        for source, source_frame in language_frame.groupby("_source_corpus", sort=True):
            source_key = str(source)
            eligible_source = source_frame[context.candidate_mask.loc[source_frame.index]]
            similarity_safe_source = source_frame[
                context.effective_candidate_mask.loc[source_frame.index]
            ]
            target_test, target_validate = context.source_targets.get(
                (language_key, source_key),
                (0, 0),
            )
            counts = Counter(source_frame["split"])
            eligible_counts = Counter(eligible_source["split"])
            synthetic_source = source_frame.get(
                "pivot_origin",
                pd.Series("original", index=source_frame.index),
            ).astype(str).eq("synthetic")
            human_source_rows = int((~synthetic_source).sum())
            assignment_tolerance = source_assignment_tolerance(
                frame,
                source_frame.index,
                context.group_ids,
                context.effective_candidate_mask,
            )
            source_reports[language_key][source_key] = {
                "input_rows": len(source_frame),
                "human_input_rows": human_source_rows,
                "synthetic_input_rows": int(synthetic_source.sum()),
                "eligible_sentence_rows": len(eligible_source),
                "similarity_safe_sentence_rows": len(similarity_safe_source),
                "eligible_human_rows": len(eligible_source),
                "eligible_synthetic_rows": 0,
                "train_rows": counts["train"],
                "test_rows": counts["test"],
                "validate_rows": counts["validate"],
                "excluded_rows": counts["excluded"],
                "eligible_train_rows": eligible_counts["train"],
                "target_test_rows": target_test,
                "target_validate_rows": target_validate,
                "assignment_tolerance_rows": assignment_tolerance,
                "test_fraction": counts["test"] / max(human_source_rows, 1),
                "validate_fraction": counts["validate"] / max(human_source_rows, 1),
                "test_fraction_of_eligible_sentences": (
                    counts["test"] / max(len(eligible_source), 1)
                ),
                "validate_fraction_of_eligible_sentences": (
                    counts["validate"] / max(len(eligible_source), 1)
                ),
                "synthetic_test_rows": int(
                    (synthetic_source & source_frame["split"].eq("test")).sum()
                ),
                "synthetic_validate_rows": int(
                    (synthetic_source & source_frame["split"].eq("validate")).sum()
                ),
            }
            if (
                abs(counts["test"] - target_test) > assignment_tolerance
                or abs(counts["validate"] - target_validate) > assignment_tolerance
            ):
                source_shortfalls[language_key][source_key] = {
                    "test": counts["test"],
                    "target_test": target_test,
                    "validate": counts["validate"],
                    "target_validate": target_validate,
                    "tolerance": assignment_tolerance,
                }
        if not source_shortfalls[language_key]:
            source_shortfalls.pop(language_key)

    lexical_like_eval_rows = int(
        (
            output["split"].isin({"test", "validate"})
            & ~evaluation_candidate_mask(
                output,
                min_formosan_tokens=context.min_formosan_tokens,
                min_target_tokens=context.min_target_tokens,
                min_combined_tokens=context.min_combined_tokens,
                min_punctuated_combined_tokens=context.min_punctuated_combined_tokens,
                max_eval_units_per_side=context.max_eval_units_per_side,
            )
        ).sum()
    )
    output_complete = len(output) == len(frame) and context.excluded.empty
    report = {
        "schema_version": 3,
        "complete": output_complete and not ratio_shortfalls and not source_shortfalls,
        "tier": TIER,
        "target_language": context.target_language,
        "evaluation_length_policy": {
            "min_formosan_tokens": context.min_formosan_tokens,
            "min_target_tokens": context.min_target_tokens,
            "min_combined_tokens": context.min_combined_tokens,
            "min_punctuated_combined_tokens": context.min_punctuated_combined_tokens,
            "max_eval_units_per_side": context.max_eval_units_per_side,
        },
        "input_rows": len(frame) + len(context.duplicate_rows),
        "deduplicated_input_rows": len(frame),
        "duplicate_rows_removed": len(context.duplicate_rows),
        "output_rows": len(output),
        "excluded_rows": len(context.excluded),
        "excluded_heldout_group_rows": int(context.heldout_group_non_eval.sum()),
        "blocked_validate_test_candidate_rows": len(context.validate_test_blocked_indexes),
        "blocked_permanent_train_candidate_rows": len(context.ngram_train_blocked_indexes),
        "blocked_train_evaluation_candidate_rows": len(context.train_eval_blocked_indexes),
        "synthetic_eval_rows": int(
            (
                output.get("pivot_origin", pd.Series("original", index=output.index)).eq("synthetic")
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "synthetic_eval_rows_by_split": {
            split_name: int(
                (
                    output.get(
                        "pivot_origin",
                        pd.Series("original", index=output.index),
                    ).eq("synthetic")
                    & output["split"].eq(split_name)
                ).sum()
            )
            for split_name in ("test", "validate")
        },
        "synthetic_eval_policy": SPLIT_DEFAULTS["synthetic_eval_policy"],
        "synthetic_train_rows": int(
            (
                output.get("pivot_origin", pd.Series("original", index=output.index)).eq("synthetic")
                & output["split"].eq("train")
            ).sum()
        ),
        "mt_ineligible_eval_rows": int(
            (
                ~bool_series(
                    output["mt_eval_eligible"],
                    context="hard-split output:mt_eval_eligible",
                )
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "ambiguous_normalization_eval_rows": int(
            (
                output["mt_normalization_confidence"].astype(str).eq("ambiguous")
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "lexeme_eval_rows": int(
            (
                output["row_type"].isin({"lexeme", "morpheme"})
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "lexical_like_eval_rows": lexical_like_eval_rows,
        "document_overlap_train_eval": len(set(train["_document_key"]) & set(evaluation["_document_key"])),
        "document_overlap_validate_test": len(set(test["_document_key"]) & set(validate["_document_key"])),
        "near_duplicate_train_eval_rows": len(context.final_near),
        "near_duplicate_validate_test_rows": len(context.validate_test_near),
        "cross_language_near_duplicate_validate_test_rows": len(
            context.validate_test_near_global - context.validate_test_near
        ),
        "near_duplicate_iterations": context.near_iterations,
        "ngram_jaccard_threshold": context.ngram_threshold,
        "overlap": overlap_summary(train, evaluation),
        "split_counts": split_counts(output),
        "split_counts_by_language": split_counts_by_language(output),
        "mt_standardization": context.mt_profile,
        "languages": context.language_reports,
        "source_strata": source_reports,
        "source_distribution_total_variation": source_distribution_tvd,
        "ratio_basis": SPLIT_DEFAULTS["ratio_basis"],
        "required_human_ratios": {
            "train": round(1.0 - context.test_ratio - context.val_ratio, 12),
            "test": context.test_ratio,
            "validate": context.val_ratio,
        },
        "ratio_shortfalls": ratio_shortfalls,
        "source_ratio_shortfalls": source_shortfalls,
        "benchmark_registry_input": str(context.registry_in) if context.registry_in else None,
        "benchmark_registry_stats": context.registry_stats,
    }
    if not output_complete or ratio_shortfalls or source_shortfalls:
        raise SystemExit(
            "Could not construct source-balanced sentence evaluation sets: "
            + json.dumps(
                {
                    "output_rows": len(output),
                    "deduplicated_input_rows": len(frame),
                    "languages": ratio_shortfalls,
                    "sources": source_shortfalls,
                },
                sort_keys=True,
            )
        )
    return report
