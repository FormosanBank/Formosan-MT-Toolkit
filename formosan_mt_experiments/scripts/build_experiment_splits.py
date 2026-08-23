#!/usr/bin/env python3
"""Build one source-stratified, similarity-controlled hard MT split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from experiment_config import load_corpus_pipeline_config
from mt_common import (
    add_normalized_columns,
    bool_series,
    evaluation_candidate_mask,
    mt_standard_contract,
    normalize_target_language,
    read_parallel_csv,
    split_counts,
    split_counts_by_language,
    target_col_for,
    target_tag_for,
    write_columnar_cache,
    write_json,
)
from split_allocation import (
    apply_registry,
    fill_assignments,
    fill_language_shortfalls,
    materialize_splits,
    source_assignment_tolerance,
    source_stratum_targets,
)
from split_similarity import (
    NgramSimilarityIndex,
    SplitNgramIndexes,
    block_candidate_neighborhood,
    block_evaluation_conflicts_with_training,
    exclude_test_conflicts_with_validation,
    group_safe_evaluation_mask,
    leakage_group_ids,
    near_candidate_conflicts,
    overlap_summary,
)

PIPELINE_DEFAULTS = load_corpus_pipeline_config()
SPLIT_DEFAULTS = PIPELINE_DEFAULTS["splits"]
MAX_TRAINING_UNITS_PER_SIDE = PIPELINE_DEFAULTS["cleaning"][
    "max_training_units_per_side"
]
TIER = SPLIT_DEFAULTS["headline_tier"]


def deduplicate_input(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = frame.copy()
    work["_dedupe_key"] = (
        work["lang_code"].astype(str)
        + "\u241f"
        + work["_pair_key"].astype(str)
    )
    work["_human_priority"] = (
        work.get("pivot_origin", pd.Series("original", index=work.index))
        .astype(str)
        .eq("synthetic")
        .astype(int)
    )
    row_priority = {"sentence": 0, "lexeme": 1, "morpheme": 2, "unknown": 3}
    work["_row_priority"] = work["row_type"].map(row_priority).fillna(4)
    work["_input_order"] = range(len(work))
    work = work.sort_values(
        ["_dedupe_key", "_human_priority", "_row_priority", "_input_order"],
        kind="stable",
    )
    canonical = work.groupby("_dedupe_key", sort=False)["row_id"].first().to_dict()
    duplicate_mask = work.duplicated("_dedupe_key", keep="first")
    duplicates = work[duplicate_mask].copy()
    duplicates["canonical_row_id"] = duplicates["_dedupe_key"].map(canonical)
    duplicates["exclusion_reason"] = "duplicate_pair"
    kept = work[~duplicate_mask].copy()
    drop = ["_dedupe_key", "_human_priority", "_row_priority", "_input_order"]
    return (
        kept.sort_values("_input_order", kind="stable").drop(columns=drop).reset_index(drop=True),
        duplicates.drop(columns=drop).reset_index(drop=True),
    )


def evaluation_masks(
    frame: pd.DataFrame,
    *,
    min_formosan_tokens: int,
    min_target_tokens: int,
    min_combined_tokens: int,
    min_punctuated_combined_tokens: int,
    max_eval_units_per_side: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    synthetic = (
        frame.get("pivot_origin", pd.Series("original", index=frame.index))
        .astype(str)
        .eq("synthetic")
    )
    candidate = evaluation_candidate_mask(
        frame,
        min_formosan_tokens=min_formosan_tokens,
        min_target_tokens=min_target_tokens,
        min_combined_tokens=min_combined_tokens,
        min_punctuated_combined_tokens=min_punctuated_combined_tokens,
        max_eval_units_per_side=max_eval_units_per_side,
    )
    human_candidate = candidate & ~synthetic
    return synthetic, human_candidate, candidate


def build_hard_split(
    frame: pd.DataFrame,
    *,
    target_col: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    min_formosan_tokens: int,
    min_target_tokens: int,
    min_combined_tokens: int,
    min_punctuated_combined_tokens: int,
    attempts: int,
    min_test_rows: int,
    min_validate_rows: int,
    ngram_threshold: float,
    registry_in: Path | None,
    preserve_internal: bool = False,
    max_eval_units_per_side: int = SPLIT_DEFAULTS["max_eval_units_per_side"],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    frame = frame.drop(
        columns=["source_bucket", "_source_bucket"],
        errors="ignore",
    )
    if "row_id" not in frame.columns or frame["row_id"].astype(str).duplicated().any():
        raise SystemExit("Input must contain unique stable row_id values")
    if "kindOf" not in frame.columns or not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Hard splitting requires every row to use kindOf=standard")
    mt_profile = mt_standard_contract(frame, context="hard-split input")

    frame = add_normalized_columns(
        frame.copy(),
        target_col=target_col,
        target_lang=(
            "chinese" if target_col == "chinese_sentence" else "english"
        ),
    )
    overlength = (
        frame["_formosan_tokens"].gt(MAX_TRAINING_UNITS_PER_SIDE)
        | frame["_target_tokens"].gt(MAX_TRAINING_UNITS_PER_SIDE)
    )
    if overlength.any():
        raise SystemExit(
            "Hard-split input contains "
            f"{int(overlength.sum()):,} rows above the "
            f"{MAX_TRAINING_UNITS_PER_SIDE}-unit training limit"
        )
    frame["_document_key"] = (
        frame["lang_code"].astype(str)
        + "\u241f"
        + frame["source"].astype(str)
    )
    frame, duplicate_rows = deduplicate_input(frame)
    frame["_document_key"] = (
        frame["lang_code"].astype(str)
        + "\u241f"
        + frame["source"].astype(str)
    )
    group_ids = leakage_group_ids(frame)
    ngram_indexes = SplitNgramIndexes(
        formosan=NgramSimilarityIndex(
            frame,
            "_formosan_key",
            by_language=True,
            threshold=ngram_threshold,
        ),
        target=NgramSimilarityIndex(
            frame,
            "_target_key",
            by_language=False,
            threshold=ngram_threshold,
        ),
    )
    synthetic, human_candidate, _ = (
        evaluation_masks(
            frame,
            min_formosan_tokens=min_formosan_tokens,
            min_target_tokens=min_target_tokens,
            min_combined_tokens=min_combined_tokens,
            min_punctuated_combined_tokens=min_punctuated_combined_tokens,
            max_eval_units_per_side=max_eval_units_per_side,
        )
    )
    candidate_mask = group_safe_evaluation_mask(
        frame,
        human_candidate,
        group_ids,
    )
    source_targets, targets = source_stratum_targets(
        frame,
        candidate_mask,
        test_ratio=test_ratio,
        val_ratio=val_ratio,
        min_test_rows=min_test_rows,
        min_validate_rows=min_validate_rows,
    )
    language_reports: dict[str, dict] = {}
    for language, language_frame in frame.groupby("lang_code", sort=True):
        index = language_frame.index
        eligible_total = int(candidate_mask.loc[index].sum())
        test_target, validate_target = targets[str(language)]
        language_reports[str(language)] = {
            "rows_total": len(language_frame),
            "eligible_sentence_rows": int(human_candidate.loc[index].sum()),
            "group_safe_human_sentence_rows": eligible_total,
            "eligible_human_sentence_rows": int(
                human_candidate.loc[index].sum()
            ),
            "eligible_synthetic_sentence_rows": 0,
            "synthetic_rows": int(synthetic.loc[index].sum()),
            "lexical_rows": int(
                language_frame["row_type"].isin({"lexeme", "morpheme"}).sum()
            ),
            "evaluation_ineligible_rows": int(
                len(language_frame) - human_candidate.loc[index].sum()
            ),
            "target_test_rows": test_target,
            "target_validate_rows": validate_target,
        }

    assignments, registry_stats = apply_registry(
        frame,
        group_ids,
        candidate_mask,
        registry_in,
    )
    registry_groups = set(assignments)
    ngram_train_blocked_indexes = near_candidate_conflicts(
        frame[~candidate_mask],
        frame[candidate_mask],
        ngram_threshold=ngram_threshold,
        target_by_language=True,
        ngram_indexes=ngram_indexes,
    )
    effective_candidate_mask = candidate_mask.copy()
    ngram_train_blocked_indexes = block_candidate_neighborhood(
        frame,
        effective_candidate_mask,
        group_ids,
        ngram_train_blocked_indexes,
        ngram_threshold=ngram_threshold,
        ngram_indexes=ngram_indexes,
    )
    validate_test_blocked_indexes: set[int] = set()
    train_eval_blocked_indexes: set[int] = set()
    near_iterations: list[dict[str, int]] = []
    max_iterations = int(group_ids.nunique()) + 1
    for _ in range(max_iterations):
        source_targets, current_targets = source_stratum_targets(
            frame,
            effective_candidate_mask,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            min_test_rows=min_test_rows,
            min_validate_rows=min_validate_rows,
        )
        if current_targets != targets:
            raise SystemExit(
                "Similarity filtering left insufficient eligible rows for "
                "the all-pair language targets"
            )
        for group_id in set(assignments) - registry_groups:
            assignments.pop(group_id)
        fill_assignments(
            frame,
            group_ids,
            effective_candidate_mask,
            source_targets,
            assignments,
            seed=seed,
            attempts=attempts,
        )
        fill_language_shortfalls(
            frame,
            group_ids,
            effective_candidate_mask,
            source_targets,
            targets,
            assignments,
            seed=seed,
            attempts=attempts,
        )
        split = materialize_splits(
            frame,
            group_ids,
            effective_candidate_mask,
            assignments,
        )
        # Exact and one-edit variants share a group and cannot cross splits.
        # Iterative stabilization only needs to resolve the wider n-gram gate.
        iteration = exclude_test_conflicts_with_validation(
            frame,
            split,
            group_ids,
            effective_candidate_mask,
            assignments,
            validate_test_blocked_indexes,
            ngram_threshold=ngram_threshold,
            include_one_edit=False,
            ngram_indexes=ngram_indexes,
        )
        if iteration["conflicting_eval_rows"]:
            iteration["train_eval_conflicting_rows"] = 0
            near_iterations.append(iteration)
            continue
        split = materialize_splits(
            frame,
            group_ids,
            effective_candidate_mask,
            assignments,
        )
        train_eval_iteration = block_evaluation_conflicts_with_training(
            frame,
            split,
            group_ids,
            effective_candidate_mask,
            assignments,
            train_eval_blocked_indexes,
            ngram_threshold=ngram_threshold,
            include_one_edit=False,
            ngram_indexes=ngram_indexes,
        )
        iteration["train_eval_conflicting_rows"] = train_eval_iteration[
            "conflicting_eval_rows"
        ]
        near_iterations.append(iteration)
        if train_eval_iteration["conflicting_eval_rows"] == 0:
            break
    else:
        raise SystemExit("Near-duplicate split stabilization did not converge")

    split = materialize_splits(
        frame,
        group_ids,
        effective_candidate_mask,
        assignments,
    )
    heldout_group_non_eval = split.eq("excluded")
    frame["split"] = split
    frame["eval_tier"] = TIER
    frame["source_corpus"] = frame["_source_corpus"]
    frame["formosan_tokens"] = frame["_formosan_tokens"].astype(int)
    frame["target_tokens"] = frame["_target_tokens"].astype(int)
    frame["short_entry"] = frame["_short_entry"].astype(bool)
    frame["document_id"] = frame["_document_key"]

    exclusion_reason = pd.Series("", index=frame.index, dtype="object")
    exclusion_reason.loc[
        heldout_group_non_eval
    ] = "heldout_group_non_evaluation_row"
    exclusion_reason.loc[
        list(validate_test_blocked_indexes)
    ] = "near_duplicate_between_test_and_validation"
    excluded = frame[split.eq("excluded")].copy()
    excluded["exclusion_reason"] = exclusion_reason.loc[
        excluded.index
    ]
    output = frame[split.isin({"train", "validate", "test"})].copy()
    train = output[output["split"].eq("train")]
    evaluation = output[output["split"].isin({"validate", "test"})]
    test = output[output["split"].eq("test")]
    validate = output[output["split"].eq("validate")]

    final_near = near_candidate_conflicts(
        train,
        evaluation,
        ngram_threshold=ngram_threshold,
        target_by_language=True,
        ngram_indexes=ngram_indexes,
    )
    validate_test_near = near_candidate_conflicts(
        test,
        validate,
        ngram_threshold=ngram_threshold,
        target_by_language=True,
        ngram_indexes=ngram_indexes,
    )
    validate_test_near_global = near_candidate_conflicts(
        test,
        validate,
        ngram_threshold=ngram_threshold,
        ngram_indexes=ngram_indexes,
    )
    if final_near or validate_test_near:
        raise SystemExit(
            f"Near-duplicate validation failed: train/eval={len(final_near)}, "
            f"validate/test={len(validate_test_near)}"
        )

    ratio_shortfalls: dict[str, dict] = {}
    for language, language_frame in output.groupby("lang_code", sort=True):
        counts = Counter(language_frame["split"])
        target_test, target_validate = targets[str(language)]
        language_reports[str(language)].update(
            {
                "output_rows": len(language_frame),
                "train_rows": counts["train"],
                "test_rows": counts["test"],
                "validate_rows": counts["validate"],
                "test_fraction_of_eligible_sentences": (
                    counts["test"]
                    / max(
                        language_reports[str(language)]["eligible_sentence_rows"],
                        1,
                    )
                ),
                "validate_fraction_of_eligible_sentences": (
                    counts["validate"]
                    / max(
                        language_reports[str(language)]["eligible_sentence_rows"],
                        1,
                    )
                ),
                "test_fraction_of_all_input_rows": (
                    counts["test"]
                    / max(language_reports[str(language)]["rows_total"], 1)
                ),
                "validate_fraction_of_all_input_rows": (
                    counts["validate"]
                    / max(language_reports[str(language)]["rows_total"], 1)
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
            ratio_shortfalls[str(language)] = {
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
        eligible_language = language_frame[
            candidate_mask.loc[language_frame.index]
        ]
        all_distribution = Counter(human_language["_source_corpus"])
        split_distributions = {
            split_name: Counter(
                eligible_language[
                    eligible_language["split"].eq(split_name)
                ]["_source_corpus"]
            )
            for split_name in ("test", "validate")
        }
        source_distribution_tvd[language_key] = {}
        for split_name, distribution in split_distributions.items():
            total = sum(distribution.values())
            all_total = sum(all_distribution.values())
            source_distribution_tvd[language_key][split_name] = (
                0.5
                * sum(
                    abs(
                        all_distribution[source] / max(all_total, 1)
                        - distribution[source] / max(total, 1)
                    )
                    for source in set(all_distribution) | set(distribution)
                )
            )
        for source, source_frame in language_frame.groupby(
            "_source_corpus", sort=True
        ):
            source_key = str(source)
            eligible_source = source_frame[
                candidate_mask.loc[source_frame.index]
            ]
            similarity_safe_source = source_frame[
                effective_candidate_mask.loc[source_frame.index]
            ]
            target_test, target_validate = source_targets.get(
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
                group_ids,
                effective_candidate_mask,
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
                "validate_fraction": (
                    counts["validate"] / max(human_source_rows, 1)
                ),
                "test_fraction_of_eligible_sentences": (
                    counts["test"] / max(len(eligible_source), 1)
                ),
                "validate_fraction_of_eligible_sentences": (
                    counts["validate"] / max(len(eligible_source), 1)
                ),
                "synthetic_test_rows": int(
                    (
                        synthetic_source
                        & source_frame["split"].eq("test")
                    ).sum()
                ),
                "synthetic_validate_rows": int(
                    (
                        synthetic_source
                        & source_frame["split"].eq("validate")
                    ).sum()
                ),
            }
            if (
                abs(counts["test"] - target_test) > assignment_tolerance
                or abs(counts["validate"] - target_validate)
                > assignment_tolerance
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
                min_formosan_tokens=min_formosan_tokens,
                min_target_tokens=min_target_tokens,
                min_combined_tokens=min_combined_tokens,
                min_punctuated_combined_tokens=min_punctuated_combined_tokens,
                max_eval_units_per_side=max_eval_units_per_side,
            )
        ).sum()
    )
    overlaps = overlap_summary(train, evaluation)
    output_complete = len(output) == len(frame) and excluded.empty
    report = {
        "schema_version": 3,
        "complete": (
            output_complete
            and not ratio_shortfalls
            and not source_shortfalls
        ),
        "tier": TIER,
        "evaluation_length_policy": {
            "min_formosan_tokens": min_formosan_tokens,
            "min_target_tokens": min_target_tokens,
            "min_combined_tokens": min_combined_tokens,
            "min_punctuated_combined_tokens": min_punctuated_combined_tokens,
            "max_eval_units_per_side": max_eval_units_per_side,
        },
        "input_rows": len(frame) + len(duplicate_rows),
        "deduplicated_input_rows": len(frame),
        "duplicate_rows_removed": len(duplicate_rows),
        "output_rows": len(output),
        "excluded_rows": len(excluded),
        "excluded_heldout_group_rows": int(
            heldout_group_non_eval.sum()
        ),
        "blocked_validate_test_candidate_rows": len(
            validate_test_blocked_indexes
        ),
        "blocked_permanent_train_candidate_rows": len(
            ngram_train_blocked_indexes
        ),
        "blocked_train_evaluation_candidate_rows": len(
            train_eval_blocked_indexes
        ),
        "synthetic_eval_rows": int(
            (
                output.get("pivot_origin", pd.Series("original", index=output.index))
                .eq("synthetic")
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
                output.get(
                    "pivot_origin",
                    pd.Series("original", index=output.index),
                ).eq("synthetic")
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
        "document_overlap_train_eval": len(
            set(train["_document_key"]) & set(evaluation["_document_key"])
        ),
        "document_overlap_validate_test": len(
            set(test["_document_key"]) & set(validate["_document_key"])
        ),
        "near_duplicate_train_eval_rows": len(final_near),
        "near_duplicate_validate_test_rows": len(validate_test_near),
        "cross_language_near_duplicate_validate_test_rows": len(
            validate_test_near_global - validate_test_near
        ),
        "near_duplicate_iterations": near_iterations,
        "ngram_jaccard_threshold": ngram_threshold,
        "overlap": overlaps,
        "split_counts": split_counts(output),
        "split_counts_by_language": split_counts_by_language(output),
        "mt_standardization": mt_profile,
        "languages": language_reports,
        "source_strata": source_reports,
        "source_distribution_total_variation": source_distribution_tvd,
        "ratio_basis": SPLIT_DEFAULTS["ratio_basis"],
        "required_ratios": {"test": test_ratio, "validate": val_ratio},
        "ratio_shortfalls": ratio_shortfalls,
        "source_ratio_shortfalls": source_shortfalls,
        "benchmark_registry_input": str(registry_in) if registry_in else None,
        "benchmark_registry_stats": registry_stats,
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

    internal_columns = [column for column in output.columns if column.startswith("_")]
    if preserve_internal:
        return output, excluded, duplicate_rows, report
    return (
        output.drop(columns=internal_columns),
        excluded.drop(columns=[column for column in internal_columns if column in excluded]),
        duplicate_rows,
        report,
    )


def validate_report(report: dict) -> None:
    failures: dict[str, object] = {}
    if report.get("complete") is not True:
        failures["complete"] = report.get("complete")
    for key in (
        "synthetic_eval_rows",
        "lexeme_eval_rows",
        "lexical_like_eval_rows",
        "mt_ineligible_eval_rows",
        "ambiguous_normalization_eval_rows",
        "near_duplicate_train_eval_rows",
        "near_duplicate_validate_test_rows",
    ):
        if report.get(key) != 0:
            failures[key] = report.get(key)
    for family in ("exact", "skeleton"):
        for key, value in report["overlap"][family].items():
            if key in {"formosan", "target", "pair"} and value["overlap_unique"]:
                failures[f"{family}:{key}"] = value["overlap_unique"]
    if report.get("ratio_shortfalls"):
        failures["ratio_shortfalls"] = report["ratio_shortfalls"]
    if report.get("source_ratio_shortfalls"):
        failures["source_ratio_shortfalls"] = report[
            "source_ratio_shortfalls"
        ]
    if failures:
        raise SystemExit(
            "Hard-split release validation failed: "
            + json.dumps(failures, sort_keys=True)
        )


def write_registry(path: Path, output: pd.DataFrame, report: dict) -> None:
    evaluation = output[output["split"].isin({"test", "validate"})]
    payload = {
        "schema_version": 3,
        "complete": True,
        "tier": TIER,
        "ratio_basis": report["ratio_basis"],
        "mt_standardization": report["mt_standardization"],
        "evaluation_rows": [
            {
                "row_id": str(row["row_id"]),
                "split": str(row["split"]),
                "lang_code": str(row["lang_code"]),
                "source": str(row["source"]),
            }
            for _, row in evaluation.sort_values(["lang_code", "split", "row_id"]).iterrows()
        ],
    }
    write_json(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-stratified hard MT split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col")
    parser.add_argument("--output-prefix")
    parser.add_argument("--train-ratio", type=float, default=SPLIT_DEFAULTS["train_ratio"])
    parser.add_argument("--val-ratio", type=float, default=SPLIT_DEFAULTS["validate_ratio"])
    parser.add_argument("--test-ratio", type=float, default=SPLIT_DEFAULTS["test_ratio"])
    parser.add_argument("--seed", type=int, default=SPLIT_DEFAULTS["seed"])
    parser.add_argument(
        "--min-formosan-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_formosan_tokens"],
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_target_tokens"],
    )
    parser.add_argument(
        "--min-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_combined_tokens"],
    )
    parser.add_argument(
        "--min-punctuated-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_punctuated_combined_tokens"],
    )
    parser.add_argument(
        "--max-eval-units-per-side",
        type=int,
        default=SPLIT_DEFAULTS["max_eval_units_per_side"],
    )
    parser.add_argument("--min-test-rows", type=int, default=SPLIT_DEFAULTS["min_test_rows"])
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=SPLIT_DEFAULTS["min_validate_rows"],
    )
    parser.add_argument("--selection-attempts", type=int, default=200)
    parser.add_argument(
        "--ngram-jaccard-threshold",
        type=float,
        default=SPLIT_DEFAULTS["character_ngram_jaccard_threshold"],
    )
    parser.add_argument("--registry-in", type=Path)
    parser.add_argument(
        "--tiers",
        default=TIER,
        help="Compatibility option; only in_domain_hard is supported.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_tiers = {value.strip() for value in args.tiers.split(",") if value.strip()}
    if requested_tiers != {TIER}:
        raise SystemExit(f"Corpus pipeline v3 supports only --tiers {TIER}")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise SystemExit("Split ratios must sum to 1.0")
    if not 0.5 <= args.ngram_jaccard_threshold <= 1.0:
        raise SystemExit("--ngram-jaccard-threshold must be in [0.5, 1.0]")
    if min(args.min_formosan_tokens, args.min_target_tokens) < 1:
        raise SystemExit("Evaluation per-side token minimums must be positive")
    if args.max_eval_units_per_side < 1:
        raise SystemExit("Evaluation per-side unit maximum must be positive")
    if args.min_punctuated_combined_tokens > args.min_combined_tokens:
        raise SystemExit("Punctuated combined minimum cannot exceed the general combined minimum")

    target_language = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_language)
    target_tag = target_tag_for(target_language)
    short = "en" if target_tag == "eng" else target_tag
    output_prefix = args.output_prefix or f"big_corpus_{short}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_parallel_csv(args.input, target_col=target_col)
    output, excluded, duplicates, report = build_hard_split(
        raw,
        target_col=target_col,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_formosan_tokens=args.min_formosan_tokens,
        min_target_tokens=args.min_target_tokens,
        min_combined_tokens=args.min_combined_tokens,
        min_punctuated_combined_tokens=args.min_punctuated_combined_tokens,
        max_eval_units_per_side=args.max_eval_units_per_side,
        attempts=args.selection_attempts,
        min_test_rows=args.min_test_rows,
        min_validate_rows=args.min_validate_rows,
        ngram_threshold=args.ngram_jaccard_threshold,
        registry_in=args.registry_in,
        preserve_internal=True,
    )
    validate_report(report)

    internal_columns = [column for column in output.columns if column.startswith("_")]
    release_output = output.drop(columns=internal_columns)
    release_excluded = excluded.drop(
        columns=[column for column in internal_columns if column in excluded]
    )

    full_path = args.output_dir / f"{output_prefix}_{TIER}.csv"
    test_path = args.output_dir / f"{output_prefix}_{TIER}_test.csv"
    validate_path = args.output_dir / f"{output_prefix}_{TIER}_validate.csv"
    excluded_path = args.output_dir / f"{output_prefix}_{TIER}_excluded.csv"
    duplicate_path = args.output_dir / f"{output_prefix}_{TIER}_duplicates.csv"
    registry_path = args.output_dir / "benchmark_registry.json"
    release_output.to_csv(full_path, index=False)
    release_output[release_output["split"].eq("test")].to_csv(test_path, index=False)
    release_output[release_output["split"].eq("validate")].to_csv(validate_path, index=False)
    release_excluded.to_csv(excluded_path, index=False)
    duplicates.to_csv(duplicate_path, index=False)
    write_registry(registry_path, release_output, report)
    columnar_path = write_columnar_cache(output, full_path)
    report["files"] = {
        "full": str(full_path),
        "test": str(test_path),
        "validate": str(validate_path),
        "excluded": str(excluded_path),
        "duplicates": str(duplicate_path),
        "benchmark_registry": str(registry_path),
        "full_columnar": str(columnar_path),
    }
    write_json(args.output_dir / f"report_{TIER}.json", report)
    write_json(
        args.output_dir / "report_all_tiers.json",
        {
            "schema_version": 3,
            "complete": True,
            "input": str(args.input),
            "target_lang": target_language,
            "target_col": target_col,
            "tiers": {TIER: report},
        },
    )
    print(f"[{TIER}] wrote {full_path}")
    print(f"Splits: {json.dumps(report['split_counts'], sort_keys=True)}")


if __name__ == "__main__":
    main()
