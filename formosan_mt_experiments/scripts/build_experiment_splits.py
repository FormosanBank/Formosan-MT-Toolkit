#!/usr/bin/env python3
"""Build one human-only, document-aware hard MT benchmark split."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from pathlib import Path

import pandas as pd
from mt_common import (
    EASY_BUCKETS,
    add_normalized_columns,
    bucket_counts,
    normalize_target_language,
    overlap_stats,
    read_parallel_csv,
    split_counts,
    split_counts_by_language,
    target_col_for,
    target_tag_for,
    write_json,
)

TIER = "in_domain_hard"


@dataclass(frozen=True)
class GroupCandidate:
    group_id: int
    eligible_rows: int
    total_rows: int
    non_eval_rows: int
    easy_fraction: float
    average_tokens: float


def one_edit_conflicts(train: pd.DataFrame, eval_df: pd.DataFrame, column: str) -> set[int]:
    """Return train indexes whose text is within one character edit of eval."""
    conflicts: set[int] = set()
    for language, evaluation in eval_df.groupby("lang_code", sort=False):
        training = train[train["lang_code"].eq(language)]
        if training.empty:
            continue
        eval_values = {str(value) for value in evaluation[column] if str(value)}
        eval_deletions = {
            value[:position] + value[position + 1 :]
            for value in eval_values
            for position in range(len(value))
        }
        for index, raw_value in training[column].items():
            value = str(raw_value)
            if not value:
                continue
            if value in eval_values or value in eval_deletions:
                conflicts.add(int(index))
                continue
            if any(
                value[:position] + value[position + 1 :] in eval_values
                or value[:position] + value[position + 1 :] in eval_deletions
                for position in range(len(value))
            ):
                conflicts.add(int(index))
    return conflicts


def one_edit_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
) -> set[int]:
    if by_language:
        return one_edit_conflicts(candidates, reference, column)
    reference_copy = reference.copy()
    candidates_copy = candidates.copy()
    reference_copy["lang_code"] = "_global"
    candidates_copy["lang_code"] = "_global"
    return one_edit_conflicts(candidates_copy, reference_copy, column)


def char_ngrams(value: str, size: int = 4) -> frozenset[str]:
    value = str(value)
    if not value:
        return frozenset()
    if len(value) <= size:
        return frozenset({value})
    return frozenset(value[position : position + size] for position in range(len(value) - size + 1))


def jaccard_prefix(
    grams: frozenset[str],
    frequency: Counter[str],
    threshold: float,
) -> tuple[str, ...]:
    """PPJoin-style prefix that cannot miss a pair above the threshold."""
    prefix_length = len(grams) - math.ceil(threshold * len(grams)) + 1
    ordered = sorted(grams, key=lambda gram: (frequency[gram], gram))
    return tuple(ordered[: max(1, prefix_length)])


def ngram_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
    threshold: float,
) -> set[int]:
    """Deterministic prefix-filtered search with exact Jaccard verification."""
    if reference.empty or candidates.empty:
        return set()
    conflicts: set[int] = set()
    reference_groups = (
        reference.groupby("lang_code", sort=False)
        if by_language
        else [("_global", reference)]
    )
    candidate_groups = (
        {str(language): group for language, group in candidates.groupby("lang_code", sort=False)}
        if by_language
        else {"_global": candidates}
    )
    for key, reference_group in reference_groups:
        candidate_group = candidate_groups.get(str(key))
        if candidate_group is None or candidate_group.empty:
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
        for grams in chain(candidate_grams.values(), reference_grams.values()):
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
    return conflicts


def near_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    ngram_threshold: float,
) -> set[int]:
    conflicts = one_edit_candidate_conflicts(
        reference,
        candidates,
        "_formosan_skeleton",
        by_language=True,
    )
    conflicts |= one_edit_candidate_conflicts(
        reference,
        candidates,
        "_target_skeleton",
        by_language=False,
    )
    conflicts |= ngram_candidate_conflicts(
        reference,
        candidates,
        "_formosan_skeleton",
        by_language=True,
        threshold=ngram_threshold,
    )
    conflicts |= ngram_candidate_conflicts(
        reference,
        candidates,
        "_target_skeleton",
        by_language=False,
        threshold=ngram_threshold,
    )
    return conflicts


def leakage_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Use source documents as indivisible benchmark groups.

    Repeated normalized strings are made evaluation-ineligible separately. If
    they were joined transitively here, ubiquitous targets such as "yes" would
    connect hundreds of otherwise unrelated documents into one unusable group.
    """
    groups, _ = pd.factorize(frame["_document_key"].astype(str), sort=False)
    return pd.Series(groups, index=frame.index, dtype="int64")


def unique_document_value_mask(
    frame: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
) -> pd.Series:
    """Return rows whose normalized value occurs in exactly one document."""
    values = frame[column].astype(str)
    if by_language:
        keys = frame["lang_code"].astype(str) + "\u241f" + values
    else:
        keys = values
    document_counts = (
        pd.DataFrame({"key": keys, "document": frame["_document_key"]})
        .drop_duplicates()
        .groupby("key", sort=False)["document"]
        .nunique()
    )
    return keys.map(document_counts).eq(1) & values.ne("")


def split_targets(
    rows_total: int,
    eligible_total: int,
    test_ratio: float,
    val_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[int, int]:
    """Size human evaluation sets from the complete augmented corpus."""
    if eligible_total <= 0:
        return 0, 0
    desired_test = max(math.ceil(rows_total * test_ratio), min_test_rows)
    desired_validate = max(math.ceil(rows_total * val_ratio), min_validate_rows)
    if desired_test + desired_validate <= eligible_total:
        return desired_test, desired_validate
    eval_ratio = test_ratio + val_ratio
    test_share = test_ratio / eval_ratio if eval_ratio else 0.75
    test = max(1, round(eligible_total * test_share))
    validate = eligible_total - test
    if eligible_total >= 2 and validate == 0:
        test -= 1
        validate = 1
    return test, validate


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
) -> tuple[pd.Series, pd.Series, pd.Series]:
    synthetic = (
        frame.get("pivot_origin", pd.Series("original", index=frame.index))
        .astype(str)
        .eq("synthetic")
    )
    sentence = frame["row_type"].astype(str).eq("sentence")
    flags = frame.get("quality_flags", pd.Series("", index=frame.index)).astype(str)
    clean = ~flags.str.contains(r"(?:contains_unclear|unknown_row_type)", regex=True)
    human_sentence = ~synthetic & sentence & clean
    hard = (
        human_sentence
        & ~frame["_source_bucket"].isin(EASY_BUCKETS)
        & frame["_formosan_tokens"].ge(min_formosan_tokens)
        & frame["_target_tokens"].ge(min_target_tokens)
    )
    broad = (
        human_sentence
        & frame["_formosan_tokens"].ge(1)
        & frame["_target_tokens"].ge(1)
    )
    unique_to_document = pd.Series(True, index=frame.index)
    for column, by_language in (
        ("_formosan_key", True),
        ("_formosan_skeleton", True),
        ("_target_key", False),
        ("_target_skeleton", False),
        ("_pair_skeleton", True),
    ):
        unique_to_document &= unique_document_value_mask(
            frame,
            column,
            by_language=by_language,
        )
    return synthetic, human_sentence, (hard | broad) & unique_to_document


def candidate_groups(
    frame: pd.DataFrame,
    language: str,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    assignments: dict[int, str],
    blocked: set[int],
) -> list[GroupCandidate]:
    language_rows = frame["lang_code"].eq(language)
    language_frame = frame[language_rows]
    language_group_ids = group_ids.loc[language_frame.index]
    candidates: list[GroupCandidate] = []
    for group_id, group in language_frame.groupby(language_group_ids, sort=False):
        group_key = int(group_id)
        if group_key in assignments or group_key in blocked:
            continue
        eligible = group[candidate_mask.loc[group.index]]
        if eligible.empty:
            continue
        candidates.append(
            GroupCandidate(
                group_id=group_key,
                eligible_rows=len(eligible),
                total_rows=len(group),
                non_eval_rows=len(group) - len(eligible),
                easy_fraction=float(eligible["_source_bucket"].isin(EASY_BUCKETS).mean()),
                average_tokens=float(
                    (eligible["_formosan_tokens"] + eligible["_target_tokens"]).mean()
                    / 2
                ),
            )
        )
    return candidates


def choose_groups(
    candidates: list[GroupCandidate],
    target_rows: int,
    *,
    seed: int,
    attempts: int,
) -> set[int]:
    if target_rows <= 0 or not candidates:
        return set()
    rng = random.Random(seed)
    best: list[GroupCandidate] = []
    best_cost = (float("inf"), float("inf"), float("inf"), float("inf"))
    for attempt in range(max(1, attempts)):
        ordered = candidates[:]
        if attempt:
            rng.shuffle(ordered)
        ordered.sort(
            key=lambda group: (
                group.easy_fraction,
                group.non_eval_rows / max(group.eligible_rows, 1),
                -group.average_tokens,
                rng.random() if attempt else group.group_id,
            )
        )
        selected: list[GroupCandidate] = []
        rows = 0
        for candidate in ordered:
            selected.append(candidate)
            rows += candidate.eligible_rows
            if rows >= target_rows:
                break
        cost = (
            abs(rows - target_rows),
            sum(group.non_eval_rows for group in selected),
            sum(group.easy_fraction * group.eligible_rows for group in selected),
            len(selected),
        )
        if selected and cost < best_cost:
            best = selected
            best_cost = cost
    return {group.group_id for group in best}


def apply_registry(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    registry_path: Path | None,
) -> tuple[dict[int, str], dict[str, int]]:
    if registry_path is None:
        return {}, {"requested": 0, "matched": 0, "missing": 0}
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if payload.get("complete") is not True:
        raise SystemExit(f"Benchmark registry is incomplete: {registry_path}")
    requested = {
        str(row["row_id"]): str(row["split"])
        for row in payload.get("evaluation_rows", [])
    }
    row_to_index = {
        str(row_id): int(index)
        for index, row_id in frame["row_id"].astype(str).items()
    }
    assignments: dict[int, str] = {}
    matched = 0
    for row_id, split in requested.items():
        index = row_to_index.get(row_id)
        if index is None:
            continue
        if split not in {"test", "validate"} or not bool(candidate_mask.loc[index]):
            raise SystemExit(f"Registry row {row_id} is no longer eligible for {split}")
        group_id = int(group_ids.loc[index])
        previous = assignments.get(group_id)
        if previous and previous != split:
            raise SystemExit(f"Registry assigns leakage group {group_id} to two splits")
        assignments[group_id] = split
        matched += 1
    return assignments, {
        "requested": len(requested),
        "matched": matched,
        "missing": len(requested) - matched,
    }


def materialize_splits(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    assignments: dict[int, str],
) -> pd.Series:
    split = pd.Series("train", index=frame.index, dtype="object")
    assigned = group_ids.map(assignments)
    heldout = assigned.isin({"test", "validate"})
    split.loc[heldout & candidate_mask] = assigned.loc[heldout & candidate_mask]
    split.loc[heldout & ~candidate_mask] = "excluded"
    return split


def fill_assignments(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    targets: dict[str, tuple[int, int]],
    assignments: dict[int, str],
    blocked: set[int],
    *,
    seed: int,
    attempts: int,
) -> None:
    language_order = sorted(
        targets,
        key=lambda language: (
            int((frame["lang_code"].eq(language) & candidate_mask).sum()),
            language,
        ),
    )
    for offset, language in enumerate(language_order):
        language_mask = frame["lang_code"].eq(language) & candidate_mask
        current = group_ids.map(assignments)
        target_test, target_validate = targets[language]
        for split_name, target, seed_offset in (
            ("test", target_test, 0),
            ("validate", target_validate, 17),
        ):
            current_rows = int((language_mask & current.eq(split_name)).sum())
            groups = choose_groups(
                candidate_groups(
                    frame,
                    language,
                    candidate_mask,
                    group_ids,
                    assignments,
                    blocked,
                ),
                max(0, target - current_rows),
                seed=seed + offset * 997 + seed_offset,
                attempts=attempts,
            )
            for group in groups:
                assignments[group] = split_name
            current = group_ids.map(assignments)


def remove_validate_test_conflicting_groups(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    split: pd.Series,
    assignments: dict[int, str],
    blocked: set[int],
    *,
    ngram_threshold: float,
) -> dict[str, int]:
    test = frame[split.eq("test")]
    validate = frame[split.eq("validate")]
    conflicts = near_candidate_conflicts(
        test,
        validate,
        ngram_threshold=ngram_threshold,
    )
    bad_groups = {
        int(group_ids.loc[index])
        for index in conflicts
    }
    for group in bad_groups:
        assignments.pop(group, None)
        blocked.add(group)
    return {
        "conflicting_eval_rows": len(conflicts),
        "conflicting_groups": len(bad_groups),
        "validate_test_conflicting_rows": len(conflicts),
    }


def overlap_summary(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    exact = overlap_stats(train, evaluation)
    skeleton: dict[str, dict[str, int]] = {}
    for name, column in (
        ("formosan", "_formosan_skeleton"),
        ("target", "_target_skeleton"),
        ("pair", "_pair_skeleton"),
    ):
        train_values = set(train[column])
        eval_values = set(evaluation[column])
        skeleton[name] = {
            "train_unique": len(train_values),
            "eval_unique": len(eval_values),
            "overlap_unique": len(train_values & eval_values),
        }
    return {"exact": exact, "skeleton": skeleton}


def build_hard_split(
    frame: pd.DataFrame,
    *,
    target_col: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    min_formosan_tokens: int,
    min_target_tokens: int,
    attempts: int,
    min_test_rows: int,
    min_validate_rows: int,
    ngram_threshold: float,
    registry_in: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if "row_id" not in frame.columns or frame["row_id"].astype(str).duplicated().any():
        raise SystemExit("Input must contain unique stable row_id values")
    if "kindOf" not in frame.columns or not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Hard splitting requires every row to use kindOf=standard")

    frame = frame.copy()
    frame["_document_key"] = (
        frame["lang_code"].astype(str)
        + "\u241f"
        + frame["source"].astype(str)
    )
    frame, duplicate_rows = deduplicate_input(frame)
    frame = add_normalized_columns(
        frame,
        target_col=target_col,
        target_lang=(
            "chinese" if target_col == "chinese_sentence" else "english"
        ),
    )
    frame["_document_key"] = (
        frame["lang_code"].astype(str)
        + "\u241f"
        + frame["source"].astype(str)
    )
    group_ids = leakage_group_ids(frame)
    synthetic, human_sentence, candidate_mask = evaluation_masks(
        frame,
        min_formosan_tokens=min_formosan_tokens,
        min_target_tokens=min_target_tokens,
    )

    targets: dict[str, tuple[int, int]] = {}
    language_reports: dict[str, dict] = {}
    for language, language_frame in frame.groupby("lang_code", sort=True):
        index = language_frame.index
        rows_total = len(language_frame)
        human_total = int(human_sentence.loc[index].sum())
        eligible_total = int(candidate_mask.loc[index].sum())
        test_target, validate_target = split_targets(
            rows_total,
            eligible_total,
            test_ratio,
            val_ratio,
            min_test_rows,
            min_validate_rows,
        )
        targets[str(language)] = (test_target, validate_target)
        language_reports[str(language)] = {
            "rows_total": len(language_frame),
            "human_sentence_rows": human_total,
            "synthetic_rows": int(synthetic.loc[index].sum()),
            "lexical_rows": int(
                language_frame["row_type"].isin({"lexeme", "morpheme"}).sum()
            ),
            "eligible_human_eval_rows": eligible_total,
            "target_test_rows": test_target,
            "target_validate_rows": validate_target,
        }

    assignments, registry_stats = apply_registry(
        frame,
        group_ids,
        candidate_mask,
        registry_in,
    )
    blocked: set[int] = set()
    near_iterations: list[dict[str, int]] = []
    for _ in range(8):
        fill_assignments(
            frame,
            group_ids,
            candidate_mask,
            targets,
            assignments,
            blocked,
            seed=seed,
            attempts=attempts,
        )
        split = materialize_splits(frame, group_ids, candidate_mask, assignments)
        iteration = remove_validate_test_conflicting_groups(
            frame,
            group_ids,
            split,
            assignments,
            blocked,
            ngram_threshold=ngram_threshold,
        )
        near_iterations.append(iteration)
        if iteration["conflicting_groups"] == 0:
            break
    else:
        raise SystemExit("Near-duplicate split stabilization did not converge")

    split = materialize_splits(frame, group_ids, candidate_mask, assignments)
    heldout_document_non_eval = split.eq("excluded")
    evaluation = frame[split.isin({"test", "validate"})]
    training = frame[split.eq("train")]
    near_train_indexes = near_candidate_conflicts(
        evaluation,
        training,
        ngram_threshold=ngram_threshold,
    )
    split.loc[list(near_train_indexes)] = "excluded"
    frame["split"] = split
    frame["eval_tier"] = TIER
    frame["source_bucket"] = frame["_source_bucket"]
    frame["formosan_tokens"] = frame["_formosan_tokens"].astype(int)
    frame["target_tokens"] = frame["_target_tokens"].astype(int)
    frame["short_entry"] = frame["_short_entry"].astype(bool)
    frame["document_id"] = frame["_document_key"]

    exclusion_reason = pd.Series("", index=frame.index, dtype="object")
    exclusion_reason.loc[
        heldout_document_non_eval
    ] = "heldout_document_non_evaluation_row"
    exclusion_reason.loc[
        list(near_train_indexes)
    ] = "near_duplicate_of_evaluation"
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
    )
    validate_test_near = near_candidate_conflicts(
        test,
        validate,
        ngram_threshold=ngram_threshold,
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
                "test_fraction_of_human_sentences": (
                    counts["test"]
                    / max(language_reports[str(language)]["human_sentence_rows"], 1)
                ),
                "validate_fraction_of_human_sentences": (
                    counts["validate"]
                    / max(language_reports[str(language)]["human_sentence_rows"], 1)
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
            }
        )
        if counts["test"] < target_test or counts["validate"] < target_validate:
            ratio_shortfalls[str(language)] = {
                "test": counts["test"],
                "target_test": target_test,
                "validate": counts["validate"],
                "target_validate": target_validate,
            }

    overlaps = overlap_summary(train, evaluation)
    report = {
        "schema_version": 2,
        "complete": not ratio_shortfalls,
        "tier": TIER,
        "input_rows": len(frame) + len(duplicate_rows),
        "deduplicated_input_rows": len(frame),
        "duplicate_rows_removed": len(duplicate_rows),
        "output_rows": len(output),
        "excluded_rows": len(excluded),
        "excluded_document_rows": int(
            heldout_document_non_eval.sum()
        ),
        "excluded_near_duplicate_train_rows": len(
            near_train_indexes
        ),
        "synthetic_eval_rows": int(
            (
                output.get("pivot_origin", pd.Series("original", index=output.index))
                .eq("synthetic")
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "lexeme_eval_rows": int(
            (
                output["row_type"].isin({"lexeme", "morpheme"})
                & output["split"].isin({"test", "validate"})
            ).sum()
        ),
        "document_overlap_train_eval": len(
            set(train["_document_key"]) & set(evaluation["_document_key"])
        ),
        "document_overlap_validate_test": len(
            set(test["_document_key"]) & set(validate["_document_key"])
        ),
        "near_duplicate_train_eval_rows": len(final_near),
        "near_duplicate_validate_test_rows": len(validate_test_near),
        "near_duplicate_iterations": near_iterations,
        "ngram_jaccard_threshold": ngram_threshold,
        "overlap": overlaps,
        "split_counts": split_counts(output),
        "split_counts_by_language": split_counts_by_language(output),
        "bucket_counts": bucket_counts(output),
        "languages": language_reports,
        "ratio_basis": "all_deduplicated_input_rows",
        "required_ratios": {"test": test_ratio, "validate": val_ratio},
        "ratio_shortfalls": ratio_shortfalls,
        "benchmark_registry_input": str(registry_in) if registry_in else None,
        "benchmark_registry_stats": registry_stats,
    }
    if ratio_shortfalls:
        raise SystemExit(
            "Could not construct sufficiently large human-only evaluation sets: "
            + json.dumps(ratio_shortfalls, sort_keys=True)
        )

    internal_columns = [column for column in output.columns if column.startswith("_")]
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
        "document_overlap_train_eval",
        "document_overlap_validate_test",
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
    if failures:
        raise SystemExit(
            "Hard-split release validation failed: "
            + json.dumps(failures, sort_keys=True)
        )


def write_registry(path: Path, output: pd.DataFrame, report: dict) -> None:
    evaluation = output[output["split"].isin({"test", "validate"})]
    payload = {
        "schema_version": 1,
        "complete": True,
        "tier": TIER,
        "ratio_basis": report["ratio_basis"],
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
        description="Build the document-aware, human-only hard MT split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col")
    parser.add_argument("--output-prefix")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--test-ratio", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-formosan-tokens", type=int, default=4)
    parser.add_argument("--min-target-tokens", type=int, default=4)
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--min-validate-rows", type=int, default=150)
    parser.add_argument("--selection-attempts", type=int, default=200)
    parser.add_argument("--ngram-jaccard-threshold", type=float, default=0.82)
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
        raise SystemExit(f"Corpus pipeline v2 supports only --tiers {TIER}")
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-9:
        raise SystemExit("Split ratios must sum to 1.0")
    if not 0.5 <= args.ngram_jaccard_threshold <= 1.0:
        raise SystemExit("--ngram-jaccard-threshold must be in [0.5, 1.0]")

    target_language = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_language)
    target_tag = target_tag_for(target_language)
    short = "en" if target_tag == "eng" else target_tag
    output_prefix = args.output_prefix or f"big_corpus_{short}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_parallel_csv(args.input, target_col=target_col)
    normalized = add_normalized_columns(
        raw,
        target_col=target_col,
        target_lang=target_language,
    )
    output, excluded, duplicates, report = build_hard_split(
        normalized,
        target_col=target_col,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        min_formosan_tokens=args.min_formosan_tokens,
        min_target_tokens=args.min_target_tokens,
        attempts=args.selection_attempts,
        min_test_rows=args.min_test_rows,
        min_validate_rows=args.min_validate_rows,
        ngram_threshold=args.ngram_jaccard_threshold,
        registry_in=args.registry_in,
    )
    validate_report(report)

    full_path = args.output_dir / f"{output_prefix}_{TIER}.csv"
    test_path = args.output_dir / f"{output_prefix}_{TIER}_test.csv"
    validate_path = args.output_dir / f"{output_prefix}_{TIER}_validate.csv"
    excluded_path = args.output_dir / f"{output_prefix}_{TIER}_excluded.csv"
    duplicate_path = args.output_dir / f"{output_prefix}_{TIER}_duplicates.csv"
    registry_path = args.output_dir / "benchmark_registry.json"
    output.to_csv(full_path, index=False)
    output[output["split"].eq("test")].to_csv(test_path, index=False)
    output[output["split"].eq("validate")].to_csv(validate_path, index=False)
    excluded.to_csv(excluded_path, index=False)
    duplicates.to_csv(duplicate_path, index=False)
    write_registry(registry_path, output, report)
    report["files"] = {
        "full": str(full_path),
        "test": str(test_path),
        "validate": str(validate_path),
        "excluded": str(excluded_path),
        "duplicates": str(duplicate_path),
        "benchmark_registry": str(registry_path),
    }
    write_json(args.output_dir / f"report_{TIER}.json", report)
    write_json(
        args.output_dir / "report_all_tiers.json",
        {
            "schema_version": 2,
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
