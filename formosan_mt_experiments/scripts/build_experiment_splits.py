#!/usr/bin/env python3
"""Build one source-stratified, similarity-controlled hard MT split."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from columnar_cache import write_columnar_cache
from experiment_config import load_corpus_pipeline_config
from mt_common import (
    add_normalized_columns,
    bool_series,
    bucket_counts,
    evaluation_candidate_mask,
    mt_standard_contract,
    normalize_target_language,
    read_parallel_csv,
    split_counts,
    split_counts_by_language,
    target_col_for,
    target_tag_for,
    weighted_apportioned_counts,
    write_json,
)

SPLIT_DEFAULTS = load_corpus_pipeline_config()["splits"]
TIER = SPLIT_DEFAULTS["headline_tier"]


@dataclass(frozen=True)
class GroupCandidate:
    group_id: int
    eligible_rows: int
    total_rows: int
    non_eval_rows: int
    average_tokens: float
    synthetic_fraction: float = 0.0


def one_edit_conflicts(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    column: str,
    *,
    by_language: bool = True,
    ignore_same_index: bool = False,
) -> set[int]:
    """Return train indexes whose text is within one character edit of eval."""
    conflicts: set[int] = set()
    evaluation_groups = (
        eval_df.groupby("lang_code", sort=False)
        if by_language
        else [("_global", eval_df)]
    )
    training_groups = (
        {
            str(language): group
            for language, group in train.groupby("lang_code", sort=False)
        }
        if by_language
        else {"_global": train}
    )
    for language, evaluation in evaluation_groups:
        training = training_groups.get(str(language))
        if training is None:
            continue
        if training.empty:
            continue
        eval_values: Counter[str] = Counter()
        eval_value_owner: dict[str, int] = {}
        eval_deletions: Counter[str] = Counter()
        eval_deletion_owner: dict[str, int] = {}
        for eval_index, raw_value in evaluation[column].items():
            value = str(raw_value)
            if not value:
                continue
            index = int(eval_index)
            eval_values[value] += 1
            eval_value_owner.setdefault(value, index)
            for deletion in {
                value[:position] + value[position + 1 :]
                for position in range(len(value))
            }:
                eval_deletions[deletion] += 1
                eval_deletion_owner.setdefault(deletion, index)

        def matches_other(
            counts: Counter[str],
            owners: dict[str, int],
            value: str,
            index: int,
        ) -> bool:
            count = counts[value]
            if not count:
                return False
            if not ignore_same_index or owners[value] != index:
                return True
            return count > 1

        for index, raw_value in training[column].items():
            index = int(index)
            value = str(raw_value)
            if not value:
                continue
            if matches_other(eval_values, eval_value_owner, value, index) or matches_other(
                eval_deletions,
                eval_deletion_owner,
                value,
                index,
            ):
                conflicts.add(index)
                continue
            for deletion in {
                value[:position] + value[position + 1 :]
                for position in range(len(value))
            }:
                if matches_other(
                    eval_values,
                    eval_value_owner,
                    deletion,
                    index,
                ) or matches_other(
                    eval_deletions,
                    eval_deletion_owner,
                    deletion,
                    index,
                ):
                    conflicts.add(index)
                    break
    return conflicts


def one_edit_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
    ignore_same_index: bool = False,
) -> set[int]:
    return one_edit_conflicts(
        candidates,
        reference,
        column,
        by_language=by_language,
        ignore_same_index=ignore_same_index,
    )


def globally_unsafe_one_edit_candidates(
    frame: pd.DataFrame,
    candidate_mask: pd.Series,
) -> set[int]:
    candidates = frame[candidate_mask]
    conflicts = one_edit_candidate_conflicts(
        frame,
        candidates,
        "_target_skeleton",
        by_language=False,
        ignore_same_index=True,
    )
    conflicts |= one_edit_candidate_conflicts(
        frame,
        candidates,
        "_formosan_skeleton",
        by_language=True,
        ignore_same_index=True,
    )
    return conflicts


def jaccard_prefix(
    grams: frozenset[int],
    threshold: float,
) -> tuple[int, ...]:
    """PPJoin-style prefix that cannot miss a pair above the threshold."""
    prefix_length = len(grams) - math.ceil(threshold * len(grams)) + 1
    return tuple(sorted(grams)[: max(1, prefix_length)])


class NgramSimilarityIndex:
    """Reusable exact character n-gram lookup for split refinement."""

    def __init__(
        self,
        frame: pd.DataFrame,
        column: str,
        *,
        by_language: bool,
        threshold: float,
    ) -> None:
        self.threshold = threshold
        self.by_language = by_language
        self.languages = frame["lang_code"].astype(str)
        self.features: dict[int, frozenset[int]] = {}
        self.prefixes: dict[int, tuple[int, ...]] = {}
        self.postings: dict[str, dict[int, list[int]]] = {}
        values = frame[column].astype(str)
        gram_counts: Counter[str] = Counter()
        for value in values:
            if len(value) >= 8:
                gram_counts.update(
                    {
                        value[position : position + 4]
                        for position in range(len(value) - 3)
                    }
                )
        gram_ids = {
            gram: index
            for index, gram in enumerate(
                sorted(gram_counts, key=lambda gram: (gram_counts[gram], gram))
            )
        }

        for index, value in values.items():
            if len(value) < 8:
                continue
            row_index = int(index)
            grams = frozenset(
                gram_ids[value[position : position + 4]]
                for position in range(len(value) - 3)
            )
            prefix = jaccard_prefix(grams, threshold)
            group = self.languages.at[index] if by_language else "_global"
            group_postings = self.postings.setdefault(group, {})
            self.features[row_index] = grams
            self.prefixes[row_index] = prefix
            for gram in prefix:
                group_postings.setdefault(gram, []).append(row_index)

    def conflicts(
        self,
        reference: pd.Index,
        candidates: pd.Index,
        *,
        same_language: bool = False,
    ) -> set[int]:
        if reference.empty or candidates.empty:
            return set()
        reference_indexes = {int(index) for index in reference}
        reference_by_language: dict[str, set[int]] = {}
        if same_language and not self.by_language:
            for index in reference_indexes:
                reference_by_language.setdefault(
                    self.languages.at[index],
                    set(),
                ).add(index)

        conflicts: set[int] = set()
        threshold = self.threshold
        for raw_index in candidates:
            index = int(raw_index)
            grams = self.features.get(index)
            if grams is None:
                continue
            language = self.languages.at[index]
            group = language if self.by_language else "_global"
            active_reference = (
                reference_by_language.get(language, set())
                if same_language and not self.by_language
                else reference_indexes
            )
            postings = self.postings.get(group, {})
            seen: set[int] = set()
            matched = False
            for gram in self.prefixes[index]:
                for other_index in postings.get(gram, ()):
                    if (
                        other_index in seen
                        or other_index not in active_reference
                    ):
                        continue
                    seen.add(other_index)
                    other = self.features[other_index]
                    if not (
                        threshold * len(grams)
                        <= len(other)
                        <= len(grams) / threshold
                    ):
                        continue
                    intersection = len(grams & other)
                    union = len(grams) + len(other) - intersection
                    if union and intersection / union >= threshold:
                        conflicts.add(index)
                        matched = True
                        break
                if matched:
                    break
        return conflicts


@dataclass(frozen=True)
class SplitNgramIndexes:
    formosan: NgramSimilarityIndex
    target: NgramSimilarityIndex


def ngram_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    column: str,
    *,
    by_language: bool,
    threshold: float,
) -> set[int]:
    """Find candidate conflicts using the reusable exact lookup."""
    if reference.empty or candidates.empty:
        return set()
    combined = pd.concat(
        [
            reference[["lang_code", column]],
            candidates[["lang_code", column]],
        ],
        axis=0,
    )
    index = NgramSimilarityIndex(
        combined,
        column,
        by_language=by_language,
        threshold=threshold,
    )
    return index.conflicts(reference.index, candidates.index)


def near_candidate_conflicts(
    reference: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    ngram_threshold: float,
    target_by_language: bool = False,
    include_one_edit: bool = True,
    ngram_indexes: SplitNgramIndexes | None = None,
) -> set[int]:
    conflicts: set[int] = set()
    if include_one_edit:
        conflicts |= one_edit_candidate_conflicts(
            reference,
            candidates,
            "_formosan_skeleton",
            by_language=True,
        )
        conflicts |= one_edit_candidate_conflicts(
            reference,
            candidates,
            "_target_skeleton",
            by_language=target_by_language,
        )
    if ngram_indexes is None:
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
            by_language=target_by_language,
            threshold=ngram_threshold,
        )
    else:
        conflicts |= ngram_indexes.formosan.conflicts(
            reference.index,
            candidates.index,
        )
        conflicts |= ngram_indexes.target.conflicts(
            reference.index,
            candidates.index,
            same_language=target_by_language,
        )
    return conflicts


def leakage_group_ids(frame: pd.DataFrame) -> pd.Series:
    """Use stable rows as assignment units; similarity gates enforce hardness.

    Large source corpora are often serialized as one XML file. Treating the
    whole file as an indivisible document can place thousands of rows in test
    and makes source-proportional splitting impossible.
    """
    groups, _ = pd.factorize(frame["row_id"].astype(str), sort=False)
    return pd.Series(groups, index=frame.index, dtype="int64")


def split_targets(
    total_rows: int,
    eligible_total: int,
    test_ratio: float,
    val_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[int, int]:
    """Size evaluation from all pairs and fail when quality capacity is short."""
    if total_rows <= 0:
        return 0, 0
    desired_test = max(math.ceil(total_rows * test_ratio), min_test_rows)
    desired_validate = max(math.ceil(total_rows * val_ratio), min_validate_rows)
    required = desired_test + desired_validate
    if required > eligible_total:
        raise ValueError(
            f"All-pair split requires {required:,} evaluation rows from "
            f"{eligible_total:,} eligible sentences across {total_rows:,} pairs"
        )
    return desired_test, desired_validate


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


def candidate_groups(
    frame: pd.DataFrame,
    indexes: pd.Index,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    assignments: dict[int, str],
) -> list[GroupCandidate]:
    language_frame = frame.loc[indexes]
    language_group_ids = group_ids.loc[language_frame.index]
    candidates: list[GroupCandidate] = []
    for group_id, group in language_frame.groupby(language_group_ids, sort=False):
        group_key = int(group_id)
        if group_key in assignments:
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
                average_tokens=float(
                    (eligible["_formosan_tokens"] + eligible["_target_tokens"]).mean()
                    / 2
                ),
                synthetic_fraction=float(
                    eligible.get(
                        "pivot_origin",
                        pd.Series("original", index=eligible.index),
                    )
                    .astype(str)
                    .eq("synthetic")
                    .mean()
                ),
            )
        )
    return candidates


def choose_groups(
    candidates: list[GroupCandidate],
    target_rows: int,
    *,
    reserve_rows: int,
    seed: int,
    attempts: int,
) -> set[int]:
    if target_rows <= 0 or not candidates:
        return set()
    rng = random.Random(seed)
    tie_breakers = {
        candidate.group_id: rng.random()
        for candidate in candidates
    }
    ordered = sorted(
        candidates,
        key=lambda group: (
            group.synthetic_fraction,
            group.non_eval_rows / max(group.eligible_rows, 1),
            -group.average_tokens,
            tie_breakers[group.group_id],
            group.group_id,
        ),
    )
    quality_rank = {
        candidate.group_id: rank
        for rank, candidate in enumerate(ordered)
    }
    total_rows = sum(candidate.eligible_rows for candidate in candidates)
    max_selected_rows = max(0, total_rows - reserve_rows)

    if all(candidate.eligible_rows == 1 for candidate in ordered):
        selected_rows = min(target_rows, max_selected_rows, len(ordered))
        return {
            candidate.group_id
            for candidate in ordered[:selected_rows]
        }

    # Keep the best-quality subset for each reachable row count when a future
    # registry or leakage component groups more than one row together.
    states: dict[
        int,
        tuple[tuple[float, float, float, int, int], tuple[int, ...]],
    ] = {
        0: ((0.0, 0.0, 0.0, 0, 0), ()),
    }
    for candidate in ordered:
        weight = candidate.eligible_rows
        additions: dict[
            int,
            tuple[tuple[float, float, float, int, int], tuple[int, ...]],
        ] = {}
        for rows, (cost, selected) in list(states.items()):
            next_rows = rows + weight
            if next_rows > max_selected_rows:
                continue
            next_cost = (
                cost[0] + candidate.synthetic_fraction * weight,
                cost[1] + candidate.non_eval_rows,
                cost[2] - candidate.average_tokens * weight,
                cost[3] + 1,
                cost[4] + quality_rank[candidate.group_id],
            )
            next_selected = (*selected, candidate.group_id)
            previous = states.get(next_rows) or additions.get(next_rows)
            if previous is None or (next_cost, next_selected) < previous:
                additions[next_rows] = (next_cost, next_selected)
        for rows, state in additions.items():
            previous = states.get(rows)
            if previous is None or state < previous:
                states[rows] = state

    feasible = [
        (rows, cost, selected)
        for rows, (cost, selected) in states.items()
        if rows >= target_rows
    ]
    if not feasible:
        feasible = [
            (rows, cost, selected)
            for rows, (cost, selected) in states.items()
            if rows > 0
        ]
    if not feasible:
        return set()
    _, _, selected = min(
        feasible,
        key=lambda item: (
            abs(item[0] - target_rows),
            item[1],
            item[2],
        )
    )
    return set(selected)


def choose_singleton_groups(
    frame: pd.DataFrame,
    indexes: pd.Index,
    group_ids: pd.Series,
    assignments: dict[int, str],
    target_rows: int,
    *,
    reserve_rows: int,
    seed: int,
) -> set[int]:
    """Choose row-sized groups without constructing one pandas group per row."""
    if target_rows <= 0 or indexes.empty:
        return set()
    candidate_group_ids = group_ids.loc[indexes]
    if assignments:
        unassigned = ~candidate_group_ids.isin(assignments)
        indexes = indexes[unassigned.to_numpy()]
        candidate_group_ids = candidate_group_ids.loc[indexes]
    available = len(indexes)
    selected_rows = min(target_rows, max(0, available - reserve_rows))
    if selected_rows <= 0:
        return set()

    candidates = frame.loc[indexes]
    synthetic = (
        candidates.get(
            "pivot_origin",
            pd.Series("original", index=candidates.index),
        )
        .astype(str)
        .eq("synthetic")
    )
    average_tokens = (
        candidates["_formosan_tokens"] + candidates["_target_tokens"]
    ) / 2
    rng = random.Random(seed)
    ranked = pd.DataFrame(
        {
            "group_id": candidate_group_ids.astype("int64"),
            "synthetic": synthetic.astype("int8"),
            "average_tokens": average_tokens.astype("float64"),
            "tie_breaker": [rng.random() for _ in range(available)],
        },
        index=indexes,
    ).sort_values(
        ["synthetic", "average_tokens", "tie_breaker", "group_id"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    return set(ranked["group_id"].head(selected_rows).astype(int))


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


def source_stratum_targets(
    frame: pd.DataFrame,
    candidate_mask: pd.Series,
    *,
    test_ratio: float,
    val_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[
    dict[tuple[str, str], tuple[int, int]],
    dict[str, tuple[int, int]],
]:
    targets: dict[tuple[str, str], tuple[int, int]] = {}
    language_targets: dict[str, tuple[int, int]] = {}
    for language, language_frame in frame.groupby("lang_code", sort=True):
        language_key = str(language)
        eligible = language_frame[candidate_mask.loc[language_frame.index]]
        source_rows = {
            str(bucket): len(group)
            for bucket, group in language_frame.groupby("_source_corpus", sort=True)
        }
        eligible_capacities = {
            str(bucket): len(group)
            for bucket, group in eligible.groupby("_source_corpus", sort=True)
        }
        try:
            test_total, validate_total = split_targets(
                len(language_frame),
                len(eligible),
                test_ratio,
                val_ratio,
                min_test_rows,
                min_validate_rows,
            )
        except ValueError as exc:
            raise SystemExit(f"{language_key}: {exc}") from exc
        language_targets[language_key] = (test_total, validate_total)
        evaluation_by_source = weighted_apportioned_counts(
            source_rows,
            eligible_capacities,
            test_total + validate_total,
        )
        validate_by_source = weighted_apportioned_counts(
            source_rows,
            evaluation_by_source,
            validate_total,
        )
        for bucket, evaluation_rows in evaluation_by_source.items():
            validate_rows = validate_by_source.get(bucket, 0)
            targets[(language_key, bucket)] = (
                evaluation_rows - validate_rows,
                validate_rows,
            )
    return targets, language_targets


def fill_assignments(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    targets: dict[tuple[str, str], tuple[int, int]],
    assignments: dict[int, str],
    *,
    seed: int,
    attempts: int,
) -> None:
    eligible = frame[candidate_mask]
    singleton_groups = group_ids.is_unique
    stratum_indexes = {
        (str(language), str(corpus)): group.index
        for (language, corpus), group in eligible.groupby(
            ["lang_code", "_source_corpus"],
            sort=False,
        )
    }
    stratum_order = sorted(
        targets,
        key=lambda stratum: (
            len(stratum_indexes.get(stratum, ())),
            stratum,
        ),
    )
    for offset, (language, bucket) in enumerate(stratum_order):
        indexes = stratum_indexes.get((language, bucket), pd.Index([]))
        indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
        stratum_group_ids = group_ids.loc[indexes]
        current = stratum_group_ids.map(assignments)
        target_test, target_validate = targets[(language, bucket)]
        for split_name, target, reserve_target, seed_offset in (
            ("validate", target_validate, target_test, 17),
            ("test", target_test, target_validate, 0),
        ):
            current_rows = int(current.eq(split_name).sum())
            other_split = "test" if split_name == "validate" else "validate"
            reserve_rows = max(
                0,
                reserve_target - int(current.eq(other_split).sum()),
            )
            target_rows = max(0, target - current_rows)
            choice_seed = seed + offset * 997 + seed_offset
            if singleton_groups:
                groups = choose_singleton_groups(
                    frame,
                    indexes,
                    group_ids,
                    assignments,
                    target_rows,
                    reserve_rows=reserve_rows,
                    seed=choice_seed,
                )
            else:
                groups = choose_groups(
                    candidate_groups(
                        frame,
                        indexes,
                        candidate_mask,
                        group_ids,
                        assignments,
                    ),
                    target_rows,
                    reserve_rows=reserve_rows,
                    seed=choice_seed,
                    attempts=attempts,
                )
            for group in groups:
                assignments[group] = split_name
            current = stratum_group_ids.map(assignments)


def exclude_test_conflicts_with_validation(
    frame: pd.DataFrame,
    split: pd.Series,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    assignments: dict[int, str],
    blocked_indexes: set[int],
    *,
    ngram_threshold: float,
    include_one_edit: bool = True,
    ngram_indexes: SplitNgramIndexes | None = None,
) -> dict[str, int]:
    validate = frame[split.eq("validate")]
    test = frame[split.eq("test")]
    conflicts = near_candidate_conflicts(
        validate,
        test,
        ngram_threshold=ngram_threshold,
        target_by_language=True,
        include_one_edit=include_one_edit,
        ngram_indexes=ngram_indexes,
    )
    conflict_groups = {
        int(group_id)
        for group_id in group_ids.loc[list(conflicts)]
    }
    blocked = set(conflicts)
    if conflicts:
        remaining = frame[
            candidate_mask
            & ~frame.index.isin(validate.index)
        ]
        blocked |= near_candidate_conflicts(
            validate,
            remaining,
            ngram_threshold=ngram_threshold,
            target_by_language=True,
            include_one_edit=include_one_edit,
            ngram_indexes=ngram_indexes,
        )
    blocked_groups = {
        int(group_id)
        for group_id in group_ids.loc[list(blocked)]
    }
    for group_id in blocked_groups:
        if assignments.get(group_id) == "test":
            assignments.pop(group_id)
    candidate_mask.loc[list(blocked)] = False
    blocked_indexes.update(blocked)
    return {
        "conflicting_eval_rows": len(conflicts),
        "conflicting_groups": len(conflict_groups),
        "validate_test_conflicting_rows": len(conflicts),
        "blocked_candidate_rows": len(blocked),
    }


def block_evaluation_conflicts_with_training(
    frame: pd.DataFrame,
    split: pd.Series,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    assignments: dict[int, str],
    blocked_indexes: set[int],
    *,
    ngram_threshold: float,
    include_one_edit: bool = True,
    ngram_indexes: SplitNgramIndexes | None = None,
) -> dict[str, int]:
    training = frame[split.eq("train")]
    evaluation = frame[split.isin({"test", "validate"})]
    conflicts = near_candidate_conflicts(
        training,
        evaluation,
        ngram_threshold=ngram_threshold,
        include_one_edit=include_one_edit,
        ngram_indexes=ngram_indexes,
    )
    conflict_groups = {
        int(group_id)
        for group_id in group_ids.loc[list(conflicts)]
    }
    blocked = set(conflicts)
    if conflicts:
        conflict_rows = frame.loc[list(conflicts)]
        remaining = frame[
            candidate_mask
            & ~frame.index.isin(conflicts)
        ]
        blocked |= near_candidate_conflicts(
            conflict_rows,
            remaining,
            ngram_threshold=ngram_threshold,
            include_one_edit=include_one_edit,
            ngram_indexes=ngram_indexes,
        )
    blocked_groups = {
        int(group_id)
        for group_id in group_ids.loc[list(blocked)]
    }
    for group_id in blocked_groups:
        assignments.pop(group_id, None)
    candidate_mask.loc[list(blocked)] = False
    blocked_indexes.update(blocked)
    return {
        "conflicting_eval_rows": len(conflicts),
        "conflicting_groups": len(conflict_groups),
        "blocked_candidate_rows": len(blocked),
    }


def overlap_summary(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict:
    summaries: dict[str, dict[str, dict[str, int]]] = {}
    for family, columns in (
        (
            "exact",
            (
                ("formosan", "_formosan_key", True),
                ("target", "_target_key", False),
                ("pair", "_pair_key", True),
            ),
        ),
        (
            "skeleton",
            (
                ("formosan", "_formosan_skeleton", True),
                ("target", "_target_skeleton", False),
                ("pair", "_pair_skeleton", True),
            ),
        ),
    ):
        values: dict[str, dict[str, int]] = {}
        for name, column, by_language in columns:
            train_values = train[column].astype(str)
            eval_values = evaluation[column].astype(str)
            if by_language:
                train_values = (
                    train["lang_code"].astype(str)
                    + "\u241f"
                    + train_values
                )
                eval_values = (
                    evaluation["lang_code"].astype(str)
                    + "\u241f"
                    + eval_values
                )
            train_set = set(train_values)
            eval_set = set(eval_values)
            values[name] = {
                "train_unique": len(train_set),
                "eval_unique": len(eval_set),
                "overlap_unique": len(train_set & eval_set),
            }
        summaries[family] = values
    return summaries


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
            "_formosan_skeleton",
            by_language=True,
            threshold=ngram_threshold,
        ),
        target=NgramSimilarityIndex(
            frame,
            "_target_skeleton",
            by_language=False,
            threshold=ngram_threshold,
        ),
    )
    synthetic, human_candidate, candidate_mask = (
        evaluation_masks(
            frame,
            min_formosan_tokens=min_formosan_tokens,
            min_target_tokens=min_target_tokens,
            min_combined_tokens=min_combined_tokens,
            min_punctuated_combined_tokens=min_punctuated_combined_tokens,
            max_eval_units_per_side=max_eval_units_per_side,
        )
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
            "eligible_sentence_rows": eligible_total,
            "eligible_human_sentence_rows": int(
                human_candidate.loc[index].sum()
            ),
            "eligible_synthetic_sentence_rows": int(
                (candidate_mask.loc[index] & synthetic.loc[index]).sum()
            ),
            "synthetic_rows": int(synthetic.loc[index].sum()),
            "lexical_rows": int(
                language_frame["row_type"].isin({"lexeme", "morpheme"}).sum()
            ),
            "evaluation_ineligible_rows": len(language_frame) - eligible_total,
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
    one_edit_blocked_indexes = globally_unsafe_one_edit_candidates(
        frame,
        candidate_mask,
    )
    effective_candidate_mask = candidate_mask.copy()
    effective_candidate_mask.loc[list(one_edit_blocked_indexes)] = False
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
        split = materialize_splits(
            frame,
            group_ids,
            effective_candidate_mask,
            assignments,
        )
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
    frame["source_bucket"] = frame["_source_bucket"]
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
        eligible_language = language_frame[
            candidate_mask.loc[language_frame.index]
        ]
        all_distribution = Counter(language_frame["_source_corpus"])
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
                        all_distribution[bucket] / max(all_total, 1)
                        - distribution[bucket] / max(total, 1)
                    )
                    for bucket in set(all_distribution) | set(distribution)
                )
            )
        for bucket, source_frame in language_frame.groupby(
            "_source_corpus", sort=True
        ):
            bucket_key = str(bucket)
            eligible_source = source_frame[
                candidate_mask.loc[source_frame.index]
            ]
            similarity_safe_source = source_frame[
                effective_candidate_mask.loc[source_frame.index]
            ]
            target_test, target_validate = source_targets.get(
                (language_key, bucket_key),
                (0, 0),
            )
            counts = Counter(source_frame["split"])
            eligible_counts = Counter(eligible_source["split"])
            synthetic_source = eligible_source.get(
                "pivot_origin",
                pd.Series("original", index=eligible_source.index),
            ).astype(str).eq("synthetic")
            source_reports[language_key][bucket_key] = {
                "input_rows": len(source_frame),
                "eligible_sentence_rows": len(eligible_source),
                "similarity_safe_sentence_rows": len(similarity_safe_source),
                "eligible_human_rows": int((~synthetic_source).sum()),
                "eligible_synthetic_rows": int(synthetic_source.sum()),
                "train_rows": counts["train"],
                "test_rows": counts["test"],
                "validate_rows": counts["validate"],
                "excluded_rows": counts["excluded"],
                "eligible_train_rows": eligible_counts["train"],
                "target_test_rows": target_test,
                "target_validate_rows": target_validate,
                "test_fraction": counts["test"] / max(len(source_frame), 1),
                "validate_fraction": (
                    counts["validate"] / max(len(source_frame), 1)
                ),
                "test_fraction_of_eligible_sentences": (
                    counts["test"] / max(len(eligible_source), 1)
                ),
                "validate_fraction_of_eligible_sentences": (
                    counts["validate"] / max(len(eligible_source), 1)
                ),
                "synthetic_test_rows": int(
                    (synthetic_source & eligible_source["split"].eq("test")).sum()
                ),
                "synthetic_validate_rows": int(
                    (
                        synthetic_source
                        & eligible_source["split"].eq("validate")
                    ).sum()
                ),
            }
            if (
                counts["test"] != target_test
                or counts["validate"] != target_validate
            ):
                source_shortfalls[language_key][bucket_key] = {
                    "test": counts["test"],
                    "target_test": target_test,
                    "validate": counts["validate"],
                    "target_validate": target_validate,
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
        "blocked_global_one_edit_candidate_rows": len(
            one_edit_blocked_indexes
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
        "bucket_counts": bucket_counts(output),
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
