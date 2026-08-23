"""Deterministic source-stratified allocation for hard MT splits."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from experiment_config import load_corpus_pipeline_config
from mt_common import weighted_apportioned_counts

SPLIT_DEFAULTS = load_corpus_pipeline_config()["splits"]


@dataclass(frozen=True)
class GroupCandidate:
    group_id: int
    eligible_rows: int
    total_rows: int
    non_eval_rows: int
    average_tokens: float
    synthetic_fraction: float = 0.0


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


def candidate_groups(
    frame: pd.DataFrame,
    indexes: pd.Index,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    assignments: dict[int, str],
) -> list[GroupCandidate]:
    indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
    if indexes.empty:
        return []
    group_values = group_ids.loc[indexes].astype("int64")
    if assignments:
        keep = ~group_values.isin(assignments)
        indexes = indexes[keep.to_numpy()]
        group_values = group_values.loc[indexes]
    if indexes.empty:
        return []

    scores = pd.DataFrame(
        {
            "group_id": group_values,
            "average_tokens": (
                frame.loc[indexes, "_formosan_tokens"].to_numpy()
                + frame.loc[indexes, "_target_tokens"].to_numpy()
            )
            / 2,
        },
        index=indexes,
    )
    grouped = scores.groupby("group_id", sort=False)["average_tokens"].agg(
        ["size", "mean"]
    )
    return [
        GroupCandidate(
            group_id=int(row.Index),
            eligible_rows=int(row.size),
            total_rows=int(row.size),
            non_eval_rows=0,
            average_tokens=float(row.mean),
        )
        for row in grouped.itertuples()
    ]


def choose_groups(
    candidates: list[GroupCandidate],
    target_rows: int,
    *,
    reserve_rows: int,
    seed: int,
    attempts: int,
    allow_overshoot: bool = True,
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
    desired_rows = min(target_rows, max_selected_rows)
    if desired_rows <= 0:
        return set()

    singletons = [group for group in ordered if group.eligible_rows == 1]
    if len(singletons) >= desired_rows:
        return {group.group_id for group in singletons[:desired_rows]}

    selected: set[int] = set()
    selected_rows = 0
    required_multi_rows = desired_rows - len(singletons)
    for candidate in ordered:
        if candidate.eligible_rows == 1 or selected_rows >= required_multi_rows:
            continue
        if selected_rows + candidate.eligible_rows > desired_rows:
            continue
        selected.add(candidate.group_id)
        selected_rows += candidate.eligible_rows

    if allow_overshoot and selected_rows < required_multi_rows:
        remaining = [
            candidate
            for candidate in ordered
            if candidate.eligible_rows > 1
            and candidate.group_id not in selected
            and selected_rows + candidate.eligible_rows <= max_selected_rows
        ]
        if remaining:
            candidate = min(
                remaining,
                key=lambda group: (
                    abs(selected_rows + group.eligible_rows - desired_rows),
                    quality_rank[group.group_id],
                ),
            )
            selected.add(candidate.group_id)
            selected_rows += candidate.eligible_rows

    singleton_rows = min(len(singletons), desired_rows - selected_rows)
    selected.update(group.group_id for group in singletons[:singleton_rows])
    return selected


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
        synthetic = (
            language_frame.get(
                "pivot_origin",
                pd.Series("original", index=language_frame.index),
            )
            .astype(str)
            .eq("synthetic")
        )
        human_frame = language_frame[~synthetic]
        source_rows = {
            str(source): len(group)
            for source, group in human_frame.groupby("_source_corpus", sort=True)
        }
        eligible_capacities = {
            str(source): len(group)
            for source, group in eligible.groupby("_source_corpus", sort=True)
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
        for source, evaluation_rows in evaluation_by_source.items():
            validate_rows = validate_by_source.get(source, 0)
            targets[(language_key, source)] = (
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
    for offset, (language, source) in enumerate(stratum_order):
        indexes = stratum_indexes.get((language, source), pd.Index([]))
        indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
        stratum_group_ids = group_ids.loc[indexes]
        current = stratum_group_ids.map(assignments)
        target_test, target_validate = targets[(language, source)]
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
                    allow_overshoot=False,
                )
            for group in groups:
                assignments[group] = split_name
            current = stratum_group_ids.map(assignments)


def source_assignment_tolerance(
    frame: pd.DataFrame,
    indexes: pd.Index,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
) -> int:
    source_frame = frame.loc[indexes]
    synthetic = source_frame.get(
        "pivot_origin",
        pd.Series("original", index=source_frame.index),
    ).astype(str).eq("synthetic")
    eligible_indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
    group_sizes = group_ids.loc[eligible_indexes].value_counts()
    largest_group = int(group_sizes.max()) if not group_sizes.empty else 1
    return max(
        2,
        math.ceil(
            int((~synthetic).sum())
            * SPLIT_DEFAULTS["source_ratio_tolerance"]
        ),
        largest_group - 1,
    )


def fill_language_shortfalls(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    source_targets: dict[tuple[str, str], tuple[int, int]],
    language_targets: dict[str, tuple[int, int]],
    assignments: dict[int, str],
    *,
    seed: int,
    attempts: int,
) -> None:
    eligible = frame[candidate_mask]
    for offset, (language, language_frame) in enumerate(
        eligible.groupby("lang_code", sort=True)
    ):
        language_key = str(language)
        indexes = language_frame.index
        language_group_ids = group_ids.loc[indexes]
        current = language_group_ids.map(assignments)
        target_test, target_validate = language_targets[language_key]
        source_indexes = {
            str(source): source_frame.index
            for source, source_frame in language_frame.groupby(
                "_source_corpus", sort=True
            )
        }
        for split_name, target, seed_offset in (
            ("validate", target_validate, 43),
            ("test", target_test, 29),
        ):
            missing = max(0, target - int(current.eq(split_name).sum()))
            if not missing:
                continue
            other_split = "test" if split_name == "validate" else "validate"
            source_order = []
            for source, stratum_indexes in source_indexes.items():
                stratum_current = group_ids.loc[stratum_indexes].map(assignments)
                source_test, source_validate = source_targets.get(
                    (language_key, source), (0, 0)
                )
                source_target = (
                    source_test if split_name == "test" else source_validate
                )
                deficit = max(
                    0,
                    source_target - int(stratum_current.eq(split_name).sum()),
                )
                source_order.append((deficit, source))

            for _, source in sorted(
                source_order, key=lambda value: (-value[0], value[1])
            ):
                missing = max(0, target - int(current.eq(split_name).sum()))
                if not missing:
                    break
                stratum_indexes = source_indexes[source]
                stratum_current = group_ids.loc[stratum_indexes].map(assignments)
                source_test, source_validate = source_targets.get(
                    (language_key, source), (0, 0)
                )
                source_target = (
                    source_test if split_name == "test" else source_validate
                )
                source_other_target = (
                    source_validate if split_name == "test" else source_test
                )
                source_rows = int(stratum_current.eq(split_name).sum())
                headroom = (
                    source_target
                    + source_assignment_tolerance(
                        frame,
                        stratum_indexes,
                        group_ids,
                        candidate_mask,
                    )
                    - source_rows
                )
                if headroom <= 0:
                    continue
                candidates = candidate_groups(
                    frame,
                    stratum_indexes,
                    candidate_mask,
                    group_ids,
                    assignments,
                )
                available_rows = sum(group.eligible_rows for group in candidates)
                reserve_rows = max(
                    max(
                        0,
                        source_other_target
                        - int(stratum_current.eq(other_split).sum()),
                    ),
                    available_rows - min(missing, headroom),
                )
                desired_rows = min(
                    missing,
                    max(1, source_target - source_rows),
                )
                groups = choose_groups(
                    candidates,
                    desired_rows,
                    reserve_rows=reserve_rows,
                    seed=(
                        seed
                        + offset * 1543
                        + seed_offset
                        + sum(map(ord, source))
                    ),
                    attempts=attempts,
                    allow_overshoot=True,
                )
                assignments.update(
                    {group_id: split_name for group_id in groups}
                )
                current = language_group_ids.map(assignments)
