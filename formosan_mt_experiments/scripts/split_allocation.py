"""Deterministic source-stratified allocation for hard MT splits.

Evaluation rows are a seeded uniform sample of the eligible sentences in each
language and source corpus. Eligibility decides which rows may be held out;
nothing here prefers one eligible row over another, so validation, test, and
the eligible sentences left in training share one distribution.
"""

from __future__ import annotations

import hashlib
import json
import math
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
    priority: int


def selection_priorities(pair_keys: pd.Series, seed: int) -> pd.Series:
    """Give every sentence pair a stable pseudo-random rank for evaluation sampling.

    The rank depends only on the seed and the pair's language and normalized
    text. It does not move when other rows are added, removed, or blocked by
    the similarity gates, when a file is reordered upstream, or when the same
    pair arrives through a different repository. A pair therefore tends to
    stay on the same side of the split across rebuilds and across the public
    and private corpora.
    """
    return pd.Series(
        [
            int.from_bytes(
                hashlib.blake2b(f"{seed}\u241f{key}".encode(), digest_size=8).digest(),
                "big",
            )
            for key in pair_keys.astype(str)
        ],
        index=pair_keys.index,
        dtype="uint64",
    )


def split_targets(
    human_rows: int,
    eligible_total: int,
    test_ratio: float,
    val_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[int, int]:
    """Size evaluation from human pairs and fail when quality capacity is short."""
    if human_rows <= 0:
        return 0, 0
    desired_test = max(math.ceil(human_rows * test_ratio), min_test_rows)
    desired_validate = max(math.ceil(human_rows * val_ratio), min_validate_rows)
    required = desired_test + desired_validate
    if required > eligible_total:
        raise ValueError(
            f"Human-corpus split requires {required:,} evaluation rows from "
            f"{eligible_total:,} eligible sentences across {human_rows:,} human pairs"
        )
    return desired_test, desired_validate


def candidate_groups(
    frame: pd.DataFrame,
    indexes: pd.Index,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    assignments: dict[int, str],
) -> list[GroupCandidate]:
    """Return unassigned leakage groups in sampling order."""
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

    grouped = (
        pd.DataFrame(
            {
                "group_id": group_values,
                "priority": frame.loc[indexes, "_selection_priority"],
            },
            index=indexes,
        )
        .groupby("group_id", sort=False)["priority"]
        .agg(["size", "min"])
    )
    return sorted(
        (
            GroupCandidate(
                group_id=int(row.Index),
                eligible_rows=int(row.size),
                priority=int(row.min),
            )
            for row in grouped.itertuples()
        ),
        key=lambda group: (group.priority, group.group_id),
    )


def choose_groups(
    candidates: list[GroupCandidate],
    target_rows: int,
    *,
    reserve_rows: int,
    allow_overshoot: bool = True,
) -> set[int]:
    """Take groups in sampling order until the row target is met.

    Single-row groups fill the target exactly. Larger leakage groups move as
    a unit and are used only when single rows cannot reach the target.
    """
    if target_rows <= 0 or not candidates:
        return set()
    total_rows = sum(candidate.eligible_rows for candidate in candidates)
    max_selected_rows = max(0, total_rows - reserve_rows)
    desired_rows = min(target_rows, max_selected_rows)
    if desired_rows <= 0:
        return set()

    singletons = [group for group in candidates if group.eligible_rows == 1]
    if len(singletons) >= desired_rows:
        return {group.group_id for group in singletons[:desired_rows]}

    selected: set[int] = set()
    selected_rows = 0
    required_multi_rows = desired_rows - len(singletons)
    for candidate in candidates:
        if candidate.eligible_rows == 1 or selected_rows >= required_multi_rows:
            continue
        if selected_rows + candidate.eligible_rows > desired_rows:
            continue
        selected.add(candidate.group_id)
        selected_rows += candidate.eligible_rows

    if allow_overshoot and selected_rows < required_multi_rows:
        remaining = [
            (position, candidate)
            for position, candidate in enumerate(candidates)
            if candidate.eligible_rows > 1
            and candidate.group_id not in selected
            and selected_rows + candidate.eligible_rows <= max_selected_rows
        ]
        if remaining:
            _, candidate = min(
                remaining,
                key=lambda item: (
                    abs(selected_rows + item[1].eligible_rows - desired_rows),
                    item[0],
                ),
            )
            selected.add(candidate.group_id)
            selected_rows += candidate.eligible_rows

    singleton_rows = min(len(singletons), desired_rows - selected_rows)
    selected.update(group.group_id for group in singletons[:singleton_rows])
    return selected


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
                len(human_frame),
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
        # Divide every source's evaluation rows in the same proportion, so
        # validation and test keep one source mix even when a source has too
        # few eligible sentences to meet its share.
        validate_by_source = weighted_apportioned_counts(
            evaluation_by_source,
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
) -> None:
    eligible = frame[candidate_mask]
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
    for language, source in stratum_order:
        indexes = stratum_indexes.get((language, source), pd.Index([]))
        indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
        stratum_group_ids = group_ids.loc[indexes]
        current = stratum_group_ids.map(assignments)
        target_test, target_validate = targets[(language, source)]
        for split_name, target, reserve_target in (
            ("validate", target_validate, target_test),
            ("test", target_test, target_validate),
        ):
            current_rows = int(current.eq(split_name).sum())
            other_split = "test" if split_name == "validate" else "validate"
            reserve_rows = max(
                0,
                reserve_target - int(current.eq(other_split).sum()),
            )
            groups = choose_groups(
                candidate_groups(
                    frame,
                    indexes,
                    candidate_mask,
                    group_ids,
                    assignments,
                ),
                max(0, target - current_rows),
                reserve_rows=reserve_rows,
                allow_overshoot=False,
            )
            for group in groups:
                assignments[group] = split_name
            current = stratum_group_ids.map(assignments)


def group_assignment_tolerance(
    indexes: pd.Index,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
) -> int:
    eligible_indexes = indexes[candidate_mask.loc[indexes].to_numpy()]
    group_sizes = group_ids.loc[eligible_indexes].value_counts()
    largest_group = int(group_sizes.max()) if not group_sizes.empty else 1
    return largest_group - 1


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
    return max(
        2,
        math.ceil(
            int((~synthetic).sum())
            * SPLIT_DEFAULTS["source_ratio_tolerance"]
        ),
        group_assignment_tolerance(indexes, group_ids, candidate_mask),
    )


def fill_language_shortfalls(
    frame: pd.DataFrame,
    group_ids: pd.Series,
    candidate_mask: pd.Series,
    source_targets: dict[tuple[str, str], tuple[int, int]],
    language_targets: dict[str, tuple[int, int]],
    assignments: dict[int, str],
) -> None:
    eligible = frame[candidate_mask]
    for language, language_frame in eligible.groupby("lang_code", sort=True):
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
        for split_name, target in (
            ("validate", target_validate),
            ("test", target_test),
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
                    allow_overshoot=True,
                )
                assignments.update(
                    {group_id: split_name for group_id in groups}
                )
                current = language_group_ids.map(assignments)
