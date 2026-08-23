"""Similarity indexes and leakage blocking for hard MT splits."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import pandas as pd

CHARACTER_NGRAM_ORDERS = (3, 4, 5)


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


def jaccard_prefix(
    grams: frozenset[int],
    threshold: float,
) -> tuple[int, ...]:
    """PPJoin-style prefix that cannot miss a pair above the threshold."""
    prefix_length = len(grams) - math.ceil(threshold * len(grams)) + 1
    return tuple(sorted(grams)[: max(1, prefix_length)])


def character_ngrams(value: str) -> frozenset[str]:
    """Match TAME-MT's character 3/4/5-gram feature set."""
    if not value:
        return frozenset()
    if len(value) < min(CHARACTER_NGRAM_ORDERS):
        return frozenset({value})
    return frozenset(
        value[position : position + order]
        for order in CHARACTER_NGRAM_ORDERS
        if order <= len(value)
        for position in range(len(value) - order + 1)
    )


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
            gram_counts.update(character_ngrams(value))
        gram_ids = {
            gram: index
            for index, gram in enumerate(
                sorted(gram_counts, key=lambda gram: (gram_counts[gram], gram))
            )
        }

        for index, value in values.items():
            raw_grams = character_ngrams(value)
            if not raw_grams:
                continue
            row_index = int(index)
            grams = frozenset(gram_ids[gram] for gram in raw_grams)
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
        candidate_indexes = {int(index) for index in candidates}
        reference_by_language: dict[str, set[int]] = {}
        if same_language and not self.by_language:
            for index in reference_indexes:
                reference_by_language.setdefault(
                    self.languages.at[index],
                    set(),
                ).add(index)

        threshold = self.threshold
        if len(reference_indexes) < len(candidate_indexes):
            conflicts: set[int] = set()
            seen_pairs: set[tuple[int, int]] = set()
            for reference_index in reference_indexes:
                reference_grams = self.features.get(reference_index)
                if reference_grams is None:
                    continue
                language = self.languages.at[reference_index]
                group = language if self.by_language else "_global"
                postings = self.postings.get(group, {})
                for gram in self.prefixes[reference_index]:
                    for candidate_index in postings.get(gram, ()):
                        pair = (reference_index, candidate_index)
                        if (
                            pair in seen_pairs
                            or candidate_index not in candidate_indexes
                            or (
                                same_language
                                and self.languages.at[candidate_index] != language
                            )
                        ):
                            continue
                        seen_pairs.add(pair)
                        candidate_grams = self.features[candidate_index]
                        if not (
                            threshold * len(reference_grams)
                            <= len(candidate_grams)
                            <= len(reference_grams) / threshold
                        ):
                            continue
                        intersection = len(reference_grams & candidate_grams)
                        union = (
                            len(reference_grams)
                            + len(candidate_grams)
                            - intersection
                        )
                        if union and intersection / union >= threshold:
                            conflicts.add(candidate_index)
            return conflicts

        conflicts: set[int] = set()
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
    """Group same-language exact and one-edit variants for split assignment."""
    indexes = list(frame.index)
    positions = {int(index): position for position, index in enumerate(indexes)}
    parent = list(range(len(indexes)))
    sizes = [1] * len(indexes)

    def find(position: int) -> int:
        while parent[position] != position:
            parent[position] = parent[parent[position]]
            position = parent[position]
        return position

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if sizes[left_root] < sizes[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        sizes[left_root] += sizes[right_root]

    for _, language_frame in frame.groupby("lang_code", sort=False):
        for column in ("_formosan_skeleton", "_target_skeleton"):
            exact_owner: dict[str, int] = {}
            for index, raw_value in language_frame[column].items():
                value = str(raw_value)
                if not value:
                    continue
                position = positions[int(index)]
                previous = exact_owner.get(value)
                if previous is None:
                    exact_owner[value] = position
                else:
                    union(position, previous)

            deletion_owner: dict[str, int] = {}
            for value, position in exact_owner.items():
                for deletion in {
                    value[:offset] + value[offset + 1 :]
                    for offset in range(len(value))
                }:
                    exact_match = exact_owner.get(deletion)
                    if exact_match is not None:
                        union(position, exact_match)
                    deletion_match = deletion_owner.get(deletion)
                    if deletion_match is None:
                        deletion_owner[deletion] = position
                    else:
                        union(position, deletion_match)

    roots = [find(position) for position in range(len(indexes))]
    groups, _ = pd.factorize(pd.Series(roots), sort=False)
    return pd.Series(groups, index=frame.index, dtype="int64")


def group_safe_evaluation_mask(
    frame: pd.DataFrame,
    human_candidate: pd.Series,
    group_ids: pd.Series,
) -> pd.Series:
    """Keep only human groups that can move intact into evaluation."""
    fully_eligible = human_candidate.groupby(group_ids).transform("all")
    one_source = (
        frame["_source_corpus"]
        .groupby(group_ids)
        .transform("nunique")
        .eq(1)
    )
    return human_candidate & fully_eligible & one_source


def block_candidate_groups(
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    indexes: set[int],
) -> set[int]:
    if not indexes:
        return set()
    blocked_groups = set(group_ids.loc[list(indexes)].astype(int))
    blocked = set(
        candidate_mask.index[
            candidate_mask & group_ids.isin(blocked_groups)
        ].astype(int)
    )
    candidate_mask.loc[list(blocked)] = False
    return blocked


def block_candidate_neighborhood(
    frame: pd.DataFrame,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    seeds: set[int],
    *,
    ngram_threshold: float,
    ngram_indexes: SplitNgramIndexes,
    protected_indexes: pd.Index | None = None,
) -> set[int]:
    """Block the complete connected n-gram neighborhood of seed rows."""
    blocked = block_candidate_groups(candidate_mask, group_ids, seeds)
    frontier = blocked
    protected = set() if protected_indexes is None else set(protected_indexes)
    while frontier:
        remaining = frame[
            candidate_mask & ~frame.index.isin(protected)
        ]
        neighbors = near_candidate_conflicts(
            frame.loc[list(frontier)],
            remaining,
            ngram_threshold=ngram_threshold,
            target_by_language=True,
            include_one_edit=False,
            ngram_indexes=ngram_indexes,
        )
        frontier = block_candidate_groups(
            candidate_mask,
            group_ids,
            neighbors,
        )
        blocked.update(frontier)
    return blocked


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
    blocked = block_candidate_neighborhood(
        frame,
        candidate_mask,
        group_ids,
        set(conflicts),
        ngram_threshold=ngram_threshold,
        ngram_indexes=ngram_indexes,
        protected_indexes=validate.index,
    )
    blocked_groups = set(group_ids.loc[list(blocked)].astype(int))
    for group_id in blocked_groups:
        if assignments.get(group_id) == "test":
            assignments.pop(group_id)
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
        target_by_language=True,
        include_one_edit=include_one_edit,
        ngram_indexes=ngram_indexes,
    )
    conflict_groups = {
        int(group_id)
        for group_id in group_ids.loc[list(conflicts)]
    }
    blocked = block_candidate_neighborhood(
        frame,
        candidate_mask,
        group_ids,
        set(conflicts),
        ngram_threshold=ngram_threshold,
        ngram_indexes=ngram_indexes,
    )
    blocked_groups = set(group_ids.loc[list(blocked)].astype(int))
    for group_id in blocked_groups:
        assignments.pop(group_id, None)
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
                ("target", "_target_key", True),
                ("pair", "_pair_key", True),
            ),
        ),
        (
            "skeleton",
            (
                ("formosan", "_formosan_skeleton", True),
                ("target", "_target_skeleton", True),
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
