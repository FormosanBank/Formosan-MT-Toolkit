"""Independent character n-gram checks for corpus validation."""

from __future__ import annotations

import math
import unicodedata
from collections import Counter

import pandas as pd

NGRAM_ORDERS = (3, 4, 5)


def _character_ngrams(value: str) -> frozenset[str]:
    if not value:
        return frozenset()
    if len(value) < min(NGRAM_ORDERS):
        return frozenset({value})
    return frozenset(
        value[position : position + order]
        for order in NGRAM_ORDERS
        if order <= len(value)
        for position in range(len(value) - order + 1)
    )


def _prefix(grams: frozenset[int], threshold: float) -> tuple[int, ...]:
    length = len(grams) - math.ceil(threshold * len(grams)) + 1
    return tuple(sorted(grams)[: max(1, length)])


class ValidationNgramIndex:
    """Exact candidate-side PPJoin lookup for independent validation."""

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
        self.languages = frame["lang_code"]
        self.values = frame[column]

    def conflicts(
        self,
        reference: pd.Index,
        candidates: pd.Index,
        *,
        same_language: bool = False,
    ) -> set[int]:
        """Return candidate rows with exact character n-gram exposure."""
        if reference.empty or candidates.empty:
            return set()
        candidate_indexes = {int(index) for index in candidates}
        candidate_raw: dict[int, frozenset[str]] = {}
        frequencies: Counter[str] = Counter()
        for index in candidate_indexes:
            grams = _character_ngrams(str(self.values.at[index]))
            if grams:
                candidate_raw[index] = grams
                frequencies.update(grams)
        if not frequencies:
            return set()
        gram_ids = {
            gram: identifier
            for identifier, gram in enumerate(
                sorted(frequencies, key=lambda item: (frequencies[item], item))
            )
        }

        features: dict[int, frozenset[int]] = {}
        postings: dict[str, dict[int, list[int]]] = {}
        for index, raw_grams in candidate_raw.items():
            grams = frozenset(gram_ids[gram] for gram in raw_grams)
            group = (
                str(self.languages.at[index])
                if self.by_language
                else "_global"
            )
            features[index] = grams
            group_postings = postings.setdefault(group, {})
            for gram in _prefix(grams, self.threshold):
                group_postings.setdefault(gram, []).append(index)

        conflicts: set[int] = set()
        threshold = self.threshold
        for raw_index in reference:
            index = int(raw_index)
            raw_grams = _character_ngrams(str(self.values.at[index]))
            if not raw_grams:
                continue
            known_grams = frozenset(
                gram_ids[gram]
                for gram in raw_grams
                if gram in gram_ids
            )
            if not known_grams:
                continue
            language = str(self.languages.at[index])
            group = language if self.by_language else "_global"
            group_postings = postings.get(group, {})
            prefix_length = (
                len(raw_grams)
                - math.ceil(threshold * len(raw_grams))
                + 1
            )
            seen: set[int] = set()
            for gram in sorted(known_grams)[: max(1, prefix_length)]:
                for candidate in group_postings.get(gram, ()):
                    if (
                        candidate in seen
                        or (
                            same_language
                            and not self.by_language
                            and str(self.languages.at[candidate]) != language
                        )
                    ):
                        continue
                    seen.add(candidate)
                    candidate_grams = features[candidate]
                    if not (
                        threshold * len(raw_grams)
                        <= len(candidate_grams)
                        <= len(raw_grams) / threshold
                    ):
                        continue
                    intersection = len(known_grams & candidate_grams)
                    union = (
                        len(raw_grams)
                        + len(candidate_grams)
                        - intersection
                    )
                    if union and intersection / union >= threshold:
                        conflicts.add(candidate)
        return conflicts


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
    output["_formosan_task_key"] = output["lang_code"].astype(str) + "\u241f" + output["_formosan_key"]
    output["_target_task_key"] = output["lang_code"].astype(str) + "\u241f" + output["_target_key"]
    output["_formosan_task_skeleton"] = (
        output["lang_code"].astype(str) + "\u241f" + output["_formosan_skeleton"]
    )
    output["_target_task_skeleton"] = (
        output["lang_code"].astype(str) + "\u241f" + output["_target_skeleton"]
    )
    output["_pair_skeleton"] = (
        output["lang_code"].astype(str)
        + "\u241f"
        + output["_formosan_skeleton"]
        + "\u241f"
        + output["_target_skeleton"]
    )
    output["_document_key"] = output["lang_code"].astype(str) + "\u241f" + output["source"].astype(str)
    return output


def overlap_count(
    left: pd.DataFrame,
    right: pd.DataFrame,
    column: str,
) -> int:
    return len(set(left[column].astype(str)) & set(right[column].astype(str)))


def deletion_keys(value: str) -> set[str]:
    return {value[:position] + value[position + 1 :] for position in range(len(value))}


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
    formosan_key = "_formosan_task_key" if formosan_by_language else "_formosan_key"
    target_key = "_target_task_key" if target_by_language else "_target_key"
    formosan_skeleton = (
        "_formosan_task_skeleton" if formosan_by_language else "_formosan_skeleton"
    )
    target_skeleton = "_target_task_skeleton" if target_by_language else "_target_skeleton"
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
        "formosan": len(formosan_ngram_index.conflicts(left.index, right.index)),
        "target": len(
            target_ngram_index.conflicts(
                left.index,
                right.index,
                same_language=target_by_language,
            )
        ),
    }
    document_overlap = overlap_count(left, right, "_document_key")
    return {
        "exact_overlap": exact,
        "skeleton_overlap": skeleton_overlap,
        "one_edit_conflicting_rows": one_edit,
        "character_ngram_conflicting_rows": character_ngram,
        "document_overlap": document_overlap,
        "formosan_by_language": formosan_by_language,
        "target_by_language": target_by_language,
        "ok": not any(
            value
            for family in (exact, skeleton_overlap, one_edit, character_ngram)
            for value in family.values()
        ),
        "document_disjoint": document_overlap == 0,
    }
