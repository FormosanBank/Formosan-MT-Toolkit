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
    """Exact PPJoin-style lookup kept independent from split construction."""

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
        frequencies: Counter[str] = Counter()
        for value in values:
            frequencies.update(_character_ngrams(value))
        gram_ids = {
            gram: identifier
            for identifier, gram in enumerate(
                sorted(frequencies, key=lambda item: (frequencies[item], item))
            )
        }

        for raw_index, value in values.items():
            raw_grams = _character_ngrams(value)
            if not raw_grams:
                continue
            index = int(raw_index)
            grams = frozenset(gram_ids[gram] for gram in raw_grams)
            prefix = _prefix(grams, threshold)
            group = self.languages.at[raw_index] if by_language else "_global"
            postings = self.postings.setdefault(group, {})
            self.features[index] = grams
            self.prefixes[index] = prefix
            for gram in prefix:
                postings.setdefault(gram, []).append(index)

    def conflicts(
        self,
        reference: pd.Index,
        candidates: pd.Index,
        *,
        same_language: bool = False,
    ) -> set[int]:
        if reference.empty or candidates.empty:
            return set()
        references = {int(index) for index in reference}
        candidate_indexes = {int(index) for index in candidates}
        references_by_language: dict[str, set[int]] = {}
        if same_language and not self.by_language:
            for index in references:
                references_by_language.setdefault(
                    self.languages.at[index],
                    set(),
                ).add(index)

        if len(references) < len(candidate_indexes):
            return self._search_from_references(
                references,
                candidate_indexes,
                same_language=same_language,
            )

        conflicts: set[int] = set()
        for index in candidate_indexes:
            grams = self.features.get(index)
            if grams is None:
                continue
            language = self.languages.at[index]
            active_references = (
                references_by_language.get(language, set())
                if same_language and not self.by_language
                else references
            )
            group = language if self.by_language else "_global"
            if self._has_match(index, grams, active_references, self.postings.get(group, {})):
                conflicts.add(index)
        return conflicts

    def _search_from_references(
        self,
        references: set[int],
        candidates: set[int],
        *,
        same_language: bool,
    ) -> set[int]:
        conflicts: set[int] = set()
        seen: set[tuple[int, int]] = set()
        for reference in references:
            reference_grams = self.features.get(reference)
            if reference_grams is None:
                continue
            language = self.languages.at[reference]
            group = language if self.by_language else "_global"
            for gram in self.prefixes[reference]:
                for candidate in self.postings.get(group, {}).get(gram, ()):
                    pair = (reference, candidate)
                    if (
                        pair in seen
                        or candidate not in candidates
                        or (same_language and self.languages.at[candidate] != language)
                    ):
                        continue
                    seen.add(pair)
                    if self._similar(reference_grams, self.features[candidate]):
                        conflicts.add(candidate)
        return conflicts

    def _has_match(
        self,
        index: int,
        grams: frozenset[int],
        references: set[int],
        postings: dict[int, list[int]],
    ) -> bool:
        seen: set[int] = set()
        for gram in self.prefixes[index]:
            for other in postings.get(gram, ()):
                if other in seen or other not in references:
                    continue
                seen.add(other)
                if self._similar(grams, self.features[other]):
                    return True
        return False

    def _similar(self, left: frozenset[int], right: frozenset[int]) -> bool:
        threshold = self.threshold
        if not threshold * len(left) <= len(right) <= len(left) / threshold:
            return False
        intersection = len(left & right)
        union = len(left) + len(right) - intersection
        return bool(union and intersection / union >= threshold)


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
