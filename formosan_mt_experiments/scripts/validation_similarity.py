"""Independent character n-gram checks for corpus validation."""

from __future__ import annotations

import math
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
