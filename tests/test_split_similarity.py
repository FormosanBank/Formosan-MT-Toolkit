from __future__ import annotations

import random
import unittest

import pandas as pd
from split_similarity import (
    NgramSimilarityIndex,
    block_evaluation_conflicts_with_training,
    exclude_test_conflicts_with_validation,
    ngram_candidate_conflicts,
    one_edit_conflicts,
)
from validation_similarity import ValidationNgramIndex


class SplitSimilarityTests(unittest.TestCase):
    def test_one_character_variants_conflict_with_evaluation(self) -> None:
        evaluation = pd.DataFrame({"lang_code": ["ami"], "text": ["malikoda"]}, index=[100])
        training = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "ami", "bnn"],
                "text": ["malikod", "malikooda", "malikada", "faraway", "malikada"],
            },
            index=[1, 2, 3, 4, 5],
        )
        self.assertEqual(one_edit_conflicts(training, evaluation, "text"), {1, 2, 3})

    def test_one_edit_self_comparison_only_marks_real_neighbors(self) -> None:
        rows = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami"],
                "text": ["malikoda", "malikoda", "unrelated"],
            },
            index=[1, 2, 3],
        )
        self.assertEqual(
            one_edit_conflicts(
                rows,
                rows,
                "text",
                ignore_same_index=True,
            ),
            {1, 2},
        )

    def test_streaming_ngram_join_finds_conflicts_with_either_side_smaller(self) -> None:
        base = "abcdefghijklmnopqrstuvwxyz" * 4
        near = base[:50] + "X" + base[51:]
        far = "zyxwvutsrqponmlkjihgfedcba" * 4
        reference = pd.DataFrame(
            {"lang_code": ["ami"], "text": [base]},
            index=[100],
        )
        candidates = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami"],
                "text": [near, far, f"{far} extra"],
            },
            index=[1, 2, 3],
        )
        self.assertEqual(
            ngram_candidate_conflicts(
                reference,
                candidates,
                "text",
                by_language=False,
                threshold=0.82,
            ),
            {1},
        )
        self.assertEqual(
            ngram_candidate_conflicts(
                candidates,
                reference,
                "text",
                by_language=False,
                threshold=0.82,
            ),
            {100},
        )

    def test_cached_ngram_index_matches_streaming_join(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "bnn", "bnn"],
                "text": [
                    "abcdefghijklmnopqrstuvwxyz",
                    "abcdefghijklmnopqrstuvwxzz",
                    "completely unrelated value",
                    "abcdefghijklmnopqrstuvwxyz",
                    "another unrelated sentence",
                ],
            },
            index=[10, 11, 12, 13, 14],
        )
        reference = frame.loc[[10, 14]]
        candidates = frame.loc[[11, 12, 13]]
        for by_language, expected in ((False, {11, 13}), (True, {11})):
            index = NgramSimilarityIndex(
                frame,
                "text",
                by_language=by_language,
                threshold=0.82,
            )
            self.assertEqual(
                index.conflicts(reference.index, candidates.index),
                expected,
            )

    def test_independent_validation_index_matches_split_index(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "bnn", "bnn"],
                "text": [
                    "abcdefghijklmnopqrstuvwxyz",
                    "abcdefghijklmnopqrstuvwxzz",
                    "completely unrelated value",
                    "abcdefghijklmnopqrstuvwxyz",
                    "another unrelated sentence",
                ],
            },
            index=[10, 11, 12, 13, 14],
        )
        reference = pd.Index([10, 14])
        candidates = pd.Index([11, 12, 13])
        for by_language in (False, True):
            split_index = NgramSimilarityIndex(
                frame,
                "text",
                by_language=by_language,
                threshold=0.82,
            )
            validation_index = ValidationNgramIndex(
                frame,
                "text",
                by_language=by_language,
                threshold=0.82,
            )
            self.assertEqual(
                validation_index.conflicts(reference, candidates),
                split_index.conflicts(reference, candidates),
            )

    def test_candidate_validation_index_matches_split_index_randomized(self) -> None:
        rng = random.Random(917)
        alphabet = "abcdefghijklmnopqrstuvwxyz "
        for sample in range(10):
            values: list[str] = []
            languages: list[str] = []
            for _ in range(50):
                value = "".join(rng.choice(alphabet) for _ in range(rng.randint(2, 50)))
                if values and rng.random() < 0.35:
                    value = values[rng.randrange(len(values))]
                    if value and rng.random() < 0.7:
                        offset = rng.randrange(len(value))
                        value = value[:offset] + rng.choice(alphabet) + value[offset + 1 :]
                values.append(value)
                languages.append(rng.choice(["ami", "bnn", "tay"]))
            frame = pd.DataFrame(
                {"lang_code": languages, "text": values},
                index=range(sample * 100, sample * 100 + len(values)),
            )
            indexes = list(frame.index)
            rng.shuffle(indexes)
            reference = pd.Index(indexes[:30])
            candidates = pd.Index(indexes[30:])
            for threshold in (0.70, 0.82, 0.95):
                for by_language in (False, True):
                    split_index = NgramSimilarityIndex(
                        frame,
                        "text",
                        by_language=by_language,
                        threshold=threshold,
                    )
                    validation_index = ValidationNgramIndex(
                        frame,
                        "text",
                        by_language=by_language,
                        threshold=threshold,
                    )
                    self.assertEqual(
                        validation_index.conflicts(reference, candidates),
                        split_index.conflicts(reference, candidates),
                    )

    def test_ngram_index_matches_tame_high_exposure_boundary(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ckv", "ckv", "ssf", "ssf"],
                "text": [
                    "sessenan ti iku tu mai tu qunqunian temanan ti iku.",
                    "sessenan iku tu mai tu qunqunian temanan ti iku.",
                    "thithu palalawan sa suma a pakikalhian sa suma sa lalawa.",
                    "thithu palalawan sa suma a pakikalhian sa lalawa.",
                ],
            }
        )
        index = NgramSimilarityIndex(
            frame,
            "text",
            by_language=True,
            threshold=0.95,
        )

        self.assertEqual(index.conflicts(pd.Index([1, 3]), pd.Index([0, 2])), {0, 2})

    def test_split_conflicts_block_the_full_similarity_neighborhood(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami"] * 4,
                "_formosan_skeleton": [
                    "abcdefghijklmnopqrstuvwx",
                    "abcdefghijklmnopqrstuvwy",
                    "abcdefghijklmnopqrstuvwz",
                    "completelydifferentstring",
                ],
                "_target_skeleton": [
                    "thevalidationtranslationx",
                    "thevalidationtranslationy",
                    "thevalidationtranslationz",
                    "unrelatedtargettranslation",
                ],
            }
        )
        group_ids = pd.Series(range(len(frame)), index=frame.index, dtype="int64")

        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {0: "validate", 1: "test"}
        blocked: set[int] = set()
        result = exclude_test_conflicts_with_validation(
            frame,
            pd.Series(["validate", "test", "train", "train"]),
            group_ids,
            candidate_mask,
            assignments,
            blocked,
            ngram_threshold=0.82,
            include_one_edit=False,
        )
        self.assertGreater(result["blocked_candidate_rows"], 1)
        self.assertEqual(blocked, {1, 2})
        self.assertNotIn(1, assignments)
        self.assertTrue(bool(candidate_mask.iloc[3]))

        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {1: "test"}
        blocked = set()
        result = block_evaluation_conflicts_with_training(
            frame,
            pd.Series(["train", "test", "train", "train"]),
            group_ids,
            candidate_mask,
            assignments,
            blocked,
            ngram_threshold=0.82,
            include_one_edit=False,
        )
        self.assertGreater(result["blocked_candidate_rows"], 1)
        self.assertEqual(blocked, {0, 1, 2})
        self.assertNotIn(1, assignments)
        self.assertTrue(bool(candidate_mask.iloc[3]))
