from __future__ import annotations

import hashlib
import unittest

import pandas as pd
from build_experiment_splits import build_hard_split
from mt_standardization import DEFAULT_PROFILE_PATH, load_profile, profile_sha256
from split_allocation import candidate_groups, selection_priorities, source_stratum_targets
from validation_policy import validate_test_length_balance

SPLIT_OPTIONS = {
    "target_col": "english_sentence",
    "test_ratio": 0.2,
    "val_ratio": 0.1,
    "min_formosan_tokens": 2,
    "min_target_tokens": 2,
    "min_combined_tokens": 6,
    "min_punctuated_combined_tokens": 5,
    "min_test_rows": 0,
    "min_validate_rows": 0,
    "ngram_threshold": 0.95,
    "registry_in": None,
}


def unique_words(key: str, count: int) -> str:
    """Return ``count`` distinct pseudo-words so no two sentences are near duplicates."""
    return " ".join(hashlib.sha256(f"{key}:{index}".encode()).hexdigest()[:9] for index in range(count))


def corpus_with_varied_lengths(rows: int = 1200) -> pd.DataFrame:
    """Eligible human sentences in two sources whose lengths run from 3 to 42 words."""
    profile = load_profile(DEFAULT_PROFILE_PATH)
    records = []
    for index in range(rows):
        words = 3 + index % 40
        formosan = unique_words(f"formosan-{index}", words)
        source = "SourceA" if index % 3 else "SourceB"
        records.append(
            {
                "row_id": f"row-{index:05d}",
                "lang_code": "ami",
                "formosan_sentence": formosan,
                "english_sentence": unique_words(f"english-{index}", words),
                "source": f"FormosanBank/Corpora/{source}/XML/text.xml",
                "dialect": "Test",
                "row_type": "sentence",
                "pivot_origin": "original",
                "quality_flags": "",
                "kindOf": "standard",
                "standard_namespace": "formosan-mt",
                "formosan_mt_standard": formosan,
                "mt_standard_sha256": hashlib.sha256(formosan.encode()).hexdigest(),
                "mt_normalization_status": "accepted",
                "mt_normalization_confidence": "unchanged",
                "mt_eval_eligible": True,
                "mt_standard_profile": profile["profile_id"],
                "mt_standard_profile_sha256": profile_sha256(DEFAULT_PROFILE_PATH),
            }
        )
    return pd.DataFrame(records)


class EvaluationSelectionTests(unittest.TestCase):
    def test_evaluation_rows_are_not_selected_by_length(self) -> None:
        # Every row here is eligible, so validation, test, and training should
        # look alike. Ranking candidates by length would put the longest tenth
        # in validation (mean about 40 words) and leave training near 17.
        output, _, _, report = build_hard_split(corpus_with_varied_lengths(), seed=42, **SPLIT_OPTIONS)
        self.assertEqual(report["split_counts"], {"train": 840, "test": 240, "validate": 120})
        self.assertEqual(report["evaluation_selection"], "seeded_uniform_within_language_and_source")
        balance = report["evaluation_length_balance"]["ami"]
        self.assertEqual(set(balance), {"train", "validate", "test"})
        self.assertLess(max(balance.values()) / min(balance.values()), 1.15)
        mean_length = output.groupby("split")["formosan_tokens"].mean()
        overall = output["formosan_tokens"].mean()
        for split in ("train", "validate", "test"):
            self.assertLess(abs(mean_length[split] - overall) / overall, 0.15, split)
        longest = output["formosan_tokens"].ge(39)
        for split in ("validate", "test"):
            share = (output["split"].eq(split) & longest).sum() / output["split"].eq(split).sum()
            self.assertLess(share, 0.25, split)

    def test_each_source_keeps_its_share_of_evaluation(self) -> None:
        output, _, _, _ = build_hard_split(corpus_with_varied_lengths(), seed=42, **SPLIT_OPTIONS)
        shares = pd.crosstab(output["source_corpus"], output["split"], normalize="index")
        for source in ("SourceA", "SourceB"):
            self.assertAlmostEqual(shares.loc[source, "test"], 0.2, delta=0.01)
            self.assertAlmostEqual(shares.loc[source, "validate"], 0.1, delta=0.01)

    def test_seed_selects_the_sample(self) -> None:
        frame = corpus_with_varied_lengths(600)

        def test_rows(seed: int) -> set[str]:
            output, _, _, _ = build_hard_split(frame.copy(), seed=seed, **SPLIT_OPTIONS)
            return set(output.loc[output["split"].eq("test"), "row_id"])

        self.assertEqual(test_rows(42), test_rows(42))
        self.assertNotEqual(test_rows(42), test_rows(43))
        self.assertEqual(len(test_rows(43)), 120)

    def test_priorities_depend_only_on_seed_and_pair(self) -> None:
        keys = pd.Series([f"ami\u241fpair {index}" for index in range(50)])
        priorities = selection_priorities(keys, 42)
        shuffled = keys.sample(frac=1.0, random_state=7)
        reordered = selection_priorities(shuffled, 42)
        self.assertEqual(
            dict(zip(keys, priorities, strict=True)),
            dict(zip(shuffled, reordered, strict=True)),
        )
        self.assertEqual(priorities.nunique(), 50)
        self.assertFalse(priorities.equals(selection_priorities(keys, 43)))

    def test_held_out_pairs_survive_new_data_and_new_provenance(self) -> None:
        # The same pairs arrive from another repository with other row IDs,
        # plus 20% new pairs. Held-out pairs should mostly stay held out.
        first = corpus_with_varied_lengths(1000)
        grown = corpus_with_varied_lengths(1200)
        grown["row_id"] = [f"other-{index:05d}" for index in range(len(grown))]
        grown["source"] = grown["source"].str.replace("FormosanBank/Corpora/", "Formosan-", regex=False)

        def held_out(frame: pd.DataFrame) -> dict[str, set[str]]:
            output, _, _, _ = build_hard_split(frame, seed=42, **SPLIT_OPTIONS)
            return {
                split: set(output.loc[output["split"].eq(split), "formosan_sentence"])
                for split in ("train", "validate", "test")
            }

        before, after = held_out(first), held_out(grown)
        evaluation_before = before["validate"] | before["test"]
        self.assertEqual(len(evaluation_before), 300)
        self.assertGreater(len(evaluation_before - after["train"]) / len(evaluation_before), 0.9)
        self.assertGreater(len(before["test"] & after["test"]) / len(before["test"]), 0.8)

    def test_blocking_candidates_keeps_the_remaining_order(self) -> None:
        # The similarity gates remove candidates between passes. A stable order
        # means the rest of the sample does not reshuffle when that happens.
        frame = pd.DataFrame({"pair": [f"ami\u241fpair {index}" for index in range(40)]})
        frame["_selection_priority"] = selection_priorities(frame["pair"], 42)
        group_ids = pd.Series(range(40), index=frame.index, dtype="int64")
        everyone = pd.Series(True, index=frame.index)
        full = [group.group_id for group in candidate_groups(frame, frame.index, everyone, group_ids, {})]
        blocked = set(full[:5]) | {full[11], full[20]}
        remaining_mask = ~frame.index.isin(blocked)
        remaining = [
            group.group_id
            for group in candidate_groups(frame, frame.index, pd.Series(remaining_mask, index=frame.index), group_ids, {})
        ]
        self.assertEqual(remaining, [group_id for group_id in full if group_id not in blocked])
        self.assertNotEqual(full, sorted(full))



class SourceMixTests(unittest.TestCase):
    def test_validation_and_test_share_one_source_mix_when_capacity_binds(self) -> None:
        # SourceB has 1,000 human pairs but only 60 eligible sentences, so it
        # cannot supply its share of evaluation. Its 60 held-out rows must
        # still divide between validation and test like every other source.
        # Apportioning validation by human pairs sent all 60 to validation.
        frame = pd.DataFrame(
            {
                "lang_code": "ami",
                "_source_corpus": ["SourceA"] * 1000 + ["SourceB"] * 1000,
                "pivot_origin": "original",
            }
        )
        eligible = pd.Series([True] * 900 + [False] * 100 + [True] * 60 + [False] * 940)
        targets, language_targets = source_stratum_targets(
            frame,
            eligible,
            test_ratio=0.2,
            val_ratio=0.1,
            min_test_rows=0,
            min_validate_rows=0,
        )
        self.assertEqual(language_targets, {"ami": (400, 200)})
        self.assertEqual(targets[("ami", "SourceA")], (360, 180))
        self.assertEqual(targets[("ami", "SourceB")], (40, 20))


class LengthBalanceGateTests(unittest.TestCase):
    @staticmethod
    def frame(validate_lengths: list[int], test_lengths: list[int]) -> tuple[pd.DataFrame, pd.Series]:
        lengths = validate_lengths + test_lengths + [10] * 50
        keyed = pd.DataFrame(
            {
                "lang_code": "ami",
                "_formosan_tokens": lengths,
                "_target_tokens": lengths,
            }
        )
        split = pd.Series(["validate"] * len(validate_lengths) + ["test"] * len(test_lengths) + ["train"] * 50)
        return keyed, split

    def test_matching_validation_and_test_lengths_pass(self) -> None:
        validate = [3 + index % 30 for index in range(210)]
        test = [3 + index % 30 for index in range(420)]
        report, failures = validate_test_length_balance(*self.frame(validate, test))
        self.assertEqual(failures, {})
        self.assertLess(abs(report["ami"]["standardized_difference"]), 0.05)

    def test_longer_validation_than_test_fails(self) -> None:
        # The old length-ranked selection put the longest rows in validation.
        validate = [25 + index % 8 for index in range(200)]
        test = [8 + index % 17 for index in range(400)]
        report, failures = validate_test_length_balance(*self.frame(validate, test))
        self.assertEqual(set(failures), {"ami"})
        self.assertGreater(report["ami"]["standardized_difference"], 1.0)
        self.assertGreater(report["ami"]["validate_mean_units"], report["ami"]["test_mean_units"])

    def test_small_samples_are_not_failed_on_chance_gaps(self) -> None:
        report, failures = validate_test_length_balance(*self.frame([12, 14, 30, 9], [10, 11, 8, 16, 12, 9, 13, 15]))
        self.assertGreater(report["ami"]["standardized_difference"], 0.2)
        self.assertEqual(failures, {})
