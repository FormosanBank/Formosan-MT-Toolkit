from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

from build_experiment_splits import build_hard_split  # noqa: E402
from mt_standardization import DEFAULT_PROFILE_PATH, load_profile, profile_sha256  # noqa: E402
from split_allocation import candidate_groups, selection_priorities  # noqa: E402

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

    def test_priorities_depend_only_on_seed_and_row_id(self) -> None:
        row_ids = pd.Series([f"row-{index}" for index in range(50)])
        priorities = selection_priorities(row_ids, 42)
        shuffled = row_ids.sample(frac=1.0, random_state=7)
        reordered = selection_priorities(shuffled, 42)
        self.assertEqual(
            dict(zip(row_ids, priorities, strict=True)),
            dict(zip(shuffled, reordered, strict=True)),
        )
        self.assertEqual(priorities.nunique(), 50)
        self.assertFalse(priorities.equals(selection_priorities(row_ids, 43)))

    def test_blocking_candidates_keeps_the_remaining_order(self) -> None:
        # The similarity gates remove candidates between passes. A stable order
        # means the rest of the sample does not reshuffle when that happens.
        frame = pd.DataFrame({"row_id": [f"row-{index}" for index in range(40)]})
        frame["_selection_priority"] = selection_priorities(frame["row_id"], 42)
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


if __name__ == "__main__":
    unittest.main()
