from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

from compare_predictions import format_report, paired_predictions  # noqa: E402
from evaluate_directional import group_scores, score_predictions  # noqa: E402
from mt_metrics import (  # noqa: E402
    METRIC_NAMES,
    bleu_tokenizer_for,
    bootstrap_confidence_intervals,
    macro_average,
    paired_bootstrap_comparison,
    score_translations,
    sentence_statistics,
)

WORDS = "ko a i no to ira kako kiso cingra mafana maolah tayni tayra loma wawa ina mama".split()


def synthetic_corpus(size: int, error_rate: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Return hypotheses, references, and languages with a known error rate."""
    rng = random.Random(seed)
    references = [" ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 12))) + "." for _ in range(size)]
    hypotheses = [
        " ".join(word if rng.random() > error_rate else rng.choice(WORDS) for word in reference.split())
        for reference in references
    ]
    languages = [("ami", "tay", "bnn")[index % 3] for index in range(size)]
    return hypotheses, references, languages


class SentenceStatisticsTests(unittest.TestCase):
    def test_scores_equal_direct_corpus_scoring(self) -> None:
        hypotheses, references, _ = synthetic_corpus(120, 0.4, seed=1)
        metrics = score_translations(hypotheses, references)
        self.assertAlmostEqual(
            metrics["BLEU"],
            BLEU(tokenize="13a", effective_order=True).corpus_score(hypotheses, [references]).score,
        )
        self.assertAlmostEqual(metrics["chrF2"], CHRF().corpus_score(hypotheses, [references]).score)
        self.assertAlmostEqual(metrics["TER"], TER(normalized=True).corpus_score(hypotheses, [references]).score)

    def test_chinese_scores_use_character_tokenization(self) -> None:
        references = ["我今天去學校上課。", "他們在山上打獵", "這是用來測試的句子"]
        hypotheses = ["我今天去學校。", "他們在山上打獵", "這是測試的句子"]
        metrics = score_translations(hypotheses, references, bleu_tokenize="zh")
        self.assertAlmostEqual(
            metrics["BLEU"],
            BLEU(tokenize="zh", effective_order=True).corpus_score(hypotheses, [references]).score,
        )
        self.assertAlmostEqual(
            metrics["TER"],
            TER(normalized=True, asian_support=True).corpus_score(hypotheses, [references]).score,
        )
        self.assertEqual(bleu_tokenizer_for("f2zh"), "zh")
        self.assertEqual(bleu_tokenizer_for("zh2f"), "13a")

    def test_subset_scores_equal_scoring_the_subset_alone(self) -> None:
        hypotheses, references, languages = synthetic_corpus(150, 0.4, seed=2)
        statistics = sentence_statistics(hypotheses, references)
        rows = np.flatnonzero(np.array(languages) == "tay")
        alone = score_translations([hypotheses[row] for row in rows], [references[row] for row in rows])
        subset = statistics.score(rows)
        for name in METRIC_NAMES:
            self.assertAlmostEqual(subset[name], alone[name])

    def test_diagnostics_are_reported(self) -> None:
        metrics = score_translations(
            ["hello world", "talima", ""],
            ["hello world", "talima", "missing"],
            sources=["different", "  TALIMA  ", "source"],
        )
        self.assertAlmostEqual(metrics["exact_match_rate"], 2 / 3)
        self.assertAlmostEqual(metrics["empty_output_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["source_copy_rate"], 1 / 3)
        self.assertAlmostEqual(metrics["character_length_ratio"], 17 / 24)
        self.assertEqual(set(metrics["signatures"]), set(METRIC_NAMES))

    def test_empty_and_mismatched_inputs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            score_translations([], [])
        with self.assertRaises(ValueError):
            score_translations(["a"], ["a", "b"])
        with self.assertRaises(ValueError):
            sentence_statistics(["a"], ["a"]).score(np.array([], dtype=int))

    def test_group_scores_use_row_positions_not_index_labels(self) -> None:
        hypotheses, references, languages = synthetic_corpus(30, 0.4, seed=3)
        predictions = pd.DataFrame(
            {"lang_code": languages},
            index=range(1000, 1000 + 7 * len(languages), 7),
        )
        groups = group_scores(predictions, "lang_code", sentence_statistics(hypotheses, references))
        self.assertEqual({name: group["samples"] for name, group in groups.items()}, {"ami": 10, "bnn": 10, "tay": 10})
        alone = score_translations(hypotheses[1::3], references[1::3])
        self.assertAlmostEqual(groups["tay"]["BLEU"], alone["BLEU"])


    def test_evaluation_report_sections_share_one_scoring_pass(self) -> None:
        hypotheses, references, languages = synthetic_corpus(60, 0.4, seed=9)
        predictions = pd.DataFrame(
            {
                "lang_code": languages,
                "source_corpus": ["ePark", "NTU"] * 30,
                "dialect": "Coastal",
                "length_bin": ["004_008", "009_016", "017_032"] * 20,
                "src": references,
                "ref": references,
                "hyp_default": hypotheses,
                "hyp_oracle": references,
            },
            index=range(500, 560),
        )
        report = score_predictions(
            predictions,
            ["default", "oracle"],
            lowercase=False,
            bleu_tokenize="13a",
            bootstrap_samples=50,
            bootstrap_seed=42,
        )
        direct = score_translations(hypotheses, references, sources=references)
        self.assertEqual(report["global"], {"samples": 60, **direct})
        self.assertEqual(report["metadata_modes"]["default"], report["global"])
        self.assertAlmostEqual(report["metadata_modes"]["oracle"]["exact_match_rate"], 1.0)
        self.assertEqual(report["macro_by_language"]["groups"], 3)
        self.assertEqual(set(report["by_language"]), {"ami", "bnn", "tay"})
        self.assertEqual(set(report["by_source_corpus"]), {"ePark", "NTU"})
        self.assertEqual(sum(group["samples"] for group in report["by_length_bin"].values()), 60)
        confidence = report["bootstrap_95_ci"]
        self.assertEqual(
            (confidence["status"], confidence["stratification"], confidence["samples"]),
            ("complete", "language", 50),
        )
        self.assertEqual(set(confidence["by_stratum"]), {"ami", "bnn", "tay"})

        skipped = score_predictions(
            predictions,
            ["default"],
            lowercase=False,
            bleu_tokenize="13a",
            bootstrap_samples=0,
            bootstrap_seed=42,
        )
        self.assertEqual(skipped["bootstrap_95_ci"]["status"], "skipped")


class BootstrapTests(unittest.TestCase):
    def test_intervals_are_deterministic_and_contain_the_estimate(self) -> None:
        hypotheses, references, languages = synthetic_corpus(300, 0.4, seed=4)
        statistics = sentence_statistics(hypotheses, references)
        first = bootstrap_confidence_intervals(statistics, strata=languages, samples=200, seed=7)
        self.assertEqual(first, bootstrap_confidence_intervals(statistics, strata=languages, samples=200, seed=7))
        self.assertNotEqual(first, bootstrap_confidence_intervals(statistics, strata=languages, samples=200, seed=8))
        self.assertEqual(list(first["by_stratum"]), ["ami", "bnn", "tay"])
        pooled = statistics.score()
        macro = macro_average(statistics, languages)
        self.assertEqual(macro["groups"], 3)
        for name in METRIC_NAMES:
            for interval, estimate in ((first["metrics"][name], pooled[name]), (first["macro"][name], macro[name])):
                self.assertLess(interval["lower"], estimate)
                self.assertLess(estimate, interval["upper"])
                self.assertLessEqual(interval["lower"], interval["median"])
                self.assertLessEqual(interval["median"], interval["upper"])

    def test_resampling_keeps_every_stratum_at_its_own_size(self) -> None:
        # A stratum with one sentence can only be drawn as itself, so its
        # interval collapses. Unstratified resampling would not guarantee that.
        statistics = sentence_statistics(
            ["one two three four", "a b c d", "e f g h"],
            ["one two three four", "a b x d", "e f g z"],
        )
        intervals = bootstrap_confidence_intervals(statistics, strata=["ami", "bnn", "bnn"], samples=50, seed=1)
        self.assertAlmostEqual(intervals["by_stratum"]["ami"]["BLEU"]["lower"], 100.0)
        self.assertAlmostEqual(intervals["by_stratum"]["ami"]["BLEU"]["upper"], 100.0)
        self.assertLess(intervals["by_stratum"]["bnn"]["BLEU"]["lower"], intervals["by_stratum"]["bnn"]["BLEU"]["upper"])

    def test_paired_comparison_separates_real_gains_from_noise(self) -> None:
        _, references, languages = synthetic_corpus(400, 0.0, seed=5)
        weak, _, _ = synthetic_corpus(400, 0.5, seed=5)
        strong, _, _ = synthetic_corpus(400, 0.2, seed=5)
        baseline = sentence_statistics(weak, references)
        better = paired_bootstrap_comparison(
            baseline,
            sentence_statistics(strong, references),
            strata=languages,
            samples=300,
            seed=3,
        )
        self.assertEqual(better["sentences"], 400)
        for view in (better["metrics"], better["macro"], better["by_stratum"]["tay"]):
            self.assertGreater(view["BLEU"]["delta"], 0)
            self.assertGreater(view["BLEU"]["delta_lower"], 0)
            self.assertLess(view["TER"]["delta_upper"], 0)
            self.assertLess(view["BLEU"]["p_value"], 0.01)

        same = paired_bootstrap_comparison(baseline, baseline, strata=languages, samples=50, seed=3)
        self.assertEqual(same["metrics"]["chrF2"]["delta"], 0.0)
        self.assertEqual(same["metrics"]["chrF2"]["p_value"], 1.0)

    def test_paired_comparison_does_not_call_resampling_noise_significant(self) -> None:
        # Two systems with the same error rate differ only by chance. The
        # seeds are fixed, so this checks one ordinary draw, not calibration.
        _, references, languages = synthetic_corpus(400, 0.0, seed=6)
        noise = [random.Random(seed) for seed in (11, 12)]
        first, second = (
            [
                " ".join(word if rng.random() > 0.4 else rng.choice(WORDS) for word in reference.split())
                for reference in references
            ]
            for rng in noise
        )
        report = paired_bootstrap_comparison(
            sentence_statistics(first, references),
            sentence_statistics(second, references),
            strata=languages,
            samples=300,
            seed=3,
        )
        self.assertGreater(report["metrics"]["BLEU"]["p_value"], 0.05)
        self.assertLess(report["metrics"]["BLEU"]["delta_lower"], 0)
        self.assertGreater(report["metrics"]["BLEU"]["delta_upper"], 0)

    def test_paired_comparison_rejects_incomparable_systems(self) -> None:
        hypotheses, references, _ = synthetic_corpus(20, 0.3, seed=7)
        statistics = sentence_statistics(hypotheses, references)
        with self.assertRaises(ValueError):
            paired_bootstrap_comparison(statistics, sentence_statistics(hypotheses[:10], references[:10]))
        with self.assertRaises(ValueError):
            paired_bootstrap_comparison(statistics, sentence_statistics(hypotheses, references, lowercase=True))


class PredictionComparisonTests(unittest.TestCase):
    def write_predictions(self, directory: Path, name: str, references: list[str], shuffle: bool = False) -> Path:
        frame = pd.DataFrame(
            {
                "row_id": [f"row-{index}" for index in range(len(references))],
                "lang_code": ["ami", "tay"] * (len(references) // 2),
                "direction": "f2en",
                "ref": references,
                "hyp_default": references,
            }
        )
        if shuffle:
            frame = frame.iloc[::-1]
        path = directory / name
        frame.to_csv(path, index=False)
        return path

    def test_predictions_are_aligned_by_row_id(self) -> None:
        references = ["one two", "three four", "five six", "seven eight"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            baseline, system = paired_predictions(
                self.write_predictions(directory, "a.csv", references),
                self.write_predictions(directory, "b.csv", references, shuffle=True),
            )
        self.assertEqual(baseline["row_id"].tolist(), system["row_id"].tolist())
        self.assertEqual(baseline["ref"].tolist(), system["ref"].tolist())

    def test_different_test_sets_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            with self.assertRaises(SystemExit):
                paired_predictions(
                    self.write_predictions(directory, "a.csv", ["one two", "three four"]),
                    self.write_predictions(directory, "b.csv", ["one two", "changed reference"]),
                )

    def test_report_marks_significant_differences(self) -> None:
        _, references, languages = synthetic_corpus(200, 0.0, seed=8)
        weak, _, _ = synthetic_corpus(200, 0.5, seed=8)
        report = paired_bootstrap_comparison(
            sentence_statistics(weak, references),
            sentence_statistics(references, references),
            strata=languages,
            samples=100,
            seed=1,
        )
        table = format_report(report)
        self.assertIn("| All test rows |", table)
        self.assertIn("| Macro average over languages |", table)
        self.assertIn("| `tay` |", table)
        self.assertIn("*", table.splitlines()[2])


if __name__ == "__main__":
    unittest.main()
