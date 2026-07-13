from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local/scripts/pivot"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

from build_experiment_splits import one_edit_conflicts, split_targets  # noqa: E402
from mt_metrics import score_translations  # noqa: E402
from pivot import discover_api_key_envs, parse_api_key_envs  # noqa: E402
from train_directional_nllb import metric_improved  # noqa: E402


class DeepLKeyDiscoveryTests(unittest.TestCase):
    def test_discovers_numbered_keys_in_numeric_order(self) -> None:
        environ = {
            "DEEPL_API_KEY_10": "ten",
            "DEEPL_API_KEY": "one",
            "DEEPL_API_KEY_2": "two",
            "DEEPL_API_KEY_3": "",
            "DEEPL_API_KEY_BAD": "ignored",
        }
        self.assertEqual(
            discover_api_key_envs(environ),
            ["DEEPL_API_KEY", "DEEPL_API_KEY_2", "DEEPL_API_KEY_10"],
        )

    def test_explicit_key_list_remains_supported(self) -> None:
        self.assertEqual(parse_api_key_envs("SECOND,FIRST,SECOND"), ["SECOND", "FIRST"])


class LeakageTests(unittest.TestCase):
    def test_one_character_variants_conflict_with_evaluation(self) -> None:
        evaluation = pd.DataFrame(
            {"lang_code": ["ami"], "text": ["malikoda"]}, index=[100]
        )
        training = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "ami", "bnn"],
                "text": ["malikod", "malikooda", "malikada", "faraway", "malikada"],
            },
            index=[1, 2, 3, 4, 5],
        )
        self.assertEqual(one_edit_conflicts(training, evaluation, "text"), {1, 2, 3})

    def test_split_targets_use_every_final_row_as_denominator(self) -> None:
        self.assertEqual(
            split_targets(
                rows_total=100_000,
                eligible_total=20_000,
                test_ratio=0.075,
                val_ratio=0.025,
                min_test_rows=500,
                min_validate_rows=150,
            ),
            (7_500, 2_500),
        )


class TrainingMetricTests(unittest.TestCase):
    def test_perfect_generation_metrics_and_diagnostics(self) -> None:
        metrics = score_translations(["hello world", "talima"], ["hello world", "talima"])
        self.assertAlmostEqual(metrics["chrF2"], 100.0)
        self.assertAlmostEqual(metrics["exact_match_rate"], 1.0)
        self.assertAlmostEqual(metrics["empty_output_rate"], 0.0)
        self.assertAlmostEqual(metrics["character_length_ratio"], 1.0)

    def test_metric_direction_and_minimum_delta(self) -> None:
        self.assertTrue(metric_improved(20.1, 20.0, "chrF2", 0.05))
        self.assertFalse(metric_improved(20.01, 20.0, "chrF2", 0.05))
        self.assertTrue(metric_improved(1.8, 2.0, "mean_token_loss", 0.05))
        self.assertFalse(metric_improved(1.98, 2.0, "mean_token_loss", 0.05))
        self.assertTrue(metric_improved(49.0, 50.0, "TER", 0.05))


if __name__ == "__main__":
    unittest.main()
