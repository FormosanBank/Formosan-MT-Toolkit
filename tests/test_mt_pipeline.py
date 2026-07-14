from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

from build_experiment_splits import one_edit_conflicts, split_targets  # noqa: E402
from mt_metrics import score_translations  # noqa: E402
from pivot import discover_api_key_envs, parse_api_key_envs  # noqa: E402
from train_directional_nllb import metric_improved  # noqa: E402
from training_code_inventory import build_code_inventory  # noqa: E402
from verify_experiment_manifest import manifest_errors  # noqa: E402
from write_submission_manifest import build_job_graph, read_job_ids  # noqa: E402


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


class ExperimentManifestTests(unittest.TestCase):
    def test_submitter_uses_scratch_logs_and_handles_completed_trainers(self) -> None:
        submitter = (
            ROOT
            / "formosan_mt_experiments/slurm/submit_v1_spm8k_directional.sh"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'LOGS_DIR="${LOGS_DIR:-${SCRATCH}/formosan_mt_experiments/logs/${RUN_STAMP}}"',
            submitter,
        )
        self.assertIn('--output="${LOGS_DIR}/%x-%j.out"', submitter)
        self.assertIn('--error="${LOGS_DIR}/%x-%j.err"', submitter)
        self.assertIn("COMPLETED*)", submitter)
        self.assertIn('eval_dependency+=(--dependency="afterok:${train_id}")', submitter)

    def test_setup_script_checksum_pins_match_source(self) -> None:
        setup = ROOT / "formosan_mt_experiments/scripts/setup_formosan_nllb200.py"
        expected = hashlib.sha256(setup.read_bytes()).hexdigest()
        launchers = {
            "setup_spm_sweep.sl": "SETUP_SCRIPT_SHA256",
            "submit_v1_spm8k_directional.sh": "SETUP_IMPLEMENTATION_SHA256",
        }
        for filename, variable in launchers.items():
            path = ROOT / "formosan_mt_experiments/slurm" / filename
            match = re.search(rf'{variable}="\$\{{{variable}:-([0-9a-f]{{64}})\}}"', path.read_text())
            self.assertIsNotNone(match, filename)
            self.assertEqual(match.group(1), expected, filename)

    def test_active_training_code_inventory_is_complete(self) -> None:
        inventory = build_code_inventory(
            experiment_root=ROOT / "formosan_mt_experiments",
        )
        artifacts = inventory["artifacts"]
        repository_paths = {row["repository_path"] for row in artifacts}
        self.assertIn(
            "formosan_mt_experiments/scripts/setup_formosan_nllb200.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/train_directional_nllb.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/evaluate_directional.py",
            repository_paths,
        )
        self.assertTrue(all(len(row["sha256"]) == 64 for row in artifacts))

    def test_submission_graph_requires_complete_directional_chain(self) -> None:
        job_ids = {
            "validate_en": 1,
            "validate_zh": 2,
            "setup_en_spm8192": 3,
            "setup_zh_spm8192": 4,
        }
        next_id = 5
        for direction in ("f2en", "en2f", "f2zh", "zh2f"):
            for label in (
                f"train_{direction}",
                f"eval_{direction}_final",
                f"eval_{direction}_best",
            ):
                job_ids[label] = next_id
                next_id += 1

        graph = build_job_graph(job_ids)
        self.assertEqual(graph["f2en"], [5, 6, 7])
        self.assertEqual(len({job for chain in graph.values() for job in (chain if isinstance(chain, list) else [chain])}), 16)

    def test_submission_state_rejects_non_numeric_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state / "validate_en.id").write_text("2714139\n", encoding="utf-8")
            self.assertEqual(read_job_ids(state), {"validate_en": 2714139})
            (state / "validate_zh.id").write_text("not-a-job\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_job_ids(state)

    def test_tracked_current_manifest_is_arithmetically_complete(self) -> None:
        path = ROOT / "formosan_mt_experiments/manifests/no_bible_v1_20260712.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_errors(manifest), [])
        self.assertEqual(set(manifest["corpora"]), {
            "public_no_bible_en",
            "public_no_bible_zh",
            "private_no_bible_en",
            "private_no_bible_zh",
        })
        for corpus in manifest["corpora"].values():
            self.assertEqual(sum(corpus["splits"].values()), corpus["rows"])
            self.assertGreaterEqual(corpus["splits"]["test"] / corpus["rows"], 0.075)
            self.assertGreaterEqual(corpus["splits"]["validate"] / corpus["rows"], 0.025)
            self.assertTrue(all(value == 0 for value in corpus["validation"].values()))

        jobs = manifest["jobs"]
        all_ids: list[int] = []
        for scope in jobs.values():
            for value in scope.values():
                all_ids.extend(value if isinstance(value, list) else [value])
        self.assertEqual(len(all_ids), 32)
        self.assertEqual(len(set(all_ids)), 32)


if __name__ == "__main__":
    unittest.main()
