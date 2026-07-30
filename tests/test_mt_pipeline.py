from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

from audit_corpus_exposure import (  # noqa: E402
    audit_direction,
    build_tame_config,
    gate_errors,
)
from build_experiment_splits import (  # noqa: E402
    GroupCandidate,
    build_hard_split,
    choose_groups,
    one_edit_conflicts,
    split_targets,
)
from experiment_config import DEFAULT_PROFILE, load_profile  # noqa: E402
from mt_common import add_normalized_columns  # noqa: E402
from mt_metrics import bootstrap_confidence_intervals, score_translations  # noqa: E402
from pivot import discover_api_key_envs, parse_api_key_envs  # noqa: E402
from setup_formosan_nllb200 import realign_embeddings  # noqa: E402
from train_directional_nllb import metric_improved  # noqa: E402
from training_code_inventory import build_code_inventory  # noqa: E402
from validate_experiment import validate_provenance, validate_splits  # noqa: E402
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
    def test_group_selection_reserves_the_other_evaluation_split(self) -> None:
        candidates = [
            GroupCandidate(
                group_id=group_id,
                eligible_rows=rows,
                total_rows=rows,
                non_eval_rows=0,
                easy_fraction=0,
                average_tokens=10,
            )
            for group_id, rows in enumerate((104, 86, 16, 4))
        ]
        validation = choose_groups(
            candidates,
            33,
            reserve_rows=98,
            seed=42,
            attempts=20,
        )
        validation_rows = sum(
            candidate.eligible_rows
            for candidate in candidates
            if candidate.group_id in validation
        )
        remaining = [
            candidate
            for candidate in candidates
            if candidate.group_id not in validation
        ]
        test = choose_groups(
            remaining,
            98,
            reserve_rows=0,
            seed=43,
            attempts=20,
        )
        test_rows = sum(
            candidate.eligible_rows
            for candidate in remaining
            if candidate.group_id in test
        )

        self.assertGreaterEqual(validation_rows, 33)
        self.assertGreaterEqual(test_rows, 98)

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

    @staticmethod
    def hard_split_fixture() -> pd.DataFrame:
        rows = []
        for document in range(60):
            for sentence in range(3):
                digest = hashlib.sha256(
                    f"{document}:{sentence}".encode()
                ).hexdigest()
                rows.append(
                    {
                        "row_id": f"human-{document}-{sentence}",
                        "source_record_id": f"record-{document}-{sentence}",
                        "lang_code": "ami",
                        "formosan_sentence": (
                            f"{digest[:8]} {digest[8:16]} {digest[16:24]} "
                            f"{digest[24:32]} {digest[32:40]}"
                        ),
                        "english_sentence": (
                            f"{digest[40:48]} {digest[48:56]} sentence "
                            f"{document} number {sentence}"
                        ),
                        "source": f"FormosanBank/Corpora/Test/XML/doc-{document}.xml",
                        "repository": "FormosanBank",
                        "repository_commit": "a" * 40,
                        "xml_path": f"Corpora/Test/XML/doc-{document}.xml",
                        "xml_id": f"S-{sentence}",
                        "xml_element_index": sentence,
                        "kindOf": "standard",
                        "standard_origin": "provided",
                        "standard_after_qc_sha256": digest,
                        "qc_transform_id": f"transform-{document}-{sentence}",
                        "qc_revision": "b" * 40,
                        "dialect": "Test",
                        "row_type": "sentence",
                        "pivot_origin": "original",
                        "quality_flags": "",
                    }
                )
            rows.append(
                {
                    **rows[-1],
                    "row_id": f"synthetic-{document}",
                    "source_record_id": f"synthetic-record-{document}",
                    "formosan_sentence": f"synthetic unique {document}",
                    "english_sentence": f"synthetic target {document}",
                    "xml_id": f"pivot-{document}",
                    "pivot_origin": "synthetic",
                }
            )
            rows.append(
                {
                    **rows[-2],
                    "row_id": f"lexeme-{document}",
                    "source_record_id": f"lexeme-record-{document}",
                    "formosan_sentence": f"lexeme{document}",
                    "english_sentence": f"entry{document}",
                    "xml_id": f"W-{document}",
                    "row_type": "lexeme",
                }
            )
        rows.append({**rows[0], "row_id": "duplicate-pair"})
        return pd.DataFrame(rows)

    def test_hard_split_is_human_document_disjoint_and_large(self) -> None:
        raw = self.hard_split_fixture()
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, excluded, duplicates, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.075,
            val_ratio=0.025,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )
        evaluation = output[output["split"].isin({"test", "validate"})]
        train = output[output["split"].eq("train")]
        self.assertTrue(evaluation["row_type"].eq("sentence").all())
        self.assertFalse(evaluation["pivot_origin"].eq("synthetic").any())
        self.assertFalse(
            set(train["document_id"]) & set(evaluation["document_id"])
        )
        self.assertEqual(len(duplicates), 1)
        self.assertGreater(len(excluded), 0)
        self.assertTrue(report["complete"])

        provenance = validate_provenance(output)
        validation = validate_splits(
            output,
            target_col="english_sentence",
            min_test_ratio=0.075,
            min_validate_ratio=0.025,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
        )
        self.assertTrue(provenance["ok"], provenance)
        self.assertTrue(validation["ok"], validation)

    def test_hard_split_is_deterministic_for_identical_input(self) -> None:
        keyed = add_normalized_columns(
            self.hard_split_fixture(),
            target_col="english_sentence",
            target_lang="english",
        )
        kwargs = {
            "target_col": "english_sentence",
            "test_ratio": 0.075,
            "val_ratio": 0.025,
            "seed": 42,
            "min_formosan_tokens": 1,
            "min_target_tokens": 1,
            "attempts": 20,
            "min_test_rows": 5,
            "min_validate_rows": 2,
            "ngram_threshold": 0.82,
            "registry_in": None,
        }
        first, _, _, _ = build_hard_split(keyed.copy(), **kwargs)
        second, _, _, _ = build_hard_split(keyed.copy(), **kwargs)
        first_map = dict(zip(first["row_id"], first["split"], strict=True))
        second_map = dict(zip(second["row_id"], second["split"], strict=True))
        self.assertEqual(first_map, second_map)

    def test_near_synthetic_training_rows_are_excluded_not_eval(self) -> None:
        raw = self.hard_split_fixture()
        near_synthetic = []
        human = raw[
            raw["pivot_origin"].eq("original")
            & raw["row_type"].eq("sentence")
        ]
        for _, row in human.iterrows():
            near_synthetic.append(
                {
                    **row.to_dict(),
                    "row_id": f"near-{row['row_id']}",
                    "source_record_id": (
                        f"near-{row['source_record_id']}"
                    ),
                    "formosan_sentence": (
                        f"{row['formosan_sentence']}x"
                    ),
                    "english_sentence": (
                        f"{row['english_sentence']}x"
                    ),
                    "source": (
                        "FormosanBank/Corpora/Pivot/XML/"
                        f"near-{row['row_id']}.xml"
                    ),
                    "xml_path": (
                        "Corpora/Pivot/XML/"
                        f"near-{row['row_id']}.xml"
                    ),
                    "xml_id": f"near-{row['xml_id']}",
                    "pivot_origin": "synthetic",
                }
            )
        raw = pd.concat(
            [raw, pd.DataFrame(near_synthetic)],
            ignore_index=True,
        )
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, excluded, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.075,
            val_ratio=0.025,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        self.assertTrue(report["complete"])
        self.assertGreater(
            report["excluded_near_duplicate_train_rows"],
            0,
        )
        self.assertTrue(
            excluded["exclusion_reason"]
            .eq("near_duplicate_of_evaluation")
            .any()
        )
        evaluation = output[
            output["split"].isin({"test", "validate"})
        ]
        self.assertFalse(
            evaluation["pivot_origin"].eq("synthetic").any()
        )
        self.assertGreaterEqual(
            len(output[output["split"].eq("test")])
            / len(output),
            0.075,
        )
        self.assertGreaterEqual(
            len(output[output["split"].eq("validate")])
            / len(output),
            0.025,
        )

    def test_independent_validator_rejects_synthetic_same_document_eval(self) -> None:
        frame = self.hard_split_fixture().iloc[:3].copy()
        frame["split"] = ["train", "test", "validate"]
        frame.loc[frame.index[1], "pivot_origin"] = "synthetic"
        validation = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
        )
        self.assertFalse(validation["ok"])
        self.assertEqual(validation["synthetic_eval_rows"], 1)
        self.assertGreater(
            validation["train_evaluation"]["document_overlap"],
            0,
        )

    def test_validate_test_target_overlap_is_task_conditioned(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "row_id": "ami-train",
                    "lang_code": "ami",
                    "formosan_sentence": "ami training phrase qzx",
                    "english_sentence": "unique training target alpha",
                    "source": "repo/ami-train.xml",
                    "dialect": "A",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "train",
                },
                {
                    "row_id": "bnn-train",
                    "lang_code": "bnn",
                    "formosan_sentence": "bnn training phrase vjk",
                    "english_sentence": "unique training target beta",
                    "source": "repo/bnn-train.xml",
                    "dialect": "B",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "train",
                },
                {
                    "row_id": "ami-test",
                    "lang_code": "ami",
                    "formosan_sentence": "ami heldout phrase mnp",
                    "english_sentence": "shared multilingual template",
                    "source": "repo/ami-test.xml",
                    "dialect": "A",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "test",
                },
                {
                    "row_id": "bnn-validate",
                    "lang_code": "bnn",
                    "formosan_sentence": "bnn heldout phrase rst",
                    "english_sentence": "shared multilingual template",
                    "source": "repo/bnn-validate.xml",
                    "dialect": "B",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "validate",
                },
            ]
        )
        validation = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
        )

        self.assertTrue(validation["ok"], validation)
        self.assertEqual(
            validation["validate_test"]["exact_overlap"]["target"],
            0,
        )
        self.assertEqual(
            validation[
                "validate_test_cross_language_diagnostic"
            ]["exact_overlap"]["target"],
            1,
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

    def test_bootstrap_intervals_are_deterministic_and_ordered(self) -> None:
        kwargs = {
            "hypotheses": ["one", "two", "bad", "four"],
            "references": ["one", "two", "three", "four"],
            "strata": ["ami", "ami", "bnn", "bnn"],
            "samples": 20,
            "seed": 7,
        }
        first = bootstrap_confidence_intervals(**kwargs)
        second = bootstrap_confidence_intervals(**kwargs)
        self.assertEqual(first, second)
        for interval in first["metrics"].values():
            self.assertLessEqual(interval["lower"], interval["median"])
            self.assertLessEqual(interval["median"], interval["upper"])


class TokenizerSetupTests(unittest.TestCase):
    def test_recipe_is_pinned_to_train_only_standard_formosan_spm8k(self) -> None:
        profile = load_profile(DEFAULT_PROFILE)
        self.assertEqual(profile["tokenizer"]["default_spm_vocab"], 8192)
        self.assertEqual(profile["tokenizer"]["setup_splits"], ["train"])
        self.assertEqual(
            profile["tokenizer"]["training_columns"],
            ["formosan_sentence"],
        )
        self.assertEqual(profile["splits"]["tiers"], ["in_domain_hard"])
        self.assertEqual(len(profile["base_model"]["revision"]), 40)

    def test_embedding_realignment_uses_token_identity(self) -> None:
        class Tokenizer:
            unk_token_id = 0

            def __init__(self, vocab, pieces=None):
                self.vocab = vocab
                self.pieces = pieces or {}

            def get_vocab(self):
                return self.vocab

            def __len__(self):
                return len(self.vocab)

            def __call__(self, text, **_kwargs):
                return {"input_ids": self.pieces.get(text, [0])}

        class Model:
            def __init__(self):
                self.embedding = torch.nn.Embedding(5, 2)
                with torch.no_grad():
                    self.embedding.weight.copy_(
                        torch.tensor(
                            [
                                [0.0, 0.0],
                                [1.0, 1.0],
                                [2.0, 2.0],
                                [3.0, 3.0],
                                [4.0, 4.0],
                            ]
                        )
                    )

            def get_input_embeddings(self):
                return self.embedding

            def resize_token_embeddings(self, size):
                previous = self.embedding.weight.detach().clone()
                self.embedding = torch.nn.Embedding(size, 2)
                with torch.no_grad():
                    self.embedding.weight[: len(previous)].copy_(previous)
                return self.embedding

            def tie_weights(self):
                return None

        old = Tokenizer(
            {"<unk>": 0, "eng_Latn": 1, "shared": 2, "part_a": 3, "part_b": 4},
            {"newpiece": [3, 4]},
        )
        new = Tokenizer(
            {
                "shared": 0,
                "<unk>": 1,
                "eng_Latn": 2,
                "newpiece": 3,
                "ami_Latn": 4,
                "part_a": 5,
                "part_b": 6,
            }
        )
        model = Model()
        report = realign_embeddings(model, old, new, {"ami_Latn"})
        rows = model.get_input_embeddings().weight.detach()
        self.assertTrue(torch.equal(rows[0], torch.tensor([2.0, 2.0])))
        self.assertTrue(torch.equal(rows[3], torch.tensor([3.5, 3.5])))
        self.assertTrue(torch.equal(rows[4], torch.tensor([1.0, 1.0])))
        self.assertEqual(report["shared_tokens_realigned"], 5)
        self.assertEqual(report["new_piece_rows_initialized"], 1)
        self.assertEqual(report["formosan_language_rows_seeded_from_english"], 1)


class ExposureAuditTests(unittest.TestCase):
    @staticmethod
    def frame(*, duplicate_test: bool = False) -> pd.DataFrame:
        rows = []
        for index in range(12):
            rows.append(
                {
                    "row_id": f"train-{index}",
                    "lang_code": "ami",
                    "formosan_sentence": f"train source phrase number {index} alpha",
                    "english_sentence": f"train target phrase number {index} omega",
                    "split": "train",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "is_synthetic": "false",
                }
            )
        for split, offset in (("test", 100), ("validate", 200)):
            for index in range(4):
                rows.append(
                    {
                        "row_id": f"{split}-{index}",
                        "lang_code": "ami",
                        "formosan_sentence": (
                            "train source phrase number 0 alpha"
                            if duplicate_test and split == "test" and index == 0
                            else f"{split} heldout expression {offset + index} cedar"
                        ),
                        "english_sentence": (
                            "train target phrase number 0 omega"
                            if duplicate_test and split == "test" and index == 0
                            else f"{split} reference wording {offset + index} quartz"
                        ),
                        "split": split,
                        "kindOf": "standard",
                        "row_type": "sentence",
                        "is_synthetic": "false",
                    }
                )
        return pd.DataFrame(rows)

    def test_tame_release_gate_accepts_distant_human_evaluation(self) -> None:
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            self.frame(),
            direction="f2en",
            target_col="english_sentence",
            config=config,
        )
        self.assertEqual(
            gate_errors(
                {"f2en": payload},
                high_threshold="0.95",
                max_high_exposure_rate=0.0,
            ),
            [],
        )

    def test_tame_release_gate_rejects_exact_train_test_pair(self) -> None:
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            self.frame(duplicate_test=True),
            direction="f2en",
            target_col="english_sentence",
            config=config,
        )
        errors = gate_errors(
            {"f2en": payload},
            high_threshold="0.95",
            max_high_exposure_rate=0.0,
        )
        self.assertTrue(any("exact_overlap" in error for error in errors))


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
