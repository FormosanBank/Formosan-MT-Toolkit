from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiment_config import (
    DEFAULT_PROFILE,
    NLLB_1_3B_PROFILE,
    NLLB_3_3B_PROFILE,
    load_profile,
)
from publish_huggingface_models import (
    DIRECTIONS,
    nllb_usage,
    public_metrics,
    render_card,
    repo_name,
    validate_checkpoint,
)
from training_code_inventory import build_code_inventory
from write_submission_manifest import (
    build_job_graph,
    corpus_record,
    read_job_ids,
)

ROOT = Path(__file__).resolve().parents[1]


class ExperimentManifestTests(unittest.TestCase):
    def test_submission_manifest_reads_current_validation_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            corpus_dir = Path(temporary)
            provenance = corpus_dir / "provenance"
            provenance.mkdir()
            (provenance / "mt_build_manifest.json").write_text(
                json.dumps(
                    {
                        "artifacts": {
                            "big_corpus_en_in_domain_hard": {
                                "rows": 100,
                                "sha256": "a" * 64,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            split_validation = {
                "ok": True,
                "ratios_by_language": {"ami": {"train": 90, "test": 8, "validate": 2}},
                "synthetic_eval_rows": 0,
                "train_evaluation": {
                    "exact_overlap": {"formosan": 0, "target": 0, "pair": 0},
                    "skeleton_overlap": {"formosan": 0, "target": 0, "pair": 0},
                    "one_edit_conflicting_rows": {"formosan": 0, "target": 0},
                    "character_ngram_conflicting_rows": {
                        "formosan": 0,
                        "target": 0,
                    },
                    "document_overlap": 0,
                },
                "lexical_eval_rows": 0,
                "ratio_failures": [],
            }
            (provenance / "validate_en_in_domain_hard.json").write_text(
                json.dumps({"split_validation": split_validation}),
                encoding="utf-8",
            )

            record = corpus_record(corpus_dir, "en")

            self.assertEqual(record["splits"], {"train": 90, "test": 8, "validate": 2})
            self.assertTrue(record["validation"].pop("ok"))
            self.assertTrue(all(value == 0 for value in record["validation"].values()))

    def test_publication_accepts_sharded_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            for name in (
                "config.json",
                "experiment_metadata.json",
                "generation_config.json",
                "tokenizer_config.json",
                "spiece.model",
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
                "model.safetensors.index.json",
            ):
                (checkpoint / name).write_text("{}", encoding="utf-8")
            files = validate_checkpoint(checkpoint)
            self.assertEqual(len(files), 8)

    def test_submitter_uses_scratch_logs_and_handles_completed_trainers(self) -> None:
        submitter = (ROOT / "formosan_mt_experiments/slurm/submit_directional_experiment.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            'LOGS_DIR="${LOGS_DIR:-${SCRATCH}/formosan_mt_experiments/logs/${RUN_STAMP}}"',
            submitter,
        )
        self.assertIn('--output="${LOGS_DIR}/%x-%j.out"', submitter)
        self.assertIn('--error="${LOGS_DIR}/%x-%j.err"', submitter)
        self.assertIn("COMPLETED*)", submitter)
        self.assertIn('eval_dependency="--dependency=afterok:${train_id}"', submitter)
        self.assertIn(
            'read -r -a checkpoints <<<"${EVAL_CHECKPOINTS:-best}"',
            submitter,
        )
        self.assertIn(
            '--time="${EVAL_TIME:-${PROFILE_EVAL_TIME}}"',
            submitter,
        )
        self.assertNotIn("for checkpoint in final best", submitter)
        self.assertIn(
            "tokenizer_sweep_${RECIPE_SLUG}_${short}_spm8192",
            submitter,
        )
        self.assertIn("nllb-3.3b)", submitter)
        self.assertIn('PROFILE_GPU_CONSTRAINT="vr144g"', submitter)
        self.assertIn("BOOTSTRAP_SAMPLES=${BOOTSTRAP_SAMPLES:-1000}", submitter)

    def test_full_evaluation_defaults_are_resource_conservative(self) -> None:
        defaults = load_profile(DEFAULT_PROFILE)["generation_defaults"]
        self.assertEqual(defaults["metadata_modes"], ["default"])
        self.assertEqual(defaults["bootstrap_samples"], 0)

    def test_nllb_setup_checksum_is_computed_then_enforced(self) -> None:
        submitter = (ROOT / "formosan_mt_experiments/slurm/submit_directional_experiment.sh").read_text(
            encoding="utf-8"
        )
        setup = (ROOT / "formosan_mt_experiments/slurm/setup_spm_sweep.sl").read_text(encoding="utf-8")
        self.assertIn(
            'setup_sha="$(sha256sum "${NLLB_SETUP_IMPLEMENTATION}"',
            submitter,
        )
        self.assertIn(
            ': "${SETUP_SCRIPT_SHA256:?SETUP_SCRIPT_SHA256 is required}"',
            setup,
        )

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
            "formosan_mt_experiments/configs/nllb_1_3b_experiment.json",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/configs/nllb_3_3b_experiment.json",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/nllb_runtime.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/milmmt_runtime.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/setup_milmmt.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/train_directional.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/training_data.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/training_state.py",
            repository_paths,
        )
        self.assertIn(
            "scripts/shared/columnar_io.py",
            repository_paths,
        )
        self.assertIn("scripts/shared/reproducibility.py", repository_paths)
        self.assertIn("scripts/shared/split_policy.py", repository_paths)
        self.assertIn(
            "formosan_mt_experiments/scripts/evaluate_directional.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/formosan_mt_inference.py",
            repository_paths,
        )
        self.assertIn("config/mt_standardization.json", repository_paths)
        self.assertIn("config/corpus_pipeline.json", repository_paths)
        self.assertIn("scripts/local/corpus_quality.py", repository_paths)
        self.assertIn("scripts/local/mt_standardization.py", repository_paths)
        self.assertIn(
            "formosan_mt_experiments/slurm/train_directional.sl",
            repository_paths,
        )
        self.assertTrue(all(len(row["sha256"]) == 64 for row in artifacts))

    def test_publication_examples_apply_formosan_mt_standardization(self) -> None:
        formosan_source = nllb_usage(DIRECTIONS["f2en"])
        major_source = nllb_usage(DIRECTIONS["en2f"])
        self.assertIn(
            "from formosan_mt_inference import normalize_formosan",
            formosan_source,
        )
        self.assertIn(
            "text = normalize_formosan(text, lang_code)",
            formosan_source,
        )
        self.assertNotIn("source_bucket", formosan_source)
        self.assertNotIn("<dom_", formosan_source)
        self.assertNotIn("normalize_formosan", major_source)

    def test_publication_card_records_release_and_hard_test_contract(self) -> None:
        profile = {
            "recipe_id": "nllb200-spm8k-directional-v3",
            "model_variant": "nllb-600m",
            "base_model": {"name": "base/model", "revision": "abc123"},
            "mt_standardization": {"id": "formosan-mt-standard-v3"},
            "training_defaults": {
                "steps": 300000,
                "effective_batch_size": 64,
                "max_length": 384,
                "learning_rate": 2e-5,
                "precision": "bf16",
                "best_metric": "chrF2",
                "language_sampling_alpha": 0.5,
                "lexical_row_sampling_weight": 0.25,
                "dialect_tag_dropout": 0.25,
            },
            "splits": {
                "ratios_by_target": {
                    "english": {"train": 0.7, "validate": 0.1, "test": 0.2}
                },
                "synthetic_eval_policy": "train_only_after_human_split",
            },
        }
        metrics = {
            "global": {
                "BLEU": 9.1,
                "chrF2": 27.2,
                "TER": 90.3,
                "empty_output_rate": 0.0,
                "signatures": {
                    "BLEU": "bleu-signature",
                    "chrF2": "chrf-signature",
                    "TER": "ter-signature",
                },
            },
            "by_language": {"ami": {"samples": 100, "BLEU": 9.1, "chrF2": 27.2, "TER": 90.3}},
            "headline_metadata_mode": "default",
            "profile": {"sha256": "b" * 64},
        }
        metadata = {
            "step": 210000,
            "validation": {"generation": {"global": {"BLEU": 8.0, "chrF2": 25.0, "TER": 92.0}}},
        }
        manifest = {
            "corpora": {
                "english": {
                    "path": "/scratch/data/releases/20260824T042425Z_v3/private_no_bible/corpus.csv",
                    "rows": 1000,
                    "sha256": "a" * 64,
                    "splits": {"train": 900, "test": 75, "validate": 25},
                    "validation": {
                        "exact_overlap": 0,
                        "skeleton_overlap": 0,
                        "one_edit_conflicts": 0,
                        "character_ngram_conflicts": 0,
                        "document_overlap": 0,
                    },
                }
            }
        }
        card = render_card(
            spec=DIRECTIONS["f2en"],
            repo_id="FormosanBank/nllb200-formosan-en-spm8k",
            profile=profile,
            metrics=metrics,
            metadata=metadata,
            manifest=manifest,
            run_stamp="20260809-210523",
        )
        self.assertIn("model-index:", card)
        self.assertIn("headline result uses `default` metadata", card)
        self.assertIn("Document overlap is diagnostic", card)
        self.assertIn("70% train / 10% validate / 20% test", card)
        self.assertIn("Synthetic pivots and\nlexical entries are train-only", card)
        self.assertIn("domain or repository tags", card)
        self.assertNotIn("<dom_", card)
        self.assertIn("Language sampling alpha | 0.5", card)
        self.assertIn("Sentence / lexical sampling weight | 1.0 / 0.25", card)
        self.assertIn("`20260824T042425Z_v3`", card)
        self.assertIn("FormosanBank/formosan-mt-private", card)
        self.assertIn("`20260809-210523`", card)
        self.assertIn("nllb-200", card)

    def test_publication_names_keep_600m_compatibility_and_isolate_larger_models(self) -> None:
        profiles = {
            "600m": load_profile(DEFAULT_PROFILE),
            "1.3b": load_profile(NLLB_1_3B_PROFILE),
            "3.3b": load_profile(NLLB_3_3B_PROFILE),
        }
        self.assertEqual(
            repo_name("f2en", profiles["600m"]),
            "nllb200-formosan-en-spm8k",
        )
        self.assertEqual(
            repo_name("f2en", profiles["1.3b"]),
            "nllb200-1.3b-formosan-en-spm8k",
        )
        self.assertEqual(
            repo_name("f2en", profiles["3.3b"]),
            "nllb200-3.3b-formosan-en-spm8k",
        )

    def test_public_metrics_remove_cluster_paths(self) -> None:
        metrics = {
            "input": "/projects/private.csv",
            "model": "/scratch/model",
            "tokenizer": "/scratch/model",
            "profile": {"path": "/home/user/profile.json", "sha256": "a" * 64},
        }
        output = public_metrics(
            metrics,
            repo_id="FormosanBank/model",
            corpus_name="private_no_bible",
            target_lang="english",
        )
        self.assertEqual(output["input"], "private_no_bible:english")
        self.assertEqual(output["model"], "FormosanBank/model")
        self.assertEqual(output["profile"]["path"], "training_profile.json")
        self.assertEqual(metrics["input"], "/projects/private.csv")

    def test_submission_graph_records_available_evaluations(self) -> None:
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
        self.assertEqual(
            graph["directions"]["f2en"],
            {
                "train": 5,
                "evaluations": {"best": 7, "final": 6},
            },
        )

    def test_submission_graph_accepts_default_best_only_flight(self) -> None:
        job_ids = {"validate_en": 1, "validate_zh": 2}
        next_id = 3
        for direction in ("f2en", "en2f", "f2zh", "zh2f"):
            job_ids[f"train_{direction}"] = next_id
            job_ids[f"eval_{direction}_best"] = next_id + 1
            next_id += 2

        graph = build_job_graph(job_ids)

        self.assertEqual(
            graph["directions"]["zh2f"]["evaluations"],
            {"best": 10},
        )
        self.assertEqual(graph["setup"], {})

    def test_submission_graph_records_shared_milmmt_setup(self) -> None:
        job_ids = {
            "validate_en": 1,
            "validate_zh": 2,
            "setup_milmmt": 3,
        }
        next_id = 4
        for direction in ("f2en", "en2f", "f2zh", "zh2f"):
            job_ids[f"train_{direction}"] = next_id
            job_ids[f"eval_{direction}_best"] = next_id + 1
            next_id += 2
        graph = build_job_graph(job_ids)
        self.assertEqual(graph["setup"], {"shared": 3})

    def test_submission_state_rejects_non_numeric_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state / "validate_en.id").write_text("2714139\n", encoding="utf-8")
            self.assertEqual(read_job_ids(state), {"validate_en": 2714139})
            (state / "validate_zh.id").write_text("not-a-job\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_job_ids(state)
