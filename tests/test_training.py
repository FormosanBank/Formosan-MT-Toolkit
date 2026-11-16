from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch
from experiment_config import (
    DEFAULT_PROFILE,
    MILMMT_PROFILE,
    NLLB_1_3B_PROFILE,
    NLLB_3_3B_PROFILE,
    SHARED_SPLIT_FIELDS,
    load_corpus_pipeline_config,
    load_profile,
    target_split_ratios,
)
from mt_common import (
    language_sampling_probs,
    row_type_sampling_probs,
    special_tokens_from_corpus,
)
from setup_formosan_nllb200 import realign_embeddings
from train_directional import (
    metric_improved,
    metric_value,
)


class TrainingSamplingTests(unittest.TestCase):
    def test_control_tokens_include_languages_and_dialects_only(self) -> None:
        tokens = special_tokens_from_corpus(pd.DataFrame({"dialect": ["Coastal", "Coastal", "Coastal"]}))
        self.assertIn("<src_ami>", tokens)
        self.assertIn("<to_eng>", tokens)
        self.assertIn("<dialect_coastal>", tokens)
        self.assertFalse(any(token.startswith("<dom_") for token in tokens))

    def test_language_temperature_sampling_uses_only_language_counts(self) -> None:
        probabilities = language_sampling_probs({"ami": 100, "ssf": 25}, 0.5)
        self.assertAlmostEqual(probabilities["ami"], 2 / 3)
        self.assertAlmostEqual(probabilities["ssf"], 1 / 3)

    def test_row_sampling_uses_only_explicit_structural_type(self) -> None:
        probabilities = row_type_sampling_probs(
            ["sentence", "sentence", "lexeme", "morpheme"],
            lexical_weight=0.25,
        )
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertAlmostEqual(probabilities[0], probabilities[2] * 4)
        self.assertAlmostEqual(probabilities[1], probabilities[3] * 4)

    def test_row_sampling_rejects_unknown_structural_type(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported row types"):
            row_type_sampling_probs(["sentence", "dictionary"], lexical_weight=0.25)


class TrainingMetricTests(unittest.TestCase):
    def test_metric_direction_and_minimum_delta(self) -> None:
        self.assertTrue(metric_improved(20.1, 20.0, "chrF2", 0.05))
        self.assertFalse(metric_improved(20.01, 20.0, "chrF2", 0.05))
        self.assertTrue(metric_improved(1.8, 2.0, "mean_token_loss", 0.05))
        self.assertFalse(metric_improved(1.98, 2.0, "mean_token_loss", 0.05))
        self.assertTrue(metric_improved(49.0, 50.0, "TER", 0.05))
        self.assertTrue(metric_improved(49.0, 50.0, "macro_TER", 0.05))

    def test_macro_metric_weights_languages_equally(self) -> None:
        metrics = {
            "generation": {
                "global": {"chrF2": 80.0},
                "by_language": {
                    "ami": {"chrF2": 40.0},
                    "ssf": {"chrF2": 20.0},
                },
            }
        }
        self.assertEqual(metric_value(metrics, "chrF2"), 80.0)
        self.assertEqual(metric_value(metrics, "macro_chrF2"), 30.0)


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
        self.assertEqual(
            profile["training_defaults"]["validation_metadata_mode"],
            "default",
        )
        self.assertEqual(len(profile["base_model"]["revision"]), 40)
        defaults = profile["training_defaults"]
        self.assertEqual(defaults["batch_size"], 32)
        self.assertEqual(defaults["grad_accum_steps"], 2)
        self.assertEqual(defaults["language_sampling_chunk_size"], 8)
        self.assertEqual(defaults["effective_batch_size"], 64)
        self.assertEqual(defaults["generation_eval_batch_size"], 32)
        self.assertEqual(
            defaults["batch_size"] // defaults["language_sampling_chunk_size"] * defaults["grad_accum_steps"],
            8,
        )

    def test_nllb_size_profiles_share_the_data_and_evaluation_contract(self) -> None:
        profiles = [
            load_profile(path)
            for path in (
                DEFAULT_PROFILE,
                NLLB_1_3B_PROFILE,
                NLLB_3_3B_PROFILE,
            )
        ]
        self.assertEqual(
            [profile["model_variant"] for profile in profiles],
            ["nllb-600m", "nllb-1.3b", "nllb-3.3b"],
        )
        self.assertEqual(
            [profile["base_model"] for profile in profiles],
            [
                {
                    "name": "facebook/nllb-200-distilled-600M",
                    "revision": "f8d333a098d19b4fd9a8b18f94170487ad3f821d",
                },
                {
                    "name": "facebook/nllb-200-1.3B",
                    "revision": "b0de46b488af0cf31749cd8da5ed3171e11b2309",
                },
                {
                    "name": "facebook/nllb-200-3.3B",
                    "revision": "1a07f7d195896b2114afcb79b7b57ab512e7b43e",
                },
            ],
        )
        baseline = profiles[0]
        for profile in profiles[1:]:
            self.assertEqual(profile["splits"], baseline["splits"])
            self.assertEqual(profile["tokenizer"], baseline["tokenizer"])
            self.assertEqual(
                profile["generation_defaults"],
                baseline["generation_defaults"],
            )
            self.assertEqual(profile["training_defaults"]["steps"], 300000)
            self.assertEqual(
                profile["training_defaults"]["effective_batch_size"],
                64,
            )
            self.assertEqual(
                profile["training_defaults"]["batch_size"]
                // profile["training_defaults"]["language_sampling_chunk_size"]
                * profile["training_defaults"]["grad_accum_steps"],
                8,
            )
            self.assertTrue(
                profile["training_defaults"]["gradient_checkpointing"]
            )
        self.assertEqual(
            [profile["training_defaults"]["batch_size"] for profile in profiles],
            [32, 16, 8],
        )
        self.assertEqual(
            [profile["training_defaults"]["grad_accum_steps"] for profile in profiles],
            [2, 4, 8],
        )
        self.assertEqual(
            len({profile["recipe_id"] for profile in profiles}),
            3,
        )

    def test_experiment_profile_rejects_variant_model_disagreement(self) -> None:
        profile = json.loads(NLLB_1_3B_PROFILE.read_text(encoding="utf-8"))
        profile["base_model"]["name"] = "facebook/nllb-200-3.3B"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(
                SystemExit,
                "variant, family, and base model disagree",
            ):
                load_profile(path)

    def test_experiment_profiles_match_canonical_split_policy(self) -> None:
        pipeline_splits = load_corpus_pipeline_config()["splits"]
        profile_splits = load_profile(DEFAULT_PROFILE)["splits"]
        self.assertEqual(
            {field: profile_splits[field] for field in SHARED_SPLIT_FIELDS},
            {field: pipeline_splits[field] for field in SHARED_SPLIT_FIELDS},
        )
        self.assertEqual(
            target_split_ratios(pipeline_splits, "english"),
            {"train": 0.70, "validate": 0.10, "test": 0.20},
        )
        self.assertEqual(
            target_split_ratios(pipeline_splits, "chinese"),
            {"train": 0.85, "validate": 0.05, "test": 0.10},
        )

    def test_experiment_profile_rejects_split_policy_drift(self) -> None:
        profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        profile["splits"]["ratios_by_target"]["english"]["test"] = 0.05
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "split policies differ"):
                load_profile(path)

    def test_experiment_profile_rejects_unsupported_family(self) -> None:
        profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        profile["model_family"] = "unsupported"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "unsupported model_family"):
                load_profile(path)

    def test_milmmt_recipe_is_pinned_to_matched_native_sft(self) -> None:
        profile = load_profile(MILMMT_PROFILE)
        baseline = load_profile(DEFAULT_PROFILE)
        self.assertEqual(profile["model_family"], "milmmt")
        self.assertEqual(
            profile["base_model"],
            {
                "name": "xiaomi-research/MiLMMT-46-1B-v1.0",
                "revision": "4fc480b6c58dec29c159dcdf9fde0f6d5c354995",
            },
        )
        self.assertEqual(profile["tokenizer"], {"mode": "native"})
        self.assertEqual(profile["training_defaults"]["optimizer"], "adamw")
        self.assertEqual(profile["training_defaults"]["lr_scheduler"], "inverse_sqrt")
        self.assertEqual(profile["training_defaults"]["best_metric"], "chrF2")
        self.assertTrue(profile["training_defaults"]["use_tags"])
        self.assertEqual(profile["training_defaults"]["lexical_row_sampling_weight"], 0.25)
        self.assertFalse(profile["training_defaults"]["gradient_checkpointing"])
        self.assertTrue(profile["training_defaults"]["fused_optimizer"])
        self.assertEqual(profile["training_defaults"]["generation_eval_batch_size"], 16)
        self.assertEqual(
            profile["training_defaults"]["batch_size"]
            // profile["training_defaults"]["language_sampling_chunk_size"]
            * profile["training_defaults"]["grad_accum_steps"],
            8,
        )
        self.assertEqual(profile["generation_defaults"]["beam"], 4)
        comparison = profile["comparison"]
        self.assertEqual(
            comparison["sample_presentations"],
            profile["training_defaults"]["steps"] * profile["training_defaults"]["effective_batch_size"],
        )
        for field in comparison["matched_training_fields"]:
            self.assertEqual(
                profile["training_defaults"][field],
                baseline["training_defaults"][field],
            )
        for field in comparison["matched_generation_fields"]:
            self.assertEqual(
                profile["generation_defaults"][field],
                baseline["generation_defaults"][field],
            )

    def test_experiment_profile_rejects_invalid_language_chunk_geometry(self) -> None:
        profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        profile["training_defaults"]["language_sampling_chunk_size"] = 7
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(SystemExit, "language sampling chunk"):
                load_profile(path)

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
