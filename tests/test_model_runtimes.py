from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import milmmt_runtime as milmmt
import nllb_runtime as nllb
import numpy as np
import pandas as pd
import tokenizer_audit
import torch
from formosan_mt_inference import normalize_formosan
from train_directional import (
    compute_training_loss,
    training_source_texts,
)
from training_data import collate_encoded_batches
from training_state import clone_model_checkpoint
from validate_experiment import (
    validate_tags,
)


class FakeTokenizer:
    def __init__(self) -> None:
        values = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<to_eng>",
        ]
        self.vocab = {token: index for index, token in enumerate(values)}
        self.unk_token_id = self.vocab["<unk>"]
        self.pad_token_id = self.vocab["<pad>"]
        self.eos_token_id = self.vocab["</s>"]

    def __len__(self) -> int:
        return len(self.vocab)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def convert_tokens_to_ids(self, value):
        if isinstance(value, list):
            return [self.convert_tokens_to_ids(token) for token in value]
        return self.vocab.get(value, self.unk_token_id)

    def convert_ids_to_tokens(self, value: int) -> str:
        return next(token for token, token_id in self.vocab.items() if token_id == value)


class NllbRuntimeTests(unittest.TestCase):
    def test_inference_uses_the_pinned_mt_standardizer(self) -> None:
        self.assertEqual(normalize_formosan("ma-ku", "ami"), "maku")
        with self.assertRaisesRegex(ValueError, "not safe"):
            normalize_formosan("https://example.org/a/b", "ami")

    def test_nllb_generation_contract_is_unchanged(self) -> None:
        class NllbTokenizer:
            eos_token_id = 2
            pad_token_id = 1
            unk_token_id = 3
            vocab = {
                "ami_Latn": 10,
                "eng_Latn": 11,
            }

            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, self.unk_token_id)

            def convert_ids_to_tokens(self, token_id):
                return next(token for token, value in self.vocab.items() if value == token_id)

        tokenizer = NllbTokenizer()
        model = SimpleNamespace(
            config=SimpleNamespace(decoder_start_token_id=None),
            generation_config=SimpleNamespace(decoder_start_token_id=None),
        )
        nllb.configure_model(model, tokenizer)
        task = nllb.task_spec(
            "ami",
            "f2en",
            target_lang="english",
        )
        self.assertEqual(
            nllb.generation_kwargs(tokenizer, task),
            {
                "forced_bos_token_id": 11,
                "decoder_start_token_id": 2,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
        )

    def test_nllb_controls_cover_all_directions(self) -> None:
        row = {
            "lang_code": "ami",
            "dialect": "Coastal",
        }
        expected = {
            ("f2en", "english"): (
                "<to_eng> <src_ami> <dialect_coastal>",
                ("ami_Latn", "eng_Latn"),
            ),
            ("en2f", "english"): (
                "<to_ami> <src_eng> <dialect_coastal>",
                ("eng_Latn", "ami_Latn"),
            ),
            ("f2zh", "chinese"): (
                "<to_zh> <src_ami> <dialect_coastal>",
                ("ami_Latn", "zho_Hant"),
            ),
            ("zh2f", "chinese"): (
                "<to_ami> <src_zh> <dialect_coastal>",
                ("zho_Hant", "ami_Latn"),
            ),
        }
        for (direction, target_lang), (prefix, lids) in expected.items():
            self.assertEqual(
                nllb.source_prefix(
                    row,
                    direction,
                    target_lang=target_lang,
                    use_tags=True,
                ),
                prefix,
            )
            task = nllb.task_spec(
                "ami",
                direction,
                target_lang=target_lang,
            )
            self.assertEqual((task.source_lid, task.target_lid), lids)

    def test_tag_validation_uses_nllb_runtime(self) -> None:
        tokenizer = FakeTokenizer()
        frame = pd.DataFrame([{"source": "fixture.xml", "lang_code": "ami"}])
        with (
            mock.patch(
                "validate_experiment.nllb.load_tokenizer",
                return_value=tokenizer,
            ) as load_tokenizer,
            mock.patch(
                "validate_experiment.nllb.source_prefix",
                return_value="<to_eng>",
            ),
        ):
            report = validate_tags(
                frame,
                Path("tokenizer"),
                "en2f",
                "english",
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["model_family"], "nllb")
        load_tokenizer.assert_called_once_with(Path("tokenizer"))

    def test_unseen_evaluation_metadata_falls_back_to_setup_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.vocab.update(
            {
                "<dialect_default>": len(tokenizer.vocab),
                "<dialect_coastal>": len(tokenizer.vocab) + 1,
            }
        )
        frame = pd.DataFrame(
            {
                "dialect": ["Coastal", "Held Out", ""],
            }
        )
        normalized, report = nllb.normalize_control_metadata(
            frame,
            tokenizer,
        )
        self.assertEqual(
            normalized["dialect"].tolist(),
            ["Coastal", "default", "default"],
        )
        self.assertEqual(
            report,
            {"dialect_fallback_rows": 1},
        )

    def test_training_metadata_dropout_builds_default_control_paths(self) -> None:
        batch = pd.DataFrame(
            {
                "lang_code": ["ami", "ami"],
                "dialect": ["Coastal", "Coastal"],
                "formosan_sentence": ["mako ko loma niyam.", "mako ko loma nira."],
                "english_sentence": ["This is our house.", "This is their house."],
            }
        )
        args = SimpleNamespace(
            use_tags=True,
            dialect_tag_dropout=0.5,
            direction="f2en",
            target_col="english_sentence",
            target_lang="english",
        )
        with mock.patch(
            "train_directional.np.random.random",
            return_value=np.array([0.9, 0.1]),
        ):
            texts, counts = training_source_texts(
                batch,
                args=args,
            )
        self.assertEqual(
            texts,
            [
                "<to_eng> <src_ami> <dialect_coastal> mako ko loma niyam.",
                "<to_eng> <src_ami> <dialect_default> mako ko loma nira.",
            ],
        )
        self.assertEqual(counts, {"dialect": 1})


class MilmmtRuntimeTests(unittest.TestCase):
    def test_configure_model_uses_deterministic_generation(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(pad_token_id=None, use_cache=True),
            generation_config=SimpleNamespace(
                do_sample=True,
                top_k=50,
                top_p=0.95,
                cache_implementation="hybrid",
            ),
        )
        tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1)

        milmmt.configure_model(model, tokenizer)

        self.assertEqual(model.config.pad_token_id, 0)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.generation_config.do_sample)
        self.assertIsNone(model.generation_config.top_k)
        self.assertIsNone(model.generation_config.top_p)
        self.assertIsNone(model.generation_config.cache_implementation)

    def test_official_prompt_covers_all_directions(self) -> None:
        row = {
            "lang_code": "ami",
            "dialect": "Coastal",
        }
        expected = {
            ("f2en", "english"): ("Amis", "English"),
            ("en2f", "english"): ("English", "Amis"),
            ("f2zh", "chinese"): ("Amis", "Chinese (Traditional)"),
            ("zh2f", "chinese"): ("Chinese (Traditional)", "Amis"),
        }
        for (direction, target_lang), (source_name, target_name) in expected.items():
            prompt = milmmt.format_source(
                row,
                "sample",
                direction,
                target_lang=target_lang,
                use_tags=False,
            )
            self.assertEqual(
                prompt,
                f"Translate this from {source_name} to {target_name}:\n{source_name}: sample\n{target_name}:",
            )

    def test_prompt_uses_canonical_formosan_names(self) -> None:
        self.assertEqual(
            milmmt.task_spec("tao", "f2en", target_lang="english").source_name,
            "Tao",
        )
        self.assertEqual(
            milmmt.task_spec("trv", "f2en", target_lang="english").source_name,
            "Seediq",
        )

    def test_metadata_prompt_contains_dialect_but_no_domain(self) -> None:
        prompt = milmmt.format_source(
            {"lang_code": "ami", "dialect": "Coastal"},
            "sample",
            "f2en",
            target_lang="english",
            use_tags=True,
        )
        self.assertIn("Dialect: coastal", prompt)
        self.assertNotIn("domain", prompt.casefold())

    def test_causal_labels_mask_the_prompt(self) -> None:
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, texts, **_kwargs):
                return {"input_ids": [[ord(character) + 2 for character in text] for text in texts]}

        tokenizer = Tokenizer()
        task = milmmt.task_spec("ami", "f2en", target_lang="english")
        encoded, labels = milmmt.encode_batch(
            tokenizer,
            ["prompt"],
            ["answer"],
            task,
            max_length=32,
            device=torch.device("cpu"),
        )
        target = [ord(character) + 2 for character in "answer"] + [1]
        sequence_length = int(encoded["attention_mask"].sum())
        self.assertEqual(
            labels[0, sequence_length - len(target) : sequence_length].tolist(),
            target,
        )
        self.assertTrue((labels[0, : sequence_length - len(target)] == -100).all())
        self.assertTrue((labels[0, sequence_length:] == -100).all())
        self.assertEqual(sequence_length, 13)
        self.assertEqual(encoded["input_ids"].shape[1] % 8, 0)

    def test_chunk_collation_pads_inputs_and_labels_independently(self) -> None:
        chunks = [
            (
                {
                    "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
                },
                torch.tensor([[7, 8], [9, -100]]),
            ),
            (
                {
                    "input_ids": torch.tensor([[6, 7, 8, 9, 10]]),
                    "attention_mask": torch.ones((1, 5), dtype=torch.long),
                },
                torch.tensor([[11, 12, 13, 14]]),
            ),
        ]
        encoded, labels, token_count = collate_encoded_batches(
            chunks,
            pad_token_id=0,
            pin_memory=False,
        )
        self.assertEqual(encoded["input_ids"].shape, (3, 5))
        self.assertEqual(labels.shape, (3, 4))
        self.assertEqual(encoded["attention_mask"].sum().item(), 10)
        self.assertEqual(token_count, 7)
        self.assertEqual(labels[0].tolist(), [7, 8, -100, -100])

    def test_nllb_smoothed_loss_does_not_pass_labels_to_model(self) -> None:
        class Model:
            def __init__(self):
                self.call = None
                self.config = SimpleNamespace(
                    pad_token_id=0,
                    decoder_start_token_id=2,
                )

            def __call__(self, **kwargs):
                self.call = kwargs
                return SimpleNamespace(logits=torch.randn(1, 3, 5))

        model = Model()
        labels = torch.tensor([[1, 2, -100]])
        encoded = {
            "input_ids": torch.tensor([[3, 4, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
        loss = compute_training_loss(
            model,
            encoded,
            labels,
            model_family="nllb",
            label_smoothing=0.1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertNotIn("labels", model.call)
        self.assertIn("decoder_input_ids", model.call)
        self.assertEqual(model.call["decoder_input_ids"].tolist(), [[2, 1, 2]])

    def test_checkpoint_clone_excludes_optimizer_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "resume"
            source.mkdir()
            (source / "model.safetensors").write_bytes(b"weights")
            (source / "tokenizer.json").write_text("{}", encoding="utf-8")
            (source / "trainer_state.pt").write_bytes(b"optimizer")
            (source / "experiment_metadata.json").write_text("{}", encoding="utf-8")
            destination = root / "best"
            clone_model_checkpoint(source, destination, {"step": 10000})
            self.assertEqual((destination / "model.safetensors").read_bytes(), b"weights")
            self.assertFalse((destination / "trainer_state.pt").exists())
            self.assertEqual(
                json.loads((destination / "experiment_metadata.json").read_text()),
                {"step": 10000},
            )

    def test_text_model_allows_only_gemma_image_token_outside_embeddings(self) -> None:
        class Tokenizer:
            def __len__(self):
                return 11

            def get_added_vocab(self):
                return {"<image_soft_token>": 10}

        model = SimpleNamespace(get_input_embeddings=lambda: SimpleNamespace(num_embeddings=10))
        milmmt.validate_model_tokenizer(model, Tokenizer())

        class BadTokenizer(Tokenizer):
            def get_added_vocab(self):
                return {"<unexpected>": 10}

        with self.assertRaisesRegex(SystemExit, "unexpected mismatch"):
            milmmt.validate_model_tokenizer(model, BadTokenizer())


class TokenizerAuditTests(unittest.TestCase):
    def test_milmmt_audit_reports_sequence_truncation(self) -> None:
        class Tokenizer:
            unk_token_id = 99

            def __call__(self, text, **_kwargs):
                return {
                    "input_ids": [
                        99 if character == "?" else ord(character) for character in str(text) if not character.isspace()
                    ]
                }

            def tokenize(self, word):
                return list(word)

        frame = pd.DataFrame(
            {
                "lang_code": ["ami"],
                "row_id": ["ami-1"],
                "source": ["repo/example.xml"],
                "formosan_sentence": ["ab cd"],
                "english_sentence": ["xy"],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "audit.json"
            with (
                mock.patch.object(
                    tokenizer_audit.AutoTokenizer,
                    "from_pretrained",
                    return_value=Tokenizer(),
                ),
                mock.patch.object(
                    tokenizer_audit,
                    "read_parallel_csv",
                    return_value=frame,
                ),
            ):
                report = tokenizer_audit.audit_tokenizer(
                    tokenizer_dir=Path("tokenizer"),
                    input_csv=Path("corpus.csv"),
                    output_json=output,
                    output_csv=None,
                    max_rows_per_lang=0,
                    model_family="milmmt",
                    direction="f2en",
                    max_length=10,
                )

        amis = next(row for row in report["languages"] if row["lang_code"] == "ami")
        self.assertEqual(amis["pieces_per_sentence"], 4.0)
        self.assertEqual(amis["formosan_to_target_piece_ratio"], 2.0)
        self.assertEqual(amis["training_examples_over_max_length"], 1)
        self.assertEqual(
            report["training_examples_over_max_length"][0]["row_id"],
            "ami-1",
        )
