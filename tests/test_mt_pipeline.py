from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
from columnar_cache import read_csv_or_columnar, write_columnar_cache  # noqa: E402
from experiment_config import DEFAULT_PROFILE, load_profile  # noqa: E402
from formosan_mt_inference import normalize_formosan  # noqa: E402
from model_backends import get_backend, normalize_control_metadata  # noqa: E402
from mt_common import add_normalized_columns  # noqa: E402
from mt_metrics import bootstrap_confidence_intervals, score_translations  # noqa: E402
from mt_standardization import (  # noqa: E402
    DEFAULT_PROFILE_PATH as MT_STANDARD_PROFILE_PATH,
)
from mt_standardization import (
    load_profile as load_mt_standard_profile,
)
from mt_standardization import (
    profile_sha256 as mt_profile_sha256,
)
from pivot import discover_api_key_envs, parse_api_key_envs  # noqa: E402
from publish_huggingface_models import (  # noqa: E402
    DIRECTIONS,
    madlad_usage,
    nllb_usage,
    validate_checkpoint,
)
from setup_formosan_madlad400 import resize_and_initialize  # noqa: E402
from setup_formosan_nllb200 import realign_embeddings  # noqa: E402
from train_directional import metric_improved  # noqa: E402
from training_code_inventory import build_code_inventory  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402
from validate_experiment import (  # noqa: E402
    validate_provenance,
    validate_splits,
    validate_tags,
)
from write_submission_manifest import (  # noqa: E402
    build_job_graph,
    corpus_record,
    read_job_ids,
)

MT_STANDARD_PROFILE = load_mt_standard_profile(MT_STANDARD_PROFILE_PATH)
MT_STANDARD_PROFILE_HASH = mt_profile_sha256(MT_STANDARD_PROFILE_PATH)


class ColumnarCacheTests(unittest.TestCase):
    def test_round_trip_is_bound_to_canonical_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "corpus.csv"
            release = pd.DataFrame({"row_id": ["a", "b"], "value": ["x", "y"]})
            cached = release.assign(_normalized=["x", "y"])
            release.to_csv(csv_path, index=False)
            write_columnar_cache(cached, csv_path)
            pd.testing.assert_frame_equal(
                read_csv_or_columnar(csv_path, keep_default_na=False),
                cached,
            )
            csv_path.write_text("row_id,value\na,changed\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                read_csv_or_columnar(csv_path, keep_default_na=False)


def add_mt_contract(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["kindOf"] = "standard"
    output["standard_namespace"] = "formosan-mt"
    output["formosan_mt_standard"] = output["formosan_sentence"].astype(str)
    output["formosan_source_standard"] = output["formosan_sentence"].astype(str)
    output["formosan_original_raw"] = output["formosan_sentence"].astype(str)
    output["mt_standard_sha256"] = output["formosan_sentence"].map(
        lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    )
    output["source_standard_sha256"] = output["mt_standard_sha256"]
    output["mt_normalization_status"] = "accepted"
    output["mt_normalization_confidence"] = "unchanged"
    output["mt_eval_eligible"] = (
        output.get("pivot_origin", pd.Series("original", index=output.index))
        .astype(str)
        .eq("original")
        & output.get("row_type", pd.Series("sentence", index=output.index))
        .astype(str)
        .eq("sentence")
    )
    output["mt_normalization_reason"] = ""
    output["mt_standard_profile"] = MT_STANDARD_PROFILE["profile_id"]
    output["mt_standard_profile_sha256"] = MT_STANDARD_PROFILE_HASH
    return output


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


class FakeTokenizer:
    def __init__(self) -> None:
        values = [
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<2mi>",
            "<2ms>",
            "<2id>",
            "<2haw>",
            "<2sm>",
            "<2to>",
            "<2ami>",
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
        return next(
            token
            for token, token_id in self.vocab.items()
            if token_id == value
        )

    def __call__(self, *_args, **_kwargs) -> dict[str, list[int]]:
        return {"input_ids": [self.vocab["<2mi>"]]}


class ModelBackendTests(unittest.TestCase):
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
                return next(
                    token
                    for token, value in self.vocab.items()
                    if value == token_id
                )

        backend = get_backend("nllb")
        tokenizer = NllbTokenizer()
        model = SimpleNamespace(
            config=SimpleNamespace(decoder_start_token_id=None),
            generation_config=SimpleNamespace(
                decoder_start_token_id=None
            ),
        )
        backend.configure_model(model, tokenizer)
        task = backend.task_spec(
            "ami",
            "f2en",
            target_lang="english",
        )
        self.assertEqual(
            backend.generation_kwargs(tokenizer, model, task),
            {
                "forced_bos_token_id": 11,
                "decoder_start_token_id": 2,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
        )

    def test_madlad_prefixes_cover_all_directions(self) -> None:
        backend = get_backend("madlad400")
        row = {
            "lang_code": "ami",
            "source_bucket": "ntu",
            "dialect": "Coastal",
        }
        expected = {
            ("f2en", "english"): "<2en> <to_eng> <src_ami> <dom_ntu> <dialect_coastal>",
            ("en2f", "english"): "<2ami> <to_ami> <src_eng> <dom_ntu> <dialect_coastal>",
            ("f2zh", "chinese"): "<2zh_Hant> <to_zh> <src_ami> <dom_ntu> <dialect_coastal>",
            ("zh2f", "chinese"): "<2ami> <to_ami> <src_zh> <dom_ntu> <dialect_coastal>",
        }
        for (direction, target_lang), prefix in expected.items():
            self.assertEqual(
                backend.source_prefix(
                    row,
                    direction,
                    target_lang=target_lang,
                    use_tags=True,
                ),
                prefix,
            )

    def test_madlad_generation_never_forces_nllb_bos(self) -> None:
        backend = get_backend("madlad400")
        model = SimpleNamespace(
            config=SimpleNamespace(
                decoder_start_token_id=0,
                pad_token_id=1,
                eos_token_id=2,
            ),
            generation_config=SimpleNamespace(),
        )
        tokenizer = FakeTokenizer()
        backend.configure_model(model, tokenizer)
        task = backend.task_spec(
            "ami",
            "en2f",
            target_lang="english",
        )
        kwargs = backend.generation_kwargs(tokenizer, model, task)
        self.assertEqual(
            kwargs,
            {
                "decoder_start_token_id": 0,
                "eos_token_id": 2,
                "pad_token_id": 1,
            },
        )
        self.assertNotIn("forced_bos_token_id", kwargs)

    def test_tag_validation_uses_selected_backend(self) -> None:
        tokenizer = FakeTokenizer()
        backend = SimpleNamespace(
            family="madlad400",
            load_tokenizer=mock.Mock(return_value=tokenizer),
            source_prefix=mock.Mock(return_value="<2ami> <to_eng>"),
        )
        frame = pd.DataFrame(
            [{"source": "fixture.xml", "lang_code": "ami"}]
        )
        with mock.patch(
            "validate_experiment.get_backend",
            return_value=backend,
        ):
            report = validate_tags(
                frame,
                Path("tokenizer"),
                "en2f",
                "english",
                {"model_family": "madlad400"},
            )
        self.assertTrue(report["ok"])
        self.assertEqual(report["model_family"], "madlad400")
        backend.load_tokenizer.assert_called_once_with(Path("tokenizer"))

    def test_unseen_evaluation_metadata_falls_back_to_setup_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        tokenizer.vocab.update(
            {
                "<dom_unknown>": len(tokenizer.vocab),
                "<dom_ntu>": len(tokenizer.vocab) + 1,
                "<dialect_default>": len(tokenizer.vocab) + 2,
                "<dialect_coastal>": len(tokenizer.vocab) + 3,
            }
        )
        frame = pd.DataFrame(
            {
                "source_bucket": ["ntu", "held_out_source", None],
                "dialect": ["Coastal", "Held Out", ""],
            }
        )
        normalized, report = normalize_control_metadata(
            frame,
            tokenizer,
        )
        self.assertEqual(
            normalized["source_bucket"].tolist(),
            ["ntu", "unknown", "unknown"],
        )
        self.assertEqual(
            normalized["dialect"].tolist(),
            ["Coastal", "default", "default"],
        )
        self.assertEqual(
            report,
            {
                "domain_fallback_rows": 1,
                "dialect_fallback_rows": 1,
            },
        )

    def test_madlad_resize_updates_untied_input_and_output_vocab(self) -> None:
        tokenizer = FakeTokenizer()
        model = T5ForConditionalGeneration(
            T5Config(
                vocab_size=10,
                d_model=8,
                d_ff=16,
                num_layers=1,
                num_decoder_layers=1,
                num_heads=2,
                decoder_start_token_id=0,
                pad_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
            )
        )
        report = resize_and_initialize(
            model,
            tokenizer,
            ["<2ami>", "<to_eng>"],
            old_size=10,
        )
        self.assertEqual(report["new_vocab_size"], 12)
        self.assertEqual(model.get_input_embeddings().num_embeddings, 12)
        self.assertEqual(model.get_output_embeddings().out_features, 12)


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
        return add_mt_contract(pd.DataFrame(rows))

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
        raw = add_mt_contract(raw)
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
        frame = add_mt_contract(frame)
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

    def test_madlad_profile_uses_native_train_only_tokenizer(self) -> None:
        profile = load_profile(
            ROOT
            / "formosan_mt_experiments/configs/madlad400_3b_native.json"
        )
        self.assertEqual(profile["model_family"], "madlad400")
        self.assertEqual(profile["tokenizer"]["mode"], "native")
        self.assertEqual(profile["tokenizer"]["setup_splits"], ["train"])
        self.assertTrue(profile["training_defaults"]["gradient_checkpointing"])

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
                    "pivot_origin": "original",
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
                        "pivot_origin": "original",
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

    def test_tame_release_gate_conditions_retrieval_on_language_task(self) -> None:
        frame = self.frame(duplicate_test=True)
        train_indexes = frame.index[frame["split"].eq("train")]
        evaluation_indexes = frame.index[~frame["split"].eq("train")]
        frame.loc[train_indexes[6:], "lang_code"] = "bnn"
        frame.loc[evaluation_indexes, "lang_code"] = "bnn"
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            frame,
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
        self.assertEqual(
            payload["combined_evaluation"]["task_conditioning"],
            "lang_code",
        )


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
                "ratios_by_language": {
                    "ami": {"train": 90, "test": 8, "validate": 2}
                },
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
        submitter = (
            ROOT
            / "formosan_mt_experiments/slurm/submit_directional_experiment.sh"
        ).read_text(encoding="utf-8")
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
        self.assertIn('--time="${EVAL_TIME:-08:00:00}"', submitter)
        self.assertNotIn("for checkpoint in final best", submitter)

        bootstrap = (
            ROOT
            / "formosan_mt_experiments/slurm/bootstrap_metrics.sl"
        ).read_text(encoding="utf-8")
        self.assertIn("--cpus-per-task=8", bootstrap)
        self.assertNotIn("--gres", bootstrap)

    def test_evaluator_checkpoints_outputs_before_bootstrap(self) -> None:
        evaluator = (
            ROOT
            / "formosan_mt_experiments/scripts/evaluate_directional.py"
        ).read_text(encoding="utf-8")
        predictions_write = evaluator.index(
            "predictions.to_csv(args.output_csv, index=False)"
        )
        completed = evaluator.index('metrics["complete"] = True')
        metrics_write = evaluator.index("write_json(args.output_json, metrics)")
        bootstrap = evaluator.index(
            "bootstrap_confidence_intervals("
        )

        self.assertLess(predictions_write, completed)
        self.assertLess(completed, metrics_write)
        self.assertLess(metrics_write, bootstrap)
        self.assertIn("if args.bootstrap_samples > 0:", evaluator)

    def test_full_evaluation_defaults_are_resource_conservative(self) -> None:
        for profile_name in (
            "default_experiment.json",
            "madlad400_3b_native.json",
        ):
            profile = load_profile(
                ROOT / "formosan_mt_experiments/configs" / profile_name
            )
            defaults = profile["generation_defaults"]
            self.assertEqual(defaults["metadata_modes"], ["default"])
            self.assertEqual(defaults["bootstrap_samples"], 0)

    def test_nllb_setup_checksum_is_computed_then_enforced(self) -> None:
        submitter = (
            ROOT
            / "formosan_mt_experiments/slurm/submit_directional_experiment.sh"
        ).read_text(encoding="utf-8")
        setup = (
            ROOT
            / "formosan_mt_experiments/slurm/setup_spm_sweep.sl"
        ).read_text(encoding="utf-8")
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
            "formosan_mt_experiments/scripts/setup_formosan_madlad400.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/train_directional.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/columnar_cache.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/evaluate_directional.py",
            repository_paths,
        )
        self.assertIn(
            "formosan_mt_experiments/scripts/formosan_mt_inference.py",
            repository_paths,
        )
        self.assertIn("config/mt_standardization.json", repository_paths)
        self.assertIn("scripts/local/mt_standardization.py", repository_paths)
        self.assertIn(
            "formosan_mt_experiments/slurm/train_directional.sl",
            repository_paths,
        )
        self.assertTrue(all(len(row["sha256"]) == 64 for row in artifacts))

    def test_publication_examples_apply_formosan_mt_standardization(self) -> None:
        for usage_builder in (nllb_usage, madlad_usage):
            formosan_source = usage_builder(DIRECTIONS["f2en"])
            major_source = usage_builder(DIRECTIONS["en2f"])
            self.assertIn(
                "from formosan_mt_inference import normalize_formosan",
                formosan_source,
            )
            self.assertIn(
                "text = normalize_formosan(text, lang_code)",
                formosan_source,
            )
            self.assertNotIn("normalize_formosan", major_source)

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

    def test_madlad_submission_graph_uses_one_shared_setup(self) -> None:
        job_ids = {
            "validate_en": 1,
            "validate_zh": 2,
            "setup_madlad400": 3,
        }
        next_id = 4
        for direction in ("f2en", "en2f", "f2zh", "zh2f"):
            for label in (
                f"train_{direction}",
                f"eval_{direction}_final",
                f"eval_{direction}_best",
            ):
                job_ids[label] = next_id
                next_id += 1
        graph = build_job_graph(job_ids, model_family="madlad400")
        self.assertEqual(graph["setup_shared"], 3)
        self.assertNotIn("setup_en", graph)

    def test_submission_state_rejects_non_numeric_job_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            (state / "validate_en.id").write_text("2714139\n", encoding="utf-8")
            self.assertEqual(read_job_ids(state), {"validate_en": 2714139})
            (state / "validate_zh.id").write_text("not-a-job\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_job_ids(state)

if __name__ == "__main__":
    unittest.main()
