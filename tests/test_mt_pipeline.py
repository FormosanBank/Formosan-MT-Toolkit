from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))
sys.path.insert(0, str(ROOT / "formosan_mt_experiments/scripts"))

import milmmt_runtime as milmmt  # noqa: E402
import nllb_runtime as nllb  # noqa: E402
import tokenizer_audit  # noqa: E402
from audit_corpus_exposure import (  # noqa: E402
    audit_direction,
    build_tame_config,
    gate_errors,
)
from build_experiment_splits import (  # noqa: E402
    GroupCandidate,
    NgramSimilarityIndex,
    block_evaluation_conflicts_with_training,
    build_hard_split,
    choose_groups,
    exclude_test_conflicts_with_validation,
    fill_language_shortfalls,
    ngram_candidate_conflicts,
    one_edit_conflicts,
    split_targets,
)
from columnar_cache import read_csv_or_columnar, write_columnar_cache  # noqa: E402
from experiment_config import (  # noqa: E402
    DEFAULT_PROFILE,
    MILMMT_PROFILE,
    SHARED_SPLIT_FIELDS,
    load_corpus_pipeline_config,
    load_profile,
)
from formosan_mt_inference import normalize_formosan  # noqa: E402
from mt_common import (  # noqa: E402
    add_normalized_columns,
    evaluation_candidate_mask,
    source_bucket,
    source_corpus,
    special_tokens_from_corpus,
    weighted_apportioned_counts,
)
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
    nllb_usage,
    public_metrics,
    render_card,
    validate_checkpoint,
)
from setup_formosan_nllb200 import realign_embeddings  # noqa: E402
from train_directional import metric_improved, metric_value, training_source_texts  # noqa: E402
from training_code_inventory import build_code_inventory  # noqa: E402
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
    row_type = output.get(
        "row_type",
        pd.Series("sentence", index=output.index),
    ).astype(str)
    if "xml_unit_context" not in output:
        output["xml_unit_context"] = row_type.map(
            {
                "sentence": "sentence",
                "lexeme": "standalone_word",
                "morpheme": "standalone_morpheme",
            }
        ).fillna("unknown")
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
    output["mt_eval_eligible"] = output.get("pivot_origin", pd.Series("original", index=output.index)).astype(str).eq(
        "original"
    ) & output.get("row_type", pd.Series("sentence", index=output.index)).astype(str).eq("sentence")
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
            "source_bucket": "narrative",
            "dialect": "Coastal",
        }
        expected = {
            ("f2en", "english"): (
                "<to_eng> <src_ami> <dom_narrative> <dialect_coastal>",
                ("ami_Latn", "eng_Latn"),
            ),
            ("en2f", "english"): (
                "<to_ami> <src_eng> <dom_narrative> <dialect_coastal>",
                ("eng_Latn", "ami_Latn"),
            ),
            ("f2zh", "chinese"): (
                "<to_zh> <src_ami> <dom_narrative> <dialect_coastal>",
                ("ami_Latn", "zho_Hant"),
            ),
            ("zh2f", "chinese"): (
                "<to_ami> <src_zh> <dom_narrative> <dialect_coastal>",
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
                "<dom_unknown>": len(tokenizer.vocab),
                "<dom_narrative>": len(tokenizer.vocab) + 1,
                "<dialect_default>": len(tokenizer.vocab) + 2,
                "<dialect_coastal>": len(tokenizer.vocab) + 3,
            }
        )
        frame = pd.DataFrame(
            {
                "source_bucket": ["narrative", "held_out_source", None],
                "dialect": ["Coastal", "Held Out", ""],
            }
        )
        normalized, report = nllb.normalize_control_metadata(
            frame,
            tokenizer,
        )
        self.assertEqual(
            normalized["source_bucket"].tolist(),
            ["narrative", "unknown", "unknown"],
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

    def test_training_metadata_dropout_builds_default_control_paths(self) -> None:
        batch = pd.DataFrame(
            {
                "lang_code": ["ami", "ami"],
                "source_bucket": ["narrative", "culture"],
                "dialect": ["Coastal", "Coastal"],
                "formosan_sentence": ["mako ko loma niyam.", "mako ko loma nira."],
                "english_sentence": ["This is our house.", "This is their house."],
            }
        )
        args = SimpleNamespace(
            use_tags=True,
            domain_tag_dropout=0.5,
            dialect_tag_dropout=0.5,
            direction="f2en",
            target_col="english_sentence",
            target_lang="english",
        )
        with mock.patch(
            "train_directional.np.random.random",
            side_effect=[np.array([0.1, 0.9]), np.array([0.9, 0.1])],
        ):
            texts, counts = training_source_texts(
                batch,
                args=args,
            )
        self.assertEqual(
            texts,
            [
                "<to_eng> <src_ami> <dom_unknown> <dialect_coastal> mako ko loma niyam.",
                "<to_eng> <src_ami> <dom_culture> <dialect_default> mako ko loma nira.",
            ],
        )
        self.assertEqual(counts, {"domain": 1, "dialect": 1})


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
            "source_bucket": "narrative",
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

    def test_causal_labels_mask_the_prompt(self) -> None:
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 1

            def __call__(self, text, **_kwargs):
                return {"input_ids": [ord(character) + 2 for character in text]}

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
        self.assertEqual(labels[0, -len(target) :].tolist(), target)
        self.assertTrue((labels[0, : -len(target)] == -100).all())
        self.assertEqual(encoded["attention_mask"].sum().item(), 13)

    def test_text_model_allows_only_gemma_image_token_outside_embeddings(self) -> None:
        class Tokenizer:
            def __len__(self):
                return 11

            def get_added_vocab(self):
                return {"<image_soft_token>": 10}

        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(num_embeddings=10)
        )
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
                        99 if character == "?" else ord(character)
                        for character in str(text)
                        if not character.isspace()
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


class LeakageTests(unittest.TestCase):
    def test_source_bucket_uses_only_coarse_domains(self) -> None:
        self.assertEqual(
            source_bucket("FormosanBank/Corpora/NTUFormosanCorpus/XML/Stories/Amis/a.xml"),
            "narrative",
        )
        self.assertEqual(
            source_bucket("Formosan-Zheng-ACL-2024/Final_XML/Atayal/parallel.xml"),
            "linguistic",
        )
        self.assertEqual(
            source_bucket("Formosan-Glosbe/Final_XML/Amis/entries.xml"),
            "dictionary",
        )
        tokens = special_tokens_from_corpus(pd.DataFrame({"source_bucket": ["Formosan-Zheng-ACL-2024"]}))
        self.assertNotIn("<dom_formosan_zheng_acl_2024>", tokens)
        self.assertIn("<dom_unknown>", tokens)

    def test_source_corpus_preserves_exact_public_and_private_identity(self) -> None:
        self.assertEqual(
            source_corpus("FormosanBank/Corpora/NTUFormosanCorpus/XML/Stories/Amis/a.xml"),
            "NTUFormosanCorpus",
        )
        self.assertEqual(
            source_corpus("Formosan-Zheng-ACL-2024/Final_XML/Atayal/parallel.xml"),
            "Formosan-Zheng-ACL-2024",
        )
        self.assertEqual(
            source_corpus("PrivateRepo/XML/Bunun/examples.xml"),
            "PrivateRepo",
        )

    def test_group_selection_reserves_the_other_evaluation_split(self) -> None:
        candidates = [
            GroupCandidate(
                group_id=group_id,
                eligible_rows=rows,
                total_rows=rows,
                non_eval_rows=0,
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
        validation_rows = sum(candidate.eligible_rows for candidate in candidates if candidate.group_id in validation)
        remaining = [candidate for candidate in candidates if candidate.group_id not in validation]
        test = choose_groups(
            remaining,
            98,
            reserve_rows=0,
            seed=43,
            attempts=20,
        )
        test_rows = sum(candidate.eligible_rows for candidate in remaining if candidate.group_id in test)

        self.assertGreaterEqual(validation_rows, 33)
        self.assertGreaterEqual(test_rows, 98)

    def test_one_character_variants_conflict_with_evaluation(self) -> None:
        evaluation = pd.DataFrame({"lang_code": ["ami"], "text": ["malikoda"]}, index=[100])
        training = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "ami", "bnn"],
                "text": ["malikod", "malikooda", "malikada", "faraway", "malikada"],
            },
            index=[1, 2, 3, 4, 5],
        )
        self.assertEqual(one_edit_conflicts(training, evaluation, "text"), {1, 2, 3})

    def test_one_edit_self_comparison_only_marks_real_neighbors(self) -> None:
        rows = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami"],
                "text": ["malikoda", "malikoda", "unrelated"],
            },
            index=[1, 2, 3],
        )
        self.assertEqual(
            one_edit_conflicts(
                rows,
                rows,
                "text",
                ignore_same_index=True,
            ),
            {1, 2},
        )

    def test_streaming_ngram_join_finds_conflicts_with_either_side_smaller(self) -> None:
        base = "abcdefghijklmnopqrstuvwxyz" * 4
        near = base[:50] + "X" + base[51:]
        far = "zyxwvutsrqponmlkjihgfedcba" * 4
        reference = pd.DataFrame(
            {"lang_code": ["ami"], "text": [base]},
            index=[100],
        )
        candidates = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami"],
                "text": [near, far, f"{far} extra"],
            },
            index=[1, 2, 3],
        )
        self.assertEqual(
            ngram_candidate_conflicts(
                reference,
                candidates,
                "text",
                by_language=False,
                threshold=0.82,
            ),
            {1},
        )
        self.assertEqual(
            ngram_candidate_conflicts(
                candidates,
                reference,
                "text",
                by_language=False,
                threshold=0.82,
            ),
            {100},
        )

    def test_cached_ngram_index_matches_streaming_join(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami", "ami", "ami", "bnn", "bnn"],
                "text": [
                    "abcdefghijklmnopqrstuvwxyz",
                    "abcdefghijklmnopqrstuvwxzz",
                    "completely unrelated value",
                    "abcdefghijklmnopqrstuvwxyz",
                    "another unrelated sentence",
                ],
            },
            index=[10, 11, 12, 13, 14],
        )
        reference = frame.loc[[10, 14]]
        candidates = frame.loc[[11, 12, 13]]
        for by_language, expected in ((False, {11, 13}), (True, {11})):
            index = NgramSimilarityIndex(
                frame,
                "text",
                by_language=by_language,
                threshold=0.82,
            )
            self.assertEqual(
                index.conflicts(reference.index, candidates.index),
                expected,
            )

    def test_ngram_index_matches_tame_high_exposure_boundary(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ckv", "ckv", "ssf", "ssf"],
                "text": [
                    "sessenan ti iku tu mai tu qunqunian temanan ti iku.",
                    "sessenan iku tu mai tu qunqunian temanan ti iku.",
                    "thithu palalawan sa suma a pakikalhian sa suma sa lalawa.",
                    "thithu palalawan sa suma a pakikalhian sa lalawa.",
                ],
            }
        )
        index = NgramSimilarityIndex(
            frame,
            "text",
            by_language=True,
            threshold=0.95,
        )

        self.assertEqual(index.conflicts(pd.Index([1, 3]), pd.Index([0, 2])), {0, 2})

    def test_split_targets_use_all_pairs_as_denominator(self) -> None:
        self.assertEqual(
            split_targets(
                total_rows=1_000,
                eligible_total=1_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (100, 50),
        )
        self.assertEqual(
            split_targets(
                total_rows=20_000,
                eligible_total=20_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (2_000, 1_000),
        )
        self.assertEqual(
            split_targets(
                total_rows=20_000,
                eligible_total=5_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (2_000, 1_000),
        )
        with self.assertRaisesRegex(ValueError, "requires 3,000"):
            split_targets(
                total_rows=20_000,
                eligible_total=2_999,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            )

    def test_weighted_apportionment_redistributes_capacity_shortfall(self) -> None:
        self.assertEqual(
            weighted_apportioned_counts(
                {"lexical": 900, "narrative": 100},
                {"lexical": 20, "narrative": 100},
                100,
            ),
            {"lexical": 20, "narrative": 80},
        )

    def test_source_domain_does_not_override_row_eligibility(self) -> None:
        row = self.hard_split_fixture().iloc[[0]].copy()
        row["source"] = "Formosan-ILRDF-42-Language-Practice-Word-Lists/Final_XML/Atayal/word-list.xml"
        row["formosan_sentence"] = "one two three four"
        row["english_sentence"] = "first second third fourth"
        normalized = add_normalized_columns(
            row,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=4,
            min_target_tokens=4,
            min_combined_tokens=8,
            min_punctuated_combined_tokens=8,
        )
        self.assertTrue(bool(candidates.iloc[0]))

        normalized.loc[normalized.index[0], "row_type"] = "lexeme"
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=4,
            min_target_tokens=4,
            min_combined_tokens=8,
            min_punctuated_combined_tokens=8,
        )
        self.assertFalse(bool(candidates.iloc[0]))

    def test_evaluation_candidate_respects_model_length_limit(self) -> None:
        row = self.hard_split_fixture().iloc[[0]].copy()
        row["formosan_sentence"] = " ".join(["word"] * 385)
        row["english_sentence"] = "a complete translation"
        normalized = add_normalized_columns(
            row,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=2,
            min_target_tokens=2,
            min_combined_tokens=6,
            min_punctuated_combined_tokens=5,
            max_eval_units_per_side=384,
        )
        self.assertFalse(bool(candidates.iloc[0]))

    def test_hard_split_rejects_training_length_overflow(self) -> None:
        raw = self.hard_split_fixture()
        long_source = " ".join(["word"] * 385)
        raw.loc[raw.index[0], "formosan_sentence"] = long_source
        raw.loc[raw.index[0], "formosan_mt_standard"] = long_source
        with self.assertRaisesRegex(SystemExit, "training limit"):
            build_hard_split(
                raw,
                target_col="english_sentence",
                test_ratio=0.2,
                val_ratio=0.1,
                seed=42,
                min_formosan_tokens=2,
                min_target_tokens=2,
                min_combined_tokens=6,
                min_punctuated_combined_tokens=5,
                attempts=10,
                min_test_rows=0,
                min_validate_rows=0,
                ngram_threshold=0.82,
                registry_in=None,
            )

    def test_compact_sentence_eligibility_uses_joint_content(self) -> None:
        frame = add_mt_contract(
            pd.DataFrame(
                [
                    {
                        "row_id": "balanced",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su",
                        "english_sentence": "Please come here",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "short-question",
                        "lang_code": "ami",
                        "formosan_sentence": "“ima su?”",
                        "english_sentence": "\"Who are you?\"",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "short-fragment",
                        "lang_code": "ami",
                        "formosan_sentence": "ima su",
                        "english_sentence": "who are you",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "compact-long-target",
                        "lang_code": "ami",
                        "formosan_sentence": "ima su?",
                        "english_sentence": "Who are you today?",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "single-unit",
                        "lang_code": "ami",
                        "formosan_sentence": "millemungku",
                        "english_sentence": "I have got a parasol",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "definition",
                        "lang_code": "ami",
                        "formosan_sentence": "aru ku su",
                        "english_sentence": "a long lexical explanation",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "definition_like_sentence",
                    },
                    {
                        "row_id": "uncertain-language",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su anini",
                        "english_sentence": "Heavy rainfall flooded downtown streets",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "english_language_uncertain",
                    },
                    {
                        "row_id": "unbalanced-target",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su anini",
                        "english_sentence": 'He said "please come here',
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "unbalanced_target_delimiters",
                    },
                ]
            )
        )
        normalized = add_normalized_columns(
            frame,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=2,
            min_target_tokens=2,
            min_combined_tokens=6,
            min_punctuated_combined_tokens=5,
        )

        self.assertEqual(
            set(normalized.loc[candidates, "row_id"]),
            {"balanced", "compact-long-target"},
        )

    def test_split_conflicts_block_the_full_similarity_neighborhood(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami"] * 4,
                "_formosan_skeleton": [
                    "abcdefghijklmnopqrstuvwx",
                    "abcdefghijklmnopqrstuvwy",
                    "abcdefghijklmnopqrstuvwz",
                    "completelydifferentstring",
                ],
                "_target_skeleton": [
                    "thevalidationtranslationx",
                    "thevalidationtranslationy",
                    "thevalidationtranslationz",
                    "unrelatedtargettranslation",
                ],
            }
        )
        group_ids = pd.Series(range(len(frame)), index=frame.index, dtype="int64")

        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {0: "validate", 1: "test"}
        blocked: set[int] = set()
        result = exclude_test_conflicts_with_validation(
            frame,
            pd.Series(["validate", "test", "train", "train"]),
            group_ids,
            candidate_mask,
            assignments,
            blocked,
            ngram_threshold=0.82,
            include_one_edit=False,
        )
        self.assertGreater(result["blocked_candidate_rows"], 1)
        self.assertEqual(blocked, {1, 2})
        self.assertNotIn(1, assignments)
        self.assertTrue(bool(candidate_mask.iloc[3]))

        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {1: "test"}
        blocked = set()
        result = block_evaluation_conflicts_with_training(
            frame,
            pd.Series(["train", "test", "train", "train"]),
            group_ids,
            candidate_mask,
            assignments,
            blocked,
            ngram_threshold=0.82,
            include_one_edit=False,
        )
        self.assertGreater(result["blocked_candidate_rows"], 1)
        self.assertEqual(blocked, {0, 1, 2})
        self.assertNotIn(1, assignments)
        self.assertTrue(bool(candidate_mask.iloc[3]))

    @staticmethod
    def hard_split_fixture() -> pd.DataFrame:
        rows = []
        for document in range(60):
            for sentence in range(3):
                digest = hashlib.sha256(f"{document}:{sentence}".encode()).hexdigest()
                rows.append(
                    {
                        "row_id": f"human-{document}-{sentence}",
                        "source_record_id": f"record-{document}-{sentence}",
                        "lang_code": "ami",
                        "formosan_sentence": (
                            f"{digest[:8]} {digest[8:16]} {digest[16:24]} {digest[24:32]} {digest[32:40]}"
                        ),
                        "english_sentence": (
                            f"{digest[40:48]} {digest[48:56]} the sentence "
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

    def test_hard_split_prefers_human_rows_and_excludes_lexemes(self) -> None:
        raw = self.hard_split_fixture()
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, excluded, duplicates, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertTrue(evaluation["row_type"].eq("sentence").all())
        self.assertFalse(evaluation["pivot_origin"].eq("synthetic").any())
        self.assertEqual(report["lexical_like_eval_rows"], 0)
        self.assertGreaterEqual(report["document_overlap_train_eval"], 0)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(excluded), 0)
        self.assertTrue(report["complete"])
        language = report["languages"]["ami"]
        self.assertEqual(language["rows_total"], 300)
        self.assertEqual(language["test_rows"], 30)
        self.assertEqual(language["validate_rows"], 15)
        self.assertGreaterEqual(language["test_fraction_of_all_input_rows"], 0.10)
        self.assertGreaterEqual(
            language["validate_fraction_of_all_input_rows"],
            0.05,
        )

        provenance = validate_provenance(output)
        validation = validate_splits(
            output,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            split_report=report,
        )
        self.assertTrue(provenance["ok"], provenance)
        self.assertTrue(validation["ok"], validation)
        self.assertFalse(validation["split_report_errors"])

        contaminated = output.copy()
        contaminated["translation_kind"] = ""
        contaminated.loc[contaminated.index[0], "translation_kind"] = "interlinear-gloss"
        contaminated_validation = validate_splits(
            contaminated,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
        )
        self.assertFalse(contaminated_validation["ok"])
        self.assertEqual(contaminated_validation["gloss_translation_rows"], 1)

        lexical_contamination = output.copy()
        lexeme_index = lexical_contamination.index[
            lexical_contamination["row_type"].eq("lexeme")
        ][0]
        lexical_contamination.loc[lexeme_index, "english_sentence"] = "wash-face"
        lexical_validation = validate_splits(
            lexical_contamination,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
        )
        self.assertFalse(lexical_validation["ok"])
        self.assertEqual(lexical_validation["lexical_quality_rows"], 1)

    def test_hard_split_is_deterministic_for_identical_input(self) -> None:
        keyed = add_normalized_columns(
            self.hard_split_fixture(),
            target_col="english_sentence",
            target_lang="english",
        )
        kwargs = {
            "target_col": "english_sentence",
            "test_ratio": 0.10,
            "val_ratio": 0.05,
            "seed": 42,
            "min_formosan_tokens": 1,
            "min_target_tokens": 1,
            "min_combined_tokens": 2,
            "min_punctuated_combined_tokens": 2,
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

    def test_hard_split_apportions_each_source_corpus(self) -> None:
        raw = self.hard_split_fixture()
        document_numbers = raw["source"].str.extract(r"doc-(\d+)")[0]
        first_source = document_numbers.fillna("0").astype(int).lt(30)
        raw.loc[first_source, "source"] = raw.loc[first_source, "source"].str.replace(
            "/Test/", "/SourceA/", regex=False
        )
        raw.loc[~first_source, "source"] = raw.loc[~first_source, "source"].str.replace(
            "/Test/", "/SourceB/", regex=False
        )
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        _, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        for source in ("SourceA", "SourceB"):
            values = report["source_strata"]["ami"][source]
            self.assertEqual(values["test_rows"], values["target_test_rows"])
            self.assertEqual(
                values["validate_rows"],
                values["target_validate_rows"],
            )
        self.assertLess(
            report["source_distribution_total_variation"]["ami"]["test"],
            0.025,
        )

    def test_language_shortfall_refill_respects_source_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami"] * 12,
                "_source_corpus": ["SourceA"] * 6 + ["SourceB"] * 6,
                "_formosan_tokens": [100] * 6 + [10] * 6,
                "_target_tokens": [100] * 6 + [10] * 6,
                "pivot_origin": ["original"] * 12,
            }
        )
        group_ids = pd.Series(range(12), index=frame.index, dtype="int64")
        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {0: "test", 1: "test"}

        fill_language_shortfalls(
            frame,
            group_ids,
            candidate_mask,
            {
                ("ami", "SourceA"): (2, 0),
                ("ami", "SourceB"): (4, 0),
            },
            {"ami": (6, 0)},
            assignments,
            seed=42,
            attempts=20,
        )

        split = group_ids.map(assignments).fillna("train")
        self.assertEqual(int(split.iloc[:6].eq("test").sum()), 2)
        self.assertEqual(int(split.iloc[6:].eq("test").sum()), 4)

    def test_hard_split_redistributes_lexical_source_shortfall(self) -> None:
        raw = self.hard_split_fixture()
        source_a = raw.index[:150]
        raw.loc[source_a, "source"] = "FormosanBank/Corpora/SourceA/XML/lexical.xml"
        raw.loc[source_a, "row_type"] = "lexeme"
        raw.loc[~raw.index.isin(source_a), "source"] = (
            "FormosanBank/Corpora/SourceB/XML/sentences.xml"
        )
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
            registry_in=None,
        )

        source_a_report = report["source_strata"]["ami"]["SourceA"]
        source_b_report = report["source_strata"]["ami"]["SourceB"]
        self.assertEqual(source_a_report["target_test_rows"], 0)
        self.assertEqual(source_a_report["target_validate_rows"], 0)
        self.assertEqual(
            source_b_report["target_test_rows"]
            + source_b_report["target_validate_rows"],
            45,
        )
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertTrue(evaluation["row_type"].eq("sentence").all())

    def test_synthetic_only_source_is_train_only(self) -> None:
        raw = self.hard_split_fixture()
        synthetic = raw["pivot_origin"].eq("synthetic")
        raw.loc[synthetic, "source"] = (
            "FormosanBank/Corpora/PivotOnly/XML/synthetic.xml"
        )
        raw.loc[~synthetic, "source"] = (
            "FormosanBank/Corpora/Human/XML/human.xml"
        )
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.95,
            registry_in=None,
        )

        pivot_source = report["source_strata"]["ami"]["PivotOnly"]
        self.assertEqual(pivot_source["target_test_rows"], 0)
        self.assertEqual(pivot_source["target_validate_rows"], 0)
        self.assertEqual(pivot_source["synthetic_input_rows"], 60)
        self.assertTrue(
            output.loc[output["pivot_origin"].eq("synthetic"), "split"]
            .eq("train")
            .all()
        )
        self.assertEqual(report["split_counts"]["test"], 30)
        self.assertEqual(report["split_counts"]["validate"], 15)
        self.assertEqual(report["synthetic_eval_rows"], 0)
        validation = validate_splits(
            output,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.95,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            split_report=report,
        )
        self.assertTrue(validation["ok"], validation)

    def test_hard_split_fails_when_human_evaluation_capacity_is_too_small(self) -> None:
        raw = self.hard_split_fixture()
        sentence_indexes = raw.index[raw["row_type"].eq("sentence")]
        raw.loc[sentence_indexes[2:], "pivot_origin"] = "synthetic"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        with self.assertRaisesRegex(SystemExit, "eligible sentences"):
            build_hard_split(
                keyed,
                target_col="english_sentence",
                test_ratio=0.10,
                val_ratio=0.05,
                seed=42,
                min_formosan_tokens=1,
                min_target_tokens=1,
                min_combined_tokens=2,
                min_punctuated_combined_tokens=2,
                attempts=20,
                min_test_rows=5,
                min_validate_rows=2,
                ngram_threshold=0.82,
                registry_in=None,
            )

    def test_unsafe_evaluation_candidates_are_replaced_not_excluded(self) -> None:
        raw = self.hard_split_fixture()
        near_synthetic = []
        human = raw[
            raw["pivot_origin"].eq("original")
            & raw["row_type"].eq("sentence")
        ].head(20)
        for _, row in human.iterrows():
            near_synthetic.append(
                {
                    **row.to_dict(),
                    "row_id": f"near-{row['row_id']}",
                    "source_record_id": (f"near-{row['source_record_id']}"),
                    "formosan_sentence": (f"{row['formosan_sentence']}x"),
                    "english_sentence": (f"{row['english_sentence']}x"),
                    "source": (f"FormosanBank/Corpora/Pivot/XML/near-{row['row_id']}.xml"),
                    "xml_path": (f"Corpora/Pivot/XML/near-{row['row_id']}.xml"),
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
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        self.assertTrue(report["complete"])
        language = report["languages"]["ami"]
        self.assertLess(
            language["group_safe_human_sentence_rows"],
            language["eligible_human_sentence_rows"],
        )
        self.assertTrue(excluded.empty)
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertFalse(evaluation["pivot_origin"].eq("synthetic").any())
        self.assertTrue(
            output[output["pivot_origin"].eq("synthetic")]["split"]
            .eq("train")
            .all()
        )
        self.assertEqual(len(output), report["deduplicated_input_rows"])
        self.assertGreaterEqual(
            language["test_fraction_of_all_input_rows"],
            0.10,
        )
        self.assertGreaterEqual(
            language["validate_fraction_of_all_input_rows"],
            0.05,
        )

    def test_validate_test_conflicts_are_reallocated_across_sources(self) -> None:
        raw = self.hard_split_fixture()
        source_a = raw["source"].str.contains(r"doc-(?:[0-9]|1[0-4])\.xml", regex=True)
        raw.loc[source_a, "source"] = "FormosanBank/Corpora/SourceA/XML/a.xml"
        raw.loc[~source_a, "source"] = "FormosanBank/Corpora/SourceB/XML/b.xml"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            attempts=20,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        language = report["languages"]["ami"]
        self.assertEqual(language["test_rows"], language["target_test_rows"])
        self.assertEqual(
            language["validate_rows"],
            language["target_validate_rows"],
        )
        self.assertFalse(report["ratio_shortfalls"])
        self.assertFalse(report["source_ratio_shortfalls"])
        for source in report["source_strata"]["ami"].values():
            self.assertEqual(source["test_rows"], source["target_test_rows"])
            self.assertEqual(
                source["validate_rows"],
                source["target_validate_rows"],
            )
        self.assertTrue(
            output[output["split"].isin({"test", "validate"})]["row_type"]
            .eq("sentence")
            .all()
        )

    def test_independent_validator_reports_synthetic_and_document_overlap(self) -> None:
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
        self.assertFalse(validation["ok"], validation)
        self.assertEqual(validation["synthetic_eval_rows"], 1)
        self.assertFalse(validation["synthetic_eval_allowed"])
        self.assertGreater(
            validation["train_evaluation"]["document_overlap"],
            0,
        )
        diagnostic = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            require_human_eval=False,
        )
        self.assertTrue(diagnostic["ok"], diagnostic)

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
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
        )

        self.assertTrue(validation["ok"], validation)
        self.assertEqual(
            validation["validate_test"]["exact_overlap"]["target"],
            0,
        )
        self.assertEqual(
            validation["validate_test_cross_language_diagnostic"]["exact_overlap"]["target"],
            1,
        )


class TrainingMetricTests(unittest.TestCase):
    def test_perfect_generation_metrics_and_diagnostics(self) -> None:
        metrics = score_translations(
            ["hello world", "talima"],
            ["hello world", "talima"],
            sources=["different", "  TALIMA  "],
        )
        self.assertAlmostEqual(metrics["chrF2"], 100.0)
        self.assertAlmostEqual(metrics["exact_match_rate"], 1.0)
        self.assertAlmostEqual(metrics["empty_output_rate"], 0.0)
        self.assertAlmostEqual(metrics["character_length_ratio"], 1.0)
        self.assertAlmostEqual(metrics["source_copy_rate"], 0.5)

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
        self.assertEqual(
            profile["training_defaults"]["validation_metadata_mode"],
            "default",
        )
        self.assertEqual(len(profile["base_model"]["revision"]), 40)

    def test_experiment_profiles_match_canonical_split_policy(self) -> None:
        pipeline_splits = load_corpus_pipeline_config()["splits"]
        profile_splits = load_profile(DEFAULT_PROFILE)["splits"]
        self.assertEqual(
            {field: profile_splits[field] for field in SHARED_SPLIT_FIELDS},
            {field: pipeline_splits[field] for field in SHARED_SPLIT_FIELDS},
        )

    def test_experiment_profile_rejects_split_policy_drift(self) -> None:
        profile = json.loads(DEFAULT_PROFILE.read_text(encoding="utf-8"))
        profile["splits"]["test_ratio"] = 0.05
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
        self.assertFalse(profile["training_defaults"]["use_tags"])
        self.assertEqual(profile["generation_defaults"]["beam"], 4)
        comparison = profile["comparison"]
        self.assertEqual(
            comparison["sample_presentations"],
            profile["training_defaults"]["steps"]
            * profile["training_defaults"]["effective_batch_size"],
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
        self.assertEqual(
            payload["by_split"]["test"]["by_source_corpus"]["unknown"]["rows"],
            4,
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
        self.assertIn('--time="${EVAL_TIME:-08:00:00}"', submitter)
        self.assertNotIn("for checkpoint in final best", submitter)

        bootstrap = (ROOT / "formosan_mt_experiments/slurm/bootstrap_metrics.sl").read_text(encoding="utf-8")
        self.assertIn("--cpus-per-task=8", bootstrap)
        self.assertNotIn("--gres", bootstrap)

    def test_evaluator_checkpoints_outputs_before_bootstrap(self) -> None:
        evaluator = (ROOT / "formosan_mt_experiments/scripts/evaluate_directional.py").read_text(encoding="utf-8")
        predictions_write = evaluator.index("predictions.to_csv(args.output_csv, index=False)")
        completed = evaluator.index('metrics["complete"] = True')
        metrics_write = evaluator.index("write_json(args.output_json, metrics)")
        bootstrap = evaluator.index("bootstrap_confidence_intervals(")

        self.assertLess(predictions_write, completed)
        self.assertLess(completed, metrics_write)
        self.assertLess(metrics_write, bootstrap)
        self.assertIn("if args.bootstrap_samples > 0:", evaluator)

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
        self.assertNotIn("normalize_formosan", major_source)

    def test_publication_card_records_release_and_hard_test_contract(self) -> None:
        profile = {
            "recipe_id": "nllb200-spm8k-directional-v3",
            "base_model": {"name": "base/model", "revision": "abc123"},
            "mt_standardization": {"id": "formosan-mt-standard-v3"},
            "training_defaults": {
                "effective_batch_size": 64,
                "max_length": 384,
                "learning_rate": 2e-5,
                "precision": "bf16",
                "best_metric": "chrF2",
            },
        }
        metrics = {
            "global": {
                "BLEU": 9.1,
                "chrF2": 27.2,
                "TER": 90.3,
                "empty_output_rate": 0.0,
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
        self.assertIn("capacity-aware source", card)
        self.assertIn("Synthetic pivots and\nlexical entries are train-only", card)
        self.assertIn("FormosanBank/formosan-mt-private", card)
        self.assertIn("`20260809-210523`", card)
        self.assertIn("nllb-200", card)

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
                "bootstrap": {},
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


if __name__ == "__main__":
    unittest.main()
