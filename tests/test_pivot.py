from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import stage_cache
from build_big_corpus import (
    discover_inputs,
)
from build_context import BuildPaths
from build_release import package_training_provenance
from pivot import (
    Direction,
    discover_api_key_envs,
    load_cache,
    load_cache_chain,
    make_cache_key,
    parse_api_key_envs,
    pivot_candidate_reason,
    synthetic_row,
    write_pivot_output,
)
from stage_cache import (
    cached_stage_valid,
    file_inventory,
    load_stage_cache,
    record_cached_stage,
)
from support import mt_contract_fields, text_sha256


class PivotContractTests(unittest.TestCase):
    def direction(self) -> Direction:
        return Direction(
            name="zh2en",
            source_path=Path("zh.csv"),
            original_target_path=Path("en.csv"),
            source_text_col="chinese_sentence",
            target_text_col="english_sentence",
            source_language="chinese",
            deepl_source_lang="ZH",
            deepl_target_lang="EN-US",
            output_filename="big_corpus_en_pivot.csv",
            cache_filename="deepl_zh_to_en.jsonl",
        )

    def source_row(self) -> pd.Series:
        return pd.Series(
            {
                **mt_contract_fields("mako ko tawki niyam."),
                "row_id": "human-row",
                "source_record_id": "source-row",
                "content_sha256": "old",
                "lang_code": "ami",
                "formosan_sentence": "mako ko tawki niyam.",
                "chinese_sentence": "今天真的很好。",
                "source": "FormosanBank/Corpora/Test/XML/sample.xml",
                "kindOf": "standard",
                "dialect": "Coastal",
                "row_type": "sentence",
                "xml_unit_context": "sentence",
                "pivot_origin": "original",
                "quality_flags": "",
            }
        )

    def test_layered_cache_uses_later_record_and_audits_provider_variation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            public_cache = root / "public.jsonl"
            private_cache = root / "private.jsonl"
            key = make_cache_key(
                provider="deepl",
                source_lang="EN",
                target_lang="ZH-HANT",
                text="cry",
                split_sentences="0",
                preserve_formatting=True,
                model_type="prefer_quality_optimized",
            )

            def record(translation: str, created_at: str) -> dict[str, object]:
                return {
                    "key": key,
                    "provider": "deepl",
                    "source_lang": "EN",
                    "target_lang": "ZH-HANT",
                    "text": "cry",
                    "translation": translation,
                    "split_sentences": "0",
                    "preserve_formatting": True,
                    "model_type_requested": "prefer_quality_optimized",
                    "created_at": created_at,
                }

            public_cache.write_text(
                json.dumps(record("哭", "2026-08-09T04:30:43Z"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            private_cache.write_text(
                json.dumps(record("哭泣", "2026-07-12T22:36:04Z"), ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            cache, conflicts = load_cache_chain([public_cache, private_cache])

            self.assertEqual(cache[key]["translation"], "哭泣")
            self.assertEqual(len(conflicts), 1)
            self.assertEqual(conflicts[0]["lower_priority_translation"], "哭")
            self.assertEqual(conflicts[0]["selected_translation"], "哭泣")
            self.assertEqual(conflicts[0]["selection_policy"], "later_cache_wins")

    def test_synthetic_output_is_train_only_and_retains_source_provenance(self) -> None:
        row, reason = synthetic_row(
            self.source_row(),
            {
                "translation": "It is good.",
                "text": "今天真的很好。",
                "detected_source_language": "ZH",
                "model_type_used": "quality_optimized",
            },
            self.direction(),
            "cache-key",
        )
        self.assertEqual(reason, "")
        self.assertIsNotNone(row)
        self.assertEqual(row["source_record_id"], "source-row")
        self.assertEqual(row["split"], "train")
        self.assertEqual(row["pivot_origin"], "synthetic")
        self.assertEqual(row["english_sentence"], "It is good.")
        self.assertNotIn("chinese_sentence", row)

    def test_wrong_script_pivot_translation_is_rejected(self) -> None:
        row, reason = synthetic_row(
            self.source_row(),
            {
                "translation": "這是錯的。",
                "text": "今天真的很好。",
                "detected_source_language": "ZH",
            },
            self.direction(),
            "cache-key",
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "pivot_quality:english_target_script_mismatch")

    def test_quality_quarantine_is_accounted_without_becoming_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "zh.csv"
            original_path = root / "en.csv"
            output_dir = root / "pivot"

            source = self.source_row()
            pd.DataFrame([source.to_dict()]).to_csv(
                source_path,
                index=False,
            )
            original = source.copy()
            original["row_id"] = "original-row"
            original["source_record_id"] = "original-source"
            original["formosan_sentence"] = "misa."
            original["formosan_mt_standard"] = "misa."
            original["mt_standard_sha256"] = text_sha256("misa.")
            original["english_sentence"] = "Go."
            original = original.drop(labels=["chinese_sentence"])
            pd.DataFrame([original.to_dict()]).to_csv(
                original_path,
                index=False,
            )

            direction = self.direction()
            direction.source_path = source_path
            direction.original_target_path = original_path
            key = make_cache_key(
                provider="deepl",
                source_lang="ZH",
                target_lang="EN-US",
                text="今天真的很好。",
                split_sentences="0",
                preserve_formatting=True,
                model_type="prefer_quality_optimized",
            )
            result = write_pivot_output(
                direction,
                args=SimpleNamespace(
                    out_dir=output_dir,
                    quiet=True,
                    dedupe=True,
                    skip_target_overlaps=True,
                    split_sentences="0",
                    preserve_formatting=True,
                    model_type="prefer_quality_optimized",
                ),
                cache={
                    key: {
                        "translation": "這是錯的。",
                        "text": "今天真的很好。",
                        "detected_source_language": "ZH",
                    }
                },
            )

            self.assertEqual(result.synthetic_rows_missing, 0)
            self.assertEqual(result.synthetic_rows_quarantined, 1)
            self.assertIsNone(result.incomplete_path)
            self.assertTrue(
                (output_dir / "big_corpus_en_pivot.csv").is_file()
            )
            self.assertTrue(Path(result.quarantine_path or "").is_file())
            self.assertTrue(result.quarantine_sha256)

    def test_missing_pivot_cache_entry_remains_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "zh.csv"
            original_path = root / "en.csv"
            output_dir = root / "pivot"

            source = self.source_row()
            pd.DataFrame([source.to_dict()]).to_csv(
                source_path,
                index=False,
            )
            original = source.copy()
            original["row_id"] = "original-row"
            original["source_record_id"] = "original-source"
            original["formosan_sentence"] = "misa."
            original["formosan_mt_standard"] = "misa."
            original["mt_standard_sha256"] = text_sha256("misa.")
            original["english_sentence"] = "Go."
            original = original.drop(labels=["chinese_sentence"])
            pd.DataFrame([original.to_dict()]).to_csv(
                original_path,
                index=False,
            )

            direction = self.direction()
            direction.source_path = source_path
            direction.original_target_path = original_path
            result = write_pivot_output(
                direction,
                args=SimpleNamespace(
                    out_dir=output_dir,
                    quiet=True,
                    dedupe=True,
                    skip_target_overlaps=True,
                    split_sentences="0",
                    preserve_formatting=True,
                    model_type="prefer_quality_optimized",
                ),
                cache={},
            )

            self.assertEqual(result.synthetic_rows_missing, 1)
            self.assertEqual(result.synthetic_rows_quarantined, 0)
            self.assertTrue(Path(result.incomplete_path or "").is_file())
            self.assertFalse(
                (output_dir / "big_corpus_en_pivot.csv").exists()
            )

    def test_pivot_eligibility_is_structural_and_content_based(self) -> None:
        sentence = self.source_row()
        self.assertEqual(pivot_candidate_reason(sentence, self.direction()), "")

        sentence["source"] = (
            "Formosan-ILRDF_Dicts/Formosan-ILRDF_Dicts/Final_XML/Amis/sample.xml"
        )
        self.assertEqual(pivot_candidate_reason(sentence, self.direction()), "")

        lexeme = sentence.copy()
        lexeme["row_type"] = "lexeme"
        self.assertEqual(
            pivot_candidate_reason(lexeme, self.direction()),
            "non_sentence",
        )

        short = sentence.copy()
        short["formosan_sentence"] = "malu."
        short["formosan_mt_standard"] = "malu."
        self.assertEqual(
            pivot_candidate_reason(short, self.direction()),
            "short_formosan",
        )

    def test_training_bundle_includes_hashed_pivot_ledgers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            processed = root / "processed"
            final = root / "final"
            splits = root / "splits"
            pivot_dir = processed / "pivot"
            pivot_dir.mkdir(parents=True)
            final.mkdir()

            quarantine = pivot_dir / "pivot_rejections_zh2en.csv"
            quarantine.write_text(
                "direction,reason\nzh2en,target_script\n",
                encoding="utf-8",
            )
            quarantine_hash = hashlib.sha256(
                quarantine.read_bytes()
            ).hexdigest()
            conflicts = pivot_dir / "pivot_cache_conflicts_zh2en.jsonl"
            conflicts.write_text(
                '{"cache_key":"fixture","selection_policy":"later_cache_wins"}\n',
                encoding="utf-8",
            )
            conflicts_hash = hashlib.sha256(conflicts.read_bytes()).hexdigest()
            (pivot_dir / "pivot_manifest.json").write_text(
                json.dumps(
                    {
                        "complete": True,
                        "stats": [
                            {
                                "direction": "zh2en",
                                "quarantine_path": str(quarantine),
                                "quarantine_sha256": quarantine_hash,
                                "cache_conflict_path": str(conflicts),
                                "cache_conflict_sha256": conflicts_hash,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = root / "mt_build_manifest.json"
            snapshot = root / "source_repository_snapshot.json"
            manifest.write_text(
                json.dumps({"corpus_name": "fixture"}),
                encoding="utf-8",
            )
            snapshot.write_text("{}", encoding="utf-8")
            (final / "aggregate_manifest.json").write_text(
                "{}",
                encoding="utf-8",
            )
            for language in ("en", "zh"):
                split_dir = splits / f"splits_{language}_v1"
                split_dir.mkdir(parents=True)
                for prefix in ("report", "validation", "exposure"):
                    (
                        split_dir
                        / f"{prefix}_in_domain_hard.json"
                    ).write_text("{}", encoding="utf-8")

            paths = BuildPaths(
                root=root,
                raw_dir=root / "raw",
                processed_dir=processed,
                final_dir=final,
                split_root=splits,
                manifest_path=manifest,
                source_snapshot_path=snapshot,
            )
            bundle_path = package_training_provenance(paths, final)
            bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

            copied = final / "provenance" / quarantine.name
            self.assertTrue(copied.is_file())
            self.assertEqual(
                bundle["artifacts"][quarantine.name]["sha256"],
                quarantine_hash,
            )
            copied_conflicts = final / "provenance" / conflicts.name
            self.assertTrue(copied_conflicts.is_file())
            self.assertEqual(
                bundle["artifacts"][conflicts.name]["sha256"],
                conflicts_hash,
            )

    def test_aggregate_discovery_ignores_only_pivot_quarantine_ledgers(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pivot = root / "big_corpus_en_pivot.csv"
            pivot.write_text("row_id\nfixture\n", encoding="utf-8")
            (
                root / "pivot_rejections_zh2en.csv"
            ).write_text("reason\nfixture\n", encoding="utf-8")

            self.assertEqual(discover_inputs(root, set()), [pivot])

            unexpected = root / "unexpected.csv"
            unexpected.write_text("column\nvalue\n", encoding="utf-8")
            self.assertEqual(
                discover_inputs(root, set()),
                [pivot, unexpected],
            )

    def test_stage_cache_rejects_modified_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "prepared" / "sample.xml"
            output.parent.mkdir()
            output.write_text("original", encoding="utf-8")
            cache_path = root / ".stage_cache" / "ami.json"
            cache: dict[str, object] = {"schema_version": 2, "stages": {}}
            paths = BuildPaths(
                root=root,
                raw_dir=root / "raw",
                processed_dir=root / "processed",
                final_dir=root / "final",
                split_root=root / "splits",
                manifest_path=root / "manifest.json",
                source_snapshot_path=root / "snapshot.json",
            )
            with mock.patch.object(
                stage_cache,
                "sha256_file",
                wraps=stage_cache.sha256_file,
            ) as hasher:
                record_cached_stage(
                    paths.root,
                    cache_path,
                    cache,
                    "qc",
                    "stage-key",
                    [output],
                    "ami",
                )
                hashed_calls = hasher.call_count
                self.assertTrue(
                    cached_stage_valid(paths.root, cache, "qc", "stage-key")
                )
                self.assertEqual(hasher.call_count, hashed_calls)

                file_inventory([output], paths.root)
                file_inventory([output], paths.root)
                self.assertEqual(hasher.call_count, hashed_calls)

                output.write_text("modified", encoding="utf-8")
                self.assertFalse(
                    cached_stage_valid(paths.root, cache, "qc", "stage-key")
                )
                self.assertGreater(hasher.call_count, hashed_calls)

    def test_hash_cache_reuses_digest_for_hard_links(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "split.csv"
            linked = root / "final.csv"
            source.write_text("row_id\nfixture\n", encoding="utf-8")
            linked.hardlink_to(source)

            with mock.patch.object(
                stage_cache,
                "sha256_file",
                wraps=stage_cache.sha256_file,
            ) as hasher:
                source_hash = stage_cache.cached_sha256(source, root)
                linked_hash = stage_cache.cached_sha256(linked, root)

            self.assertEqual(source_hash, linked_hash)
            self.assertEqual(hasher.call_count, 1)

    def test_stage_cache_upgrades_verified_v1_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "processed" / "sample.csv"
            output.parent.mkdir()
            output.write_text("row_id\nfixture\n", encoding="utf-8")
            cache_path = root / ".stage_cache" / "build.json"
            cache_path.parent.mkdir()
            cache_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stages": {
                            "aggregate": {
                                "key": "stage-key",
                                "outputs": {
                                    "processed/sample.csv": hashlib.sha256(
                                        output.read_bytes()
                                    ).hexdigest()
                                },
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            cache = load_stage_cache(cache_path)

            self.assertEqual(cache["schema_version"], 2)
            self.assertTrue(
                cached_stage_valid(root, cache, "aggregate", "stage-key")
            )
            persisted = json.loads(cache_path.read_text(encoding="utf-8"))
            record = persisted["stages"]["aggregate"]["outputs"][
                "processed/sample.csv"
            ]
            self.assertEqual(record["bytes"], output.stat().st_size)
            self.assertEqual(
                record["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )

    def test_malformed_cache_is_a_hard_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            path.write_text("{bad json}\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_cache(path)

    def test_cache_key_integrity_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.jsonl"
            key = make_cache_key(
                provider="deepl",
                source_lang="ZH",
                target_lang="EN-US",
                text="很好。",
                split_sentences="0",
                preserve_formatting=True,
                model_type="prefer_quality_optimized",
            )
            record = {
                "provider": "deepl",
                "source_lang": "ZH",
                "target_lang": "EN-US",
                "text": "很好。",
                "translation": "It is good.",
                "split_sentences": "0",
                "preserve_formatting": True,
                "model_type_requested": "prefer_quality_optimized",
                "key": key,
            }
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            self.assertEqual(load_cache(path)[key]["translation"], "It is good.")


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
