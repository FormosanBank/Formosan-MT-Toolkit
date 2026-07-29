from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

from clean_xml import audit_standard_tiers, ensure_standard_tiers  # noqa: E402
from corpus_quality import (  # noqa: E402
    apply_quality_rules,
    deduplicate_pairs,
    normalize_dataframe,
    normalize_text,
)
from fetch_xml import classify_xml, git_blob_sha  # noqa: E402
from filter_split_corpus import read_csv  # noqa: E402
from make_corpus import extract_file  # noqa: E402
from pipeline_common import load_pipeline_config  # noqa: E402
from pivot import (  # noqa: E402
    Direction,
    load_cache,
    make_cache_key,
    synthetic_row,
)


class StandardTierTests(unittest.TestCase):
    def write_xml(self, directory: Path, body: str) -> Path:
        path = directory / "sample.xml"
        path.write_text(
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" xml:lang="ami">'
            f"{body}</TEXT>",
            encoding="utf-8",
        )
        return path

    def test_existing_standard_tier_is_never_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="s1"><FORM kindOf="original">ma-lu</FORM>'
                '<FORM kindOf="standard">malu<UNCLEAR/></FORM>'
                '<TRANSL xml:lang="eng">good</TRANSL></S>',
            )
            stats = ensure_standard_tiers(directory)
            root = ET.parse(path).getroot()
            standard = root.find("./S/FORM[@kindOf='standard']")
            self.assertIsNotNone(standard)
            self.assertEqual(standard.text, "malu")
            self.assertEqual([child.tag for child in standard], ["UNCLEAR"])
            self.assertEqual(stats["existing_standard"], 1)
            audit_standard_tiers(directory)

    def test_only_missing_tiers_are_completed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="original-only"><FORM kindOf="original">ma-lu</FORM></S>'
                '<S id="standard-only"><FORM kindOf="standard">malu</FORM></S>',
            )
            stats = ensure_standard_tiers(directory)
            root = ET.parse(path).getroot()
            original_only = root.find("./S[@id='original-only']")
            standard_only = root.find("./S[@id='standard-only']")
            self.assertEqual(
                original_only.find("FORM[@kindOf='standard']").text,
                "ma-lu",
            )
            self.assertEqual(
                standard_only.find("FORM[@kindOf='original']").text,
                "malu",
            )
            self.assertEqual(stats["standard_copied_from_original"], 1)
            self.assertEqual(stats["original_copied_from_standard"], 1)


class AcquisitionTests(unittest.TestCase):
    def test_malformed_xml_is_not_a_language_mismatch(self) -> None:
        status, language, dialect, error = classify_xml(
            b"<TEXT><S>",
            "ami",
            None,
            None,
        )
        self.assertEqual(status, "parse_error")
        self.assertEqual(language, "")
        self.assertTrue(error)

    def test_git_blob_hash_matches_git_object_format(self) -> None:
        self.assertEqual(
            git_blob_sha(b"hello\n"),
            "ce013625030ba8dba906f756967f9e9ca394464a",
        )

    def test_pipeline_pins_full_formosanbank_revision(self) -> None:
        config = load_pipeline_config()
        revision = config["formosanbank"]["qc_revision"]
        self.assertEqual(len(revision), 40)
        self.assertTrue(all(character in "0123456789abcdef" for character in revision))
        json.loads((ROOT / "config/corpus_pipeline.json").read_text(encoding="utf-8"))


class ExtractionAndCleaningTests(unittest.TestCase):
    def test_extraction_uses_standard_and_keeps_original_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "FormosanBank" / "Final_XML" / "sample.xml"
            path.parent.mkdir(parents=True)
            path.write_text(
                '<?xml version="1.0"?>'
                '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                'id="corpus" xml:lang="ami" dialect="Coastal">'
                '<S id="s1"><FORM kindOf="original">ma-lu=na.</FORM>'
                '<FORM kindOf="standard">maluna.</FORM>'
                '<TRANSL xml:lang="eng">It is good (today).</TRANSL></S></TEXT>',
                encoding="utf-8",
            )
            rows, stats = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FormosanBank",
                    "repository_commit": "a" * 40,
                    "source_path": "Final_XML/sample.xml",
                },
                target_codes={"eng"},
                tags={"S"},
            )
            self.assertEqual(stats["pairs"], 1)
            self.assertEqual(rows[0].formosan_sentence, "maluna.")
            self.assertEqual(rows[0].formosan_original, "ma-lu=na.")
            self.assertEqual(rows[0].kind_of, "standard")
            self.assertEqual(rows[0].row_type, "sentence")

    def test_cleaning_preserves_parentheses_and_structural_sentence_type(self) -> None:
        self.assertEqual(normalize_text("  It is good (today).  ").text, "It is good (today).")
        frame = pd.DataFrame(
            [
                {
                    "row_id": "r1",
                    "formosan": "hay.",
                    "english": "Right!",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/sample.xml",
                }
            ]
        )
        normalized, _ = normalize_dataframe(frame, "formosan", "english")
        self.assertEqual(normalized.loc[0, "row_type"], "sentence")
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="formosan",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 0)

    def test_literal_none_is_not_treated_as_missing_csv_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "data.csv"
            path.write_text("ami,english\nmana,None\n", encoding="utf-8")
            frame = read_csv(path)
            self.assertEqual(frame.loc[0, "english"], "None")

    def test_quality_and_dedupe_conserve_every_input_row(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "row_id": "r1",
                    "ami": "malu.",
                    "english": "Good.",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/a.xml",
                },
                {
                    "row_id": "r2",
                    "ami": "malu.",
                    "english": "Good.",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/b.xml",
                },
                {
                    "row_id": "r3",
                    "ami": "中文",
                    "english": "Wrong source.",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/c.xml",
                },
            ]
        )
        normalized, _ = normalize_dataframe(frame, "ami", "english")
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )
        accepted, duplicates = deduplicate_pairs(
            accepted,
            source_column="ami",
            target_column="english",
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(accepted) + len(rejected) + len(duplicates), len(frame))


class PivotContractTests(unittest.TestCase):
    def direction(self) -> Direction:
        return Direction(
            name="zh2en",
            source_path=Path("zh.csv"),
            original_target_path=Path("en.csv"),
            source_text_col="chinese_sentence",
            target_text_col="english_sentence",
            deepl_source_lang="ZH",
            deepl_target_lang="EN-US",
            output_filename="big_corpus_en_pivot.csv",
            cache_filename="deepl_zh_to_en.jsonl",
        )

    def source_row(self) -> pd.Series:
        return pd.Series(
            {
                "row_id": "human-row",
                "source_record_id": "source-row",
                "content_sha256": "old",
                "lang_code": "ami",
                "formosan_sentence": "malu.",
                "chinese_sentence": "很好。",
                "source": "FormosanBank/Corpora/Test/XML/sample.xml",
                "kindOf": "standard",
                "dialect": "Coastal",
                "row_type": "sentence",
                "quality_flags": "",
            }
        )

    def test_synthetic_output_is_train_only_and_retains_source_provenance(self) -> None:
        row, reason = synthetic_row(
            self.source_row(),
            {
                "translation": "It is good.",
                "text": "很好。",
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
                "text": "很好。",
                "detected_source_language": "ZH",
            },
            self.direction(),
            "cache-key",
        )
        self.assertIsNone(row)
        self.assertEqual(reason, "pivot_quality:english_target_script_mismatch")

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


class EndToEndCorpusPipelineTests(unittest.TestCase):
    def run_script(self, script: Path, *arguments: object) -> None:
        result = subprocess.run(
            [sys.executable, str(script), *(str(value) for value in arguments)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        if result.returncode:
            self.fail(
                f"{script.name} failed ({result.returncode})\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    @staticmethod
    def cache_record(
        *,
        source_lang: str,
        target_lang: str,
        text: str,
        translation: str,
    ) -> dict[str, object]:
        model_type = "prefer_quality_optimized"
        key = make_cache_key(
            provider="deepl",
            source_lang=source_lang,
            target_lang=target_lang,
            text=text,
            split_sentences="0",
            preserve_formatting=True,
            model_type=model_type,
        )
        return {
            "provider": "deepl",
            "direction": (
                "zh2en" if source_lang == "ZH" else "en2zh"
            ),
            "key": key,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "text": text,
            "translation": translation,
            "detected_source_language": source_lang,
            "split_sentences": "0",
            "preserve_formatting": True,
            "model_type_requested": model_type,
        }

    def test_network_free_release_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            downloaded = root / "downloaded_ami"
            raw = root / "raw"
            processed = root / "processed"
            pivot = root / "pivot"
            final = root / "final"
            splits = root / "splits"
            downloaded.mkdir()

            inventory = []
            for document in range(20):
                digest = hashlib.sha256(
                    f"document-{document}".encode()
                ).hexdigest()
                standard = " ".join(
                    digest[offset : offset + 8]
                    for offset in range(0, 40, 8)
                )
                original = standard.replace(" ", "-")
                if document % 2 == 0:
                    translation = (
                        f'<TRANSL xml:lang="eng">{digest[40:48]} '
                        f'{digest[48:56]} distinct English sentence '
                        f'{document}</TRANSL>'
                    )
                else:
                    chinese = "".join(
                        chr(0x4E00 + document * 16 + offset)
                        for offset in range(8)
                    )
                    translation = (
                        f'<TRANSL xml:lang="zho">{chinese}</TRANSL>'
                    )
                relative = Path("FixtureRepo") / "Final_XML" / f"doc-{document}.xml"
                path = downloaded / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    '<?xml version="1.0" encoding="utf-8"?>'
                    '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                    'id="fixture" xml:lang="ami" dialect="Fixture">'
                    f'<S id="s-{document}">'
                    f'<FORM kindOf="original">{original}</FORM>'
                    f'<FORM kindOf="standard">{standard}</FORM>'
                    f"{translation}</S></TEXT>",
                    encoding="utf-8",
                )
                inventory.append(
                    {
                        "repository": "FixtureRepo",
                        "commit_sha": "a" * 40,
                        "source_path": f"Final_XML/doc-{document}.xml",
                        "status": "kept",
                        "destination": str(relative),
                        "sha256": "fixture",
                    }
                )
            (downloaded / "_fetch_inventory.jsonl").write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in inventory
                ),
                encoding="utf-8",
            )
            (downloaded / "_fetch_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "source_language": "ami",
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )
            ensure_standard_tiers(downloaded)

            make = ROOT / "scripts/local/make_corpus.py"
            clean = ROOT / "scripts/local/filter_split_corpus.py"
            aggregate = ROOT / "scripts/local/build_big_corpus.py"
            pivot_script = ROOT / "scripts/local/pivot.py"
            split_script = (
                ROOT
                / "formosan_mt_experiments/scripts/build_experiment_splits.py"
            )
            validate_script = (
                ROOT
                / "formosan_mt_experiments/scripts/validate_experiment.py"
            )

            for target, short in (("english", "en"), ("chinese", "zh")):
                raw_path = raw / f"ami_{short}.csv"
                processed_path = processed / f"ami_{short}_processed.csv"
                self.run_script(
                    make,
                    "--xml-dir",
                    downloaded,
                    "--target",
                    target,
                    "--out",
                    raw_path,
                    "--units",
                    "sentences",
                )
                self.run_script(
                    clean,
                    "--input",
                    raw_path,
                    "--output",
                    processed_path,
                    "--no-split",
                )
            self.run_script(
                aggregate,
                "--input-dir",
                processed,
                "--output-dir",
                processed,
            )

            cache_dir = pivot / "cache"
            cache_dir.mkdir(parents=True)
            english = pd.read_csv(
                processed / "big_corpus_en.csv",
                keep_default_na=False,
            )
            chinese = pd.read_csv(
                processed / "big_corpus_zh.csv",
                keep_default_na=False,
            )
            zh_to_en = [
                self.cache_record(
                    source_lang="ZH",
                    target_lang="EN-US",
                    text=str(row["chinese_sentence"]),
                    translation=(
                        f"translated fixture sentence for "
                        f"{row['source_record_id']}"
                    ),
                )
                for _, row in chinese.iterrows()
            ]
            en_to_zh = [
                self.cache_record(
                    source_lang="EN",
                    target_lang="ZH-HANT",
                    text=str(row["english_sentence"]),
                    translation="".join(
                        chr(0x7000 + index + offset)
                        for offset in range(8)
                    ),
                )
                for index, (_, row) in enumerate(english.iterrows())
            ]
            for name, records in (
                ("deepl_zh_to_en.jsonl", zh_to_en),
                ("deepl_en_to_zh.jsonl", en_to_zh),
            ):
                (cache_dir / name).write_text(
                    "".join(
                        json.dumps(record, ensure_ascii=False, sort_keys=True)
                        + "\n"
                        for record in records
                    ),
                    encoding="utf-8",
                )

            self.run_script(
                pivot_script,
                "--big-corpus-en",
                processed / "big_corpus_en.csv",
                "--big-corpus-zh",
                processed / "big_corpus_zh.csv",
                "--out-dir",
                pivot,
                "--cache-dir",
                cache_dir,
                "--skip-translation",
                "--splits",
                "all",
            )
            self.run_script(
                aggregate,
                "--input-dir",
                pivot,
                "--output-dir",
                final,
            )

            for target, column, short in (
                ("english", "english_sentence", "en"),
                ("chinese", "chinese_sentence", "zh"),
            ):
                output_dir = splits / short
                self.run_script(
                    split_script,
                    "--input",
                    final / f"big_corpus_{short}.csv",
                    "--output-dir",
                    output_dir,
                    "--target-lang",
                    target,
                    "--target-col",
                    column,
                    "--output-prefix",
                    f"big_corpus_{short}",
                    "--min-formosan-tokens",
                    1,
                    "--min-target-tokens",
                    1,
                    "--min-test-rows",
                    1,
                    "--min-validate-rows",
                    1,
                    "--selection-attempts",
                    10,
                    "--tiers",
                    "in_domain_hard",
                )
                hard = output_dir / f"big_corpus_{short}_in_domain_hard.csv"
                validation = output_dir / "validation.json"
                self.run_script(
                    validate_script,
                    "--input",
                    hard,
                    "--target-lang",
                    target,
                    "--target-col",
                    column,
                    "--min-test-rows",
                    1,
                    "--min-validate-rows",
                    1,
                    "--report",
                    validation,
                )
                report = json.loads(validation.read_text(encoding="utf-8"))
                self.assertTrue(report["complete"])
                frame = pd.read_csv(hard, keep_default_na=False)
                self.assertTrue(
                    frame.loc[
                        frame["split"].isin({"test", "validate"}),
                        "pivot_origin",
                    ].ne("synthetic").all()
                )


if __name__ == "__main__":
    unittest.main()
