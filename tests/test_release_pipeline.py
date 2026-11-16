from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from build_big_corpus import (
    estimated_output_bytes,
    write_csv_atomic,
)
from build_context import replace_with_hardlink
from clean_xml import (
    ensure_standard_tiers,
)
from mt_standardization import (
    DEFAULT_PROFILE_PATH,
)
from pivot import (
    make_cache_key,
)
from support import text_sha256

ROOT = Path(__file__).resolve().parents[1]


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
            fetch_inventory_path = downloaded / "_fetch_inventory.jsonl"
            fetch_inventory_path.write_text(
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
                        "inventory_sha256": hashlib.sha256(
                            fetch_inventory_path.read_bytes()
                        ).hexdigest(),
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )
            ensure_standard_tiers(downloaded)
            qc_records = []
            for document in range(20):
                digest = hashlib.sha256(
                    f"document-{document}".encode()
                ).hexdigest()
                standard = " ".join(
                    digest[offset : offset + 8]
                    for offset in range(0, 40, 8)
                )
                original = standard.replace(" ", "-")
                qc_records.append(
                    {
                        "transform_id": f"transform-{document}",
                        "xml_path": str(
                            Path("FixtureRepo")
                            / "Final_XML"
                            / f"doc-{document}.xml"
                        ),
                        "element_tag": "S",
                        "xml_id": f"s-{document}",
                        "final_xml_id": f"s-{document}",
                        "source_element_index": 0,
                        "final_element_index": 0,
                        "standard_origin": "provided",
                        "provided_standard_present": True,
                        "formosan_original_raw": original,
                        "formosan_source_standard": standard,
                        "contains_unclear_source": False,
                        "original_before_qc_sha256": text_sha256(original),
                        "standard_before_qc_sha256": text_sha256(standard),
                        "standard_after_qc_sha256": text_sha256(standard),
                        "disposition": "retained",
                    }
                )
            qc_inventory_path = downloaded / "_qc_transform_inventory.jsonl"
            qc_inventory_path.write_text(
                "".join(
                    json.dumps(row, sort_keys=True) + "\n"
                    for row in qc_records
                ),
                encoding="utf-8",
            )
            (downloaded / "_qc_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 3,
                        "pipeline_version": "formosan-mt-corpus-v3",
                        "source_language": "ami",
                        "source_immutable": True,
                        "formosanbank_qc": {"revision": "d" * 40},
                        "transform_inventory": {
                            "path": qc_inventory_path.name,
                            "sha256": hashlib.sha256(
                                qc_inventory_path.read_bytes()
                            ).hexdigest(),
                        },
                        "complete": True,
                    }
                ),
                encoding="utf-8",
            )

            make = ROOT / "scripts/local/make_corpus.py"
            standardize = ROOT / "scripts/local/standardize_mt_corpus.py"
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

            self.run_script(
                standardize,
                "--xml-dir",
                downloaded,
                "--src-lang",
                "ami",
                "--profile",
                DEFAULT_PROFILE_PATH,
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
                signature_columns = [
                    "row_id",
                    "source_record_id",
                    "lang_code",
                    "formosan_sentence",
                    column,
                    "row_type",
                    "pivot_origin",
                    "split",
                ]
                signature = hashlib.sha256(
                    json.dumps(
                        frame[signature_columns]
                        .fillna("")
                        .astype(str)
                        .to_dict("records"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()
                expected_signatures = {
                    "en": "606b28bb0efa37166e04b5907f75ddfab228ceb3af779e93d696e8892cdb6664",
                    "zh": "303a7c1947578cba758c1c822ee02c212fa9cfe46d7943f5d369021a94ca3b36",
                }
                self.assertEqual(signature, expected_signatures[short])


class LargeArtifactSafetyTests(unittest.TestCase):
    def test_output_estimate_scales_with_input_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.csv"
            path.write_bytes(b"x" * 1024)
            self.assertGreater(estimated_output_bytes([path]), path.stat().st_size)

    def test_atomic_csv_write_removes_incomplete_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "corpus.csv"
            frame = mock.Mock()

            def fail(path: Path, *, index: bool) -> None:
                Path(path).write_text("partial", encoding="utf-8")
                raise OSError("disk full")

            frame.to_csv.side_effect = fail
            with self.assertRaisesRegex(OSError, "disk full"):
                write_csv_atomic(frame, output)

            self.assertFalse(output.exists())
            self.assertFalse((output.parent / ".corpus.csv.incomplete").exists())

    def test_final_split_uses_one_physical_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "split.csv"
            destination = root / "release.csv"
            source.write_text("a,b\n1,2\n", encoding="utf-8")

            replace_with_hardlink(source, destination)

            self.assertEqual(destination.read_bytes(), source.read_bytes())
            self.assertEqual(destination.stat().st_ino, source.stat().st_ino)
