from __future__ import annotations

import concurrent.futures
import contextlib
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import fetch_xml  # noqa: E402
from clean_xml import (  # noqa: E402
    audit_standard_tiers,
    classify_translation_version_repairs,
    ensure_standard_tiers,
    finalize_transform_inventory,
    tag_transform_sources,
)
from corpus_quality import (  # noqa: E402
    apply_quality_rules,
    deduplicate_pairs,
    normalize_dataframe,
    normalize_text,
)
from fetch_xml import (  # noqa: E402
    classify_xml,
    get_tree,
    git_blob_sha,
    load_or_create_repository_snapshot,
    repository_selection,
    write_blob_cache,
)
from filter_split_corpus import (  # noqa: E402
    filter_rule_counts,
    print_filter_rule_summary,
    read_csv,
)
from make_corpus import extract_file  # noqa: E402
from pipeline_common import load_pipeline_config  # noqa: E402
from pivot import (  # noqa: E402
    Direction,
    load_cache,
    make_cache_key,
    synthetic_row,
)
from qc_change_audit import (  # noqa: E402
    classify_cleaner_field_changes,
)
from qc_reporting import (  # noqa: E402
    parse_cleaner_transformation,
    run_cleaner_command,
    summarize_validator_findings,
)
from xml_repairs import repair_mt_xml_structure  # noqa: E402


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

    def test_qc_transform_inventory_distinguishes_provided_and_derived(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_xml(
                directory,
                '<S id="provided"><FORM kindOf="original">ma-lu</FORM>'
                '<FORM kindOf="standard">malu</FORM></S>'
                '<S id="derived"><FORM kindOf="original">pa-su</FORM></S>',
            )
            records = tag_transform_sources(directory)
            ensure_standard_tiers(directory)
            inventory = finalize_transform_inventory(directory, records)
            origins = {
                row["xml_id"]: row["standard_origin"]
                for row in inventory
            }
            self.assertEqual(origins["provided"], "provided")
            self.assertEqual(origins["derived"], "derived_from_original")
            self.assertTrue(
                all(
                    row["standard_after_qc_sha256"]
                    for row in inventory
                )
            )
            self.assertNotIn(
                "_mt_toolkit_transform_id",
                (directory / "sample.xml").read_text(encoding="utf-8"),
            )

    def test_empty_morpheme_standard_is_counted_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_xml(
                directory,
                '<S id="s1"><FORM kindOf="standard">malu</FORM>'
                '<W id="w1"><FORM kindOf="standard">malu</FORM>'
                '<M id="m1"><FORM kindOf="standard" />'
                '<TRANSL xml:lang="zho">零形態</TRANSL></M>'
                "</W></S>",
            )
            stats = audit_standard_tiers(directory)
            self.assertEqual(stats["empty_m_standard_tiers"], 1)
            self.assertEqual(stats["m_standard_tiers"], 1)

    def test_untranscribed_audio_sentence_is_retained_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="audio-only"><FORM kindOf="original"/>'
                '<PHON kindOf="original"/><FORM kindOf="standard"/>'
                '<PHON kindOf="standard"/>'
                '<TRANSL xml:lang="zho">尚未轉寫的錄音</TRANSL>'
                '<AUDIO file="audio.wav" start="0" end="1.5"/></S>',
            )
            records = tag_transform_sources(directory)
            stats, repairs, dispositions = repair_mt_xml_structure(
                directory
            )
            inventory = finalize_transform_inventory(
                directory,
                records,
                dispositions,
            )
            sentence = ET.parse(path).getroot().find("./S")

            self.assertIsNotNone(sentence)
            self.assertEqual(len(sentence.findall("FORM")), 2)
            self.assertEqual(len(sentence.findall("PHON")), 2)
            self.assertIsNotNone(sentence.find("AUDIO"))
            self.assertEqual(
                sentence.findtext("TRANSL"),
                "尚未轉寫的錄音",
            )
            self.assertEqual(
                stats["untranscribed_audio_sentences_preserved"], 1
            )
            self.assertEqual(repairs, [])
            self.assertEqual(dispositions, {})
            self.assertEqual(inventory[0]["disposition"], "retained")
            audit = audit_standard_tiers(directory)
            self.assertEqual(
                audit[
                    "untranscribed_audio_sentence_standard_tiers"
                ],
                1,
            )

            rows, extraction = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FixtureRepo",
                    "repository_commit": "a" * 40,
                    "source_path": "sample.xml",
                },
                target_codes={"zho"},
                tags={"S"},
            )
            self.assertEqual(rows, [])
            self.assertEqual(
                extraction[
                    "untranscribed_or_unclear_sentences_skipped"
                ],
                1,
            )

    def test_unclear_audio_sentence_is_retained_and_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="unclear"><FORM kindOf="original"><UNCLEAR/>'
                '</FORM><PHON kindOf="original"/>'
                '<FORM kindOf="standard"><UNCLEAR/></FORM>'
                '<PHON kindOf="standard"/>'
                '<AUDIO file="audio.wav" start="0" end="1.5"/></S>',
            )
            stats, repairs, dispositions = repair_mt_xml_structure(
                directory
            )
            sentence = ET.parse(path).getroot().find("./S")

            self.assertIsNotNone(sentence)
            self.assertEqual(len(sentence.findall("FORM")), 2)
            self.assertEqual(len(sentence.findall("PHON")), 2)
            self.assertEqual(
                stats["unclear_source_sentences_preserved"], 1
            )
            self.assertEqual(repairs, [])
            self.assertEqual(dispositions, {})
            audit = audit_standard_tiers(directory)
            self.assertEqual(
                audit["unclear_sentence_standard_tiers"], 1
            )

            rows, extraction = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FixtureRepo",
                    "repository_commit": "a" * 40,
                    "source_path": "sample.xml",
                },
                target_codes={"zho"},
                tags={"S"},
            )
            self.assertEqual(rows, [])
            self.assertEqual(
                extraction[
                    "untranscribed_or_unclear_sentences_skipped"
                ],
                1,
            )

    def test_mt_xml_repairs_are_narrow_and_auditable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="duplicate"><FORM kindOf="standard">malu</FORM>'
                '<TRANSL xml:lang="zho">好</TRANSL>、'
                '<TRANSL xml:lang="zho" ver="alt">很好</TRANSL></S>'
                '<S id="duplicate"><FORM kindOf="standard"> masalu </FORM>'
                '<TRANSL xml:lang="eng">\ufeffhappy</TRANSL></S>'
                '<M id="empty"><FORM kindOf="standard" />'
                '<TRANSL xml:lang="zho">零形態</TRANSL></M>'
                '<S id="annotated"><FORM kindOf="original">*malu</FORM>'
                '<FORM kindOf="standard">*malu</FORM></S>'
                '<W id="variant"><FORM kindOf="original">'
                "'arup(a)-ara</FORM><FORM kindOf=\"standard\">"
                "'arup(a)-ara</FORM><TRANSL xml:lang=\"zho\">互相-拿"
                "</TRANSL><M id=\"variant-m\"><FORM kindOf=\"original\">"
                "usa/bi(n</FORM><FORM kindOf=\"standard\">usa/bi(n"
                '</FORM><TRANSL xml:lang="zho">那裡-改變狀態</TRANSL>'
                "</M></W>"
                '<S id="null"><FORM kindOf="standard">∅</FORM></S>'
                '<S id="empty-sentence"><FORM kindOf="original">'
                '<UNCLEAR/></FORM><FORM kindOf="standard">'
                '<UNCLEAR/></FORM></S>',
            )
            records = tag_transform_sources(directory)
            stats, repairs, dispositions = repair_mt_xml_structure(
                directory
            )
            inventory = finalize_transform_inventory(
                directory,
                records,
                dispositions,
            )
            root = ET.parse(path).getroot()
            self.assertIsNotNone(root.find("./M[@id='empty']"))
            self.assertEqual(
                [sentence.get("id") for sentence in root.findall("./S")],
                [
                    "duplicate",
                    "duplicate__mtdup2",
                    "annotated",
                    "null",
                    "empty-sentence",
                ],
            )
            self.assertEqual(
                stats,
                {
                    "untyped_punctuation_removed": 1,
                    "duplicate_ids_disambiguated": 1,
                    "empty_source_lexical_units_preserved": 1,
                    "source_annotation_units_preserved": 1,
                    "form_boundary_whitespace_trimmed": 1,
                    "zero_width_fields_repaired": 1,
                    "null_source_sentences_preserved": 1,
                    "unclear_source_sentences_preserved": 1,
                },
            )
            self.assertEqual(len(repairs), 4)
            self.assertTrue(
                all(row["disposition"] == "retained" for row in inventory)
            )
            variant = root.find("./W[@id='variant']")
            self.assertIsNotNone(variant)
            self.assertEqual(
                variant.find("FORM[@kindOf='standard']").text,
                "'arup(a)-ara",
            )
            self.assertEqual(
                variant.find(
                    "./M[@id='variant-m']/FORM[@kindOf='standard']"
                ).text,
                "usa/bi(n",
            )
            pairs, extraction = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FixtureRepo",
                    "repository_commit": "a" * 40,
                    "source_path": "sample.xml",
                },
                target_codes={"zho"},
                tags={"W", "M"},
            )
            extracted = {
                pair.xml_id: pair.formosan_standard for pair in pairs
            }
            self.assertEqual(extracted["variant"], "'arup(a)-ara")
            self.assertEqual(extracted["variant-m"], "usa/bi(n")
            self.assertEqual(extraction["w_units_seen"], 1)
            self.assertEqual(extraction["m_units_seen"], 2)
            self.assertEqual(extraction["empty_lexical_units_skipped"], 1)
            unclear = next(
                row
                for row in inventory
                if row.get("final_xml_id") == "empty-sentence"
            )
            self.assertEqual(unclear["disposition"], "retained")
            duplicate = next(
                row
                for row in inventory
                if row.get("final_xml_id") == "duplicate__mtdup2"
            )
            self.assertEqual(duplicate["xml_id"], "duplicate")
            self.assertEqual(
                root.find(
                    "./S[@id='duplicate__mtdup2']/FORM"
                ).text,
                "masalu",
            )
            self.assertEqual(
                root.find(
                    "./S[@id='duplicate__mtdup2']/TRANSL"
                ).text,
                "happy",
            )

    def test_mt_xml_repair_rejects_substantive_untyped_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_xml(
                directory,
                '<S id="s1">untyped words'
                '<FORM kindOf="standard">malu</FORM></S>',
            )
            tag_transform_sources(directory)
            with self.assertRaisesRegex(
                SystemExit,
                "Substantive untyped content",
            ):
                repair_mt_xml_structure(directory)

    def test_invalid_audio_span_is_preserved_for_source_review(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = self.write_xml(
                directory,
                '<S id="s1"><FORM kindOf="standard">qibaq.</FORM>'
                '<TRANSL xml:lang="eng">Good.</TRANSL>'
                '<AUDIO file="bad.mp3" start="10.02" end="10.01"/>'
                '<AUDIO file="good.mp3" start="10.02" end="11.01"/>'
                "</S>",
            )
            tag_transform_sources(directory)
            stats, repairs, _ = repair_mt_xml_structure(directory)
            sentence = ET.parse(path).getroot().find("./S")
            self.assertEqual(
                [audio.get("file") for audio in sentence.findall("AUDIO")],
                ["bad.mp3", "good.mp3"],
            )
            self.assertEqual(
                sentence.find("FORM[@kindOf='standard']").text,
                "qibaq.",
            )
            self.assertEqual(
                sentence.find("TRANSL").text,
                "Good.",
            )
            self.assertEqual(stats, {})
            self.assertEqual(repairs, [])


class PipelineReportingTests(unittest.TestCase):
    def test_cleaner_filename_chatter_is_logged_not_printed(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"
            corpus.mkdir()
            (corpus / "sample.xml").write_text(
                "<TEXT/>",
                encoding="utf-8",
            )
            log_path = corpus / "_qc_logs" / "clean_xml.log"
            script = (
                "print('Processing file: sample.xml');"
                "print('File cleaned: /tmp/sample.xml');"
                "print(\"  '’' → \\\"'\\\" : 2\")"
            )
            terminal = io.StringIO()
            with (
                contextlib.redirect_stdout(terminal),
                contextlib.redirect_stderr(terminal),
            ):
                result = run_cleaner_command(
                    [sys.executable, "-c", script],
                    root,
                    corpus_dir=corpus,
                    log_path=log_path,
                )
            self.assertNotIn("Processing file", terminal.getvalue())
            self.assertNotIn("File cleaned", terminal.getvalue())
            self.assertIn("Processing file", log_path.read_text())
            self.assertEqual(result["files_scanned"], 1)
            self.assertEqual(result["files_cleaned"], 1)
            self.assertEqual(
                result["character_transformations"][0]["count"],
                2,
            )

    def test_cleaner_field_changes_are_classified_by_rule(self) -> None:
        before = {
            "token:FORM:0": {
                "xml_path": "sample.xml",
                "xml_id": "s1",
                "unit_tag": "S",
                "field_tag": "FORM",
                "field_kind": "standard",
                "language": "ami",
                "text": " ma-lu=na！！ ",
            },
            "token:TRANSL:0": {
                "xml_path": "sample.xml",
                "xml_id": "s1",
                "unit_tag": "S",
                "field_tag": "TRANSL",
                "field_kind": "",
                "language": "zho",
                "explicit_language": "zh",
                "text": "「很好」",
            },
        }
        after = {
            "token:FORM:0": {
                **before["token:FORM:0"],
                "text": "maluna!",
            },
            "token:TRANSL:0": {
                **before["token:TRANSL:0"],
                "language": "zho",
                "explicit_language": "zho",
                "text": "＂很好＂",
            },
        }
        summary = classify_cleaner_field_changes(before, after)
        self.assertEqual(summary["fields_modified"], 2)
        self.assertEqual(summary["metadata_fields_modified"], 1)
        self.assertEqual(
            summary["rule_counts"],
            {
                "normalize_chinese_double_quotes": 1,
                "normalize_punctuation": 1,
                "normalize_translation_language_zh_to_zho": 1,
                "normalize_whitespace": 1,
                "remove_standard_segmentation_markers": 1,
                "trim_repeated_punctuation": 1,
            },
        )
        self.assertEqual(summary["unclassified_examples"], [])

    def test_translation_version_repairs_are_audited(self) -> None:
        before = {
            "token:TRANSL:0": {
                "xml_path": "dictionary.xml",
                "xml_id": "S1",
                "element_tag": "S",
                "language": "eng",
                "ver": "",
            },
        }
        after = {
            "token:TRANSL:0": {
                **before["token:TRANSL:0"],
                "ver": "alt",
            },
        }

        self.assertEqual(
            classify_translation_version_repairs(before, after),
            [
                {
                    "repair": "mark_alternate_translation",
                    "xml_path": "dictionary.xml",
                    "element_tag": "S",
                    "xml_id": "S1",
                    "language": "eng",
                    "before": "",
                    "after": "alt",
                },
            ],
        )

    def test_cleaner_transformations_are_parsed_without_file_noise(
        self,
    ) -> None:
        self.assertEqual(
            parse_cleaner_transformation(
                "  '’' → \"'\" : 1,234\n"
            ),
            {
                "input": "’",
                "output": "'",
                "count": 1234,
            },
        )
        self.assertEqual(
            parse_cleaner_transformation(
                "  '\\u200b' → '<deleted>' : 8"
            ),
            {
                "input": "\u200b",
                "output": "",
                "count": 8,
            },
        )
        self.assertIsNone(
            parse_cleaner_transformation(
                "Processing file: noisy.xml"
            )
        )

    def test_validator_findings_are_summarized_by_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "findings.csv"
            path.write_text(
                "file,severity,rule_id,title\n"
                "a.xml,SOFT,V122,parentheses\n"
                "a.xml,SOFT,V122,parentheses\n"
                "b.xml,WARN,V999,example\n",
                encoding="utf-8",
            )
            summary = summarize_validator_findings(path)
            self.assertEqual(summary["records"], 3)
            self.assertEqual(summary["files_with_findings"], 2)
            self.assertEqual(
                summary["by_severity"]["SOFT"]["rules"]["V122"][
                    "count"
                ],
                2,
            )

    def test_filter_summary_lists_each_disposition_and_rule(
        self,
    ) -> None:
        rows = pd.DataFrame(
            {
                "disposition": [
                    "rejected",
                    "quarantine",
                    "deduplicated",
                ],
                "disposition_reason": [
                    "missing_translation_marker",
                    "url",
                    "duplicate_pair",
                ],
            }
        )
        counts = filter_rule_counts(rows)
        report = {
            "input": "/tmp/tay_zh.csv",
            "initial_rows": 10,
            "accepted_rows": 7,
            "transformation_counts": {"unicode_nfc": 2},
            "filter_rule_counts": counts,
            "rejection_ledger": "/tmp/rejected.csv",
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_filter_rule_summary(report)
        rendered = output.getvalue()
        self.assertIn(
            "rejected / missing translation marker: 1",
            rendered,
        )
        self.assertIn("quarantine / url: 1", rendered)
        self.assertIn(
            "deduplicated / duplicate pair: 1",
            rendered,
        )
        self.assertIn("unicode nfc: 2", rendered)


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

    def test_concurrent_blob_cache_writes_are_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_path = Path(temporary) / "blob.xml"
            content = b"<TEXT xml:lang=\"ami\"/>"
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                writes = [
                    executor.submit(write_blob_cache, cache_path, content)
                    for _ in range(100)
                ]
                for write in writes:
                    write.result()
            self.assertEqual(cache_path.read_bytes(), content)
            self.assertEqual(list(cache_path.parent.glob("*.tmp")), [])

    def test_tree_discovery_is_scoped_to_final_xml(self) -> None:
        commit_sha = "a" * 40
        final_xml_sha = "b" * 40

        def response(payload: dict) -> mock.Mock:
            result = mock.Mock()
            result.json.return_value = payload
            return result

        def api_get(url: str, *, params: dict | None = None) -> mock.Mock:
            if url.endswith(commit_sha):
                self.assertIsNone(params)
                return response(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": "Final_XML",
                                "type": "tree",
                                "sha": final_xml_sha,
                            },
                            {
                                "path": "large-unrelated-data",
                                "type": "tree",
                                "sha": "c" * 40,
                            },
                        ],
                    }
                )
            if url.endswith(final_xml_sha):
                self.assertEqual(params, {"recursive": "1"})
                return response(
                    {
                        "truncated": False,
                        "tree": [
                            {
                                "path": "Paiwan/sample.xml",
                                "type": "blob",
                                "sha": "d" * 40,
                            }
                        ],
                    }
                )
            raise AssertionError(f"Unexpected API request: {url}")

        with tempfile.TemporaryDirectory() as temporary:
            with (
                mock.patch.object(fetch_xml, "CACHE_DIR", Path(temporary)),
                mock.patch.object(
                    fetch_xml,
                    "api_get",
                    side_effect=api_get,
                ) as get,
            ):
                tree = get_tree(
                    "FormosanBank",
                    "example",
                    commit_sha,
                    root_path="Final_XML",
                )
        self.assertEqual(tree[0]["path"], "Final_XML/Paiwan/sample.xml")
        self.assertEqual(get.call_count, 2)

    def test_repository_snapshot_is_resolved_once_and_reused(self) -> None:
        selection = repository_selection(
            org="FormosanBank",
            public=False,
            branch=None,
            discovered=["RepoB", "RepoA"],
            selected=["RepoB", "RepoA"],
            excluded=[],
        )
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "source_repository_snapshot.json"
            with (
                mock.patch.object(
                    fetch_xml,
                    "get_default_branch",
                    return_value="main",
                ) as branch,
                mock.patch.object(
                    fetch_xml,
                    "resolve_commit",
                    side_effect=["a" * 40, "b" * 40],
                ) as resolve,
            ):
                first = load_or_create_repository_snapshot(
                    snapshot,
                    selection=selection,
                    refresh_metadata=True,
                )
            self.assertEqual(branch.call_count, 2)
            self.assertEqual(resolve.call_count, 2)

            with (
                mock.patch.object(
                    fetch_xml,
                    "get_default_branch",
                    side_effect=AssertionError("snapshot should be reused"),
                ),
                mock.patch.object(
                    fetch_xml,
                    "resolve_commit",
                    side_effect=AssertionError("snapshot should be reused"),
                ),
            ):
                second = load_or_create_repository_snapshot(
                    snapshot,
                    selection=selection,
                    refresh_metadata=True,
                )
        self.assertEqual(first, second)

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

    def test_extraction_ids_remain_unique_without_xml_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "sample.xml"
            path.write_text(
                '<?xml version="1.0"?>'
                '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                'xml:lang="ami">'
                '<S><FORM kindOf="standard">first sentence</FORM>'
                '<TRANSL xml:lang="eng">first target</TRANSL></S>'
                '<S><FORM kindOf="standard">second sentence</FORM>'
                '<TRANSL xml:lang="eng">second target</TRANSL></S>'
                "</TEXT>",
                encoding="utf-8",
            )
            rows, _ = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FixtureRepo",
                    "repository_commit": "a" * 40,
                    "source_path": "sample.xml",
                },
                target_codes={"eng"},
                tags={"S"},
            )
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row.row_id for row in rows}), 2)
            self.assertEqual(
                {row.xml_element_index for row in rows},
                {0, 1},
            )

    def test_extraction_skips_empty_lexical_units_with_accounting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "sample.xml"
            path.write_text(
                '<?xml version="1.0"?>'
                '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                'xml:lang="ami"><M id="m1">'
                '<FORM kindOf="standard" />'
                '<TRANSL xml:lang="zho">零形態</TRANSL>'
                "</M></TEXT>",
                encoding="utf-8",
            )
            rows, stats = extract_file(
                path,
                xml_dir=directory,
                provenance={
                    "repository": "FixtureRepo",
                    "repository_commit": "a" * 40,
                    "source_path": "sample.xml",
                },
                target_codes={"zho"},
                tags={"M"},
            )
            self.assertEqual(rows, [])
            self.assertEqual(stats["empty_standard"], 1)
            self.assertEqual(stats["empty_lexical_units_skipped"], 1)

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

    def test_source_annotations_are_rejected_at_row_level(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "row_id": "asterisk",
                    "ami": "*malu",
                    "english": "bad",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/a.xml",
                },
                {
                    "row_id": "artifact",
                    "ami": "456otca",
                    "english": "artifact",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/b.xml",
                },
            ]
        )
        normalized, _ = normalize_dataframe(
            frame,
            "ami",
            "english",
        )
        accepted, rejected, counts = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )
        self.assertEqual(len(accepted), 0)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(counts["rejected:source_annotation_marker"], 1)
        self.assertEqual(counts["rejected:source_artifact_marker"], 1)

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
            qc_records = [
                {
                    "transform_id": f"transform-{document}",
                    "xml_path": str(
                        Path("FixtureRepo")
                        / "Final_XML"
                        / f"doc-{document}.xml"
                    ),
                    "element_tag": "S",
                    "xml_id": f"s-{document}",
                    "source_element_index": 0,
                    "final_element_index": 0,
                    "standard_origin": "provided",
                    "original_before_qc_sha256": "a" * 64,
                    "standard_before_qc_sha256": "b" * 64,
                    "standard_after_qc_sha256": "c" * 64,
                    "disposition": "retained",
                }
                for document in range(20)
            ]
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
                        "schema_version": 2,
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
