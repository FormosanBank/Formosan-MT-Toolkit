from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from clean_xml import (
    classify_translation_version_repairs,
)
from filter_split_corpus import (
    filter_rule_counts,
    print_filter_rule_summary,
)
from qc_change_audit import (
    classify_cleaner_field_changes,
)
from qc_reporting import (
    parse_cleaner_transformation,
    print_qc_rule_summary,
    run_cleaner_command,
    summarize_validator_findings,
)


class PipelineReportingTests(unittest.TestCase):
    def test_v3_qc_summary_does_not_require_legacy_cleaner(self) -> None:
        result = {
            "tier_completion": {
                "standard_copied_from_original": 2,
            },
            "repair_inventory": {
                "path": "_qc_repair_inventory.jsonl",
                "counts": {"complete_missing_dialect": 3},
            },
            "transform_inventory": {
                "path": "_qc_transform_inventory.jsonl",
                "records": 10,
                "retained": 10,
                "removed_by_cleaner": 0,
            },
            "semantic_text_cleaning": {
                "applied": False,
                "authority": "formosan-mt-standardization",
            },
            "validators": [],
        }
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            print_qc_rule_summary("ami", result)
        output = terminal.getvalue()
        self.assertIn("QC rule summary [ami]", output)
        self.assertIn("standard copied from original: 2", output)
        self.assertIn("complete missing dialect: 3", output)
        self.assertIn(
            "Semantic text cleaning: not applied during XML preparation",
            output,
        )

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
