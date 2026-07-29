from __future__ import annotations

import json
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

from clean_xml import audit_standard_tiers, ensure_standard_tiers  # noqa: E402
from fetch_xml import classify_xml, git_blob_sha  # noqa: E402
from pipeline_common import load_pipeline_config  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
