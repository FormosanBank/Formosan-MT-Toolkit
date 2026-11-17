from __future__ import annotations

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from clean_xml import (
    SYNC_FILES,
    audit_standard_tiers,
    ensure_standard_tiers,
    finalize_transform_inventory,
    tag_transform_sources,
)
from make_corpus import extract_file
from support import mt_records_for_xml
from xml_repairs import repair_mt_xml_structure


class StandardTierTests(unittest.TestCase):
    def test_pinned_qc_snapshot_includes_required_root_registries(self) -> None:
        self.assertEqual(
            SYNC_FILES,
            {"dialects.csv", "languages.csv", "standards.csv"},
        )

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
                mt_records=mt_records_for_xml(path, directory),
            )
            self.assertEqual(rows, [])
            self.assertEqual(
                extraction["mt_ineligible:empty_source_standard"],
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
                mt_records=mt_records_for_xml(path, directory),
            )
            self.assertEqual(rows, [])
            self.assertEqual(
                extraction["mt_ineligible:contains_unclear"],
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
                mt_records=mt_records_for_xml(path, directory),
            )
            extracted = {
                pair.xml_id: pair.formosan_sentence for pair in pairs
            }
            self.assertEqual(extracted["variant"], "'arupaara")
            self.assertEqual(extracted["variant-m"], "usa")
            self.assertEqual(extraction["w_units_seen"], 1)
            self.assertEqual(extraction["m_units_seen"], 2)
            self.assertEqual(
                extraction["mt_ineligible:empty_source_standard"],
                1,
            )
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

    def test_alt_version_forms_do_not_count_as_duplicate_tiers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_xml(
                directory,
                '<S id="s1"><FORM kindOf="original">Ci panay.</FORM>'
                '<FORM kindOf="original" ver="alt">Ci panay k-u.</FORM>'
                "</S>",
            )
            records = tag_transform_sources(directory)
            self.assertEqual(
                [row["formosan_source_standard"] for row in records.values()],
                ["Ci panay."],
            )

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
