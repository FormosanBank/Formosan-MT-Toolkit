from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from make_corpus import extract_file, extract_file_targets
from support import mt_records_for_xml


class ExtractionTests(unittest.TestCase):
    def test_extraction_excludes_sentence_words_and_classifies_standalone_words(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "sample.xml"
            path.write_text(
                '<?xml version="1.0"?>'
                '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                'xml:lang="ami">'
                '<S id="s1"><FORM kindOf="standard">malu cira.</FORM>'
                '<TRANSL xml:lang="eng">He is well.</TRANSL>'
                '<W id="nested"><FORM kindOf="standard">malu</FORM>'
                '<TRANSL xml:lang="eng">STAT-good</TRANSL></W></S>'
                '<ENTRY><W id="standalone"><FORM kindOf="standard">mafu</FORM>'
                '<TRANSL xml:lang="eng">can be swallowed</TRANSL></W></ENTRY>'
                '<W id="outer"><FORM kindOf="standard">outer</FORM>'
                '<W id="ambiguous"><FORM kindOf="standard">inner</FORM>'
                '<TRANSL xml:lang="eng">inside</TRANSL></W></W>'
                '</TEXT>',
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
                target_codes={"eng"},
                tags={"W"},
                mt_records=mt_records_for_xml(path, directory),
            )

            self.assertEqual(
                {row.xml_id: row.xml_unit_context for row in rows},
                {
                    "standalone": "standalone_word",
                    "ambiguous": "ambiguous_word",
                },
            )
            self.assertEqual(stats["w_units_seen"], 4)
            self.assertEqual(
                stats["sentence_nested_word_units_excluded"],
                1,
            )

    def test_combined_extraction_matches_independent_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "sample.xml"
            path.write_text(
                '<?xml version="1.0"?>'
                '<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
                'xml:lang="ami"><S id="s1">'
                '<FORM kindOf="standard">malu.</FORM>'
                '<TRANSL xml:lang="eng">Good.</TRANSL>'
                '<TRANSL xml:lang="zho">很好。</TRANSL>'
                "</S></TEXT>",
                encoding="utf-8",
            )
            provenance = {
                "repository": "FixtureRepo",
                "repository_commit": "a" * 40,
                "source_path": "sample.xml",
            }
            mt_records = mt_records_for_xml(path, directory)
            combined, combined_stats = extract_file_targets(
                path,
                xml_dir=directory,
                provenance=provenance,
                targets={"english": {"eng"}, "chinese": {"zho"}},
                tags={"S"},
                mt_records=mt_records,
            )
            english, english_stats = extract_file(
                path,
                xml_dir=directory,
                provenance=provenance,
                target_codes={"eng"},
                tags={"S"},
                mt_records=mt_records,
            )
            chinese, chinese_stats = extract_file(
                path,
                xml_dir=directory,
                provenance=provenance,
                target_codes={"zho"},
                tags={"S"},
                mt_records=mt_records,
            )
            self.assertEqual(combined["english"], english)
            self.assertEqual(combined["chinese"], chinese)
            self.assertEqual(combined_stats["english"], english_stats)
            self.assertEqual(combined_stats["chinese"], chinese_stats)

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
                mt_records=mt_records_for_xml(path, directory),
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
                mt_records=mt_records_for_xml(path, directory),
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
                mt_records=mt_records_for_xml(path, directory),
            )
            self.assertEqual(rows, [])
            self.assertEqual(stats["mt_ineligible:empty_source_standard"], 1)
