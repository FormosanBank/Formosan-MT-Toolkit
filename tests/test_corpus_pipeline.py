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
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

import fetch_xml  # noqa: E402
import github_snapshot  # noqa: E402
import stage_cache  # noqa: E402
from build_big_corpus import (  # noqa: E402
    corpus_frame,
    discover_inputs,
    estimated_output_bytes,
    require_clean_pairs,
    write_csv_atomic,
    write_target,
)
from build_context import BuildPaths, replace_with_hardlink  # noqa: E402
from build_release import package_training_provenance  # noqa: E402
from clean_xml import (  # noqa: E402
    SYNC_FILES,
    audit_standard_tiers,
    classify_translation_version_repairs,
    ensure_standard_tiers,
    finalize_transform_inventory,
    tag_transform_sources,
)
from corpus_quality import (  # noqa: E402
    alignment_quality,
    apply_quality_rules,
    deduplicate_pairs,
    has_annotation_gloss_structure,
    has_lexical_morphological_gloss,
    lexical_quality_reason,
    normalize_dataframe,
    normalize_text,
    target_metadata_reason,
    target_units,
)
from fetch_xml import (  # noqa: E402
    classify_xml,
    download_blob_for_languages,
    get_tree,
    git_blob_sha,
    load_or_create_repository_snapshot,
    repository_selection,
    resolve_default_repository_refs,
    write_blob_cache,
)
from filter_split_corpus import (  # noqa: E402
    filter_rule_counts,
    print_filter_rule_summary,
    read_csv,
)
from make_corpus import extract_file, extract_file_targets  # noqa: E402
from mt_standardization import (  # noqa: E402
    DEFAULT_PROFILE_PATH,
    StandardizationContext,
    assert_idempotent,
    standardize_text,
)
from mt_standardization import (
    load_profile as load_mt_standard_profile,
)
from mt_standardization import (
    profile_sha256 as mt_profile_sha256,
)
from pipeline_common import load_pipeline_config, write_columnar_cache  # noqa: E402
from pivot import (  # noqa: E402
    Direction,
    load_cache,
    load_cache_chain,
    make_cache_key,
    pivot_candidate_reason,
    synthetic_row,
    write_pivot_output,
)
from publish_huggingface_dataset import validate_release_frame  # noqa: E402
from qc_change_audit import (  # noqa: E402
    classify_cleaner_field_changes,
)
from qc_reporting import (  # noqa: E402
    parse_cleaner_transformation,
    print_qc_rule_summary,
    run_cleaner_command,
    summarize_validator_findings,
)
from stage_cache import (  # noqa: E402
    cached_stage_valid,
    file_inventory,
    load_stage_cache,
    record_cached_stage,
)
from xml_repairs import repair_mt_xml_structure  # noqa: E402

MT_PROFILE = load_mt_standard_profile(DEFAULT_PROFILE_PATH)
MT_PROFILE_HASH = mt_profile_sha256(DEFAULT_PROFILE_PATH)


class HuggingFaceDatasetPublisherTests(unittest.TestCase):
    def release_frame(self, repository: str, source: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "lang_code": ["ami"],
                "formosan_sentence": ["O maan ko faloco'?"],
                "english_sentence": ["What are you thinking?"],
                "dialect": ["Coastal"],
                "source": [source],
                "split": ["test"],
                "repository": [repository],
                "pivot_origin": ["original"],
                "mt_eval_eligible": [True],
                "row_type": ["sentence"],
                **{
                    column: ["value"]
                    for column in (
                        "row_id",
                        "source_record_id",
                        "content_sha256",
                        "repository_commit",
                        "xml_path",
                        "xml_id",
                        "kindOf",
                        "standard_namespace",
                        "standard_origin",
                        "pivot_provider",
                        "pivot_direction",
                        "eval_tier",
                        "document_id",
                    )
                },
            }
        )

    def test_private_release_allows_private_repository_rows(self) -> None:
        frame = self.release_frame("Private-Dev-Repo", "Private-Dev-Repo/Final_XML/a.xml")
        validate_release_frame(frame, "english_sentence", private_release=True)

    def test_public_release_rejects_private_repository_rows(self) -> None:
        frame = self.release_frame("Private-Dev-Repo", "Private-Dev-Repo/Final_XML/a.xml")
        with self.assertRaisesRegex(SystemExit, "non-public repository"):
            validate_release_frame(frame, "english_sentence", private_release=False)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mt_contract_fields(
    text: str,
    *,
    row_type: str = "sentence",
    pivot_origin: str = "original",
    confidence: str = "unchanged",
) -> dict[str, object]:
    return {
        "kindOf": "standard",
        "standard_namespace": "formosan-mt",
        "formosan_mt_standard": text,
        "mt_standard_sha256": text_sha256(text),
        "mt_normalization_status": "accepted",
        "mt_normalization_confidence": confidence,
        "mt_eval_eligible": (
            pivot_origin == "original"
            and row_type == "sentence"
            and confidence in {"unchanged", "safe"}
        ),
        "mt_normalization_reason": "",
        "mt_standard_profile": MT_PROFILE["profile_id"],
        "mt_standard_profile_sha256": MT_PROFILE_HASH,
    }


def mt_records_for_xml(
    path: Path,
    directory: Path,
    *,
    language: str = "ami",
) -> dict[tuple[str, str, int, str], dict[str, object]]:
    relative = str(path.relative_to(directory))
    repository = Path(relative).parts[0] if Path(relative).parts else "FixtureRepo"
    records: dict[tuple[str, str, int, str], dict[str, object]] = {}
    unit_index = 0
    for element in ET.parse(path).getroot().iter():
        if element.tag not in {"S", "W", "M"}:
            continue
        element_index = unit_index
        unit_index += 1
        xml_id = (element.get("id") or "").strip()
        standard = element.find("FORM[@kindOf='standard']")
        original = element.find("FORM[@kindOf='original']")
        selected = standard if standard is not None else original
        source_standard = (
            "" if selected is None else "".join(selected.itertext()).strip()
        )
        original_text = (
            "" if original is None else "".join(original.itertext()).strip()
        )
        contains_unclear = (
            selected is not None
            and selected.find(".//UNCLEAR") is not None
        )
        row_type = {"S": "sentence", "W": "lexeme", "M": "morpheme"}[
            element.tag
        ]
        context = StandardizationContext(
            language=language,
            row_type=row_type,
            repository=repository,
            xml_path=relative,
        )
        result = standardize_text(
            source_standard,
            context=context,
            profile=MT_PROFILE,
            contains_unclear=contains_unclear,
        )
        assert_idempotent(result, context=context, profile=MT_PROFILE)
        records[(relative, element.tag, element_index, xml_id)] = {
            "standard_origin": (
                "provided" if standard is not None else "derived_from_original"
            ),
            "formosan_original_raw": original_text,
            "formosan_source_standard": source_standard,
            "formosan_mt_standard": result.text,
            "source_standard_sha256": text_sha256(source_standard),
            "mt_standard_sha256": text_sha256(result.text),
            "contains_unclear_source": contains_unclear,
            "mt_normalization_status": result.status,
            "mt_normalization_confidence": result.confidence,
            "mt_eval_eligible": result.eval_eligible,
            "mt_normalization_reason": result.reason,
            "mt_transformations": json.dumps(result.transformations),
            "mt_unresolved_markers": "|".join(result.unresolved_markers),
            "speaker_label": result.speaker_label,
            "mt_standard_profile": MT_PROFILE["profile_id"],
            "mt_standard_profile_sha256": MT_PROFILE_HASH,
        }
    return records


class MTStandardizationTests(unittest.TestCase):
    def test_profile_pins_standardizer_implementation(self) -> None:
        implementation = ROOT / "scripts/local/mt_standardization.py"
        self.assertEqual(
            MT_PROFILE["implementation_sha256"],
            hashlib.sha256(implementation.read_bytes()).hexdigest(),
        )

    def standardize(
        self,
        value: str,
        *,
        language: str = "ami",
        row_type: str = "sentence",
        profile: dict[str, object] | None = None,
    ):
        context = StandardizationContext(
            language=language,
            row_type=row_type,
            repository="FixtureRepo",
            xml_path="Final_XML/fixture.xml",
        )
        result = standardize_text(
            value,
            context=context,
            profile=profile or MT_PROFILE,
        )
        assert_idempotent(
            result,
            context=context,
            profile=profile or MT_PROFILE,
        )
        return result

    def test_observed_notation_families_are_deterministic(self) -> None:
        cases = {
            "ma-ku-ta-mul": ("makutamul", "safe", True),
            "k<om>aen": ("komaen", "safe", True),
            "cinim∅kee": ("cinimkee", "safe", True),
            "lemang(e)da": ("lemangeda", "ambiguous", False),
            "imatiya/hatini": ("imatiya", "ambiguous", False),
            "lali:ma": ("lali:ma", "unchanged", True),
            "== tjevus ==": ("tjevus", "safe", True),
            "mn_gluw": ("mngluw", "ambiguous", False),
            "Speaker: malu": (
                "Speaker: malu",
                "unchanged",
                True,
            ),
            "a ~ b": ("a", "ambiguous", False),
            "itaial ~ 'taial ~ taial": ("itaial", "ambiguous", False),
            "{um}ali": ("umali", "safe", True),
            "mha oy~~~ binah": ("mha oy binah", "safe", True),
            "a~ sawni qaniy ga _~ aw yaqu": (
                "a sawni qaniy ga yaqu",
                "ambiguous",
                False,
            ),
            "kalin(-na)-lumah=in": (
                "kalinnalumahin",
                "ambiguous",
                False,
            ),
            "kali(n(na)luma)hin": (
                "kalinnalumahin",
                "ambiguous",
                False,
            ),
            "  Speaker: malu  ": (
                "Speaker: malu",
                "safe",
                True,
            ),
            "東壘(turuy)- kn-bong": (
                "東壘turuy knbong",
                "ambiguous",
                False,
            ),
        }
        for source, expected in cases.items():
            with self.subTest(source=source):
                result = self.standardize(source)
                self.assertEqual(
                    (result.text, result.confidence, result.eval_eligible),
                    expected,
                )
                self.assertEqual(result.status, "accepted")

    def test_initial_label_is_metadata_not_deleted_text(self) -> None:
        source = 'Sowal ni Yis: "Ano cima ko Kawas?"'
        result = self.standardize(source)
        self.assertEqual(result.text, source)
        self.assertEqual(result.speaker_label, "Sowal ni Yis")
        self.assertEqual(result.transformations, ())

    def test_unresolved_or_nonlinguistic_input_is_not_accepted(self) -> None:
        self.assertEqual(
            self.standardize("https://example.org/a/b").status,
            "quarantine",
        )
        self.assertEqual(
            self.standardize("a____b").status,
            "quarantine",
        )
        self.assertEqual(self.standardize("∅").status, "ineligible")

    def test_profile_applies_consistently_to_all_formosan_languages(self) -> None:
        for language in (
            "ami", "bnn", "ckv", "dru", "pwn", "pyu", "ssf", "sxr",
            "szy", "tao", "tay", "trv", "tsu", "xnb", "xsy",
        ):
            with self.subTest(language=language):
                result = self.standardize("ma-ku", language=language)
                self.assertEqual(result.text, "maku")
                self.assertTrue(result.eval_eligible)

    def test_reviewed_source_override_can_enable_ambiguous_eval(self) -> None:
        profile = json.loads(json.dumps(MT_PROFILE))
        profile["source_overrides"] = [
            {
                "repository": "FixtureRepo",
                "path_regex": "Final_XML/fixture\\.xml$",
                "reviewed_ambiguous": True,
                "policy": {},
            }
        ]
        result = self.standardize("lemang(e)da", profile=profile)
        self.assertEqual(result.confidence, "ambiguous")
        self.assertTrue(result.eval_eligible)


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


class AcquisitionTests(unittest.TestCase):
    def test_multi_language_fetch_parses_and_routes_one_cached_blob(self) -> None:
        xml_bytes = (
            b'<TEXT xmlns:xml="http://www.w3.org/XML/1998/namespace" '
            b'xml:lang="ami" dialect="Coastal"><S /></TEXT>'
        )
        blob = git_blob_sha(xml_bytes)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cache"
            cache.mkdir()
            (cache / f"{blob}.xml").write_bytes(xml_bytes)
            out_dirs = {"ami": root / "ami", "tay": root / "tay"}
            for path in out_dirs.values():
                path.mkdir()
            with mock.patch.object(fetch_xml, "RAW_XML_CACHE_DIR", cache):
                result = download_blob_for_languages(
                    "FormosanBank",
                    "FixtureRepo",
                    {"path": "Final_XML/Atayal/misleading-name.xml", "sha": blob},
                    ("ami", "tay"),
                    None,
                    "a" * 40,
                    out_dirs,
                    None,
                    download_retries=1,
                    retry_base_sleep=0,
                    retry_max_sleep=0,
                )
            self.assertEqual(result["ami"].status, "kept")
            self.assertEqual(result["tay"].status, "source_language_mismatch")
            self.assertTrue((out_dirs["ami"] / result["ami"].destination).is_file())
            self.assertFalse(list(out_dirs["tay"].rglob("*.xml")))

    def test_aggregation_uses_xml_language_metadata_not_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tay_zh_processed.csv"
            frame = pd.DataFrame(
                [
                    {
                        **mt_contract_fields("O maan ko faloco'?"),
                        "lang_code": "ami",
                        "formosan_sentence": "O maan ko faloco'?",
                        "english": "What are you thinking?",
                        "target_lang": "eng",
                        "row_id": "row-1",
                        "source_record_id": "record-1",
                        "source": "Repo/XML/misleading-path.xml",
                        "row_type": "sentence",
                        "source_bucket": "narrative",
                    }
                ]
            )
            frame.to_csv(path, index=False)

            target, loaded, input_type = corpus_frame(path)

            frame.loc[0, "target_lang"] = ""
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(SystemExit, "TRANSL/@xml:lang"):
                corpus_frame(path)

        self.assertEqual(target, "en")
        self.assertEqual(input_type, "pairwise")
        self.assertEqual(loaded.loc[0, "lang_code"], "ami")
        self.assertNotIn("source_bucket", loaded.columns)
        self.assertEqual(
            loaded.loc[0, "english_sentence"],
            "What are you thinking?",
        )

    def test_aggregation_normalizes_columnar_provenance_to_csv_text(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            loaded = []
            for name, row_id, xml_id in (
                ("numeric", "row-1", 0),
                ("text", "row-2", "sample_S1"),
            ):
                path = root / f"{name}.csv"
                frame = pd.DataFrame(
                    [
                        {
                            **mt_contract_fields("malu ku su."),
                            "lang_code": "ami",
                            "formosan_sentence": "malu ku su.",
                            "english": f"Good {name} example.",
                            "target_lang": "eng",
                            "row_id": row_id,
                            "source_record_id": f"record-{row_id}",
                            "source": f"Repo/XML/{name}.xml",
                            "row_type": "sentence",
                            "xml_id": xml_id,
                        }
                    ]
                )
                frame.to_csv(path, index=False)
                write_columnar_cache(frame, path)
                _, cached, _ = corpus_frame(path)
                loaded.append(cached)

            output = root / "big_corpus_en.csv"
            combined = write_target(loaded, output, "english_sentence")

            self.assertEqual(combined["xml_id"].tolist(), ["0", "sample_S1"])
            self.assertTrue(output.with_suffix(".parquet").is_file())

    def test_graphql_repository_resolution_preserves_selection_order(self) -> None:
        response = mock.Mock()
        response.json.return_value = {
            "data": {
                "organization": {
                    "repositories": {
                        "nodes": [
                            {
                                "name": "RepoA",
                                "defaultBranchRef": {
                                    "name": "main",
                                    "target": {"oid": "a" * 40},
                                },
                            },
                            {
                                "name": "RepoB",
                                "defaultBranchRef": {
                                    "name": "release",
                                    "target": {"oid": "b" * 40},
                                },
                            },
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        with mock.patch.object(
            github_snapshot,
            "api_post",
            return_value=response,
        ):
            refs = resolve_default_repository_refs(
                "FormosanBank",
                ["RepoB", "RepoA"],
            )
        self.assertEqual([ref.name for ref in refs], ["RepoB", "RepoA"])
        self.assertEqual(refs[0].requested_ref, "release")

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
                mock.patch.object(
                    github_snapshot,
                    "CACHE_DIR",
                    Path(temporary),
                ),
                mock.patch.object(
                    github_snapshot,
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

    def test_private_xml_discovery_accepts_both_supported_roots(self) -> None:
        self.assertTrue(
            fetch_xml.is_private_release_xml_path(
                "Final_XML/Paiwan/sample.xml"
            )
        )
        self.assertTrue(
            fetch_xml.is_private_release_xml_path("XML/Paiwan/sample.xml")
        )
        self.assertFalse(
            fetch_xml.is_private_release_xml_path(
                "archive/XML/Paiwan/sample.xml"
            )
        )
        self.assertFalse(
            fetch_xml.is_private_release_xml_path("Final_XML/README.md")
        )

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
                    github_snapshot,
                    "get_default_branch",
                    return_value="main",
                ) as branch,
                mock.patch.object(
                    github_snapshot,
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
                    github_snapshot,
                    "get_default_branch",
                    side_effect=AssertionError("snapshot should be reused"),
                ),
                mock.patch.object(
                    github_snapshot,
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

    def test_cleaning_preserves_parentheses_and_structural_sentence_type(self) -> None:
        self.assertEqual(normalize_text("  It is good (today).  ").text, "It is good (today).")
        frame = pd.DataFrame(
            [
                {
                    **mt_contract_fields("hay."),
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

    def test_cleaning_does_not_infer_row_type_from_provenance(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "formosan": "mako ko tawki niyam.",
                    "english": "The weather is good today.",
                    "row_type": "",
                    "source": "Formosan-ILRDF_Dicts/Final_XML/Amis/sample.xml",
                }
            ]
        )
        normalized, _ = normalize_dataframe(frame, "formosan", "english")
        self.assertEqual(normalized.loc[0, "row_type"], "unknown")

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
                    **mt_contract_fields("*malu"),
                    "row_id": "asterisk",
                    "ami": "*malu",
                    "english": "bad",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/a.xml",
                },
                {
                    **mt_contract_fields("456otca"),
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

    def test_target_glosses_are_removed_without_touching_normal_prose(self) -> None:
        rows = [
            {
                **mt_contract_fields("mi'o mici bonu to tacumu."),
                "row_id": "labelled-gloss",
                "ami": "mi'o mici bonu to tacumu.",
                "english": "AV.REAL=1SG AV-want eat.AV NTOP banana",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "interlinear-gloss",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("mo mosi to ca'hu to pooyoyo."),
                "row_id": "unlabelled-annotation",
                "ami": "mo mosi to ca'hu to pooyoyo.",
                "english": (
                    "Father put pants on the chair. "
                    "(actor=TOP, AV verb, AV AUX mo)"
                ),
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("hay ci aki anini."),
                "row_id": "normal-parenthetical",
                "ami": "hay ci aki anini.",
                "english": "Aki is here (today).",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("maolah ci aki to ISO 639."),
                "row_id": "normal-acronym",
                "ami": "maolah ci aki to ISO 639.",
                "english": "Aki uses the ISO-639 language code.",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("babalivan sanglav ita."),
                "row_id": "mixed-case-gloss",
                "ami": "babalivan sanglav ita.",
                "english": "buy-LndF-frequently vegetable that-place/there",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("nibuLubuLluan kuaDa Lulay."),
                "row_id": "gloss-chain",
                "ami": "nibuLubuLluan kuaDa Lulay.",
                "english": "That child was-being-instructed-and-instructed",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("masalaw ku takaraw anini."),
                "row_id": "normal-hyphen-chain",
                "ami": "masalaw ku takaraw anini.",
                "english": "This is a state-of-the-art method.",
                "kindOf": "standard",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Stories/example.xml",
            },
        ]
        frame = pd.DataFrame(rows)
        normalized, _ = normalize_dataframe(frame, "ami", "english")
        accepted, rejected, counts = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )

        self.assertEqual(
            set(accepted["row_id"]),
            {"normal-parenthetical", "normal-acronym", "normal-hyphen-chain"},
        )
        reasons = dict(
            zip(
                rejected["row_id"],
                rejected["disposition_reason"],
                strict=True,
            )
        )
        self.assertEqual(reasons["labelled-gloss"], "target_gloss_translation")
        self.assertEqual(reasons["unlabelled-annotation"], "target_annotation_gloss")
        self.assertEqual(reasons["mixed-case-gloss"], "target_annotation_gloss")
        self.assertEqual(reasons["gloss-chain"], "target_annotation_gloss")
        self.assertEqual(counts["rejected:target_gloss_translation"], 1)
        self.assertEqual(counts["quarantine:target_annotation_gloss"], 3)
        self.assertTrue(has_annotation_gloss_structure("catch (AF-Imp)"))
        self.assertFalse(has_annotation_gloss_structure("Aki is here (today)."))
        self.assertFalse(has_annotation_gloss_structure("The ISO-639 language code."))
        self.assertFalse(has_annotation_gloss_structure("a state-of-the-art method"))

    def test_linguistic_analyses_are_quarantined(self) -> None:
        rows = [
            {
                **mt_contract_fields("na pa cun ku vuvu nami."),
                "row_id": "english-analysis",
                "ami": "na pa cun ku vuvu nami.",
                "english": (
                    "I saw our elders. na-pa-cun: perfective (PRF), nominative "
                    "(NOM), oblique (OBL); the root is cun"
                ),
                "row_type": "sentence",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("maolah ku lamit anini."),
                "row_id": "normal-root",
                "ami": "maolah ku lamit anini.",
                "english": "The root is visible beside the road.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(pd.DataFrame(rows), "ami", "english")
        accepted, rejected, counts = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )

        self.assertEqual(set(accepted["row_id"]), {"normal-root"})
        self.assertEqual(
            rejected.iloc[0]["disposition_reason"],
            "target_linguistic_analysis",
        )
        self.assertEqual(counts["quarantine:target_linguistic_analysis"], 1)

    def test_english_language_and_escaping_rules_preserve_uncertain_training_rows(self) -> None:
        rows = [
            {
                **mt_contract_fields("phpure ppuqun bubu rudan de."),
                "row_id": "formosan-target",
                "ami": "phpure ppuqun bubu rudan de.",
                "english": "ini pdai cmuwaq iyu",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
            {
                **mt_contract_fields("malo ku lalan anini."),
                "row_id": "uncertain-english",
                "ami": "malo ku lalan anini.",
                "english": "Heavy rainfall flooded downtown streets yesterday.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("mahapinang ci aki anini."),
                "row_id": "unbalanced-quote",
                "ami": "mahapinang ci aki anini.",
                "english": 'He said "hello.',
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("masadak ci Caʉpʉ anini."),
                "row_id": "english-with-formosan-name",
                "ami": "masadak ci Caʉpʉ anini.",
                "english": "It's Caʉpʉ's.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("kmal ku taw anini."),
                "row_id": "repeated-english-word",
                "ami": "kmal ku taw anini.",
                "english": "Knock, knock, knock, knock.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("masadak ci ama anini."),
                "row_id": "bad-target-escape",
                "ami": "masadak ci ama anini.",
                "english": r"Dad said, \Don't say anything first.",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
            {
                **mt_contract_fields('malu ku lalan.""'),
                "row_id": "bad-source-quote",
                "ami": 'malu ku lalan.""',
                "english": "The road is good today.",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(pd.DataFrame(rows), "ami", "english")
        accepted, rejected, counts = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )

        self.assertEqual(
            set(accepted["row_id"]),
            {
                "uncertain-english",
                "unbalanced-quote",
                "english-with-formosan-name",
                "repeated-english-word",
            },
        )
        flags = dict(zip(accepted["row_id"], accepted["quality_flags"], strict=True))
        self.assertIn("english_language_uncertain", flags["uncertain-english"])
        self.assertIn("unbalanced_target_delimiters", flags["unbalanced-quote"])
        reasons = dict(zip(rejected["row_id"], rejected["disposition_reason"], strict=True))
        self.assertEqual(reasons["formosan-target"], "english_target_language_mismatch")
        self.assertEqual(reasons["bad-target-escape"], "malformed_target_escaping")
        self.assertEqual(reasons["bad-source-quote"], "malformed_source_escaping")
        self.assertEqual(counts["quarantine:english_target_language_mismatch"], 1)

    def test_chinese_gloss_and_obvious_alignment_failures_are_quarantined(self) -> None:
        rows = [
            {
                **mt_contract_fields("babalivan sanglav ita."),
                "row_id": "chinese-gloss",
                "ami": "babalivan sanglav ita.",
                "chinese": "互相-拿",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("one two three four five six seven eight nine ten eleven twelve"),
                "row_id": "truncated",
                "ami": "one two three four five six seven eight nine ten eleven twelve",
                "chinese": "去打獵",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("maolah ci aki to ISO 639."),
                "row_id": "normal-acronym",
                "ami": "maolah ci aki to ISO 639.",
                "chinese": "這是ISO-639語言代碼。",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("maolah ku tamdaw anini."),
                "row_id": "chinese-analysis",
                "ami": "maolah ku tamdaw anini.",
                "chinese": "那些梅子很好吃。（受事焦點：動詞-主事者-受事者）",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("mafu ku lima nira."),
                "row_id": "truncated-analysis",
                "ami": "mafu ku lima nira.",
                "chinese": "他扶著我的手。（主事焦點句主語：主格標記+[關係子句",
                "row_type": "sentence",
                "translation_kind": "free",
                "source": "Grammar/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(
            pd.DataFrame(rows),
            "ami",
            "chinese",
        )
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="chinese",
            target_language="chinese",
            keep_redactions=False,
        )

        self.assertEqual(set(accepted["row_id"]), {"normal-acronym"})
        reasons = dict(
            zip(
                rejected["row_id"],
                rejected["disposition_reason"],
                strict=True,
            )
        )
        self.assertEqual(reasons["chinese-gloss"], "target_annotation_gloss")
        self.assertEqual(reasons["truncated"], "obvious_alignment_mismatch")
        self.assertEqual(reasons["chinese-analysis"], "target_linguistic_analysis")
        self.assertEqual(reasons["truncated-analysis"], "target_linguistic_analysis")
        self.assertEqual(target_units("他是誰？", "chinese"), 3)

    def test_bilingual_prompt_and_analysis_artifacts_are_quarantined(self) -> None:
        rows = [
            {
                **mt_contract_fields("maita ku su anini."),
                "row_id": "mixed-english",
                "ami": "maita ku su anini.",
                "target": "This reference still contains 中文 text.",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
            {
                **mt_contract_fields("mita ku su anini."),
                "row_id": "prompt",
                "ami": "mita ku su anini.",
                "target": "Source: Psalm 23:1, the Lord is my shepherd.",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(pd.DataFrame(rows), "ami", "target")
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="target",
            target_language="english",
            keep_redactions=False,
        )

        self.assertTrue(accepted.empty)
        reasons = dict(zip(rejected["row_id"], rejected["disposition_reason"], strict=True))
        self.assertEqual(reasons["mixed-english"], "english_target_script_mismatch")
        self.assertEqual(reasons["prompt"], "target_prompt_scaffolding")

    def test_chinese_source_copy_and_direct_analysis_are_quarantined(self) -> None:
        rows = [
            {
                **mt_contract_fields("nanicowaay?"),
                "row_id": "copied-source",
                "ami": "nanicowaay?",
                "target": "nanicowaay?我聞到臭味。",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("uwas lmuhuw qani."),
                "row_id": "direct-analysis",
                "ami": "uwas lmuhuw qani.",
                "target": "uwas lmuhuw語彙中的詞根是lmuhuw。",
                "row_type": "sentence",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("aed ca qani."),
                "row_id": "equals-analysis",
                "ami": "aed ca qani.",
                "target": "aed = ca 這件事很好。",
                "row_type": "sentence",
                "source": "Grammar/example.xml",
            },
            {
                **mt_contract_fields("maolah ci Mayaw Aping."),
                "row_id": "proper-name",
                "ami": "maolah ci Mayaw Aping.",
                "target": "Mayaw Aping是今天的講者。",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(pd.DataFrame(rows), "ami", "target")
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="target",
            target_language="chinese",
            keep_redactions=False,
        )

        self.assertEqual(set(accepted["row_id"]), {"proper-name"})
        reasons = dict(zip(rejected["row_id"], rejected["disposition_reason"], strict=True))
        self.assertEqual(reasons["copied-source"], "target_copied_source_clause")
        self.assertEqual(reasons["direct-analysis"], "target_linguistic_analysis")
        self.assertEqual(reasons["equals-analysis"], "target_linguistic_analysis")

    def test_embedded_reference_metadata_is_detected_without_rejecting_aligned_lists(self) -> None:
        self.assertEqual(
            target_metadata_reason("A translation [translation missing] here."),
            "embedded_missing_translation_marker",
        )
        self.assertEqual(
            target_metadata_reason("Translation missing: the speaker continued."),
            "embedded_missing_translation_marker",
        )
        self.assertEqual(
            target_metadata_reason("The fox ran. (Source: Aesop)"),
            "target_provenance_note",
        )
        self.assertEqual(
            target_metadata_reason("The fox ran. Source: 1912 edition."),
            "target_provenance_note",
        )
        self.assertEqual(
            target_metadata_reason("Retold from Aesop's Fables."),
            "target_provenance_note",
        )
        self.assertEqual(
            target_metadata_reason(
                "The first five books were translated from Hebrew into Greek."
            ),
            "",
        )
        self.assertEqual(
            target_metadata_reason("Literal translation: carry on the shoulder"),
            "target_translation_commentary",
        )
        self.assertEqual(
            target_metadata_reason("Literal meaning: carry on the shoulder"),
            "target_translation_commentary",
        )
        self.assertEqual(
            target_metadata_reason(
                "1. My feelings are hurt. 2. I have heart pain.",
                source="makeluk ku sunis",
            ),
            "target_numbered_multi_reference",
        )
        self.assertEqual(
            target_metadata_reason(
                "1. First sentence. 2. Second sentence.",
                source="1. qani sa. 2. qasa sa.",
            ),
            "",
        )

    def test_model_length_overflow_is_quarantined_before_splitting(self) -> None:
        long_formosan = " ".join(["qani"] * 385)
        long_english = " ".join(["translation"] * 385)
        rows = [
            {
                **mt_contract_fields(long_formosan),
                "row_id": "long-source",
                "ami": long_formosan,
                "target": "This is a complete translation.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("qani sa kapah."),
                "row_id": "long-target",
                "ami": "qani sa kapah.",
                "target": long_english,
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
            {
                **mt_contract_fields("qani sa kapah."),
                "row_id": "normal",
                "ami": "qani sa kapah.",
                "target": "This is a complete translation.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(pd.DataFrame(rows), "ami", "target")
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="target",
            target_language="english",
            keep_redactions=False,
            max_units_per_side=384,
        )

        self.assertEqual(list(accepted["row_id"]), ["normal"])
        reasons = dict(
            zip(
                rejected["row_id"],
                rejected["disposition_reason"],
                strict=True,
            )
        )
        self.assertEqual(
            reasons["long-source"],
            "formosan_model_length_overflow",
        )
        self.assertEqual(
            reasons["long-target"],
            "target_model_length_overflow",
        )

    def test_sentence_shaped_lexical_material_is_train_only(self) -> None:
        _, english_flags = alignment_quality(
            "mafu qani",
            "(figuratively) Five only.",
            target_language="english",
            row_type="sentence",
        )
        _, narrative_flags = alignment_quality(
            "manu tjumaljan nua qaliqali.",
            "and they told him.",
            target_language="english",
            row_type="sentence",
        )
        _, chinese_flags = alignment_quality(
            "gung quzang qipu qaca qema qali",
            "gung（牛） quzang（蝦） qipu（魚） qaca（鳥） qema（狗） qali（豬）",
            target_language="chinese",
            row_type="sentence",
        )

        self.assertIn("lexical_content_sentence", english_flags)
        self.assertNotIn("lexical_content_sentence", narrative_flags)
        self.assertIn("lexical_content_sentence", chinese_flags)

        long_source_reason, _ = alignment_quality(
            "one two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen",
            "This is a short title",
            target_language="english",
            row_type="sentence",
        )
        self.assertEqual(long_source_reason, "obvious_alignment_mismatch")

    def test_alignment_flags_keep_explanations_train_only(self) -> None:
        rows = [
            {
                **mt_contract_fields("one two three four five six seven"),
                "row_id": "heading",
                "ami": "one two three four five six seven",
                "english": "Work Objectives",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
            {
                **mt_contract_fields("mukesi na puran"),
                "row_id": "definition",
                "ami": "mukesi na puran",
                "english": "a ceremonial nut; compare another traditional form (used at harvest)",
                "row_type": "sentence",
                "source": "Dictionary/example.xml",
            },
            {
                **mt_contract_fields("one two three four five"),
                "row_id": "short-heading",
                "ami": "one two three four five",
                "english": "Those Children",
                "row_type": "sentence",
                "source": "Lessons/example.xml",
            },
            {
                **mt_contract_fields("kupongluswa acele."),
                "row_id": "compact-sentence",
                "ami": "kupongluswa acele.",
                "english": "I will bring you some water.",
                "row_type": "sentence",
                "source": "Stories/example.xml",
            },
        ]
        normalized, _ = normalize_dataframe(
            pd.DataFrame(rows),
            "ami",
            "english",
        )
        accepted, rejected, _ = apply_quality_rules(
            normalized,
            source_column="ami",
            target_column="english",
            target_language="english",
            keep_redactions=False,
        )

        self.assertEqual(
            set(accepted["row_id"]),
            {"definition", "short-heading", "compact-sentence"},
        )
        self.assertEqual(
            rejected.iloc[0]["disposition_reason"],
            "target_heading_alignment_mismatch",
        )
        flags = dict(zip(accepted["row_id"], accepted["quality_flags"], strict=True))
        self.assertIn("definition_like_sentence", flags["definition"])
        self.assertNotIn("lexical_content_sentence", flags["definition"])
        self.assertIn("heading_like_target", flags["short-heading"])
        self.assertIn("target_fragment", flags["short-heading"])
        self.assertEqual(flags["compact-sentence"], "")

    def test_standalone_lexemes_require_natural_target_text(self) -> None:
        rows = [
            {
                **mt_contract_fields("mafu", row_type="lexeme"),
                "row_id": "natural",
                "ami": "mafu",
                "english": "can be swallowed",
                "row_type": "lexeme",
                "xml_unit_context": "standalone_word",
                "translation_kind": "",
                "source": "Dictionary/sample.xml",
            },
            {
                **mt_contract_fields("maicangen", row_type="lexeme"),
                "row_id": "morphological",
                "ami": "maicangen",
                "english": "NEUT-dry-EN2",
                "row_type": "lexeme",
                "xml_unit_context": "standalone_word",
                "translation_kind": "",
                "source": "Dictionary/sample.xml",
            },
            {
                **mt_contract_fields("malaliop", row_type="lexeme"),
                "row_id": "ambiguous-target",
                "ami": "malaliop",
                "english": "wash-face",
                "row_type": "lexeme",
                "xml_unit_context": "standalone_word",
                "translation_kind": "",
                "source": "Dictionary/sample.xml",
            },
            {
                **mt_contract_fields("inner", row_type="lexeme"),
                "row_id": "ambiguous-structure",
                "ami": "inner",
                "english": "inside",
                "row_type": "lexeme",
                "xml_unit_context": "ambiguous_word",
                "translation_kind": "",
                "source": "Dictionary/sample.xml",
            },
        ]
        normalized, _ = normalize_dataframe(
            pd.DataFrame(rows),
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

        self.assertEqual(set(accepted["row_id"]), {"natural"})
        reasons = dict(
            zip(
                rejected["row_id"],
                rejected["disposition_reason"],
                strict=True,
            )
        )
        self.assertEqual(
            reasons,
            {
                "morphological": "target_morphological_gloss",
                "ambiguous-target": "ambiguous_lexical_translation",
                "ambiguous-structure": "ambiguous_lexical_structure",
            },
        )
        self.assertEqual(counts["accepted:ok"], 1)
        self.assertTrue(
            has_lexical_morphological_gloss(
                "主焦-SA-什麼=完成",
                target_language="chinese",
            )
        )
        self.assertEqual(
            lexical_quality_reason(
                "互相-拿",
                row_type="lexeme",
                xml_unit_context="standalone_word",
                target_language="chinese",
            ),
            "ambiguous_lexical_translation",
        )

    def test_aggregate_gate_refuses_gloss_contamination(self) -> None:
        frame = pd.DataFrame(
            {
                "source_record_id": ["bad-gloss"],
                "english_sentence": ["catch (AF-Imp)"],
                "translation_kind": [""],
            }
        )
        with self.assertRaises(SystemExit):
            require_clean_pairs(
                frame,
                target_column="english_sentence",
                target_language="english",
                path=Path("fixture.csv"),
            )

        lexical = pd.DataFrame(
            {
                "source_record_id": ["ambiguous-lexeme"],
                "english_sentence": ["wash-face"],
                "translation_kind": [""],
                "row_type": ["lexeme"],
                "xml_unit_context": ["standalone_word"],
            }
        )
        with self.assertRaises(SystemExit):
            require_clean_pairs(
                lexical,
                target_column="english_sentence",
                target_language="english",
                path=Path("lexical.csv"),
            )

        malformed = pd.DataFrame(
            {
                "source_record_id": ["bad-escape"],
                "formosan_sentence": ["maita ku su anini."],
                "english_sentence": [r"Dad said, \Don't say anything first."],
                "translation_kind": [""],
            }
        )
        with self.assertRaises(SystemExit):
            require_clean_pairs(
                malformed,
                target_column="english_sentence",
                target_language="english",
                path=Path("malformed.csv"),
            )

        wrong_language = pd.DataFrame(
            {
                "source_record_id": ["wrong-language"],
                "formosan_sentence": ["phpure ppuqun bubu rudan de."],
                "english_sentence": ["ini pdai cmuwaq iyu"],
                "translation_kind": [""],
            }
        )
        with self.assertRaises(SystemExit):
            require_clean_pairs(
                wrong_language,
                target_column="english_sentence",
                target_language="english",
                path=Path("wrong-language.csv"),
            )

    def test_quality_and_dedupe_conserve_every_input_row(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    **mt_contract_fields("malu."),
                    "row_id": "r1",
                    "ami": "malu.",
                    "english": "Good.",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/a.xml",
                },
                {
                    **mt_contract_fields("malu."),
                    "row_id": "r2",
                    "ami": "malu.",
                    "english": "Good.",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "source": "Stories/b.xml",
                },
                {
                    **mt_contract_fields("中文"),
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
                    "en": "7fc0951c31639cef7ad16c1343d95640f51cbe4466dd778aad2108a18560afb5",
                    "zh": "44bcc8b1dc77b9ae355729dd1c7e654769510f9eceab29f9018d8497380456e6",
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


if __name__ == "__main__":
    unittest.main()
