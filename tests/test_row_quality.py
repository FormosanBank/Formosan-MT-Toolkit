from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from build_big_corpus import (
    require_clean_pairs,
)
from corpus_quality import (
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
from filter_split_corpus import (
    read_csv,
)
from support import mt_contract_fields


class RowQualityTests(unittest.TestCase):
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
