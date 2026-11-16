from __future__ import annotations

import hashlib
import unittest

import pandas as pd
from build_experiment_splits import build_hard_split
from mt_common import (
    add_normalized_columns,
    evaluation_candidate_mask,
    source_corpus,
    weighted_apportioned_counts,
)
from split_allocation import (
    GroupCandidate,
    choose_groups,
    fill_language_shortfalls,
    group_assignment_tolerance,
    split_targets,
)
from support import add_mt_contract
from validate_experiment import (
    validate_provenance,
    validate_splits,
)


class HardSplitTests(unittest.TestCase):
    def test_source_corpus_preserves_exact_public_and_private_identity(self) -> None:
        self.assertEqual(
            source_corpus("FormosanBank/Corpora/NTUFormosanCorpus/XML/Stories/Amis/a.xml"),
            "NTUFormosanCorpus",
        )
        self.assertEqual(
            source_corpus("Formosan-Zheng-ACL-2024/Final_XML/Atayal/parallel.xml"),
            "Formosan-Zheng-ACL-2024",
        )
        self.assertEqual(
            source_corpus("PrivateRepo/XML/Bunun/examples.xml"),
            "PrivateRepo",
        )

    def test_group_selection_reserves_the_other_evaluation_split(self) -> None:
        candidates = [
            GroupCandidate(group_id=group_id, eligible_rows=rows, priority=group_id)
            for group_id, rows in enumerate((104, 86, 16, 4))
        ]
        validation = choose_groups(candidates, 33, reserve_rows=98)
        validation_rows = sum(candidate.eligible_rows for candidate in candidates if candidate.group_id in validation)
        remaining = [candidate for candidate in candidates if candidate.group_id not in validation]
        test = choose_groups(remaining, 98, reserve_rows=0)
        test_rows = sum(candidate.eligible_rows for candidate in remaining if candidate.group_id in test)

        self.assertGreaterEqual(validation_rows, 33)
        self.assertGreaterEqual(test_rows, 98)

    def test_group_assignment_tolerance_matches_indivisible_group_size(self) -> None:
        indexes = pd.Index(range(6))
        group_ids = pd.Series([10, 10, 10, 10, 20, 30], index=indexes)
        candidates = pd.Series(True, index=indexes)

        self.assertEqual(
            group_assignment_tolerance(indexes, group_ids, candidates),
            3,
        )

    def test_split_targets_use_human_pairs_as_denominator(self) -> None:
        self.assertEqual(
            split_targets(
                human_rows=1_000,
                eligible_total=1_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (100, 50),
        )
        self.assertEqual(
            split_targets(
                human_rows=20_000,
                eligible_total=20_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (2_000, 1_000),
        )
        self.assertEqual(
            split_targets(
                human_rows=20_000,
                eligible_total=5_000,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            ),
            (2_000, 1_000),
        )
        with self.assertRaisesRegex(ValueError, "requires 3,000"):
            split_targets(
                human_rows=20_000,
                eligible_total=2_999,
                test_ratio=0.10,
                val_ratio=0.05,
                min_test_rows=0,
                min_validate_rows=0,
            )

    def test_weighted_apportionment_redistributes_capacity_shortfall(self) -> None:
        self.assertEqual(
            weighted_apportioned_counts(
                {"source_a": 900, "source_b": 100},
                {"source_a": 20, "source_b": 100},
                100,
            ),
            {"source_a": 20, "source_b": 80},
        )

    def test_source_path_does_not_override_row_eligibility(self) -> None:
        row = self.hard_split_fixture().iloc[[0]].copy()
        row["source"] = "Formosan-ILRDF-42-Language-Practice-Word-Lists/Final_XML/Atayal/word-list.xml"
        row["formosan_sentence"] = "one two three four"
        row["english_sentence"] = "first second third fourth"
        normalized = add_normalized_columns(
            row,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=4,
            min_target_tokens=4,
            min_combined_tokens=8,
            min_punctuated_combined_tokens=8,
        )
        self.assertTrue(bool(candidates.iloc[0]))

        normalized.loc[normalized.index[0], "row_type"] = "lexeme"
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=4,
            min_target_tokens=4,
            min_combined_tokens=8,
            min_punctuated_combined_tokens=8,
        )
        self.assertFalse(bool(candidates.iloc[0]))

    def test_evaluation_candidate_respects_model_length_limit(self) -> None:
        row = self.hard_split_fixture().iloc[[0]].copy()
        row["formosan_sentence"] = " ".join(["word"] * 385)
        row["english_sentence"] = "a complete translation"
        normalized = add_normalized_columns(
            row,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=2,
            min_target_tokens=2,
            min_combined_tokens=6,
            min_punctuated_combined_tokens=5,
            max_eval_units_per_side=384,
        )
        self.assertFalse(bool(candidates.iloc[0]))

    def test_hard_split_rejects_training_length_overflow(self) -> None:
        raw = self.hard_split_fixture()
        long_source = " ".join(["word"] * 385)
        raw.loc[raw.index[0], "formosan_sentence"] = long_source
        raw.loc[raw.index[0], "formosan_mt_standard"] = long_source
        with self.assertRaisesRegex(SystemExit, "training limit"):
            build_hard_split(
                raw,
                target_col="english_sentence",
                test_ratio=0.2,
                val_ratio=0.1,
                seed=42,
                min_formosan_tokens=2,
                min_target_tokens=2,
                min_combined_tokens=6,
                min_punctuated_combined_tokens=5,
                min_test_rows=0,
                min_validate_rows=0,
                ngram_threshold=0.82,
                registry_in=None,
            )

    def test_compact_sentence_eligibility_uses_joint_content(self) -> None:
        frame = add_mt_contract(
            pd.DataFrame(
                [
                    {
                        "row_id": "balanced",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su",
                        "english_sentence": "Please come here",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "short-question",
                        "lang_code": "ami",
                        "formosan_sentence": "“ima su?”",
                        "english_sentence": '"Who are you?"',
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "short-fragment",
                        "lang_code": "ami",
                        "formosan_sentence": "ima su",
                        "english_sentence": "who are you",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "compact-long-target",
                        "lang_code": "ami",
                        "formosan_sentence": "ima su?",
                        "english_sentence": "Who are you today?",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "single-unit",
                        "lang_code": "ami",
                        "formosan_sentence": "millemungku",
                        "english_sentence": "I have got a parasol",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "",
                    },
                    {
                        "row_id": "definition",
                        "lang_code": "ami",
                        "formosan_sentence": "aru ku su",
                        "english_sentence": "a long lexical explanation",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "definition_like_sentence",
                    },
                    {
                        "row_id": "uncertain-language",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su anini",
                        "english_sentence": "Heavy rainfall flooded downtown streets",
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "english_language_uncertain",
                    },
                    {
                        "row_id": "unbalanced-target",
                        "lang_code": "ami",
                        "formosan_sentence": "maita ku su anini",
                        "english_sentence": 'He said "please come here',
                        "source": "fixture.xml",
                        "row_type": "sentence",
                        "quality_flags": "unbalanced_target_delimiters",
                    },
                ]
            )
        )
        normalized = add_normalized_columns(
            frame,
            target_col="english_sentence",
            target_lang="english",
        )
        candidates = evaluation_candidate_mask(
            normalized,
            min_formosan_tokens=2,
            min_target_tokens=2,
            min_combined_tokens=6,
            min_punctuated_combined_tokens=5,
        )

        self.assertEqual(
            set(normalized.loc[candidates, "row_id"]),
            {"balanced", "compact-long-target"},
        )

    @staticmethod
    def hard_split_fixture() -> pd.DataFrame:
        rows = []
        for document in range(60):
            for sentence in range(3):
                digest = hashlib.sha256(f"{document}:{sentence}".encode()).hexdigest()
                rows.append(
                    {
                        "row_id": f"human-{document}-{sentence}",
                        "source_record_id": f"record-{document}-{sentence}",
                        "lang_code": "ami",
                        "formosan_sentence": (
                            f"{digest[:8]} {digest[8:16]} {digest[16:24]} {digest[24:32]} {digest[32:40]}"
                        ),
                        "english_sentence": (
                            f"{digest[40:48]} {digest[48:56]} the sentence {document} number {sentence}"
                        ),
                        "source": f"FormosanBank/Corpora/Test/XML/doc-{document}.xml",
                        "repository": "FormosanBank",
                        "repository_commit": "a" * 40,
                        "xml_path": f"Corpora/Test/XML/doc-{document}.xml",
                        "xml_id": f"S-{sentence}",
                        "xml_element_index": sentence,
                        "kindOf": "standard",
                        "standard_origin": "provided",
                        "standard_after_qc_sha256": digest,
                        "qc_transform_id": f"transform-{document}-{sentence}",
                        "qc_revision": "b" * 40,
                        "dialect": "Test",
                        "row_type": "sentence",
                        "pivot_origin": "original",
                        "quality_flags": "",
                    }
                )
            rows.append(
                {
                    **rows[-1],
                    "row_id": f"synthetic-{document}",
                    "source_record_id": f"synthetic-record-{document}",
                    "formosan_sentence": f"synthetic unique {document}",
                    "english_sentence": f"synthetic target {document}",
                    "xml_id": f"pivot-{document}",
                    "pivot_origin": "synthetic",
                }
            )
            rows.append(
                {
                    **rows[-2],
                    "row_id": f"lexeme-{document}",
                    "source_record_id": f"lexeme-record-{document}",
                    "formosan_sentence": f"lexeme{document}",
                    "english_sentence": f"entry{document}",
                    "xml_id": f"W-{document}",
                    "row_type": "lexeme",
                }
            )
        rows.append({**rows[0], "row_id": "duplicate-pair"})
        return add_mt_contract(pd.DataFrame(rows))

    def test_hard_split_prefers_human_rows_and_excludes_lexemes(self) -> None:
        raw = self.hard_split_fixture()
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, excluded, duplicates, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.20,
            val_ratio=0.10,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertTrue(evaluation["row_type"].eq("sentence").all())
        self.assertFalse(evaluation["pivot_origin"].eq("synthetic").any())
        self.assertEqual(report["lexical_like_eval_rows"], 0)
        self.assertGreaterEqual(report["document_overlap_train_eval"], 0)
        self.assertEqual(len(duplicates), 1)
        self.assertEqual(len(excluded), 0)
        self.assertTrue(report["complete"])
        language = report["languages"]["ami"]
        self.assertEqual(language["rows_total"], 300)
        self.assertEqual(language["human_rows"], 240)
        self.assertEqual(language["test_rows"], 48)
        self.assertEqual(language["validate_rows"], 24)
        self.assertEqual(language["human_test_fraction"], 0.20)
        self.assertEqual(language["human_validate_fraction"], 0.10)
        self.assertEqual(
            report["required_human_ratios"],
            {"train": 0.70, "test": 0.20, "validate": 0.10},
        )

        provenance = validate_provenance(output)
        validation = validate_splits(
            output,
            target_col="english_sentence",
            min_test_ratio=0.20,
            min_validate_ratio=0.10,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            split_report=report,
        )
        self.assertTrue(provenance["ok"], provenance)
        self.assertTrue(validation["ok"], validation)
        self.assertFalse(validation["split_report_errors"])

        contaminated = output.copy()
        contaminated["translation_kind"] = ""
        contaminated.loc[contaminated.index[0], "translation_kind"] = "interlinear-gloss"
        contaminated_validation = validate_splits(
            contaminated,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
        )
        self.assertFalse(contaminated_validation["ok"])
        self.assertEqual(contaminated_validation["gloss_translation_rows"], 1)

        lexical_contamination = output.copy()
        lexeme_index = lexical_contamination.index[lexical_contamination["row_type"].eq("lexeme")][0]
        lexical_contamination.loc[lexeme_index, "english_sentence"] = "wash-face"
        lexical_validation = validate_splits(
            lexical_contamination,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
        )
        self.assertFalse(lexical_validation["ok"])
        self.assertEqual(lexical_validation["lexical_quality_rows"], 1)

    def test_hard_split_is_deterministic_for_identical_input(self) -> None:
        keyed = add_normalized_columns(
            self.hard_split_fixture(),
            target_col="english_sentence",
            target_lang="english",
        )
        kwargs = {
            "target_col": "english_sentence",
            "test_ratio": 0.10,
            "val_ratio": 0.05,
            "seed": 42,
            "min_formosan_tokens": 1,
            "min_target_tokens": 1,
            "min_combined_tokens": 2,
            "min_punctuated_combined_tokens": 2,
            "min_test_rows": 5,
            "min_validate_rows": 2,
            "ngram_threshold": 0.82,
            "registry_in": None,
        }
        first, _, _, _ = build_hard_split(keyed.copy(), **kwargs)
        second, _, _, _ = build_hard_split(keyed.copy(), **kwargs)
        first_map = dict(zip(first["row_id"], first["split"], strict=True))
        second_map = dict(zip(second["row_id"], second["split"], strict=True))
        self.assertEqual(first_map, second_map)

    def test_hard_split_apportions_each_source_corpus(self) -> None:
        raw = self.hard_split_fixture()
        document_numbers = raw["source"].str.extract(r"doc-(\d+)")[0]
        first_source = document_numbers.fillna("0").astype(int).lt(30)
        raw.loc[first_source, "source"] = raw.loc[first_source, "source"].str.replace(
            "/Test/", "/SourceA/", regex=False
        )
        raw.loc[~first_source, "source"] = raw.loc[~first_source, "source"].str.replace(
            "/Test/", "/SourceB/", regex=False
        )
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        _, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        for source in ("SourceA", "SourceB"):
            values = report["source_strata"]["ami"][source]
            self.assertEqual(values["test_rows"], values["target_test_rows"])
            self.assertEqual(
                values["validate_rows"],
                values["target_validate_rows"],
            )
        self.assertLess(
            report["source_distribution_total_variation"]["ami"]["test"],
            0.025,
        )

    def test_language_shortfall_refill_respects_source_targets(self) -> None:
        frame = pd.DataFrame(
            {
                "lang_code": ["ami"] * 12,
                "_source_corpus": ["SourceA"] * 6 + ["SourceB"] * 6,
                "_formosan_tokens": [100] * 6 + [10] * 6,
                "_target_tokens": [100] * 6 + [10] * 6,
                "_selection_priority": range(12),
                "pivot_origin": ["original"] * 12,
            }
        )
        group_ids = pd.Series(range(12), index=frame.index, dtype="int64")
        candidate_mask = pd.Series(True, index=frame.index)
        assignments = {0: "test", 1: "test"}

        fill_language_shortfalls(
            frame,
            group_ids,
            candidate_mask,
            {
                ("ami", "SourceA"): (2, 0),
                ("ami", "SourceB"): (4, 0),
            },
            {"ami": (6, 0)},
            assignments,
        )

        split = group_ids.map(assignments).fillna("train")
        self.assertEqual(int(split.iloc[:6].eq("test").sum()), 2)
        self.assertEqual(int(split.iloc[6:].eq("test").sum()), 4)

    def test_hard_split_redistributes_lexical_source_shortfall(self) -> None:
        raw = self.hard_split_fixture()
        source_a = raw.index[:150]
        raw.loc[source_a, "source"] = "FormosanBank/Corpora/SourceA/XML/lexical.xml"
        raw.loc[source_a, "row_type"] = "lexeme"
        raw.loc[~raw.index.isin(source_a), "source"] = "FormosanBank/Corpora/SourceB/XML/sentences.xml"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
            registry_in=None,
        )

        source_a_report = report["source_strata"]["ami"]["SourceA"]
        source_b_report = report["source_strata"]["ami"]["SourceB"]
        self.assertEqual(source_a_report["target_test_rows"], 0)
        self.assertEqual(source_a_report["target_validate_rows"], 0)
        self.assertEqual(
            source_b_report["target_test_rows"] + source_b_report["target_validate_rows"],
            36,
        )
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertTrue(evaluation["row_type"].eq("sentence").all())

    def test_synthetic_only_source_is_train_only(self) -> None:
        raw = self.hard_split_fixture()
        synthetic = raw["pivot_origin"].eq("synthetic")
        raw.loc[synthetic, "source"] = "FormosanBank/Corpora/PivotOnly/XML/synthetic.xml"
        raw.loc[~synthetic, "source"] = "FormosanBank/Corpora/Human/XML/human.xml"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.95,
            registry_in=None,
        )

        pivot_source = report["source_strata"]["ami"]["PivotOnly"]
        self.assertEqual(pivot_source["target_test_rows"], 0)
        self.assertEqual(pivot_source["target_validate_rows"], 0)
        self.assertEqual(pivot_source["synthetic_input_rows"], 60)
        self.assertTrue(output.loc[output["pivot_origin"].eq("synthetic"), "split"].eq("train").all())
        self.assertEqual(report["split_counts"]["test"], 24)
        self.assertEqual(report["split_counts"]["validate"], 12)
        self.assertEqual(report["synthetic_eval_rows"], 0)
        validation = validate_splits(
            output,
            target_col="english_sentence",
            min_test_ratio=0.10,
            min_validate_ratio=0.05,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.95,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            split_report=report,
        )
        self.assertTrue(validation["ok"], validation)

    def test_hard_split_fails_when_human_evaluation_capacity_is_too_small(self) -> None:
        raw = self.hard_split_fixture()
        sentence_indexes = raw.index[raw["row_type"].eq("sentence")]
        raw.loc[sentence_indexes[2:], "pivot_origin"] = "synthetic"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        with self.assertRaisesRegex(SystemExit, "eligible sentences"):
            build_hard_split(
                keyed,
                target_col="english_sentence",
                test_ratio=0.10,
                val_ratio=0.05,
                seed=42,
                min_formosan_tokens=1,
                min_target_tokens=1,
                min_combined_tokens=2,
                min_punctuated_combined_tokens=2,
                min_test_rows=5,
                min_validate_rows=2,
                ngram_threshold=0.82,
                registry_in=None,
            )

    def test_unsafe_evaluation_candidates_are_replaced_not_excluded(self) -> None:
        raw = self.hard_split_fixture()
        near_synthetic = []
        human = raw[raw["pivot_origin"].eq("original") & raw["row_type"].eq("sentence")].head(20)
        for _, row in human.iterrows():
            near_synthetic.append(
                {
                    **row.to_dict(),
                    "row_id": f"near-{row['row_id']}",
                    "source_record_id": (f"near-{row['source_record_id']}"),
                    "formosan_sentence": (f"{row['formosan_sentence']}x"),
                    "english_sentence": (f"{row['english_sentence']}x"),
                    "source": (f"FormosanBank/Corpora/Pivot/XML/near-{row['row_id']}.xml"),
                    "xml_path": (f"Corpora/Pivot/XML/near-{row['row_id']}.xml"),
                    "xml_id": f"near-{row['xml_id']}",
                    "pivot_origin": "synthetic",
                }
            )
        raw = pd.concat(
            [raw, pd.DataFrame(near_synthetic)],
            ignore_index=True,
        )
        raw = add_mt_contract(raw)
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, excluded, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        self.assertTrue(report["complete"])
        language = report["languages"]["ami"]
        self.assertLess(
            language["group_safe_human_sentence_rows"],
            language["eligible_human_sentence_rows"],
        )
        self.assertTrue(excluded.empty)
        evaluation = output[output["split"].isin({"test", "validate"})]
        self.assertFalse(evaluation["pivot_origin"].eq("synthetic").any())
        self.assertTrue(output[output["pivot_origin"].eq("synthetic")]["split"].eq("train").all())
        self.assertEqual(len(output), report["deduplicated_input_rows"])
        self.assertEqual(language["human_test_fraction"], 0.10)
        self.assertEqual(language["human_validate_fraction"], 0.05)

    def test_validate_test_conflicts_are_reallocated_across_sources(self) -> None:
        raw = self.hard_split_fixture()
        source_a = raw["source"].str.contains(r"doc-(?:[0-9]|1[0-4])\.xml", regex=True)
        raw.loc[source_a, "source"] = "FormosanBank/Corpora/SourceA/XML/a.xml"
        raw.loc[~source_a, "source"] = "FormosanBank/Corpora/SourceB/XML/b.xml"
        keyed = add_normalized_columns(
            raw,
            target_col="english_sentence",
            target_lang="english",
        )
        output, _, _, report = build_hard_split(
            keyed,
            target_col="english_sentence",
            test_ratio=0.10,
            val_ratio=0.05,
            seed=42,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            min_test_rows=5,
            min_validate_rows=2,
            ngram_threshold=0.82,
            registry_in=None,
        )

        language = report["languages"]["ami"]
        self.assertEqual(language["test_rows"], language["target_test_rows"])
        self.assertEqual(
            language["validate_rows"],
            language["target_validate_rows"],
        )
        self.assertFalse(report["ratio_shortfalls"])
        self.assertFalse(report["source_ratio_shortfalls"])
        for source in report["source_strata"]["ami"].values():
            self.assertEqual(source["test_rows"], source["target_test_rows"])
            self.assertEqual(
                source["validate_rows"],
                source["target_validate_rows"],
            )
        self.assertTrue(output[output["split"].isin({"test", "validate"})]["row_type"].eq("sentence").all())

    def test_independent_validator_reports_synthetic_and_document_overlap(self) -> None:
        frame = self.hard_split_fixture().iloc[:3].copy()
        frame["split"] = ["train", "test", "validate"]
        frame.loc[frame.index[1], "pivot_origin"] = "synthetic"
        validation = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
        )
        self.assertFalse(validation["ok"], validation)
        self.assertEqual(validation["synthetic_eval_rows"], 1)
        self.assertFalse(validation["synthetic_eval_allowed"])
        self.assertGreater(
            validation["train_evaluation"]["document_overlap"],
            0,
        )
        diagnostic = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
            require_human_eval=False,
        )
        self.assertTrue(diagnostic["ok"], diagnostic)

    def test_validate_test_target_overlap_is_task_conditioned(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "row_id": "ami-train",
                    "lang_code": "ami",
                    "formosan_sentence": "ami training phrase qzx",
                    "english_sentence": "unique training target alpha",
                    "source": "repo/ami-train.xml",
                    "dialect": "A",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "train",
                },
                {
                    "row_id": "bnn-train",
                    "lang_code": "bnn",
                    "formosan_sentence": "bnn training phrase vjk",
                    "english_sentence": "unique training target beta",
                    "source": "repo/bnn-train.xml",
                    "dialect": "B",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "train",
                },
                {
                    "row_id": "ami-test",
                    "lang_code": "ami",
                    "formosan_sentence": "ami heldout phrase mnp",
                    "english_sentence": "shared multilingual template",
                    "source": "repo/ami-test.xml",
                    "dialect": "A",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "test",
                },
                {
                    "row_id": "bnn-validate",
                    "lang_code": "bnn",
                    "formosan_sentence": "bnn heldout phrase rst",
                    "english_sentence": "shared multilingual template",
                    "source": "repo/bnn-validate.xml",
                    "dialect": "B",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                    "split": "validate",
                },
            ]
        )
        frame = add_mt_contract(frame)
        validation = validate_splits(
            frame,
            target_col="english_sentence",
            min_test_ratio=0,
            min_validate_ratio=0,
            min_test_rows=0,
            min_validate_rows=0,
            ngram_threshold=0.82,
            min_formosan_tokens=1,
            min_target_tokens=1,
            min_combined_tokens=2,
            min_punctuated_combined_tokens=2,
        )

        self.assertTrue(validation["ok"], validation)
        self.assertEqual(
            validation["validate_test"]["exact_overlap"]["target"],
            0,
        )
        self.assertEqual(
            validation["validate_test_cross_language_diagnostic"]["exact_overlap"]["target"],
            1,
        )
