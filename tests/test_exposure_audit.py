from __future__ import annotations

import unittest

import pandas as pd
from audit_corpus_exposure import (
    audit_direction,
    build_tame_config,
    gate_errors,
)


class ExposureAuditTests(unittest.TestCase):
    @staticmethod
    def frame(*, duplicate_test: bool = False) -> pd.DataFrame:
        rows = []
        for index in range(12):
            rows.append(
                {
                    "row_id": f"train-{index}",
                    "lang_code": "ami",
                    "formosan_sentence": f"train source phrase number {index} alpha",
                    "english_sentence": f"train target phrase number {index} omega",
                    "split": "train",
                    "kindOf": "standard",
                    "row_type": "sentence",
                    "pivot_origin": "original",
                }
            )
        for split, offset in (("test", 100), ("validate", 200)):
            for index in range(4):
                rows.append(
                    {
                        "row_id": f"{split}-{index}",
                        "lang_code": "ami",
                        "formosan_sentence": (
                            "train source phrase number 0 alpha"
                            if duplicate_test and split == "test" and index == 0
                            else f"{split} heldout expression {offset + index} cedar"
                        ),
                        "english_sentence": (
                            "train target phrase number 0 omega"
                            if duplicate_test and split == "test" and index == 0
                            else f"{split} reference wording {offset + index} quartz"
                        ),
                        "split": split,
                        "kindOf": "standard",
                        "row_type": "sentence",
                        "pivot_origin": "original",
                    }
                )
        return pd.DataFrame(rows)

    def test_tame_release_gate_accepts_distant_human_evaluation(self) -> None:
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            self.frame(),
            direction="f2en",
            target_col="english_sentence",
            config=config,
        )
        self.assertEqual(
            gate_errors(
                {"f2en": payload},
                high_threshold="0.95",
                max_high_exposure_rate=0.0,
            ),
            [],
        )
        self.assertEqual(
            payload["by_split"]["test"]["by_source_corpus"]["unknown"]["rows"],
            4,
        )

    def test_tame_release_gate_rejects_exact_train_test_pair(self) -> None:
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            self.frame(duplicate_test=True),
            direction="f2en",
            target_col="english_sentence",
            config=config,
        )
        errors = gate_errors(
            {"f2en": payload},
            high_threshold="0.95",
            max_high_exposure_rate=0.0,
        )
        self.assertTrue(any("exact_overlap" in error for error in errors))

    def test_tame_release_gate_conditions_retrieval_on_language_task(self) -> None:
        frame = self.frame(duplicate_test=True)
        train_indexes = frame.index[frame["split"].eq("train")]
        evaluation_indexes = frame.index[~frame["split"].eq("train")]
        frame.loc[train_indexes[6:], "lang_code"] = "bnn"
        frame.loc[evaluation_indexes, "lang_code"] = "bnn"
        config = build_tame_config(
            high_threshold=0.95,
            pair_k=10,
            batch_size=64,
        )
        payload = audit_direction(
            frame,
            direction="f2en",
            target_col="english_sentence",
            config=config,
        )
        self.assertEqual(
            gate_errors(
                {"f2en": payload},
                high_threshold="0.95",
                max_high_exposure_rate=0.0,
            ),
            [],
        )
        self.assertEqual(
            payload["combined_evaluation"]["task_conditioning"],
            "lang_code",
        )
