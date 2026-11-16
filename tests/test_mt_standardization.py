from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from mt_standardization import (
    StandardizationContext,
    assert_idempotent,
    standardize_text,
)
from support import MT_PROFILE

ROOT = Path(__file__).resolve().parents[1]


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
