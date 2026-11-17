from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from build_output import (
    CommandExecutionError,
    format_language_summary,
    format_rule_summary,
    run_logged,
)
from summarize_build import load_build, summarize


class LoggedCommandTests(unittest.TestCase):
    def test_normal_mode_captures_child_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "stage.log"
            terminal = io.StringIO()
            with contextlib.redirect_stdout(terminal):
                run_logged(
                    [sys.executable, "-c", "print('raw child output')"],
                    project_root=root,
                    label="Fixture stage",
                    log_path=log_path,
                )

            self.assertNotIn("raw child output", terminal.getvalue())
            self.assertIn("[stage] Fixture stage", terminal.getvalue())
            self.assertIn("[done]  Fixture stage", terminal.getvalue())
            self.assertIn("raw child output", log_path.read_text(encoding="utf-8"))

    def test_stream_mode_mirrors_child_output_to_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "stage.log"
            terminal = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(terminal):
                run_logged(
                    [sys.executable, "-c", "print('live child output')"],
                    project_root=root,
                    label="Live stage",
                    log_path=log_path,
                    stream=True,
                )

            self.assertIn("live child output", terminal.getvalue())
            self.assertIn("live child output", log_path.read_text(encoding="utf-8"))

    def test_failure_reports_log_tail_without_full_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_path = root / "stage.log"
            with self.assertRaises(CommandExecutionError) as raised:
                run_logged(
                    [sys.executable, "-c", "print('useful failure'); raise SystemExit(7)"],
                    project_root=root,
                    label="Broken stage",
                    log_path=log_path,
                    quiet=True,
                )

            message = str(raised.exception)
            self.assertIn("Broken stage failed (exit 7)", message)
            self.assertIn("useful failure", message)
            self.assertNotIn("raise SystemExit", message)


class SummaryFormattingTests(unittest.TestCase):
    @staticmethod
    def write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_language_summary_uses_manifest_rule_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_json(
                root / "prepared_ami" / "_qc_manifest.json",
                {
                    "input": {"xml_files": 3},
                    "repair_inventory": {"counts": {"remove_zero_width_characters": 2}},
                    "validators": [{"summary": {"by_severity": {"SOFT": {"records": 4, "rules": {"V122": {}}}}}}],
                },
            )
            self.write_json(
                root / "prepared_ami" / "_mt_standard_manifest.json",
                {
                    "inventory": {
                        "records": 12,
                        "status_counts": {"accepted": 10, "quarantine": 2},
                        "rule_counts": {"remove_hyphen_boundary": 5},
                    }
                },
            )
            for target, initial, accepted in (("en", 9, 7), ("zh", 11, 8)):
                self.write_json(
                    root / "processed_corpora" / "filter_reports" / f"ami_{target}_processed" / "summary.json",
                    {
                        "initial_rows": initial,
                        "accepted_rows": accepted,
                        "filter_rule_counts": {"deduplicated:duplicate_pair": initial - accepted},
                    },
                )
            self.write_json(
                root / "raw_corpora" / "ami_en.extraction.json",
                {
                    "counts": {
                        "sentence_nested_word_units_excluded": 12,
                    }
                },
            )

            summary = format_language_summary(
                root,
                {
                    "code": "ami",
                    "language": "Amis",
                    "processed_en_rows": 7,
                    "processed_zh_rows": 8,
                    "stage_status": {"QC": "rebuilt", "extraction": "cached"},
                },
            )
            rules = format_rule_summary(root, [{"code": "ami"}])

            self.assertIn("en 9->7", summary)
            self.assertIn("zh 11->8", summary)
            self.assertIn("QC repairs: remove zero width characters=2", rules)
            self.assertIn("MT cleaning: remove hyphen boundary=5", rules)
            self.assertIn(
                "XML unit exclusions: sentence nested word units excluded=12",
                rules,
            )
            self.assertIn("QC findings: SOFT=4 across 1 rules", rules)


class BuildSummaryTests(unittest.TestCase):
    def write_build(self, root: Path, *, ami_human: int, sources: dict[str, int]) -> Path:
        manifest = {
            "complete": True,
            "created_at": "2026-08-24T04:11:52Z",
            "repository": {"commit": "573488b215ee1cf3701519a5bcd1010e8f0c6548"},
            "settings": {"public": True, "with_pivot": True, "qc_revision": "acc3ec9f3137a59b"},
        }
        root.mkdir(parents=True)
        (root / "mt_build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = {
            "complete": True,
            "languages": {
                "ami": {
                    "human_rows": ami_human,
                    "synthetic_rows": 50,
                    "lexical_rows": 5,
                    "eligible_human_sentence_rows": 60,
                    "train_rows": ami_human + 50 - 30,
                    "validate_rows": 10,
                    "test_rows": 20,
                }
            },
            "source_strata": {
                "ami": {
                    name: {
                        "human_input_rows": rows,
                        "synthetic_input_rows": 0,
                        "eligible_human_rows": rows,
                        "validate_rows": 1,
                        "test_rows": 2,
                    }
                    for name, rows in sources.items()
                }
            },
        }
        for short in ("en", "zh"):
            directory = root / "formosan_mt_experiments" / "data" / f"splits_{short}_v1"
            directory.mkdir(parents=True)
            (directory / "report_in_domain_hard.json").write_text(json.dumps(report), encoding="utf-8")
        return root

    def test_summary_lists_languages_totals_and_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            build = load_build(self.write_build(Path(tmp) / "public_no_bible", ami_human=100, sources={"ePark": 100}))
            summary = summarize(build, None, sources=True)
        self.assertIn("`public_no_bible`: built 2026-08-24 at toolkit commit `573488b`, QC `acc3ec9`", summary)
        self.assertIn("| `ami` | 100 | 50 | 5 | 60 | 120 | 10 | 20 |", summary)
        self.assertIn("| **Total** | 100 | 50 | 5 | 60 | 120 | 10 | 20 |", summary)
        self.assertIn("| `ePark` | 100 | 0 | 100 | 1 | 2 |", summary)
        self.assertEqual(summary.count("## English"), 1)
        self.assertEqual(summary.count("## Chinese"), 1)

    def test_comparison_shows_signed_changes_and_removed_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = load_build(
                self.write_build(Path(tmp) / "old", ami_human=100, sources={"ePark": 60, "Dropped": 40})
            )
            new = load_build(self.write_build(Path(tmp) / "new", ami_human=130, sources={"ePark": 130}))
            summary = summarize(new, old, sources=True)
        self.assertIn("Changes in parentheses are against `old`", summary)
        self.assertIn("| `ami` | 130 (+30) | 50 | 5 | 60 | 150 (+30) | 10 | 20 |", summary)
        self.assertIn("| `ePark` | 130 (+70) |", summary)
        self.assertIn("| `Dropped` | 0 (-40) |", summary)

    def test_same_named_builds_are_told_apart_by_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old = load_build(self.write_build(Path(tmp) / "august" / "public_no_bible", ami_human=100, sources={"ePark": 100}))
            new = load_build(self.write_build(Path(tmp) / "resplit" / "public_no_bible", ami_human=100, sources={"ePark": 100}))
            summary = summarize(new, old, sources=False)
        self.assertIn("`resplit/public_no_bible`: built", summary)
        self.assertIn("against `august/public_no_bible`: built", summary)

    def test_incomplete_builds_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = self.write_build(Path(tmp) / "build", ami_human=100, sources={"ePark": 100})
            (root / "mt_build_manifest.json").write_text(json.dumps({"complete": False}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                load_build(root)

