from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/local"))

from build_output import (  # noqa: E402
    CommandExecutionError,
    format_language_summary,
    run_logged,
)


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

            self.assertIn("QC repairs: remove zero width characters=2", summary)
            self.assertIn("MT cleaning: remove hyphen boundary=5", summary)
            self.assertIn("en: 9 input -> 7 kept", summary)
            self.assertIn("QC findings: SOFT=4 across 1 rules", summary)


if __name__ == "__main__":
    unittest.main()
