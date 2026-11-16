from __future__ import annotations

import importlib.metadata
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import stage_cache
from build_mt_corpus import environment_drift_message
from reproducibility import environment_report, pinned_requirements

ROOT = Path(__file__).resolve().parents[1]


class FakeDistribution:
    def __init__(self, version: str, *, editable: bool = False) -> None:
        self.version = version
        self._editable = editable

    def read_text(self, name: str) -> str | None:
        if name != "direct_url.json" or not self._editable:
            return None
        return json.dumps({"url": "file:///checkout", "dir_info": {"editable": True}})


def fake_environment(installed: dict[str, FakeDistribution]):
    def distribution(name: str) -> FakeDistribution:
        if name not in installed:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed[name]

    return mock.patch("importlib.metadata.distribution", side_effect=distribution)


class PinnedEnvironmentTests(unittest.TestCase):
    def write_requirements(self, directory: Path) -> Path:
        (directory / "base.txt").write_text("numpy==2.3.2\n# comment\n\npandas==2.3.2  # data frames\n")
        requirements = directory / "corpus.txt"
        requirements.write_text("-r base.txt\ntame-mt==0.2.2\n")
        return requirements

    def test_pins_follow_includes_and_ignore_comments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pins = pinned_requirements(self.write_requirements(Path(tmp)))
        self.assertEqual(pins, {"numpy": "2.3.2", "pandas": "2.3.2", "tame-mt": "0.2.2"})

    def test_unpinned_requirements_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requirements = Path(tmp) / "loose.txt"
            requirements.write_text("numpy>=2\n")
            with self.assertRaises(ValueError):
                pinned_requirements(requirements)

    def test_repository_requirement_files_are_exact_pins(self) -> None:
        for name in ("requirements-corpus.txt", "requirements-training.txt", "requirements-dev.txt"):
            self.assertTrue(pinned_requirements(ROOT / name), name)

    def test_matching_environment_has_no_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requirements = self.write_requirements(Path(tmp))
            installed = {
                "numpy": FakeDistribution("2.3.2"),
                "pandas": FakeDistribution("2.3.2"),
                "tame-mt": FakeDistribution("0.2.2"),
            }
            with fake_environment(installed):
                report = environment_report(requirements)
        self.assertEqual(report["drift"], [])
        self.assertEqual(report["requirements"], "corpus.txt")
        self.assertEqual(report["packages"]["numpy"], {"pinned": "2.3.2", "installed": "2.3.2", "editable": False})

    def test_wrong_missing_and_editable_packages_are_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            requirements = self.write_requirements(Path(tmp))
            installed = {
                "numpy": FakeDistribution("1.26.4"),
                "tame-mt": FakeDistribution("0.2.2", editable=True),
            }
            with fake_environment(installed):
                report = environment_report(requirements)
        self.assertEqual(report["drift"], ["numpy", "pandas", "tame-mt"])
        self.assertIsNone(report["packages"]["pandas"]["installed"])
        message = environment_drift_message(report)
        self.assertIn("numpy: pinned 2.3.2, installed 1.26.4", message)
        self.assertIn("pandas: pinned 2.3.2, installed nothing", message)
        self.assertIn("tame-mt: pinned 0.2.2, installed as an editable checkout", message)
        self.assertIn("pip install --force-reinstall -r corpus.txt", message)

    def test_stage_keys_follow_installed_packages_not_the_pin_file(self) -> None:
        script = ROOT / "scripts/local/stage_cache.py"
        with mock.patch.object(stage_cache, "installed_corpus_packages", return_value={"numpy": ["2.3.2", False]}):
            pinned = stage_cache.stage_key("filter", {"input": "a"}, [script])
            self.assertEqual(pinned, stage_cache.stage_key("filter", {"input": "a"}, [script]))
        with mock.patch.object(stage_cache, "installed_corpus_packages", return_value={"numpy": ["1.26.4", False]}):
            self.assertNotEqual(pinned, stage_cache.stage_key("filter", {"input": "a"}, [script]))
        with mock.patch.object(stage_cache, "installed_corpus_packages", return_value={"numpy": ["2.3.2", True]}):
            self.assertNotEqual(pinned, stage_cache.stage_key("filter", {"input": "a"}, [script]))

