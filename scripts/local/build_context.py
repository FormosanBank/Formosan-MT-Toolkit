"""Stable language, path, and filesystem contracts for corpus builds."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXACT_BIBLE_REPOS = ("Formosan-Taiwan-Bible-Society-Bibles",)


@dataclass(frozen=True)
class Language:
    name: str
    code: str


@dataclass(frozen=True)
class BuildPaths:
    root: Path
    raw_dir: Path
    processed_dir: Path
    final_dir: Path
    split_root: Path
    manifest_path: Path
    source_snapshot_path: Path

    def source_xml_dir(self, lang: Language) -> Path:
        return self.root / f"downloaded_{lang.code}"

    def prepared_xml_dir(self, lang: Language) -> Path:
        return self.root / f"prepared_{lang.code}"


LANGUAGES = (
    Language("Amis", "ami"),
    Language("Atayal", "tay"),
    Language("Bunun", "bnn"),
    Language("Kanakanavu", "xnb"),
    Language("Kavalan", "ckv"),
    Language("Paiwan", "pwn"),
    Language("Puyuma", "pyu"),
    Language("Rukai", "dru"),
    Language("Saaroa", "sxr"),
    Language("Saisiyat", "xsy"),
    Language("Sakizaya", "szy"),
    Language("Seediq", "trv"),
    Language("Thao", "ssf"),
    Language("Tsou", "tsu"),
    Language("Yami/Tao", "tao"),
)


def script(path: str) -> Path:
    return PROJECT_ROOT / path


def stage_log(paths: BuildPaths, name: str) -> Path:
    return paths.root / "logs" / f"{name}.log"


def parse_languages(raw: str) -> list[Language]:
    if raw.strip().lower() == "all":
        return list(LANGUAGES)
    by_code = {lang.code: lang for lang in LANGUAGES}
    selected: list[Language] = []
    for part in raw.split(","):
        code = part.strip().lower()
        if not code:
            continue
        if code not in by_code:
            raise SystemExit(
                f"Unknown language code {code!r}; valid codes: {', '.join(by_code)}"
            )
        selected.append(by_code[code])
    if not selected:
        raise SystemExit("No languages selected.")
    return selected


def safe_corpus_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not name:
        raise SystemExit(
            "--corpus-name must contain at least one alphanumeric character"
        )
    if name in {".", ".."}:
        raise SystemExit(f"Invalid --corpus-name: {value!r}")
    return name


def variant_name(public: bool, exclude_bible: bool) -> str:
    base = "public" if public else "private"
    return f"{base}_no_bible" if exclude_bible else base


def resolve_build_paths(args: argparse.Namespace) -> BuildPaths:
    if args.corpus_name:
        name = safe_corpus_name(args.corpus_name)
        root = (args.build_root or (PROJECT_ROOT / "corpus_builds")) / name
    elif args.build_root:
        root = args.build_root
    else:
        raise SystemExit(
            "Choose an isolated output with --corpus-name or --build-root, "
            "or use --build-public-private."
        )

    root = root.expanduser().resolve()
    return BuildPaths(
        root=root,
        raw_dir=root / "raw_corpora",
        processed_dir=root / "processed_corpora",
        final_dir=root / "pivot_corpora_final",
        split_root=root / "formosan_mt_experiments" / "data",
        manifest_path=root / "mt_build_manifest.json",
        source_snapshot_path=root / "source_repository_snapshot.json",
    )


def pivot_cache_dir(paths: BuildPaths) -> Path:
    return paths.processed_dir / "pivot" / "cache"


def pivot_read_cache_dirs(
    args: argparse.Namespace,
    paths: BuildPaths,
) -> list[Path]:
    seen: set[Path] = set()
    directories: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved == pivot_cache_dir(paths).resolve() or resolved in seen:
            return
        seen.add(resolved)
        directories.append(resolved)

    for path in getattr(args, "pivot_read_cache_dir", []):
        add(path)
    for path in getattr(args, "extra_pivot_read_cache_dirs", []):
        add(path)
    return directories


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def replace_with_hardlink(source: Path, destination: Path) -> None:
    """Avoid storing a second physical copy of a finalized split."""
    remove_path(destination)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def clean_generated_outputs(paths: BuildPaths) -> None:
    """Remove abandoned temporary files while retaining verified stage outputs."""
    pivot_dir = paths.processed_dir / "pivot"
    if not pivot_dir.exists():
        return
    for pattern in ("big_corpus*.incomplete", "*.tmp"):
        for path in pivot_dir.glob(pattern):
            remove_path(path)


def should_clean_generated_outputs(
    args: argparse.Namespace,
    paths: BuildPaths,
) -> bool:
    if args.dry_run or args.keep_build_output:
        return False
    return not (args.skip_raw or args.skip_filter or args.skip_aggregate)


def prune_unselected_language_outputs(
    paths: BuildPaths,
    languages: list[Language],
) -> None:
    selected = {lang.code for lang in languages}
    for lang in LANGUAGES:
        if lang.code in selected:
            continue
        for suffix in ("zh", "en"):
            remove_path(paths.raw_dir / f"{lang.code}_{suffix}.csv")
            remove_path(paths.raw_dir / f"{lang.code}_{suffix}.extraction.json")
            processed = paths.processed_dir / f"{lang.code}_{suffix}_processed.csv"
            remove_path(processed)
            remove_path(paths.processed_dir / "filter_reports" / processed.stem)


def language_cache_path(paths: BuildPaths, lang: Language) -> Path:
    return paths.root / ".stage_cache" / f"{lang.code}.json"


def build_cache_path(paths: BuildPaths) -> Path:
    return paths.root / ".stage_cache" / "build.json"


def require_json_manifest(
    path: Path,
    *,
    stage: str,
    expected: dict[str, object] | None = None,
) -> dict:
    if not path.is_file():
        raise SystemExit(f"{stage} did not produce its required manifest: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{stage} manifest is malformed at {path}: {exc}") from exc
    if payload.get("complete") is not True:
        raise SystemExit(f"{stage} manifest is incomplete: {path}")
    for key, value in (expected or {}).items():
        if payload.get(key) != value:
            raise SystemExit(
                f"{stage} manifest mismatch for {key}: expected {value!r}, "
                f"found {payload.get(key)!r}"
            )
    return payload
