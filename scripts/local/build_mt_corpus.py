#!/usr/bin/env python3
"""End-to-end FormosanBank XML -> MT corpus builder.

This is the reproducible replacement for the historical build_corpora.sh flow.
It keeps each stage explicit while making the default path produce hard-split
multilingual corpora suitable for the current NLLB/SPM8k directional recipes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
EXACT_BIBLE_REPOS = (
    "Formosan-Taiwan-Bible-Society-Bibles",
)


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
    legacy_layout: bool

    def xml_dir(self, lang: Language) -> Path:
        if self.legacy_layout:
            return PROJECT_ROOT / f"downloaded_{lang.code}"
        return self.root / f"downloaded_{lang.code}"


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


def run(cmd: list[str], *, dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


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
            raise SystemExit(f"Unknown language code {code!r}; valid codes: {', '.join(by_code)}")
        selected.append(by_code[code])
    if not selected:
        raise SystemExit("No languages selected.")
    return selected


def safe_corpus_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    if not name:
        raise SystemExit("--corpus-name must contain at least one alphanumeric character")
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
        legacy_layout = False
    elif args.build_root:
        root = args.build_root
        legacy_layout = False
    else:
        root = PROJECT_ROOT
        legacy_layout = True

    root = root.expanduser().resolve()
    if legacy_layout:
        raw_dir = PROJECT_ROOT / "raw_corpora"
        processed_dir = PROJECT_ROOT / "processed_corpora"
        final_dir = PROJECT_ROOT / "pivot_corpora_final"
        split_root = PROJECT_ROOT / "formosan_mt_experiments" / "data"
        manifest_path = processed_dir / "mt_build_manifest.json"
    else:
        raw_dir = root / "raw_corpora"
        processed_dir = root / "processed_corpora"
        final_dir = root / "pivot_corpora_final"
        split_root = root / "formosan_mt_experiments" / "data"
        manifest_path = root / "mt_build_manifest.json"

    return BuildPaths(
        root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        final_dir=final_dir,
        split_root=split_root,
        manifest_path=manifest_path,
        legacy_layout=legacy_layout,
    )


def pivot_cache_dir(paths: BuildPaths) -> Path:
    return paths.processed_dir / "pivot" / "cache"


def pivot_read_cache_dirs(args: argparse.Namespace, paths: BuildPaths) -> list[Path]:
    seen: set[Path] = set()
    dirs: list[Path] = []

    def add(path: Path) -> None:
        resolved = path.expanduser().resolve()
        if resolved == pivot_cache_dir(paths).resolve() or resolved in seen:
            return
        seen.add(resolved)
        dirs.append(resolved)

    for path in getattr(args, "pivot_read_cache_dir", []):
        add(path)
    if args.shared_pivot_cache:
        add(PROJECT_ROOT / "processed_corpora" / "pivot" / "cache")
    for path in getattr(args, "extra_pivot_read_cache_dirs", []):
        add(path)
    return dirs


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_generated_outputs(paths: BuildPaths) -> None:
    """Remove generated data for a named build without deleting pivot caches."""
    remove_path(paths.raw_dir)
    remove_path(paths.final_dir)
    remove_path(paths.split_root)
    remove_path(paths.processed_dir / "filter_reports")
    for pattern in ("*_processed.csv", "big_corpus*.csv"):
        for path in paths.processed_dir.glob(pattern):
            remove_path(path)
    pivot_dir = paths.processed_dir / "pivot"
    if pivot_dir.exists():
        for pattern in ("big_corpus*.csv", "pivot_manifest.json", "*.tmp"):
            for path in pivot_dir.glob(pattern):
                remove_path(path)


def should_clean_generated_outputs(args: argparse.Namespace, paths: BuildPaths) -> bool:
    if args.dry_run or paths.legacy_layout or args.keep_build_output:
        return False
    # Incremental stage skips imply the caller expects existing intermediates.
    return not (args.skip_raw or args.skip_filter or args.skip_aggregate)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_record(path: Path, *, compute_hash: bool) -> dict:
    record = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    record["bytes"] = path.stat().st_size
    if path.suffix.lower() == ".csv":
        record["rows"] = count_csv_rows(path)
    if compute_hash:
        record["sha256"] = sha256_file(path)
    return record


def count_bible_source_rows(path: Path) -> int:
    if not path.exists() or path.suffix.lower() != ".csv":
        return 0
    count = 0
    exact_repo_components = {repo.lower() for repo in EXACT_BIBLE_REPOS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "source" not in reader.fieldnames:
            return 0
        for row in reader:
            source_parts = [
                part.strip().lower()
                for part in (row.get("source") or "").replace("\\", "/").split("/")
                if part.strip()
            ]
            if any(part in exact_repo_components for part in source_parts):
                count += 1
    return count


def validate_no_bible_sources(paths: BuildPaths, final_corpus_dir: Path) -> None:
    files: list[Path] = []
    files.extend(sorted(paths.processed_dir.glob("*_processed.csv")))
    files.extend(final_corpus_dir.glob("big_corpus*.csv"))
    failures = {
        str(path): rows
        for path in files
        if (rows := count_bible_source_rows(path)) > 0
    }
    if failures:
        details = ", ".join(f"{path}: {rows}" for path, rows in sorted(failures.items()))
        raise SystemExit(f"--exclude-bible validation failed; Bible source rows remain: {details}")


def build_artifact_manifest(
    args: argparse.Namespace,
    final_corpus_dir: Path,
    paths: BuildPaths,
) -> dict[str, dict]:
    if args.dry_run:
        return {}
    artifact_paths = {
        "big_corpus_en": final_corpus_dir / "big_corpus_en.csv",
        "big_corpus_zh": final_corpus_dir / "big_corpus_zh.csv",
        "big_corpus_combined": final_corpus_dir / "big_corpus_combined.csv",
        "big_corpus_en_in_domain_hard": final_corpus_dir / "big_corpus_en_in_domain_hard.csv",
        "big_corpus_zh_in_domain_hard": final_corpus_dir / "big_corpus_zh_in_domain_hard.csv",
    }
    for split_dir in (paths.split_root / "splits_en_v1", paths.split_root / "splits_zh_v1"):
        if split_dir.exists():
            for path in sorted(split_dir.glob("*.csv")):
                artifact_paths[f"{split_dir.name}/{path.name}"] = path

    return {
        name: artifact_record(path, compute_hash=not args.skip_artifact_checksums)
        for name, path in artifact_paths.items()
    }


def build_language(lang: Language, args: argparse.Namespace, paths: BuildPaths) -> dict:
    xml_dir = paths.xml_dir(lang)
    raw_zh = paths.raw_dir / f"{lang.code}_zh.csv"
    raw_en = paths.raw_dir / f"{lang.code}_en.csv"
    proc_zh = paths.processed_dir / f"{lang.code}_zh_processed.csv"
    proc_en = paths.processed_dir / f"{lang.code}_en_processed.csv"

    if not args.skip_fetch:
        cmd = [PYTHON, str(script("scripts/local/fetch_xml.py")), "--src-lang", lang.code]
        cmd.extend(["--out-dir", str(xml_dir)])
        cmd.extend(["--workers", str(args.fetch_workers)])
        cmd.extend(["--download-retries", str(args.fetch_download_retries)])
        cmd.extend(["--retry-base-sleep", str(args.fetch_retry_base_sleep)])
        cmd.extend(["--retry-max-sleep", str(args.fetch_retry_max_sleep)])
        if args.allow_download_failures:
            cmd.append("--allow-download-failures")
        if not args.keep_downloaded:
            cmd.append("--clean-output")
        if args.public:
            cmd.append("--public")
        if args.no_public_language_path_prefilter:
            cmd.append("--no-public-language-path-prefilter")
        if args.force_branch:
            cmd.extend(["--branch", args.force_branch])
        if args.exclude_bible:
            cmd.append("--exclude-bible")
        for pattern in args.exclude_repo_pattern:
            cmd.extend(["--exclude-repo-pattern", pattern])
        for pattern in args.exclude_path_pattern:
            cmd.extend(["--exclude-path-pattern", pattern])
        run(cmd, dry_run=args.dry_run)

    if not args.skip_clean:
        cmd = [PYTHON, str(script("scripts/local/clean_xml.py")), "--src-lang", lang.code]
        cmd.extend(["--in-dir", str(xml_dir)])
        if args.formosanbank_path:
            cmd.extend(["--formosanbank-path", str(args.formosanbank_path)])
        if args.force_qc_update:
            cmd.append("--force-update")
        if args.validate_qc:
            cmd.append("--validate")
        run(cmd, dry_run=args.dry_run)

    if not args.skip_raw:
        run(
            [
                PYTHON,
                str(script("scripts/local/make_corpus.py")),
                "--xml-dir",
                str(xml_dir),
                "--target",
                "chinese",
                "--out",
                str(raw_zh),
                "--units",
                args.units,
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                PYTHON,
                str(script("scripts/local/make_corpus.py")),
                "--xml-dir",
                str(xml_dir),
                "--target",
                "english",
                "--out",
                str(raw_en),
                "--units",
                args.units,
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_filter:
        filter_base = [
            PYTHON,
            str(script("scripts/local/filter_split_corpus.py")),
            "--workers",
            str(args.workers),
            "--val-ratio",
            str(args.legacy_val_ratio),
            "--test-ratio",
            str(args.legacy_test_ratio),
        ]
        if args.keep_redactions:
            filter_base.append("--keep-redactions")
        run(filter_base + ["--input", str(raw_zh), "--output", str(proc_zh)], dry_run=args.dry_run)
        run(filter_base + ["--input", str(raw_en), "--output", str(proc_en)], dry_run=args.dry_run)

    return {
        "language": lang.name,
        "code": lang.code,
        "raw_zh_rows": count_csv_rows(raw_zh),
        "raw_en_rows": count_csv_rows(raw_en),
        "processed_zh_rows": count_csv_rows(proc_zh),
        "processed_en_rows": count_csv_rows(proc_en),
    }


def build_aggregates(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> None:
    run(
        [
            PYTHON,
            str(script("processed_corpora/helpers/big_corpus_for_tokenizer.py")),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        dry_run=args.dry_run,
    )


def run_pivot(args: argparse.Namespace, paths: BuildPaths) -> None:
    cmd = [
        PYTHON,
        str(script("scripts/local/scripts/pivot/pivot.py")),
        "--big-corpus-en",
        str(paths.processed_dir / "big_corpus_en.csv"),
        "--big-corpus-zh",
        str(paths.processed_dir / "big_corpus_zh.csv"),
        "--out-dir",
        str(paths.processed_dir / "pivot"),
        "--cache-dir",
        str(pivot_cache_dir(paths)),
        "--directions",
        args.pivot_directions,
        "--splits",
        args.pivot_splits,
        "--api-key-env",
        args.api_key_env,
    ]
    for cache_dir in pivot_read_cache_dirs(args, paths):
        cmd.extend(["--read-cache-dir", str(cache_dir)])
    if args.pivot_skip_translation:
        cmd.append("--skip-translation")
    if args.pivot_dry_run:
        cmd.append("--dry-run")
    if args.respect_usage_limit:
        cmd.append("--respect-usage-limit")
    run(cmd, dry_run=args.dry_run)


def build_hard_splits(args: argparse.Namespace, corpus_dir: Path, output_root: Path) -> None:
    split_jobs = [
        ("english", "english_sentence", "en", corpus_dir / "big_corpus_en.csv"),
        ("chinese", "chinese_sentence", "zh", corpus_dir / "big_corpus_zh.csv"),
    ]
    for target_lang, target_col, short, input_csv in split_jobs:
        if not input_csv.exists() and not args.dry_run:
            print(f"Skipping {target_lang} hard splits; missing {input_csv}")
            continue
        out_dir = output_root / f"splits_{short}_v1"
        run(
            [
                PYTHON,
                str(script("formosan_mt_experiments/scripts/build_experiment_splits.py")),
                "--input",
                str(input_csv),
                "--target-lang",
                target_lang,
                "--target-col",
                target_col,
                "--output-prefix",
                f"big_corpus_{short}",
                "--output-dir",
                str(out_dir),
                "--train-ratio",
                str(args.train_ratio),
                "--val-ratio",
                str(args.val_ratio),
                "--test-ratio",
                str(args.test_ratio),
                "--min-formosan-tokens",
                str(args.min_formosan_tokens),
                "--min-target-tokens",
                str(args.min_target_tokens),
                "--min-test-rows",
                str(args.min_test_rows),
                "--min-validate-rows",
                str(args.min_validate_rows),
                "--tiers",
                args.tiers,
            ],
            dry_run=args.dry_run,
        )
        hard_file = out_dir / f"big_corpus_{short}_in_domain_hard.csv"
        dest = corpus_dir / hard_file.name
        if hard_file.exists() and not args.dry_run:
            shutil.copy2(hard_file, dest)
            print(f"Copied headline hard split -> {dest}")


def write_manifest(
    args: argparse.Namespace,
    language_reports: list[dict],
    final_corpus_dir: Path,
    paths: BuildPaths,
) -> None:
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_name": args.corpus_name,
        "languages": language_reports,
        "settings": {
            "public": args.public,
            "units": args.units,
            "workers": args.workers,
            "hard_split_ratios": {
                "train": args.train_ratio,
                "validate": args.val_ratio,
                "test": args.test_ratio,
            },
            "hard_split_minimum_eval_rows": {
                "test": args.min_test_rows,
                "validate": args.min_validate_rows,
            },
            "with_pivot": args.with_pivot,
            "shared_pivot_cache": args.shared_pivot_cache,
            "pivot_read_cache_dirs": [
                str(path)
                for path in pivot_read_cache_dirs(args, paths)
            ],
            "keep_redactions": args.keep_redactions,
            "fresh_downloads": not args.keep_downloaded,
            "keep_build_output": args.keep_build_output,
            "fetch_workers": args.fetch_workers,
            "fetch_download_retries": args.fetch_download_retries,
            "fetch_retry_base_sleep": args.fetch_retry_base_sleep,
            "fetch_retry_max_sleep": args.fetch_retry_max_sleep,
            "allow_download_failures": args.allow_download_failures,
            "public_language_path_prefilter": not args.no_public_language_path_prefilter,
            "exclude_bible": args.exclude_bible,
            "exclude_bible_exact_repos": list(EXACT_BIBLE_REPOS) if args.exclude_bible else [],
            "exclude_repo_patterns": args.exclude_repo_pattern,
            "exclude_path_patterns": args.exclude_path_pattern,
        },
        "outputs": {
            "root": str(paths.root),
            "raw_corpora": str(paths.raw_dir),
            "processed_corpora": str(paths.processed_dir),
            "final_corpus_dir": str(final_corpus_dir),
            "experiment_splits": str(paths.split_root),
        },
        "artifacts": build_artifact_manifest(args, final_corpus_dir, paths),
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Manifest: {paths.manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Formosan MT corpora from FormosanBank XML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--languages", default="all", help="all or comma-separated language codes")
    parser.add_argument(
        "--corpus-name",
        default=None,
        help=(
            "Write all generated artifacts under corpus_builds/<name>/ instead of "
            "the legacy top-level output directories."
        ),
    )
    parser.add_argument(
        "--build-root",
        type=Path,
        default=None,
        help=(
            "Root directory for named builds. With --corpus-name this is the parent "
            "directory; without --corpus-name it is the exact build root."
        ),
    )
    parser.add_argument(
        "--build-public-private",
        action="store_true",
        help=(
            "Build public and private/all-data variants sequentially into separate "
            "named corpus_builds/ directories."
        ),
    )
    parser.add_argument("--public", action="store_true", help="Fetch from public FormosanBank/Corpora XML only")
    parser.add_argument(
        "--no-public-language-path-prefilter",
        action="store_true",
        help=(
            "Disable the conservative public-mode path-language prefilter before "
            "raw XML downloads."
        ),
    )
    parser.add_argument("--force-branch", default=None, help="Force GitHub branch for fetch_xml.py")
    parser.add_argument(
        "--exclude-bible",
        action="store_true",
        help="Exclude the exact Formosan-Taiwan-Bible-Society-Bibles repo/corpus root at fetch time",
    )
    parser.add_argument(
        "--exclude-repo-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring for fetch_xml.py to skip repos; repeat or comma-separate.",
    )
    parser.add_argument(
        "--exclude-path-pattern",
        action="append",
        default=[],
        help="Case-insensitive substring for fetch_xml.py to skip XML paths; repeat or comma-separate.",
    )
    parser.add_argument(
        "--keep-downloaded",
        action="store_true",
        help="Do not clear downloaded_<lang> before fetching; useful for manual incremental debugging.",
    )
    parser.add_argument(
        "--fetch-workers",
        type=int,
        default=4,
        help="Concurrent raw GitHub XML downloads passed to fetch_xml.py.",
    )
    parser.add_argument(
        "--fetch-download-retries",
        type=int,
        default=8,
        help="Per-file transient HTTP retry attempts passed to fetch_xml.py.",
    )
    parser.add_argument(
        "--fetch-retry-base-sleep",
        type=float,
        default=2.0,
        help="Initial raw GitHub download backoff in seconds passed to fetch_xml.py.",
    )
    parser.add_argument(
        "--fetch-retry-max-sleep",
        type=float,
        default=60.0,
        help="Maximum raw GitHub download backoff in seconds passed to fetch_xml.py.",
    )
    parser.add_argument(
        "--allow-download-failures",
        action="store_true",
        help="Allow fetch_xml.py to continue even if some XML candidates never download.",
    )
    parser.add_argument(
        "--keep-build-output",
        action="store_true",
        help=(
            "For named builds, keep existing raw/processed/final/split outputs. "
            "By default named full rebuilds remove stale generated CSVs while preserving pivot caches."
        ),
    )
    parser.add_argument("--formosanbank-path", type=Path, default=None)
    parser.add_argument("--force-qc-update", action="store_true")
    parser.add_argument("--validate-qc", action="store_true")
    parser.add_argument("--units", default="sentences,words")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--legacy-val-ratio", type=float, default=0.10)
    parser.add_argument("--legacy-test-ratio", type=float, default=0.10)
    parser.add_argument("--keep-redactions", action="store_true")

    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-raw", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--skip-hard-splits", action="store_true")
    parser.add_argument(
        "--resplit-only",
        action="store_true",
        help=(
            "Rebuild hard splits and refresh the manifest from an existing named "
            "pivot_corpora_final directory without fetching, cleaning, filtering, "
            "aggregating, or calling DeepL. Requires --corpus-name --with-pivot."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-artifact-checksums",
        action="store_true",
        help="Do not compute SHA-256 checksums for final corpus/split artifacts in the manifest.",
    )

    parser.add_argument("--with-pivot", action="store_true")
    parser.add_argument("--pivot-directions", default="both")
    parser.add_argument("--pivot-splits", default="all")
    parser.add_argument(
        "--api-key-env",
        default="auto",
        help=(
            "DeepL key environment variables to rotate through. The default 'auto' discovers "
            "DEEPL_API_KEY and all numbered DEEPL_API_KEY_N variables loaded from .env."
        ),
    )
    parser.add_argument("--pivot-skip-translation", action="store_true")
    parser.add_argument("--pivot-dry-run", action="store_true")
    parser.add_argument("--respect-usage-limit", action="store_true")
    parser.add_argument(
        "--pivot-read-cache-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Read existing DeepL cache records from this directory before the build's own cache. "
            "Can be repeated. Writes still go only to the build-local cache."
        ),
    )
    parser.add_argument(
        "--no-shared-pivot-cache",
        dest="shared_pivot_cache",
        action="store_false",
        help="Do not seed named pivot builds from processed_corpora/pivot/cache.",
    )
    parser.set_defaults(shared_pivot_cache=True)

    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--test-ratio", type=float, default=0.075)
    parser.add_argument("--min-formosan-tokens", type=int, default=4)
    parser.add_argument("--min-target-tokens", type=int, default=4)
    parser.add_argument("--min-test-rows", type=int, default=500)
    parser.add_argument("--min-validate-rows", type=int, default=150)
    parser.add_argument("--tiers", default="lexical,in_domain_hard,hard_global")
    args = parser.parse_args()
    if args.fetch_workers < 1:
        raise SystemExit("--fetch-workers must be >= 1")
    if args.fetch_download_retries < 1:
        raise SystemExit("--fetch-download-retries must be >= 1")
    if args.fetch_retry_base_sleep < 0:
        raise SystemExit("--fetch-retry-base-sleep must be >= 0")
    if args.fetch_retry_max_sleep < 0:
        raise SystemExit("--fetch-retry-max-sleep must be >= 0")
    if args.min_test_rows < 0:
        raise SystemExit("--min-test-rows must be >= 0")
    if args.min_validate_rows < 0:
        raise SystemExit("--min-validate-rows must be >= 0")
    if args.resplit_only and (not args.corpus_name or not args.with_pivot):
        raise SystemExit("--resplit-only requires --corpus-name and --with-pivot")
    return args


def run_build(args: argparse.Namespace) -> Path:
    paths = resolve_build_paths(args)
    if args.resplit_only:
        if not paths.final_dir.is_dir():
            raise SystemExit(f"Missing finalized pivot corpus for resplit: {paths.final_dir}")
        previous_languages = []
        if paths.manifest_path.is_file():
            previous_languages = json.loads(
                paths.manifest_path.read_text(encoding="utf-8")
            ).get("languages", [])
        build_hard_splits(args, paths.final_dir, paths.split_root)
        if args.exclude_bible and not args.dry_run:
            validate_no_bible_sources(paths, paths.final_dir)
        write_manifest(args, previous_languages, paths.final_dir, paths)
        return paths.root

    languages = parse_languages(args.languages)
    if not paths.legacy_layout:
        print(f"📦  Corpus build root: {paths.root}")
    if should_clean_generated_outputs(args, paths):
        print(f"🧹  Removing stale generated outputs under {paths.root} (pivot caches preserved)")
        clean_generated_outputs(paths)
    if not args.dry_run:
        paths.raw_dir.mkdir(parents=True, exist_ok=True)
        paths.processed_dir.mkdir(parents=True, exist_ok=True)

    language_reports = [build_language(lang, args, paths) for lang in languages]

    processed = paths.processed_dir
    final_corpus_dir = processed
    if not args.skip_aggregate:
        build_aggregates(args, processed, processed)

    if args.with_pivot:
        run_pivot(args, paths)
        final_corpus_dir = paths.final_dir
        if not args.dry_run:
            final_corpus_dir.mkdir(parents=True, exist_ok=True)
        build_aggregates(args, processed / "pivot", final_corpus_dir)

    if not args.skip_hard_splits:
        build_hard_splits(args, final_corpus_dir, paths.split_root)

    if args.exclude_bible and not args.dry_run:
        validate_no_bible_sources(paths, final_corpus_dir)

    write_manifest(args, language_reports, final_corpus_dir, paths)
    return paths.root


def run_public_private(args: argparse.Namespace) -> None:
    if args.corpus_name:
        raise SystemExit("--build-public-private chooses corpus names automatically; do not pass --corpus-name.")
    if args.public:
        raise SystemExit("--build-public-private builds both variants; do not also pass --public.")

    built_roots: list[Path] = []
    previous_variant_cache_dirs: list[Path] = []
    for public in (True, False):
        variant_args = copy.copy(args)
        variant_args.build_public_private = False
        variant_args.public = public
        variant_args.corpus_name = variant_name(public=public, exclude_bible=args.exclude_bible)
        variant_args.extra_pivot_read_cache_dirs = list(previous_variant_cache_dirs)
        label = "public" if public else "private/all-data"
        print("\n" + "=" * 80)
        print(f"Building {label} corpus: {variant_args.corpus_name}")
        print("=" * 80)
        built_roots.append(run_build(variant_args))
        previous_variant_cache_dirs.append(pivot_cache_dir(resolve_build_paths(variant_args)))

    print("\nBuilt corpus variants:")
    for root in built_roots:
        print(f"  - {root}")


def main() -> None:
    args = parse_args()
    if args.build_public_private:
        run_public_private(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
