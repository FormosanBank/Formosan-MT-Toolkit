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
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from pipeline_common import (
    PIPELINE_CONFIG_PATH,
    atomic_write_json,
    git_state,
    load_pipeline_config,
    sha256_file,
    utc_now,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable
PIPELINE_CONFIG = load_pipeline_config()
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
    elif args.build_root:
        root = args.build_root
    else:
        raise SystemExit("Choose an isolated output with --corpus-name or --build-root, or use --build-public-private.")

    root = root.expanduser().resolve()
    raw_dir = root / "raw_corpora"
    processed_dir = root / "processed_corpora"
    final_dir = root / "pivot_corpora_final"
    split_root = root / "formosan_mt_experiments" / "data"
    manifest_path = root / "mt_build_manifest.json"
    source_snapshot_path = root / "source_repository_snapshot.json"

    return BuildPaths(
        root=root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        final_dir=final_dir,
        split_root=split_root,
        manifest_path=manifest_path,
        source_snapshot_path=source_snapshot_path,
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
    remove_path(paths.processed_dir / "aggregate_manifest.json")
    for pattern in ("*_processed.csv", "big_corpus*.csv"):
        for path in paths.processed_dir.glob(pattern):
            remove_path(path)
    pivot_dir = paths.processed_dir / "pivot"
    if pivot_dir.exists():
        for pattern in (
            "big_corpus*.csv",
            "big_corpus*.incomplete",
            "pivot_manifest.json",
            "pivot_rejections_*.csv",
            "aggregate_manifest.json",
            "*.tmp",
        ):
            for path in pivot_dir.glob(pattern):
                remove_path(path)


def should_clean_generated_outputs(args: argparse.Namespace, paths: BuildPaths) -> bool:
    if args.dry_run or args.keep_build_output:
        return False
    # Incremental stage skips imply the caller expects existing intermediates.
    return not (args.skip_raw or args.skip_filter or args.skip_aggregate)


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


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
            raise SystemExit(f"{stage} manifest mismatch for {key}: expected {value!r}, found {payload.get(key)!r}")
    return payload


def stage_manifest_record(path: Path) -> dict:
    payload = require_json_manifest(path, stage=path.stem)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "complete": True,
    }


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
                part.strip().lower() for part in (row.get("source") or "").replace("\\", "/").split("/") if part.strip()
            ]
            if any(part in exact_repo_components for part in source_parts):
                count += 1
    return count


def validate_no_bible_sources(paths: BuildPaths, final_corpus_dir: Path) -> None:
    files: list[Path] = []
    files.extend(sorted(paths.processed_dir.glob("*_processed.csv")))
    files.extend(final_corpus_dir.glob("big_corpus*.csv"))
    failures = {str(path): rows for path in files if (rows := count_bible_source_rows(path)) > 0}
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


def package_training_provenance(
    paths: BuildPaths,
    final_corpus_dir: Path,
) -> Path:
    provenance_dir = final_corpus_dir / "provenance"
    remove_path(provenance_dir)
    provenance_dir.mkdir(parents=True)
    sources = {
        "mt_build_manifest.json": paths.manifest_path,
        "source_repository_snapshot.json": paths.source_snapshot_path,
        "corpus_pipeline.json": PIPELINE_CONFIG_PATH,
        "mt_standardization.json": PROJECT_ROOT / "config" / "mt_standardization.json",
        "aggregate_manifest.json": final_corpus_dir / "aggregate_manifest.json",
        "split_en_in_domain_hard.json": (
            paths.split_root / "splits_en_v1" / "report_in_domain_hard.json"
        ),
        "split_zh_in_domain_hard.json": (
            paths.split_root / "splits_zh_v1" / "report_in_domain_hard.json"
        ),
        "validate_en_in_domain_hard.json": (
            paths.split_root
            / "splits_en_v1"
            / "validation_in_domain_hard.json"
        ),
        "validate_zh_in_domain_hard.json": (
            paths.split_root
            / "splits_zh_v1"
            / "validation_in_domain_hard.json"
        ),
        "exposure_en_in_domain_hard.json": (
            paths.split_root
            / "splits_en_v1"
            / "exposure_in_domain_hard.json"
        ),
        "exposure_zh_in_domain_hard.json": (
            paths.split_root
            / "splits_zh_v1"
            / "exposure_in_domain_hard.json"
        ),
    }
    for prepared_dir in sorted(paths.root.glob("prepared_*")):
        language = prepared_dir.name.removeprefix("prepared_")
        for source_name, prefix in (
            ("_qc_manifest.json", "xml_preparation"),
            ("_mt_standard_manifest.json", "mt_standard"),
        ):
            source = prepared_dir / source_name
            if source.is_file():
                sources[f"{prefix}_{language}.json"] = source
    pivot_manifest = (
        paths.processed_dir / "pivot" / "pivot_manifest.json"
    )
    if pivot_manifest.is_file():
        sources["pivot_manifest.json"] = pivot_manifest
        pivot_payload = json.loads(
            pivot_manifest.read_text(encoding="utf-8")
        )
        for stats in pivot_payload.get("stats", []):
            quarantine_value = str(
                stats.get("quarantine_path") or ""
            ).strip()
            if not quarantine_value:
                continue
            quarantine_path = Path(quarantine_value)
            expected_hash = str(
                stats.get("quarantine_sha256") or ""
            ).strip()
            if not quarantine_path.is_file():
                raise SystemExit(
                    "Cannot package training provenance; missing pivot "
                    f"quarantine ledger {quarantine_path}"
                )
            actual_hash = sha256_file(quarantine_path)
            if not expected_hash or actual_hash != expected_hash:
                raise SystemExit(
                    "Cannot package training provenance; pivot quarantine "
                    f"hash mismatch for {quarantine_path}"
                )
            sources[quarantine_path.name] = quarantine_path
    missing = [
        name
        for name, source in sources.items()
        if not source.is_file()
    ]
    if missing:
        raise SystemExit(
            f"Cannot package training provenance; missing {missing}"
        )
    artifacts: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        destination = provenance_dir / name
        shutil.copy2(source, destination)
        artifacts[name] = artifact_record(
            destination,
            compute_hash=True,
        )
    bundle_manifest = provenance_dir / "bundle_manifest.json"
    atomic_write_json(
        bundle_manifest,
        {
            "schema_version": 1,
            "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
            "corpus_name": json.loads(
                paths.manifest_path.read_text(encoding="utf-8")
            ).get("corpus_name"),
            "artifacts": artifacts,
            "complete": True,
        },
    )
    return bundle_manifest


def build_language(lang: Language, args: argparse.Namespace, paths: BuildPaths) -> dict:
    source_xml_dir = paths.source_xml_dir(lang)
    prepared_xml_dir = paths.prepared_xml_dir(lang)
    raw_zh = paths.raw_dir / f"{lang.code}_zh.csv"
    raw_en = paths.raw_dir / f"{lang.code}_en.csv"
    proc_zh = paths.processed_dir / f"{lang.code}_zh_processed.csv"
    proc_en = paths.processed_dir / f"{lang.code}_en_processed.csv"
    fetch_manifest = source_xml_dir / "_fetch_manifest.json"
    qc_manifest = prepared_xml_dir / "_qc_manifest.json"
    mt_standard_manifest = prepared_xml_dir / "_mt_standard_manifest.json"

    if not args.skip_fetch:
        cmd = [PYTHON, str(script("scripts/local/fetch_xml.py")), "--src-lang", lang.code]
        cmd.extend(["--out-dir", str(source_xml_dir)])
        cmd.extend(["--workers", str(args.fetch_workers)])
        cmd.extend(["--download-retries", str(args.fetch_download_retries)])
        cmd.extend(["--retry-base-sleep", str(args.fetch_retry_base_sleep)])
        cmd.extend(["--retry-max-sleep", str(args.fetch_retry_max_sleep)])
        cmd.extend(["--repository-snapshot", str(paths.source_snapshot_path)])
        cmd.append("--refresh-repository-metadata")
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
    if not args.dry_run:
        require_json_manifest(
            fetch_manifest,
            stage=f"{lang.code} fetch",
            expected={"source_language": lang.code},
        )

    if not args.skip_clean:
        cmd = [PYTHON, str(script("scripts/local/clean_xml.py")), "--src-lang", lang.code]
        cmd.extend(["--in-dir", str(source_xml_dir)])
        cmd.extend(["--out-dir", str(prepared_xml_dir)])
        if args.formosanbank_path:
            cmd.extend(["--formosanbank-path", str(args.formosanbank_path)])
        cmd.extend(["--qc-revision", args.qc_revision])
        if args.force_qc_update:
            cmd.append("--force-update")
        if args.skip_qc_validation:
            cmd.append("--skip-validation")
        run(cmd, dry_run=args.dry_run)
    if not args.dry_run:
        qc_payload = require_json_manifest(
            qc_manifest,
            stage=f"{lang.code} QC",
            expected={"source_language": lang.code},
        )
        qc_revision = qc_payload.get("formosanbank_qc", {}).get("revision")
        if qc_revision != args.qc_revision:
            raise SystemExit(f"{lang.code} QC revision mismatch: expected {args.qc_revision}, found {qc_revision}")

    if not args.skip_mt_standardization:
        run(
            [
                PYTHON,
                str(script("scripts/local/standardize_mt_corpus.py")),
                "--xml-dir",
                str(prepared_xml_dir),
                "--src-lang",
                lang.code,
                "--profile",
                str(args.mt_standard_profile),
            ],
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        mt_payload = require_json_manifest(
            mt_standard_manifest,
            stage=f"{lang.code} MT standardization",
            expected={"source_language": lang.code},
        )
        expected_profile_hash = sha256_file(args.mt_standard_profile)
        profile_record = mt_payload.get("profile", {})
        if (
            profile_record.get("id")
            != PIPELINE_CONFIG["mt_standardization"]["profile_id"]
            or profile_record.get("sha256") != expected_profile_hash
        ):
            raise SystemExit(
                f"{lang.code} MT standardization profile does not match the build contract"
            )

    if not args.skip_raw:
        run(
            [
                PYTHON,
                str(script("scripts/local/make_corpus.py")),
                "--xml-dir",
                str(prepared_xml_dir),
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
                str(prepared_xml_dir),
                "--target",
                "english",
                "--out",
                str(raw_en),
                "--units",
                args.units,
            ],
            dry_run=args.dry_run,
        )
    if not args.dry_run:
        require_json_manifest(
            raw_zh.with_suffix(".extraction.json"),
            stage=f"{lang.code} Chinese extraction",
            expected={"source_language": lang.code, "target": "chinese"},
        )
        require_json_manifest(
            raw_en.with_suffix(".extraction.json"),
            stage=f"{lang.code} English extraction",
            expected={"source_language": lang.code, "target": "english"},
        )

    if not args.skip_filter:
        filter_base = [
            PYTHON,
            str(script("scripts/local/filter_split_corpus.py")),
            "--workers",
            str(args.workers),
            "--no-split",
        ]
        if args.keep_redactions:
            filter_base.append("--keep-redactions")
        run(filter_base + ["--input", str(raw_zh), "--output", str(proc_zh)], dry_run=args.dry_run)
        run(filter_base + ["--input", str(raw_en), "--output", str(proc_en)], dry_run=args.dry_run)
    if not args.dry_run:
        zh_filter_manifest = paths.processed_dir / "filter_reports" / proc_zh.stem / "summary.json"
        en_filter_manifest = paths.processed_dir / "filter_reports" / proc_en.stem / "summary.json"
        require_json_manifest(zh_filter_manifest, stage=f"{lang.code} Chinese cleaning")
        require_json_manifest(en_filter_manifest, stage=f"{lang.code} English cleaning")

    return {
        "language": lang.name,
        "code": lang.code,
        "raw_zh_rows": count_csv_rows(raw_zh),
        "raw_en_rows": count_csv_rows(raw_en),
        "processed_zh_rows": count_csv_rows(proc_zh),
        "processed_en_rows": count_csv_rows(proc_en),
        "manifests": {
            "fetch": stage_manifest_record(fetch_manifest) if not args.dry_run else {},
            "qc": stage_manifest_record(qc_manifest) if not args.dry_run else {},
            "mt_standard": (
                stage_manifest_record(mt_standard_manifest)
                if not args.dry_run
                else {}
            ),
            "extract_zh": (stage_manifest_record(raw_zh.with_suffix(".extraction.json")) if not args.dry_run else {}),
            "extract_en": (stage_manifest_record(raw_en.with_suffix(".extraction.json")) if not args.dry_run else {}),
            "clean_zh": (stage_manifest_record(zh_filter_manifest) if not args.dry_run else {}),
            "clean_en": (stage_manifest_record(en_filter_manifest) if not args.dry_run else {}),
        },
    }


def build_aggregates(args: argparse.Namespace, input_dir: Path, output_dir: Path) -> None:
    run(
        [
            PYTHON,
            str(script("scripts/local/build_big_corpus.py")),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        require_json_manifest(
            output_dir / "aggregate_manifest.json",
            stage=f"aggregate {input_dir.name}",
        )


def run_pivot(args: argparse.Namespace, paths: BuildPaths) -> None:
    cmd = [
        PYTHON,
        str(script("scripts/local/pivot.py")),
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
    if not args.dry_run:
        pivot_manifest = require_json_manifest(
            paths.processed_dir / "pivot" / "pivot_manifest.json",
            stage="DeepL pivot",
        )
        for stats in pivot_manifest.get("stats", []):
            if stats.get("synthetic_rows_missing") or stats.get("errors") or stats.get("stopped_reason"):
                raise SystemExit(f"DeepL pivot direction {stats.get('direction')} is incomplete: {stats}")


def build_hard_splits(args: argparse.Namespace, corpus_dir: Path, output_root: Path) -> None:
    split_jobs = [
        ("english", "english_sentence", "en", corpus_dir / "big_corpus_en.csv"),
        ("chinese", "chinese_sentence", "zh", corpus_dir / "big_corpus_zh.csv"),
    ]
    for target_lang, target_col, short, input_csv in split_jobs:
        if not input_csv.exists() and not args.dry_run:
            raise SystemExit(f"Cannot build {target_lang} hard splits; missing {input_csv}")
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
                "--ngram-jaccard-threshold",
                str(args.ngram_jaccard_threshold),
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
            run(
                [
                    PYTHON,
                    str(script("formosan_mt_experiments/scripts/validate_experiment.py")),
                    "--input",
                    str(hard_file),
                    "--target-col",
                    target_col,
                    "--target-lang",
                    target_lang,
                    "--min-test-ratio",
                    str(args.test_ratio),
                    "--min-validate-ratio",
                    str(args.val_ratio),
                    "--min-test-rows",
                    str(args.min_test_rows),
                    "--min-validate-rows",
                    str(args.min_validate_rows),
                    "--ngram-jaccard-threshold",
                    str(args.ngram_jaccard_threshold),
                    "--report",
                    str(out_dir / "validation_in_domain_hard.json"),
                    "--require-human-eval",
                    "--require-document-holdout-report",
                ],
                dry_run=False,
            )
            require_json_manifest(
                out_dir / "validation_in_domain_hard.json",
                stage=f"{target_lang} hard-split validation",
            )
            run(
                [
                    PYTHON,
                    str(
                        script(
                            "formosan_mt_experiments/scripts/"
                            "audit_corpus_exposure.py"
                        )
                    ),
                    "--input",
                    str(hard_file),
                    "--target-col",
                    target_col,
                    "--target-lang",
                    target_lang,
                    "--high-threshold",
                    str(
                        PIPELINE_CONFIG["exposure_audit"][
                            "high_threshold"
                        ]
                    ),
                    "--max-high-exposure-rate",
                    str(
                        PIPELINE_CONFIG["exposure_audit"][
                            "max_high_exposure_rate"
                        ]
                    ),
                    "--report",
                    str(out_dir / "exposure_in_domain_hard.json"),
                ],
                dry_run=False,
            )
            require_json_manifest(
                out_dir / "exposure_in_domain_hard.json",
                stage=f"{target_lang} TAME-MT exposure audit",
            )
        elif not args.dry_run:
            raise SystemExit(f"Hard split builder did not produce {hard_file}")


def write_manifest(
    args: argparse.Namespace,
    language_reports: list[dict],
    final_corpus_dir: Path,
    paths: BuildPaths,
) -> None:
    artifacts = build_artifact_manifest(args, final_corpus_dir, paths)
    required_artifacts = {
        "big_corpus_en",
        "big_corpus_zh",
        "big_corpus_combined",
        "big_corpus_en_in_domain_hard",
        "big_corpus_zh_in_domain_hard",
    }
    missing_artifacts = sorted(name for name in required_artifacts if not artifacts.get(name, {}).get("exists"))
    stage_paths = {
        "source_repository_snapshot": paths.source_snapshot_path,
        "processed_aggregate": paths.processed_dir / "aggregate_manifest.json",
        "final_aggregate": final_corpus_dir / "aggregate_manifest.json",
        "split_en": paths.split_root / "splits_en_v1" / "report_in_domain_hard.json",
        "split_zh": paths.split_root / "splits_zh_v1" / "report_in_domain_hard.json",
        "validate_en": paths.split_root / "splits_en_v1" / "validation_in_domain_hard.json",
        "validate_zh": paths.split_root / "splits_zh_v1" / "validation_in_domain_hard.json",
        "exposure_en": paths.split_root / "splits_en_v1" / "exposure_in_domain_hard.json",
        "exposure_zh": paths.split_root / "splits_zh_v1" / "exposure_in_domain_hard.json",
    }
    if args.with_pivot:
        stage_paths["pivot"] = paths.processed_dir / "pivot" / "pivot_manifest.json"
    stage_manifests: dict[str, dict] = {}
    missing_stages: list[str] = []
    if not args.dry_run:
        for name, path in stage_paths.items():
            if path.is_file():
                try:
                    stage_manifests[name] = stage_manifest_record(path)
                except SystemExit:
                    missing_stages.append(name)
            else:
                missing_stages.append(name)

    dependency_versions: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "python-dotenv",
        "requests",
        "sacrebleu",
        "sentencepiece",
        "tame-mt",
        "torch",
        "transformers",
    ):
        try:
            dependency_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependency_versions[package] = "not-installed"
    complete = not missing_artifacts and not missing_stages
    manifest = {
        "schema_version": 3,
        "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
        "created_at": utc_now(),
        "corpus_name": args.corpus_name,
        "command": [sys.executable, *sys.argv],
        "repository": git_state(PROJECT_ROOT),
        "pipeline_config": {
            "path": str(PIPELINE_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PIPELINE_CONFIG_PATH),
        },
        "mt_standardization": {
            "id": PIPELINE_CONFIG["mt_standardization"]["profile_id"],
            "sha256": sha256_file(args.mt_standard_profile),
            "namespace": PIPELINE_CONFIG["mt_standardization"]["namespace"],
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "dependencies": dependency_versions,
        },
        "languages": language_reports,
        "settings": {
            "public": args.public,
            "qc_revision": args.qc_revision,
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
            "hard_split_character_ngram_jaccard_threshold": args.ngram_jaccard_threshold,
            "with_pivot": args.with_pivot,
            "pivot_read_cache_dirs": [str(path) for path in pivot_read_cache_dirs(args, paths)],
            "keep_redactions": args.keep_redactions,
            "fresh_downloads": not args.keep_downloaded,
            "source_repository_snapshot": str(paths.source_snapshot_path),
            "mt_standardization_profile": str(args.mt_standard_profile),
            "keep_build_output": args.keep_build_output,
            "allow_dirty_repository": args.allow_dirty_repository,
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
        "stage_manifests": stage_manifests,
        "artifacts": artifacts,
        "release_gate": {
            "required_artifacts": sorted(required_artifacts),
            "missing_artifacts": missing_artifacts,
            "missing_or_incomplete_stages": missing_stages,
        },
        "complete": complete,
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return
    atomic_write_json(paths.manifest_path, manifest)
    print(f"Manifest: {paths.manifest_path}")
    if not complete:
        raise SystemExit(f"Corpus build did not pass release gates; inspect {paths.manifest_path}")
    try:
        bundle_manifest = package_training_provenance(
            paths,
            final_corpus_dir,
        )
    except (Exception, SystemExit) as exc:
        manifest["complete"] = False
        manifest["release_gate"]["provenance_bundle_error"] = str(exc)
        atomic_write_json(paths.manifest_path, manifest)
        raise
    print(f"Training provenance bundle: {bundle_manifest}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Formosan MT corpora from FormosanBank XML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--languages", default="all", help="all or comma-separated language codes")
    parser.add_argument(
        "--corpus-name",
        default=None,
        help=("Write all generated artifacts under corpus_builds/<name>/."),
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
            "Build public and private/all-data variants sequentially into separate named corpus_builds/ directories."
        ),
    )
    parser.add_argument("--public", action="store_true", help="Fetch from public FormosanBank/Corpora XML only")
    parser.add_argument(
        "--no-public-language-path-prefilter",
        action="store_true",
        help=("Disable the conservative public-mode path-language prefilter before raw XML downloads."),
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
    parser.add_argument(
        "--qc-revision",
        default=PIPELINE_CONFIG["formosanbank"]["qc_revision"],
        help="Pinned FormosanBank commit used for QC.",
    )
    parser.add_argument(
        "--mt-standard-profile",
        type=Path,
        default=PROJECT_ROOT / PIPELINE_CONFIG["mt_standardization"]["profile"],
        help="Versioned toolkit profile used to derive model-facing Formosan text.",
    )
    parser.add_argument("--force-qc-update", action="store_true")
    parser.add_argument(
        "--skip-qc-validation",
        action="store_true",
        help="Diagnostic only; production builds run hard FormosanBank validators.",
    )
    parser.add_argument(
        "--validate-qc",
        action="store_true",
        help="Deprecated compatibility flag; QC validation is now enabled by default.",
    )
    parser.add_argument("--units", default="sentences,words")
    parser.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    parser.add_argument("--keep-redactions", action="store_true")

    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-clean", action="store_true")
    parser.add_argument("--skip-mt-standardization", action="store_true")
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
    parser.add_argument(
        "--allow-dirty-repository",
        action="store_true",
        help=(
            "Diagnostic only: permit a non-dry-run build from a dirty Git checkout. "
            "Production releases fail closed by default."
        ),
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
        "--train-ratio",
        type=float,
        default=PIPELINE_CONFIG["splits"]["train_ratio"],
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=PIPELINE_CONFIG["splits"]["validate_ratio"],
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=PIPELINE_CONFIG["splits"]["test_ratio"],
    )
    parser.add_argument("--min-formosan-tokens", type=int, default=4)
    parser.add_argument("--min-target-tokens", type=int, default=4)
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_test_rows"],
    )
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_validate_rows"],
    )
    parser.add_argument(
        "--ngram-jaccard-threshold",
        type=float,
        default=PIPELINE_CONFIG["splits"]["character_ngram_jaccard_threshold"],
    )
    parser.add_argument(
        "--tiers",
        default=PIPELINE_CONFIG["splits"]["headline_tier"],
    )
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
    if not 0.5 <= args.ngram_jaccard_threshold <= 1.0:
        raise SystemExit("--ngram-jaccard-threshold must be in [0.5, 1.0]")
    if args.tiers != PIPELINE_CONFIG["splits"]["headline_tier"]:
        raise SystemExit(
            "Corpus pipeline v3 supports only "
            f"--tiers {PIPELINE_CONFIG['splits']['headline_tier']}"
        )
    if len(args.qc_revision) != 40 or any(
        character not in "0123456789abcdef" for character in args.qc_revision.lower()
    ):
        raise SystemExit("--qc-revision must be a full 40-character commit SHA")
    args.mt_standard_profile = args.mt_standard_profile.expanduser().resolve()
    if not args.mt_standard_profile.is_file():
        raise SystemExit(
            f"MT standardization profile does not exist: {args.mt_standard_profile}"
        )
    split_total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(split_total - 1.0) > 1e-9:
        raise SystemExit(f"Hard split ratios must sum to 1.0, found {split_total:.12f}")
    if args.with_pivot and args.pivot_splits.strip().lower() not in {"all", "*"}:
        raise SystemExit("Pivoting now occurs before the single hard split and must use --pivot-splits all")
    if args.skip_artifact_checksums and not args.dry_run:
        raise SystemExit("Production corpus builds require artifact checksums")
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
            previous_languages = json.loads(paths.manifest_path.read_text(encoding="utf-8")).get("languages", [])
        build_hard_splits(args, paths.final_dir, paths.split_root)
        if args.exclude_bible and not args.dry_run:
            validate_no_bible_sources(paths, paths.final_dir)
        write_manifest(args, previous_languages, paths.final_dir, paths)
        return paths.root

    languages = parse_languages(args.languages)
    print(f"📦  Corpus build root: {paths.root}")
    if should_clean_generated_outputs(args, paths):
        print(f"🧹  Removing stale generated outputs under {paths.root} (pivot caches preserved)")
        clean_generated_outputs(paths)
    if not args.dry_run:
        paths.raw_dir.mkdir(parents=True, exist_ok=True)
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run and not args.skip_fetch and not args.keep_downloaded:
        remove_path(paths.source_snapshot_path)

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
    repository = git_state(PROJECT_ROOT)
    if (
        not args.dry_run
        and not args.allow_dirty_repository
        and repository.get("dirty")
    ):
        paths = ", ".join(repository.get("dirty_paths", []))
        raise SystemExit(
            "Production corpus builds require a clean Git checkout. "
            f"Commit or remove local changes first: {paths}"
        )
    if args.build_public_private:
        run_public_private(args)
    else:
        run_build(args)


if __name__ == "__main__":
    main()
