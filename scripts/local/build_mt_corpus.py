#!/usr/bin/env python3
"""End-to-end FormosanBank XML -> MT corpus builder.

This is the reproducible replacement for the historical build_corpora.sh flow.
It keeps each stage explicit while making the default path produce hard-split
multilingual corpora suitable for the current NLLB/SPM8k directional recipes.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import copy
import csv
import importlib.metadata
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from build_output import (
    CommandExecutionError,
    format_aggregate_summary,
    format_fetch_summary,
    format_language_summary,
    format_pivot_summary,
    format_rule_summary,
    format_split_summary,
    run_logged,
)
from pipeline_common import (
    PIPELINE_CONFIG_PATH,
    atomic_write_json,
    git_state,
    load_pipeline_config,
    sha256_file,
    utc_now,
)
from stage_cache import (
    cached_stage_valid,
    file_inventory,
    load_stage_cache,
    record_cached_stage,
    stage_key,
)
from tqdm import tqdm

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


def run(
    cmd: list[str],
    *,
    label: str,
    log_path: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
    append: bool = False,
) -> None:
    run_logged(
        cmd,
        project_root=PROJECT_ROOT,
        label=label,
        log_path=log_path,
        dry_run=dry_run,
        env=env,
        verbose=verbose,
        quiet=quiet,
        append=append,
    )


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
    if pivot_dir.exists():
        for pattern in (
            "big_corpus*.incomplete",
            "*.tmp",
        ):
            for path in pivot_dir.glob(pattern):
                remove_path(path)


def should_clean_generated_outputs(args: argparse.Namespace, paths: BuildPaths) -> bool:
    if args.dry_run or args.keep_build_output:
        return False
    # Incremental stage skips imply the caller expects existing intermediates.
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


def language_cache_path(paths: BuildPaths, lang: Language) -> Path:
    return paths.root / ".stage_cache" / f"{lang.code}.json"


def build_cache_path(paths: BuildPaths) -> Path:
    return paths.root / ".stage_cache" / "build.json"


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
            for path_field, hash_field, label in (
                ("quarantine_path", "quarantine_sha256", "quarantine"),
                ("cache_conflict_path", "cache_conflict_sha256", "cache conflict"),
            ):
                value = str(stats.get(path_field) or "").strip()
                if not value:
                    continue
                artifact_path = Path(value)
                expected_hash = str(stats.get(hash_field) or "").strip()
                if not artifact_path.is_file():
                    raise SystemExit(
                        "Cannot package training provenance; missing pivot "
                        f"{label} ledger {artifact_path}"
                    )
                actual_hash = sha256_file(artifact_path)
                if not expected_hash or actual_hash != expected_hash:
                    raise SystemExit(
                        f"Cannot package training provenance; pivot {label} "
                        f"hash mismatch for {artifact_path}"
                    )
                sources[artifact_path.name] = artifact_path
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
    zh_filter_manifest = paths.processed_dir / "filter_reports" / proc_zh.stem / "summary.json"
    en_filter_manifest = paths.processed_dir / "filter_reports" / proc_en.stem / "summary.json"
    cache_path = language_cache_path(paths, lang)
    cache = load_stage_cache(cache_path)
    language_log = stage_log(paths, f"language_{lang.code}")
    stage_status: dict[str, str] = {}
    log_started = False

    def run_language_stage(cmd: list[str], stage: str) -> None:
        nonlocal log_started
        run(
            cmd,
            label=f"{lang.code} {stage}",
            log_path=language_log,
            dry_run=args.dry_run,
            verbose=args.verbose,
            quiet=not args.verbose,
            append=log_started,
        )
        log_started = True

    if not args.dry_run:
        fetch_payload = require_json_manifest(
            fetch_manifest,
            stage=f"{lang.code} fetch",
            expected={"source_language": lang.code},
        )
    else:
        fetch_payload = {}

    if not args.skip_clean:
        qc_key = stage_key(
            "qc",
            {
                "fetch_inventory_records_sha256": fetch_payload.get("inventory_records_sha256"),
                "source_language": lang.code,
                "qc_revision": args.qc_revision,
                "skip_validation": args.skip_qc_validation,
                "pipeline_config_sha256": sha256_file(PIPELINE_CONFIG_PATH),
            },
            [
                script("scripts/local/clean_xml.py"),
                script("scripts/local/pipeline_common.py"),
                script("scripts/local/qc_change_audit.py"),
                script("scripts/local/xml_repairs.py"),
                script("scripts/local/qc_reporting.py"),
            ],
        )
        qc_cached = (
            not args.dry_run
            and not args.no_stage_cache
            and not args.force_qc_update
            and cached_stage_valid(paths.root, cache, "qc", qc_key)
        )
        if qc_cached:
            stage_status["QC"] = "cached"
        else:
            stage_status["QC"] = "planned" if args.dry_run else "rebuilt"
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
            run_language_stage(cmd, "QC")
    if not args.dry_run:
        qc_payload = require_json_manifest(
            qc_manifest,
            stage=f"{lang.code} QC",
            expected={
                "source_language": lang.code,
                "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
            },
        )
        qc_revision = qc_payload.get("formosanbank_qc", {}).get("revision")
        if qc_revision != args.qc_revision:
            raise SystemExit(f"{lang.code} QC revision mismatch: expected {args.qc_revision}, found {qc_revision}")
        if not args.skip_clean and not qc_cached:
            record_cached_stage(
                paths.root,
                cache_path,
                cache,
                "qc",
                qc_key,
                [path for path in prepared_xml_dir.rglob("*") if path.is_file()],
                lang.code,
            )

    if not args.skip_mt_standardization:
        mt_key = stage_key(
            "mt_standard",
            {
                "qc_manifest_sha256": sha256_file(qc_manifest) if qc_manifest.is_file() else "",
                "qc_transform_inventory_sha256": (
                    qc_payload.get("transform_inventory", {}).get("sha256")
                    if not args.dry_run
                    else ""
                ),
                "profile_sha256": sha256_file(args.mt_standard_profile),
                "source_language": lang.code,
            },
            [
                script("scripts/local/standardize_mt_corpus.py"),
                script("scripts/local/mt_standardization.py"),
                script("scripts/local/pipeline_common.py"),
            ],
        )
        mt_cached = (
            not args.dry_run
            and not args.no_stage_cache
            and cached_stage_valid(paths.root, cache, "mt_standard", mt_key)
        )
        if mt_cached:
            stage_status["MT standardization"] = "cached"
        else:
            stage_status["MT standardization"] = "planned" if args.dry_run else "rebuilt"
            run_language_stage(
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
                "MT standardization",
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
        if not args.skip_mt_standardization and not mt_cached:
            record_cached_stage(
                paths.root,
                cache_path,
                cache,
                "mt_standard",
                mt_key,
                [
                    mt_standard_manifest,
                    prepared_xml_dir / str(mt_payload["inventory"]["path"]),
                ],
                lang.code,
            )

    if not args.skip_raw:
        extract_key = stage_key(
            "extract",
            {
                "fetch_inventory_sha256": fetch_payload.get("inventory_sha256"),
                "qc_manifest_sha256": sha256_file(qc_manifest) if qc_manifest.is_file() else "",
                "mt_manifest_sha256": (
                    sha256_file(mt_standard_manifest)
                    if mt_standard_manifest.is_file()
                    else ""
                ),
                "units": sorted(part.strip() for part in args.units.split(",") if part.strip()),
            },
            [
                script("scripts/local/make_corpus.py"),
                script("scripts/local/pipeline_common.py"),
            ],
        )
        extract_cached = (
            not args.dry_run
            and not args.no_stage_cache
            and cached_stage_valid(paths.root, cache, "extract", extract_key)
        )
        if extract_cached:
            stage_status["extraction"] = "cached"
        else:
            stage_status["extraction"] = "planned" if args.dry_run else "rebuilt"
            run_language_stage(
                [
                    PYTHON,
                    str(script("scripts/local/make_corpus.py")),
                    "--xml-dir",
                    str(prepared_xml_dir),
                    "--output",
                    f"chinese={raw_zh}",
                    "--output",
                    f"english={raw_en}",
                    "--units",
                    args.units,
                ],
                "bilingual extraction",
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
        if not args.skip_raw and not extract_cached:
            record_cached_stage(
                paths.root,
                cache_path,
                cache,
                "extract",
                extract_key,
                [
                    raw_zh,
                    raw_zh.with_suffix(".extraction.json"),
                    raw_en,
                    raw_en.with_suffix(".extraction.json"),
                ],
                lang.code,
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
        for target, raw_path, output_path, report_path in (
            ("zh", raw_zh, proc_zh, zh_filter_manifest),
            ("en", raw_en, proc_en, en_filter_manifest),
        ):
            filter_key = stage_key(
                f"filter_{target}",
                {
                    "input_sha256": sha256_file(raw_path) if raw_path.is_file() else "",
                    "extraction_report_sha256": (
                        sha256_file(raw_path.with_suffix(".extraction.json"))
                        if raw_path.with_suffix(".extraction.json").is_file()
                        else ""
                    ),
                    "keep_redactions": args.keep_redactions,
                    "pipeline_config_sha256": sha256_file(PIPELINE_CONFIG_PATH),
                },
                [
                    script("scripts/local/filter_split_corpus.py"),
                    script("scripts/local/corpus_quality.py"),
                    script("scripts/local/pipeline_common.py"),
                ],
            )
            cache_name = f"filter_{target}"
            filter_cached = (
                not args.dry_run
                and not args.no_stage_cache
                and cached_stage_valid(paths.root, cache, cache_name, filter_key)
            )
            if filter_cached:
                stage_status[f"{target} filtering"] = "cached"
                continue
            stage_status[f"{target} filtering"] = "planned" if args.dry_run else "rebuilt"
            if not args.dry_run:
                remove_path(report_path.parent)
            run_language_stage(
                filter_base + ["--input", str(raw_path), "--output", str(output_path)],
                f"{target} filtering",
            )
            if not args.dry_run:
                require_json_manifest(
                    report_path,
                    stage=f"{lang.code} {target} cleaning",
                )
                record_cached_stage(
                    paths.root,
                    cache_path,
                    cache,
                    cache_name,
                    filter_key,
                    [output_path, *report_path.parent.glob("*")],
                    lang.code,
                )
    if not args.dry_run:
        require_json_manifest(zh_filter_manifest, stage=f"{lang.code} Chinese cleaning")
        require_json_manifest(en_filter_manifest, stage=f"{lang.code} English cleaning")

    return {
        "language": lang.name,
        "code": lang.code,
        "raw_zh_rows": count_csv_rows(raw_zh),
        "raw_en_rows": count_csv_rows(raw_en),
        "processed_zh_rows": count_csv_rows(proc_zh),
        "processed_en_rows": count_csv_rows(proc_en),
        "stage_status": stage_status,
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


def fetch_languages(
    languages: list[Language],
    args: argparse.Namespace,
    paths: BuildPaths,
) -> None:
    if args.skip_fetch:
        return
    cmd = [
        PYTHON,
        str(script("scripts/local/fetch_xml.py")),
        "--src-langs",
        ",".join(lang.code for lang in languages),
        "--out-root",
        str(paths.root),
        "--workers",
        str(args.fetch_workers),
        "--download-retries",
        str(args.fetch_download_retries),
        "--retry-base-sleep",
        str(args.fetch_retry_base_sleep),
        "--retry-max-sleep",
        str(args.fetch_retry_max_sleep),
        "--repository-snapshot",
        str(paths.source_snapshot_path),
        "--refresh-repository-metadata",
    ]
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
    run(
        cmd,
        label=f"Acquire XML snapshot for {len(languages)} languages",
        log_path=stage_log(paths, "fetch"),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if not args.dry_run:
        print(format_fetch_summary(paths.root, [lang.code for lang in languages]))


def build_languages(
    languages: list[Language],
    args: argparse.Namespace,
    paths: BuildPaths,
) -> list[dict]:
    language_workers = 1 if args.force_qc_update or args.verbose else args.language_workers
    if args.force_qc_update and args.language_workers > 1:
        print("Language concurrency disabled during forced QC cache refresh.")
    if args.verbose and args.language_workers > 1:
        print("Language concurrency disabled in verbose mode to keep output ordered.")
    reports: dict[str, dict] = {}
    progress = tqdm(
        total=len(languages),
        desc="Prepare languages",
        unit="lang",
        dynamic_ncols=True,
        disable=args.verbose,
    )
    try:
        if language_workers == 1 or len(languages) == 1 or args.dry_run:
            for lang in languages:
                reports[lang.code] = build_language(lang, args, paths)
                progress.update()
        else:
            with futures.ThreadPoolExecutor(max_workers=language_workers) as executor:
                jobs = {executor.submit(build_language, lang, args, paths): lang for lang in languages}
                for job in futures.as_completed(jobs):
                    lang = jobs[job]
                    reports[lang.code] = job.result()
                    progress.update()
    finally:
        progress.close()

    ordered = [reports[lang.code] for lang in languages]
    print("\nLanguage preparation summary")
    for report in ordered:
        print(format_language_summary(paths.root, report))
    print(format_rule_summary(paths.root, ordered))
    for report in ordered:
        report.pop("stage_status", None)
    return ordered


def build_aggregates(
    args: argparse.Namespace,
    input_dir: Path,
    output_dir: Path,
    paths: BuildPaths,
    cache: dict[str, object],
    cache_name: str,
) -> None:
    inputs = list(input_dir.glob("*_processed.csv")) + [
        path
        for path in (
            input_dir / "big_corpus_en_pivot.csv",
            input_dir / "big_corpus_zh_pivot.csv",
        )
        if path.is_file()
    ]
    key = stage_key(
        cache_name,
        {"inputs": file_inventory(inputs, paths.root)},
        [
            script("scripts/local/build_big_corpus.py"),
            script("scripts/local/corpus_quality.py"),
            script("scripts/local/pipeline_common.py"),
        ],
    )
    cached = not args.dry_run and not args.no_stage_cache and cached_stage_valid(paths.root, cache, cache_name, key)
    label = (
        "Aggregate cleaned bilingual corpora"
        if cache_name == "processed_aggregate"
        else "Aggregate finalized pivot corpora"
    )
    if cached:
        print(f"[cache] {label}")
    else:
        run(
            [
                PYTHON,
                str(script("scripts/local/build_big_corpus.py")),
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
            label=label,
            log_path=stage_log(paths, cache_name),
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
    if not args.dry_run:
        manifest_path = output_dir / "aggregate_manifest.json"
        require_json_manifest(
            manifest_path,
            stage=f"aggregate {input_dir.name}",
        )
        if not cached:
            record_cached_stage(
                paths.root,
                build_cache_path(paths),
                cache,
                cache_name,
                key,
                [
                    output_dir / "aggregate_manifest.json",
                    output_dir / "big_corpus_en.csv",
                    output_dir / "big_corpus_zh.csv",
                    output_dir / "big_corpus_combined.csv",
                ],
                "build",
            )
        print(format_aggregate_summary(manifest_path, label))


def run_pivot(
    args: argparse.Namespace,
    paths: BuildPaths,
    cache: dict[str, object],
) -> None:
    def current_key() -> str:
        cache_files: list[Path] = []
        for directory in [pivot_cache_dir(paths), *pivot_read_cache_dirs(args, paths)]:
            if directory.is_dir():
                cache_files.extend(directory.glob("*.jsonl"))
        return stage_key(
            "pivot",
            {
                "inputs": file_inventory(
                    [
                        paths.processed_dir / "big_corpus_en.csv",
                        paths.processed_dir / "big_corpus_zh.csv",
                    ],
                    paths.root,
                ),
                "cache_files": [
                    {
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                    for path in sorted(set(cache_files))
                ],
                "directions": args.pivot_directions,
                "splits": args.pivot_splits,
                "skip_translation": args.pivot_skip_translation,
                "pivot_dry_run": args.pivot_dry_run,
                "respect_usage_limit": args.respect_usage_limit,
                "pipeline_config_sha256": sha256_file(PIPELINE_CONFIG_PATH),
            },
            [
                script("scripts/local/pivot.py"),
                script("scripts/local/corpus_quality.py"),
                script("scripts/local/pipeline_common.py"),
            ],
        )

    key = current_key()
    cached = not args.dry_run and not args.no_stage_cache and cached_stage_valid(paths.root, cache, "pivot", key)
    manifest_path = paths.processed_dir / "pivot" / "pivot_manifest.json"
    if cached:
        print("[cache] Assemble DeepL pivot rows")
        print(format_pivot_summary(manifest_path))
        return
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
    run(
        cmd,
        label="Assemble DeepL pivot rows",
        log_path=stage_log(paths, "pivot"),
        dry_run=args.dry_run,
        verbose=args.verbose,
    )
    if not args.dry_run:
        pivot_manifest = require_json_manifest(
            manifest_path,
            stage="DeepL pivot",
        )
        for stats in pivot_manifest.get("stats", []):
            if stats.get("synthetic_rows_missing") or stats.get("errors") or stats.get("stopped_reason"):
                raise SystemExit(f"DeepL pivot direction {stats.get('direction')} is incomplete: {stats}")
        pivot_dir = paths.processed_dir / "pivot"
        key = current_key()
        record_cached_stage(
            paths.root,
            build_cache_path(paths),
            cache,
            "pivot",
            key,
            [
                path
                for path in pivot_dir.iterdir()
                if path.is_file()
                and (
                    path.name == "pivot_manifest.json"
                    or path.name.startswith("big_corpus_")
                    or path.name.startswith("pivot_rejections_")
                    or path.name.startswith("pivot_cache_conflicts_")
                )
            ],
            "build",
        )
        print(format_pivot_summary(manifest_path))


def build_hard_splits(
    args: argparse.Namespace,
    corpus_dir: Path,
    output_root: Path,
    paths: BuildPaths,
    cache: dict[str, object],
) -> None:
    key = stage_key(
        "hard_splits",
        {
            "inputs": file_inventory(
                [
                    corpus_dir / "big_corpus_en.csv",
                    corpus_dir / "big_corpus_zh.csv",
                ],
                paths.root,
            ),
            "train_ratio": args.train_ratio,
            "validate_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "min_formosan_tokens": args.min_formosan_tokens,
            "min_target_tokens": args.min_target_tokens,
            "min_test_rows": args.min_test_rows,
            "min_validate_rows": args.min_validate_rows,
            "ngram_jaccard_threshold": args.ngram_jaccard_threshold,
            "source_ratio_tolerance": args.source_ratio_tolerance,
            "tiers": args.tiers,
            "pipeline_config_sha256": sha256_file(PIPELINE_CONFIG_PATH),
        },
        [
            script("formosan_mt_experiments/scripts/build_experiment_splits.py"),
            script("formosan_mt_experiments/scripts/validate_experiment.py"),
            script("formosan_mt_experiments/scripts/audit_corpus_exposure.py"),
            script("formosan_mt_experiments/scripts/experiment_config.py"),
            script("formosan_mt_experiments/scripts/mt_common.py"),
            script("formosan_mt_experiments/scripts/columnar_cache.py"),
            script("formosan_mt_experiments/configs/default_experiment.json"),
        ],
    )
    cached = not args.dry_run and not args.no_stage_cache and cached_stage_valid(paths.root, cache, "hard_splits", key)
    if cached:
        print("[cache] Hard splits and audits")
        for short, target in (("en", "English"), ("zh", "Chinese")):
            print(format_split_summary(output_root / f"splits_{short}_v1", target))
        return
    if not args.dry_run:
        remove_path(output_root)
        remove_path(corpus_dir / "big_corpus_en_in_domain_hard.csv")
        remove_path(corpus_dir / "big_corpus_zh_in_domain_hard.csv")
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
            label=f"Build {target_lang} hard split",
            log_path=stage_log(paths, f"split_{short}"),
            dry_run=args.dry_run,
            verbose=args.verbose,
        )
        hard_file = out_dir / f"big_corpus_{short}_in_domain_hard.csv"
        dest = corpus_dir / hard_file.name
        if hard_file.exists() and not args.dry_run:
            replace_with_hardlink(hard_file, dest)
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
                    "--min-formosan-tokens",
                    str(args.min_formosan_tokens),
                    "--min-target-tokens",
                    str(args.min_target_tokens),
                    "--source-ratio-tolerance",
                    str(args.source_ratio_tolerance),
                    "--split-report",
                    str(out_dir / "report_in_domain_hard.json"),
                    "--report",
                    str(out_dir / "validation_in_domain_hard.json"),
                ],
                label=f"Validate {target_lang} hard split",
                log_path=stage_log(paths, f"validate_{short}"),
                dry_run=False,
                verbose=args.verbose,
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
                label=f"Audit {target_lang} train-test exposure",
                log_path=stage_log(paths, f"exposure_{short}"),
                dry_run=False,
                verbose=args.verbose,
            )
            require_json_manifest(
                out_dir / "exposure_in_domain_hard.json",
                stage=f"{target_lang} TAME-MT exposure audit",
            )
            print(format_split_summary(out_dir, target_lang.title()))
        elif not args.dry_run:
            raise SystemExit(f"Hard split builder did not produce {hard_file}")
    if not args.dry_run:
        record_cached_stage(
            paths.root,
            build_cache_path(paths),
            cache,
            "hard_splits",
            key,
            [
                *output_root.rglob("*"),
                corpus_dir / "big_corpus_en_in_domain_hard.csv",
                corpus_dir / "big_corpus_zh_in_domain_hard.csv",
            ],
            "build",
        )


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
            "language_workers": args.language_workers,
            "incremental_stage_cache": not args.no_stage_cache,
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
            "hard_split_source_ratio_tolerance": args.source_ratio_tolerance,
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
        print(f"[plan]  Write build manifest: {paths.manifest_path}")
        return
    atomic_write_json(paths.manifest_path, manifest)
    print(f"Build manifest: {paths.manifest_path}")
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
    print(f"Training provenance: {bundle_manifest}")


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
            "Keep abandoned temporary pivot outputs for debugging. Completed outputs "
            "are retained and reused only after checksum verification by default."
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
    parser.add_argument(
        "--language-workers",
        type=int,
        default=3,
        help="Language preparation pipelines to run concurrently after acquisition.",
    )
    parser.add_argument(
        "--no-stage-cache",
        action="store_true",
        help="Force language QC, standardization, extraction, and filtering to rerun.",
    )
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
        "--verbose",
        action="store_true",
        help=(
            "Stream child-process output and full commands. Normal mode writes "
            "raw stage output under <build-root>/logs/."
        ),
    )
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
    parser.add_argument(
        "--min-formosan-tokens",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_formosan_tokens"],
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_target_tokens"],
    )
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
        "--source-ratio-tolerance",
        type=float,
        default=PIPELINE_CONFIG["splits"]["source_ratio_tolerance"],
    )
    parser.add_argument(
        "--tiers",
        default=PIPELINE_CONFIG["splits"]["headline_tier"],
    )
    args = parser.parse_args()
    if args.fetch_workers < 1:
        raise SystemExit("--fetch-workers must be >= 1")
    if args.language_workers < 1:
        raise SystemExit("--language-workers must be >= 1")
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
    if not 0 <= args.source_ratio_tolerance <= 1.0:
        raise SystemExit("--source-ratio-tolerance must be in [0, 1]")
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
        build_cache = load_stage_cache(build_cache_path(paths))
        build_cache.get("stages", {}).pop("hard_splits", None)
        build_hard_splits(
            args,
            paths.final_dir,
            paths.split_root,
            paths,
            build_cache,
        )
        if args.exclude_bible and not args.dry_run:
            validate_no_bible_sources(paths, paths.final_dir)
        write_manifest(args, previous_languages, paths.final_dir, paths)
        return paths.root

    languages = parse_languages(args.languages)
    print(f"\nCorpus build: {args.corpus_name or paths.root.name}")
    print(f"Output: {paths.root}")
    print(f"Detailed logs: {paths.root / 'logs'}")
    if should_clean_generated_outputs(args, paths):
        print("Removing abandoned temporary outputs.")
        clean_generated_outputs(paths)
    if not args.dry_run:
        paths.raw_dir.mkdir(parents=True, exist_ok=True)
        paths.processed_dir.mkdir(parents=True, exist_ok=True)
        prune_unselected_language_outputs(paths, languages)
    if not args.dry_run and not args.skip_fetch and not args.keep_downloaded:
        remove_path(paths.source_snapshot_path)

    fetch_languages(languages, args, paths)
    language_reports = build_languages(languages, args, paths)
    build_cache = load_stage_cache(build_cache_path(paths))

    processed = paths.processed_dir
    final_corpus_dir = processed
    if not args.skip_aggregate:
        build_aggregates(
            args,
            processed,
            processed,
            paths,
            build_cache,
            "processed_aggregate",
        )

    if args.with_pivot:
        run_pivot(args, paths, build_cache)
        final_corpus_dir = paths.final_dir
        if not args.dry_run:
            final_corpus_dir.mkdir(parents=True, exist_ok=True)
        build_aggregates(
            args,
            processed / "pivot",
            final_corpus_dir,
            paths,
            build_cache,
            "final_aggregate",
        )

    if not args.skip_hard_splits:
        build_hard_splits(
            args,
            final_corpus_dir,
            paths.split_root,
            paths,
            build_cache,
        )

    if args.exclude_bible and not args.dry_run:
        validate_no_bible_sources(paths, final_corpus_dir)

    write_manifest(args, language_reports, final_corpus_dir, paths)
    if not args.dry_run:
        print(f"Build complete: {paths.root}")
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
    try:
        if args.build_public_private:
            run_public_private(args)
        else:
            run_build(args)
    except CommandExecutionError as exc:
        raise SystemExit(str(exc)) from None


if __name__ == "__main__":
    main()
