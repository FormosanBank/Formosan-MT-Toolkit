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
import json
import sys
from pathlib import Path

from build_analysis import build_hard_splits
from build_cli import parse_args
from build_context import (
    BuildPaths,
    Language,
    build_cache_path,
    clean_generated_outputs,
    language_cache_path,
    parse_languages,
    pivot_cache_dir,
    pivot_read_cache_dirs,
    prune_unselected_language_outputs,
    remove_path,
    require_json_manifest,
    resolve_build_paths,
    script,
    should_clean_generated_outputs,
    stage_log,
    variant_name,
)
from build_output import (
    CommandExecutionError,
    format_aggregate_summary,
    format_fetch_summary,
    format_language_summary,
    format_pivot_summary,
    format_rule_summary,
    run_logged,
)
from build_release import (
    count_csv_rows,
    stage_manifest_record,
    validate_no_bible_sources,
    write_manifest,
)
from pipeline_common import (
    PIPELINE_CONFIG_PATH,
    git_state,
    load_pipeline_config,
    sha256_file,
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
                    script("scripts/shared/columnar_io.py"),
                    script("scripts/shared/reproducibility.py"),
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
                    [
                        output_path,
                        output_path.with_suffix(".parquet"),
                        output_path.with_suffix(".parquet.json"),
                        *report_path.parent.glob("*"),
                    ],
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
            script("scripts/shared/columnar_io.py"),
            script("scripts/shared/reproducibility.py"),
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
                    output_dir / "big_corpus_en.parquet",
                    output_dir / "big_corpus_en.parquet.json",
                    output_dir / "big_corpus_zh.csv",
                    output_dir / "big_corpus_zh.parquet",
                    output_dir / "big_corpus_zh.parquet.json",
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
