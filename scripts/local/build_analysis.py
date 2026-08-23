"""Build, validate, and audit the English and Chinese hard corpora."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from build_context import (
    PROJECT_ROOT,
    BuildPaths,
    build_cache_path,
    remove_path,
    replace_with_hardlink,
    require_json_manifest,
    script,
    stage_log,
)
from build_output import format_split_summary, run_logged
from pipeline_common import PIPELINE_CONFIG_PATH, load_pipeline_config, sha256_file
from stage_cache import (
    cached_stage_valid,
    file_inventory,
    record_cached_stage,
    stage_key,
)

PYTHON = sys.executable
PIPELINE_CONFIG = load_pipeline_config()


@dataclass(frozen=True)
class AnalysisJob:
    target_lang: str
    target_col: str
    short: str
    input_csv: Path


def run_stage(
    command: list[str],
    *,
    label: str,
    log_path: Path,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
) -> None:
    run_logged(
        command,
        project_root=PROJECT_ROOT,
        label=label,
        log_path=log_path,
        dry_run=dry_run,
        verbose=verbose,
        quiet=quiet,
    )


def run_analysis_job(
    job: AnalysisJob,
    args: argparse.Namespace,
    corpus_dir: Path,
    output_root: Path,
    paths: BuildPaths,
    *,
    quiet: bool = False,
) -> str:
    if not job.input_csv.exists() and not args.dry_run:
        raise SystemExit(
            f"Cannot build {job.target_lang} hard splits; missing {job.input_csv}"
        )
    out_dir = output_root / f"splits_{job.short}_v1"
    run_stage(
        [
            PYTHON,
            str(script("formosan_mt_experiments/scripts/build_experiment_splits.py")),
            "--input",
            str(job.input_csv),
            "--target-lang",
            job.target_lang,
            "--target-col",
            job.target_col,
            "--output-prefix",
            f"big_corpus_{job.short}",
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
            "--min-combined-tokens",
            str(args.min_combined_tokens),
            "--min-punctuated-combined-tokens",
            str(args.min_punctuated_combined_tokens),
            "--max-eval-units-per-side",
            str(args.max_eval_units_per_side),
            "--min-test-rows",
            str(args.min_test_rows),
            "--min-validate-rows",
            str(args.min_validate_rows),
            "--ngram-jaccard-threshold",
            str(args.ngram_jaccard_threshold),
            "--tiers",
            args.tiers,
        ],
        label=f"Build {job.target_lang} hard split",
        log_path=stage_log(paths, f"split_{job.short}"),
        dry_run=args.dry_run,
        verbose=args.verbose,
        quiet=quiet,
    )
    hard_file = out_dir / f"big_corpus_{job.short}_in_domain_hard.csv"
    if args.dry_run:
        return ""
    if not hard_file.exists():
        raise SystemExit(f"Hard split builder did not produce {hard_file}")

    replace_with_hardlink(hard_file, corpus_dir / hard_file.name)
    run_stage(
        [
            PYTHON,
            str(script("formosan_mt_experiments/scripts/validate_experiment.py")),
            "--input",
            str(hard_file),
            "--target-col",
            job.target_col,
            "--target-lang",
            job.target_lang,
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
            "--min-combined-tokens",
            str(args.min_combined_tokens),
            "--min-punctuated-combined-tokens",
            str(args.min_punctuated_combined_tokens),
            "--max-eval-units-per-side",
            str(args.max_eval_units_per_side),
            "--source-ratio-tolerance",
            str(args.source_ratio_tolerance),
            "--split-report",
            str(out_dir / "report_in_domain_hard.json"),
            "--report",
            str(out_dir / "validation_in_domain_hard.json"),
        ],
        label=f"Validate {job.target_lang} hard split",
        log_path=stage_log(paths, f"validate_{job.short}"),
        verbose=args.verbose,
        quiet=quiet,
    )
    require_json_manifest(
        out_dir / "validation_in_domain_hard.json",
        stage=f"{job.target_lang} hard-split validation",
    )

    exposure_config = PIPELINE_CONFIG["exposure_audit"]
    run_stage(
        [
            PYTHON,
            str(script("formosan_mt_experiments/scripts/audit_corpus_exposure.py")),
            "--input",
            str(hard_file),
            "--target-col",
            job.target_col,
            "--target-lang",
            job.target_lang,
            "--high-threshold",
            str(exposure_config["high_threshold"]),
            "--max-high-exposure-rate",
            str(exposure_config["max_high_exposure_rate"]),
            "--report",
            str(out_dir / "exposure_in_domain_hard.json"),
        ],
        label=f"Audit {job.target_lang} train-test exposure",
        log_path=stage_log(paths, f"exposure_{job.short}"),
        verbose=args.verbose,
        quiet=quiet,
    )
    require_json_manifest(
        out_dir / "exposure_in_domain_hard.json",
        stage=f"{job.target_lang} TAME-MT exposure audit",
    )
    return format_split_summary(out_dir, job.target_lang.title())


def analysis_stage_key(
    args: argparse.Namespace,
    corpus_dir: Path,
    paths: BuildPaths,
) -> str:
    return stage_key(
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
            "min_combined_tokens": args.min_combined_tokens,
            "min_punctuated_combined_tokens": args.min_punctuated_combined_tokens,
            "max_eval_units_per_side": args.max_eval_units_per_side,
            "min_test_rows": args.min_test_rows,
            "min_validate_rows": args.min_validate_rows,
            "ngram_jaccard_threshold": args.ngram_jaccard_threshold,
            "source_ratio_tolerance": args.source_ratio_tolerance,
            "tiers": args.tiers,
            "pipeline_config_sha256": sha256_file(PIPELINE_CONFIG_PATH),
        },
        [
            script("scripts/local/build_analysis.py"),
            script("formosan_mt_experiments/scripts/build_experiment_splits.py"),
            script("formosan_mt_experiments/scripts/validate_experiment.py"),
            script("formosan_mt_experiments/scripts/audit_corpus_exposure.py"),
            script("formosan_mt_experiments/scripts/experiment_config.py"),
            script("formosan_mt_experiments/scripts/mt_common.py"),
            script("formosan_mt_experiments/scripts/split_allocation.py"),
            script("formosan_mt_experiments/scripts/split_similarity.py"),
            script("formosan_mt_experiments/scripts/validation_similarity.py"),
            script("scripts/shared/columnar_io.py"),
            script("scripts/shared/reproducibility.py"),
            script("formosan_mt_experiments/configs/default_experiment.json"),
        ],
    )


def build_hard_splits(
    args: argparse.Namespace,
    corpus_dir: Path,
    output_root: Path,
    paths: BuildPaths,
    cache: dict[str, object],
) -> None:
    key = analysis_stage_key(args, corpus_dir, paths)
    cached = (
        not args.dry_run
        and not args.no_stage_cache
        and cached_stage_valid(paths.root, cache, "hard_splits", key)
    )
    if cached:
        print("[cache] Hard splits and audits")
        for short, target in (("en", "English"), ("zh", "Chinese")):
            print(format_split_summary(output_root / f"splits_{short}_v1", target))
        return

    if not args.dry_run:
        remove_path(output_root)
        remove_path(corpus_dir / "big_corpus_en_in_domain_hard.csv")
        remove_path(corpus_dir / "big_corpus_zh_in_domain_hard.csv")
    jobs = [
        AnalysisJob(
            "english",
            "english_sentence",
            "en",
            corpus_dir / "big_corpus_en.csv",
        ),
        AnalysisJob(
            "chinese",
            "chinese_sentence",
            "zh",
            corpus_dir / "big_corpus_zh.csv",
        ),
    ]
    workers = 1 if args.verbose or args.dry_run else min(args.analysis_workers, len(jobs))
    if workers == 1:
        for job in jobs:
            summary = run_analysis_job(job, args, corpus_dir, output_root, paths)
            if summary:
                print(summary)
    else:
        print(f"[stage] Build and audit {len(jobs)} target corpora ({workers} workers)")
        started = time.monotonic()
        summaries: dict[str, str] = {}
        with futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pending = {
                executor.submit(
                    run_analysis_job,
                    job,
                    args,
                    corpus_dir,
                    output_root,
                    paths,
                    quiet=True,
                ): job
                for job in jobs
            }
            for completed in futures.as_completed(pending):
                job = pending[completed]
                summaries[job.short] = completed.result()
                print(
                    f"[done]  {job.target_lang.title()} split, validation, "
                    "and exposure"
                )
        elapsed = time.monotonic() - started
        print(f"[done]  Target corpus analysis ({elapsed / 60:.1f}m)")
        for job in jobs:
            print(summaries[job.short])

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
