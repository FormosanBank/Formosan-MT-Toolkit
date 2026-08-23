"""Command-line contract for reproducible corpus builds."""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_common import load_pipeline_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = load_pipeline_config()


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
    parser.add_argument("--units", default="sentences,words")
    parser.add_argument(
        "--language-workers",
        type=int,
        default=3,
        help="Language preparation pipelines to run concurrently after acquisition.",
    )
    parser.add_argument(
        "--analysis-workers",
        type=int,
        default=2,
        help=(
            "English/Chinese split, validation, and exposure pipelines to run "
            "concurrently. Use 1 on machines with less than 24 GB RAM."
        ),
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
        "--min-combined-tokens",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_combined_tokens"],
    )
    parser.add_argument(
        "--min-punctuated-combined-tokens",
        type=int,
        default=PIPELINE_CONFIG["splits"]["min_punctuated_combined_tokens"],
    )
    parser.add_argument(
        "--max-eval-units-per-side",
        type=int,
        default=PIPELINE_CONFIG["splits"]["max_eval_units_per_side"],
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
    if args.analysis_workers < 1:
        raise SystemExit("--analysis-workers must be >= 1")
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
    if args.max_eval_units_per_side < 1:
        raise SystemExit("--max-eval-units-per-side must be >= 1")
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
