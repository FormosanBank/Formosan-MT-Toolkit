"""Command-line contracts for corpus splitting and independent validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_config import (
    DEFAULT_PROFILE,
    load_corpus_pipeline_config,
    load_profile,
    target_split_ratios,
)
from mt_common import direction_choices, normalize_target_language

SPLIT_DEFAULTS = load_corpus_pipeline_config()["splits"]
TIER = SPLIT_DEFAULTS["headline_tier"]


def parse_split_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-stratified hard MT split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default=None)
    parser.add_argument("--target-col")
    parser.add_argument("--output-prefix")
    parser.add_argument("--train-ratio", type=float)
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--test-ratio", type=float)
    parser.add_argument("--seed", type=int, default=SPLIT_DEFAULTS["seed"])
    parser.add_argument(
        "--min-formosan-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_formosan_tokens"],
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_target_tokens"],
    )
    parser.add_argument(
        "--min-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_combined_tokens"],
    )
    parser.add_argument(
        "--min-punctuated-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_punctuated_combined_tokens"],
    )
    parser.add_argument(
        "--max-eval-units-per-side",
        type=int,
        default=SPLIT_DEFAULTS["max_eval_units_per_side"],
    )
    parser.add_argument("--min-test-rows", type=int, default=SPLIT_DEFAULTS["min_test_rows"])
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=SPLIT_DEFAULTS["min_validate_rows"],
    )
    parser.add_argument("--selection-attempts", type=int, default=200)
    parser.add_argument(
        "--ngram-jaccard-threshold",
        type=float,
        default=SPLIT_DEFAULTS["character_ngram_jaccard_threshold"],
    )
    parser.add_argument("--registry-in", type=Path)
    parser.add_argument(
        "--tiers",
        default=TIER,
        help="Compatibility option; only in_domain_hard is supported.",
    )
    args = parser.parse_args()
    ratios = target_split_ratios(
        SPLIT_DEFAULTS,
        normalize_target_language(args.target_lang, args.target_col),
    )
    supplied = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(value is not None for value in supplied) and not all(
        value is not None for value in supplied
    ):
        raise SystemExit("Provide all three split ratios or none of them")
    if all(value is None for value in supplied):
        args.train_ratio = ratios["train"]
        args.val_ratio = ratios["validate"]
        args.test_ratio = ratios["test"]
    return args


def parse_validation_args() -> argparse.Namespace:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    known, _ = preliminary.parse_known_args()
    load_profile(known.profile)

    parser = argparse.ArgumentParser(
        parents=[preliminary],
        description="Independently validate a release Formosan MT corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default=None)
    parser.add_argument("--target-col")
    parser.add_argument(
        "--split-report",
        type=Path,
        help="Original hard-split report used to validate human-corpus targets.",
    )
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--direction", choices=direction_choices())
    parser.add_argument("--min-test-ratio", type=float)
    parser.add_argument(
        "--min-validate-ratio",
        type=float,
    )
    parser.add_argument("--min-test-rows", type=int, default=SPLIT_DEFAULTS["min_test_rows"])
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=SPLIT_DEFAULTS["min_validate_rows"],
    )
    parser.add_argument(
        "--ngram-jaccard-threshold",
        type=float,
        default=SPLIT_DEFAULTS["character_ngram_jaccard_threshold"],
    )
    parser.add_argument(
        "--min-formosan-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_formosan_tokens"],
    )
    parser.add_argument(
        "--min-target-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_target_tokens"],
    )
    parser.add_argument(
        "--min-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_combined_tokens"],
    )
    parser.add_argument(
        "--min-punctuated-combined-tokens",
        type=int,
        default=SPLIT_DEFAULTS["min_punctuated_combined_tokens"],
    )
    parser.add_argument(
        "--max-eval-units-per-side",
        type=int,
        default=SPLIT_DEFAULTS["max_eval_units_per_side"],
    )
    parser.add_argument(
        "--source-ratio-tolerance",
        type=float,
        default=SPLIT_DEFAULTS["source_ratio_tolerance"],
        help="Allowed per-source split deviation in addition to one row.",
    )
    parser.add_argument("--report", "--output-json", dest="report", type=Path)
    parser.add_argument(
        "--require-human-eval",
        action="store_true",
        default=not SPLIT_DEFAULTS["synthetic_eval"],
        help="Reject synthetic pivot references in evaluation.",
    )
    parser.add_argument(
        "--require-document-holdout-report",
        action="store_true",
        help="Require source XML documents to be disjoint across splits.",
    )
    args = parser.parse_args()
    args.profile = known.profile
    ratios = target_split_ratios(
        SPLIT_DEFAULTS,
        normalize_target_language(args.target_lang, args.target_col),
    )
    if args.min_test_ratio is None:
        args.min_test_ratio = ratios["test"]
    if args.min_validate_ratio is None:
        args.min_validate_ratio = ratios["validate"]
    return args
