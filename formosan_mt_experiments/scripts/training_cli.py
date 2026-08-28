"""Command-line contract for directional MT training."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_config import DEFAULT_PROFILE, load_profile
from mt_common import (
    direction_choices,
    normalize_target_language,
    target_col_for,
    target_language_from_direction,
)

LOAD_DTYPES = ("bf16", "fp16", "fp32")


def parse_training_args() -> tuple[argparse.Namespace, dict]:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    known, _ = preliminary.parse_known_args()
    profile = load_profile(known.profile)
    defaults = profile["training_defaults"]

    parser = argparse.ArgumentParser(parents=[preliminary])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--setup-manifest", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--direction", choices=direction_choices(), required=True)
    parser.add_argument(
        "--use-tags",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("use_tags", True)),
    )
    parser.add_argument("--validate-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=defaults["steps"], help="Optimizer update steps.")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--grad-accum-steps", type=int, default=defaults["grad_accum_steps"])
    parser.add_argument(
        "--language-sampling-chunk-size",
        type=int,
        default=defaults["language_sampling_chunk_size"],
        help="Rows sampled from one language before chunks are combined into a physical batch.",
    )
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--language-sampling-alpha",
        type=float,
        default=defaults["language_sampling_alpha"],
        help="Language sampling exponent p(lang) proportional to row_count^alpha.",
    )
    parser.add_argument(
        "--lexical-row-sampling-weight",
        type=float,
        default=defaults["lexical_row_sampling_weight"],
        help="Per-row weight for lexemes and morphemes relative to sentence weight 1.0.",
    )
    parser.add_argument(
        "--dialect-tag-dropout",
        type=float,
        default=defaults["dialect_tag_dropout"],
    )
    parser.add_argument("--label-smoothing", type=float, default=defaults["label_smoothing"])
    parser.add_argument(
        "--precision",
        choices=["bf16", "fp16", "fp32"],
        default=defaults["precision"],
    )
    parser.add_argument(
        "--load-dtype",
        choices=LOAD_DTYPES,
        default=defaults.get("load_dtype", "fp32"),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("gradient_checkpointing", False)),
    )
    parser.add_argument(
        "--fused-optimizer",
        action=argparse.BooleanOptionalAction,
        default=bool(defaults.get("fused_optimizer", False)),
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument(
        "--save-interval",
        type=int,
        default=defaults["save_interval"],
        help="Checkpoint interval. Use 0 to keep only best/final.",
    )
    parser.add_argument("--eval-interval", type=int, default=defaults["generation_eval_interval"])
    parser.add_argument("--log-interval", type=int, default=defaults["log_interval"])
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=defaults["generation_eval_samples_per_language"],
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=defaults["generation_eval_batch_size"],
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=defaults["generation_eval_batch_size"],
    )
    parser.add_argument("--validation-beam", type=int, default=defaults["validation_beam"])
    parser.add_argument(
        "--validation-metadata-mode",
        choices=["default", "oracle"],
        default=defaults["validation_metadata_mode"],
    )
    parser.add_argument("--validation-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--best-metric",
        choices=[
            "chrF2",
            "BLEU",
            "TER",
            "macro_chrF2",
            "macro_BLEU",
            "macro_TER",
            "mean_token_loss",
        ],
        default=defaults["best_metric"],
        help="Validation metric used for best checkpoint selection and early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=defaults["early_stopping_patience"],
        help="Evaluations without improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=defaults["early_stopping_min_delta"],
    )
    parser.add_argument(
        "--early-stopping-start-step",
        type=int,
        default=defaults["early_stopping_start_step"],
    )
    parser.add_argument("--resume-from", default="auto", help="Checkpoint directory, 'auto', or 'none'.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.profile = known.profile
    args.target_lang = normalize_target_language(args.target_lang, args.target_col)
    args.target_col = args.target_col or target_col_for(args.target_lang)

    direction_target = target_language_from_direction(args.direction, args.target_lang)
    if direction_target != args.target_lang:
        raise SystemExit(
            f"--direction {args.direction!r} targets {direction_target}, but --target-lang is {args.target_lang!r}."
        )
    if args.eval_interval <= 0:
        raise SystemExit("--eval-interval must be positive because best-model selection requires validation.")
    if args.eval_samples <= 0:
        raise SystemExit("--eval-samples must be positive; use a bounded, fixed validation sample per language.")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience cannot be negative.")
    if not 0.0 <= args.language_sampling_alpha <= 1.0:
        raise SystemExit("--language-sampling-alpha must be between 0 and 1.")
    if not 0.0 < args.lexical_row_sampling_weight <= 1.0:
        raise SystemExit("--lexical-row-sampling-weight must be greater than 0 and at most 1.")
    if not 0.0 <= args.dialect_tag_dropout <= 1.0:
        raise SystemExit("--dialect-tag-dropout must be between 0 and 1.")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0:
        raise SystemExit("--batch-size and --grad-accum-steps must be positive.")
    if args.language_sampling_chunk_size <= 0:
        raise SystemExit("--language-sampling-chunk-size must be positive.")
    if args.batch_size % args.language_sampling_chunk_size:
        raise SystemExit("--batch-size must be divisible by --language-sampling-chunk-size.")
    return args, profile
