#!/usr/bin/env python3
"""Independently validate a release MT corpus and its control tags."""

from __future__ import annotations

import json
from pathlib import Path

import nllb_runtime as nllb
import pandas as pd
from experiment_config import (
    profile_record,
    sha256_file,
)
from mt_common import (
    normalize_target_language,
    read_parallel_csv,
    target_col_for,
    write_json,
)
from split_cli import parse_validation_args
from validation_policy import REQUIRED_PROVENANCE, validate_provenance, validate_splits


def validate_tags(
    frame: pd.DataFrame,
    tokenizer_dir: Path,
    direction: str,
    target_lang: str,
) -> dict[str, object]:
    tokenizer = nllb.load_tokenizer(tokenizer_dir)
    tags: set[str] = set()
    for _, row in frame.iterrows():
        tags.update(
            nllb.source_prefix(
                row,
                direction,
                target_lang=target_lang,
                use_tags=True,
            ).split()
        )
    bad = []
    for token in sorted(tags):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if (
            token_id == tokenizer.unk_token_id
            or tokenizer.convert_ids_to_tokens(token_id) != token
        ):
            bad.append(token)
    return {
        "ok": not bad,
        "model_family": nllb.MODEL_FAMILY,
        "direction": direction,
        "checked_tags": len(tags),
        "bad_tags": bad,
    }


def main() -> None:
    args = parse_validation_args()
    target_lang = normalize_target_language(
        args.target_lang,
        args.target_col,
    )
    target_col = args.target_col or target_col_for(target_lang)
    frame = read_parallel_csv(
        args.input,
        target_col=target_col,
        columns=sorted(
            REQUIRED_PROVENANCE
            | {"split", "pivot_origin", "translation_kind"}
        ),
    )
    if "split" not in frame:
        raise SystemExit("Input must have a split column")

    provenance = validate_provenance(frame)
    split_report = None
    if args.split_report:
        try:
            split_report = json.loads(args.split_report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read split report {args.split_report}: {exc}") from exc
    split_validation = validate_splits(
        frame,
        target_col=target_col,
        min_test_ratio=args.min_test_ratio,
        min_validate_ratio=args.min_validate_ratio,
        min_test_rows=args.min_test_rows,
        min_validate_rows=args.min_validate_rows,
        ngram_threshold=args.ngram_jaccard_threshold,
        min_formosan_tokens=args.min_formosan_tokens,
        min_target_tokens=args.min_target_tokens,
        min_combined_tokens=args.min_combined_tokens,
        min_punctuated_combined_tokens=args.min_punctuated_combined_tokens,
        max_eval_units_per_side=args.max_eval_units_per_side,
        source_ratio_tolerance=args.source_ratio_tolerance,
        require_human_eval=args.require_human_eval,
        require_document_holdout=args.require_document_holdout_report,
        split_report=split_report,
    )
    report: dict[str, object] = {
        "schema_version": 3,
        "complete": bool(provenance["ok"] and split_validation["ok"]),
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "target_language": target_lang,
        "target_column": target_col,
        "profile": profile_record(args.profile),
        "split_report": (
            {
                "path": str(args.split_report),
                "sha256": sha256_file(args.split_report),
            }
            if args.split_report
            else None
        ),
        "rows": len(frame),
        "provenance_validation": provenance,
        "split_validation": split_validation,
    }
    if args.tokenizer or args.direction:
        if not args.tokenizer or not args.direction:
            raise SystemExit("--tokenizer and --direction must be provided together")
        report["tag_validation"] = validate_tags(
            frame,
            args.tokenizer,
            args.direction,
            target_lang,
        )
        report["complete"] = bool(
            report["complete"]
            and report["tag_validation"]["ok"]
        )
    if args.report:
        write_json(args.report, report)
    leakage = split_validation["train_evaluation"]
    print(
        f"Corpus validation: {'PASS' if report['complete'] else 'FAIL'} "
        f"({len(frame):,} rows, {target_lang})"
    )
    print(
        "  eval: "
        f"synthetic={split_validation['synthetic_eval_rows']:,}, "
        f"lexical-like={split_validation['lexical_like_eval_rows']:,}, "
        f"gloss={split_validation['gloss_translation_rows'] + split_validation['annotation_gloss_rows']:,}, "
        f"malformed={split_validation['malformed_escaping_rows']:,}, "
        f"language-mismatch={split_validation['target_language_mismatch_rows']:,}, "
        f"language-uncertain-eval={split_validation['uncertain_target_language_eval_rows']:,}, "
        f"unbalanced-eval={split_validation['unbalanced_target_eval_rows']:,}, "
        f"lexical-quality={split_validation['lexical_quality_rows']:,}"
    )
    print(
        "  train/eval conflicts: "
        f"exact={sum(leakage['exact_overlap'].values()):,}, "
        f"skeleton={sum(leakage['skeleton_overlap'].values()):,}, "
        f"one-edit={sum(leakage['one_edit_conflicting_rows'].values()):,}, "
        f"char-ngram={sum(leakage['character_ngram_conflicting_rows'].values()):,}"
    )
    print(
        "  ratio failures: "
        f"languages={len(split_validation['ratio_failures']):,}, "
        "sources="
        f"{sum(len(rows) for rows in split_validation['source_ratio_failures'].values()):,}"
    )
    if args.report:
        print(f"  report: {args.report}")
    if not report["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
