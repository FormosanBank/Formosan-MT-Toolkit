#!/usr/bin/env python3
"""Build tiered, leakage-controlled Formosan MT experiment splits.

The script ignores any existing split column and emits one full CSV per eval tier:

  big_corpus_en_lexical.csv
  big_corpus_en_in_domain_hard.csv
  big_corpus_en_hard_global.csv

Each output uses split values train/validate/test, keeps extra metadata columns,
and removes train rows that would leak exact normalized source, target, or pair
text into that tier's validation/test rows. XML lexical rows are never assigned
to validation/test tiers.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mt_common import (
    DEFAULT_INPUT,
    EASY_BUCKETS,
    add_normalized_columns,
    bucket_counts,
    normalize_target_language,
    overlap_stats,
    read_parallel_csv,
    source_bucket,
    split_counts,
    split_counts_by_language,
    target_col_for,
    target_tag_for,
    write_json,
)


TIERS = ("lexical", "in_domain_hard", "hard_global")


def keep_columns(target_col: str) -> list[str]:
    return [
        "row_id",
        "lang_code",
        "formosan_sentence",
        target_col,
        "source",
        "dialect",
        "row_type",
        "split",
        "eval_tier",
        "source_bucket",
        "formosan_tokens",
        "target_tokens",
        "short_entry",
    ]


@dataclass(frozen=True)
class SourceCandidate:
    source: str
    rows: int
    total_source_rows: int
    short_frac: float
    avg_tokens: float


def tier_mask(
    df: pd.DataFrame,
    tier: str,
    min_formosan_tokens: int,
    min_target_tokens: int,
) -> pd.Series:
    hard = (
        ~df["_is_lexeme"]
        & ~df["_source_bucket"].isin(EASY_BUCKETS)
        & df["_formosan_tokens"].ge(min_formosan_tokens)
        & df["_target_tokens"].ge(min_target_tokens)
    )
    lexical = ~df["_is_lexeme"] & (df["_source_bucket"].isin(EASY_BUCKETS) | df["_short_entry"])
    if tier == "lexical":
        return lexical
    if tier == "in_domain_hard":
        return hard
    if tier == "hard_global":
        target_group_count = df.groupby("_target_key")["_target_group_key"].transform("nunique")
        return hard & target_group_count.eq(1)
    raise ValueError(f"Unknown tier: {tier}")


def source_candidates(lang_df: pd.DataFrame, candidate_mask: pd.Series) -> list[SourceCandidate]:
    rows: list[SourceCandidate] = []
    candidate_mask = candidate_mask.reindex(lang_df.index, fill_value=False)
    for source, group in lang_df.groupby("_source_key", sort=False):
        eligible = group[candidate_mask.loc[group.index]]
        if eligible.empty:
            continue
        avg_tokens = float((eligible["_formosan_tokens"] + eligible["_target_tokens"]).mean() / 2.0)
        rows.append(
            SourceCandidate(
                source=str(source),
                rows=int(len(eligible)),
                total_source_rows=int(len(group)),
                short_frac=float(eligible["_short_entry"].mean()),
                avg_tokens=avg_tokens,
            )
        )
    return rows


def selection_cost(selected: list[SourceCandidate], target: int) -> float:
    if not selected:
        return float("inf")
    rows = sum(g.rows for g in selected)
    extra_source_rows = sum(max(0, g.total_source_rows - g.rows) for g in selected)
    short_penalty = sum(g.short_frac * g.rows for g in selected)
    length_bonus = sum(min(g.avg_tokens, 40.0) * g.rows for g in selected) * 0.01
    return abs(rows - target) + 0.05 * extra_source_rows + 0.25 * short_penalty - length_bonus


def choose_sources(
    candidates: list[SourceCandidate],
    target_rows: int,
    seed: int,
    attempts: int,
    excluded_sources: set[str] | None = None,
) -> set[str]:
    excluded_sources = excluded_sources or set()
    candidates = [c for c in candidates if c.source not in excluded_sources]
    if target_rows <= 0 or not candidates:
        return set()

    rng = random.Random(seed)
    best: list[SourceCandidate] = []
    best_cost = float("inf")

    # Deterministic first pass prefers more sentence-like, longer groups.
    ordered = sorted(candidates, key=lambda c: (c.short_frac, -c.avg_tokens, -c.rows))
    current: list[SourceCandidate] = []
    current_rows = 0
    for candidate in ordered:
        current.append(candidate)
        current_rows += candidate.rows
        if current_rows >= target_rows:
            break
    best = current
    best_cost = selection_cost(current, target_rows)

    for _ in range(attempts):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        # Bias each attempt toward higher-yield, longer sources without making the
        # split fully deterministic by source size.
        shuffled.sort(key=lambda c: rng.random() + 0.015 * c.short_frac - 0.0005 * c.avg_tokens)
        selected: list[SourceCandidate] = []
        selected_rows = 0
        for candidate in shuffled:
            selected.append(candidate)
            selected_rows += candidate.rows
            if selected_rows >= target_rows:
                break
        cost = selection_cost(selected, target_rows)
        if cost < best_cost:
            best = selected
            best_cost = cost

    return {c.source for c in best}


def build_tier(
    df: pd.DataFrame,
    tier: str,
    target_col: str,
    test_ratio: float,
    val_ratio: float,
    seed: int,
    min_formosan_tokens: int,
    min_target_tokens: int,
    attempts: int,
) -> tuple[pd.DataFrame, dict]:
    candidate_mask = tier_mask(df, tier, min_formosan_tokens, min_target_tokens)
    split = pd.Series("", index=df.index, dtype="object")
    heldout_source = pd.Series(False, index=df.index)
    language_reports: dict[str, dict] = {}

    for offset, (lang, lang_df) in enumerate(df.groupby("lang_code", sort=True)):
        lang_candidate_mask = candidate_mask.reindex(lang_df.index, fill_value=False)
        candidates = source_candidates(lang_df, lang_candidate_mask)
        lang_total = len(lang_df)
        eligible_total = int(lang_candidate_mask.sum())
        desired_eval = max(1, round(lang_total * (test_ratio + val_ratio))) if eligible_total else 0
        target_eval = min(eligible_total, desired_eval)
        if target_eval >= 2:
            test_share = test_ratio / (test_ratio + val_ratio)
            target_test = min(target_eval - 1, max(1, round(target_eval * test_share)))
            target_val = target_eval - target_test
        else:
            target_test = target_eval
            target_val = 0

        test_sources = choose_sources(
            candidates,
            target_test,
            seed=seed + 997 * offset,
            attempts=attempts,
        )
        val_sources = choose_sources(
            candidates,
            target_val,
            seed=seed + 997 * offset + 17,
            attempts=attempts,
            excluded_sources=test_sources,
        )

        is_lang_test = lang_df["_source_key"].isin(test_sources)
        is_lang_val = lang_df["_source_key"].isin(val_sources)
        is_lang_heldout = is_lang_test | is_lang_val
        heldout_source.loc[lang_df.index] = is_lang_heldout

        eligible_test = lang_candidate_mask & is_lang_test
        eligible_val = lang_candidate_mask & is_lang_val
        split.loc[lang_df.index[eligible_test]] = "test"
        split.loc[lang_df.index[eligible_val]] = "validate"

        language_reports[str(lang)] = {
            "rows_total": int(lang_total),
            "eligible_rows": int(eligible_total),
            "candidate_sources": int(len(candidates)),
            "test_sources": int(len(test_sources)),
            "validate_sources": int(len(val_sources)),
            "preclean_test_rows": int(eligible_test.sum()),
            "preclean_validate_rows": int(eligible_val.sum()),
        }

    eval_mask = split.isin(["validate", "test"])
    leakage_columns = (
        "_formosan_key",
        "_target_key",
        "_pair_key",
        "_formosan_skeleton",
        "_target_skeleton",
        "_pair_skeleton",
    )
    eval_keys = {col: set(df.loc[eval_mask, col].dropna()) for col in leakage_columns}

    train_mask = ~heldout_source | df["_is_lexeme"]
    for col, values in eval_keys.items():
        train_mask &= ~df[col].isin(values)
    split.loc[train_mask] = "train"

    # Remove rows that still share source/target/pair with another split.
    for col in leakage_columns:
        active = split.isin(["train", "validate", "test"])
        current_eval_mask = split.isin(["validate", "test"])
        current_train_mask = split.eq("train")
        active_keys = pd.DataFrame({col: df.loc[active, col], "_split": split.loc[active]})
        split_count = active_keys.groupby(col)["_split"].nunique()
        leaking_keys = set(split_count[split_count > 1].index)
        if leaking_keys:
            split.loc[current_eval_mask & df[col].isin(leaking_keys)] = ""
            split.loc[current_train_mask & df[col].isin(leaking_keys)] = ""

    out = df.loc[split.isin(["train", "validate", "test"])].copy()
    out["split"] = split.loc[out.index].values
    out["eval_tier"] = tier
    out["source_bucket"] = out["_source_bucket"]
    out["formosan_tokens"] = out["_formosan_tokens"].astype(int)
    out["target_tokens"] = out["_target_tokens"].astype(int)
    out["short_entry"] = out["_short_entry"].astype(bool)
    out["row_type"] = out["row_type"].fillna("unknown").astype(str)

    train = out[out["split"].eq("train")]
    eval_df = out[out["split"].isin(["validate", "test"])]
    overlaps = overlap_stats(train, eval_df)
    skeleton_overlaps = {}
    for name, col in (
        ("formosan", "_formosan_skeleton"),
        ("target", "_target_skeleton"),
        ("pair", "_pair_skeleton"),
    ):
        train_values = set(train[col].dropna())
        eval_values = set(eval_df[col].dropna())
        overlap = train_values & eval_values
        skeleton_overlaps[name] = {
            "train_unique": len(train_values),
            "eval_unique": len(eval_values),
            "overlap_unique": len(overlap),
        }
    hard_global_target_unique = None
    if tier == "hard_global":
        hard_global_target_unique = bool(
            df.loc[eval_df.index].groupby("_target_key")["_target_group_key"].transform("nunique").eq(1).all()
        )

    report = {
        "tier": tier,
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "dropped_rows": int(len(df) - len(out)),
        "candidate_rows": int(candidate_mask.sum()),
        "lexeme_eval_rows": int((out["row_type"].isin(["lexeme", "morpheme"]) & out["split"].isin(["validate", "test"])).sum()),
        "lexeme_rows_forced_train": int((df["_is_lexeme"] & split.eq("train")).sum()),
        "lexeme_rows_removed_for_leakage_or_heldout": int(df["_is_lexeme"].sum() - (df["_is_lexeme"] & split.eq("train")).sum()),
        "split_counts": split_counts(out),
        "split_counts_by_language": split_counts_by_language(out),
        "bucket_counts": bucket_counts(out),
        "overlap_stats_train_vs_eval": overlaps,
        "skeleton_overlap_stats_train_vs_eval": skeleton_overlaps,
        "hard_global_target_unique_eval": hard_global_target_unique,
        "languages": language_reports,
    }
    return out[keep_columns(target_col)], report


def validate_report(report: dict) -> None:
    overlaps = report["overlap_stats_train_vs_eval"]
    failures = {
        key: value["overlap_unique"]
        for key, value in overlaps.items()
        if key in {"formosan", "target", "pair"} and value["overlap_unique"] != 0
    }
    skeleton_failures = {
        key: value["overlap_unique"]
        for key, value in report.get("skeleton_overlap_stats_train_vs_eval", {}).items()
        if key in {"formosan", "target", "pair"} and value["overlap_unique"] != 0
    }
    if failures:
        raise SystemExit(f"Leakage validation failed for tier={report['tier']}: {failures}")
    if skeleton_failures:
        raise SystemExit(
            f"Near-duplicate skeleton validation failed for tier={report['tier']}: {skeleton_failures}"
        )
    if report.get("lexeme_eval_rows", 0):
        raise SystemExit(
            f"Lexeme routing validation failed for tier={report['tier']}: "
            f"{report['lexeme_eval_rows']} lexeme rows in eval"
        )
    if report["tier"] == "hard_global" and report["hard_global_target_unique_eval"] is not True:
        raise SystemExit("hard_global validation failed: eval target references are not globally unique")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--output-prefix", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--test-ratio", type=float, default=0.075)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-formosan-tokens", type=int, default=4)
    parser.add_argument("--min-target-tokens", type=int, default=4)
    parser.add_argument("--selection-attempts", type=int, default=600)
    parser.add_argument(
        "--tiers",
        default=",".join(TIERS),
        help=f"Comma-separated tiers to build. Available: {','.join(TIERS)}",
    )
    args = parser.parse_args()

    target_lang = normalize_target_language(args.target_lang, args.target_col)
    target_col = args.target_col or target_col_for(target_lang)
    target_tag = target_tag_for(target_lang)
    file_short = "en" if target_tag == "eng" else target_tag
    output_prefix = args.output_prefix or f"big_corpus_{file_short}"
    if args.output_dir is None:
        args.output_dir = Path(f"formosan_mt_experiments/data/splits_{file_short}_v1")

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise SystemExit("--train-ratio + --val-ratio + --test-ratio must equal 1.0")

    requested_tiers = tuple(t.strip() for t in args.tiers.split(",") if t.strip())
    unknown = sorted(set(requested_tiers) - set(TIERS))
    if unknown:
        raise SystemExit(f"Unknown tiers: {unknown}. Available: {TIERS}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw = read_parallel_csv(args.input, target_col=target_col)
    raw = raw.copy()
    raw["row_id"] = range(len(raw))
    df = add_normalized_columns(raw, target_col=target_col, target_lang=target_lang)

    all_reports = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "target_lang": target_lang,
        "target_col": target_col,
        "output_prefix": output_prefix,
        "ratios": {
            "train": args.train_ratio,
            "validate": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "tiers": {},
    }

    for tier in requested_tiers:
        out, report = build_tier(
            df,
            tier=tier,
            target_col=target_col,
            test_ratio=args.test_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_formosan_tokens=args.min_formosan_tokens,
            min_target_tokens=args.min_target_tokens,
            attempts=args.selection_attempts,
        )
        validate_report(report)

        full_path = args.output_dir / f"{output_prefix}_{tier}.csv"
        test_path = args.output_dir / f"{output_prefix}_{tier}_test.csv"
        val_path = args.output_dir / f"{output_prefix}_{tier}_validate.csv"
        out.to_csv(full_path, index=False)
        out[out["split"].eq("test")].to_csv(test_path, index=False)
        out[out["split"].eq("validate")].to_csv(val_path, index=False)

        report["files"] = {
            "full": str(full_path),
            "test": str(test_path),
            "validate": str(val_path),
        }
        all_reports["tiers"][tier] = report
        write_json(args.output_dir / f"report_{tier}.json", report)
        print(
            f"[{tier}] wrote {full_path} | "
            f"splits={json.dumps(report['split_counts'], sort_keys=True)}"
        )

    write_json(args.output_dir / "report_all_tiers.json", all_reports)
    print(f"Report: {args.output_dir / 'report_all_tiers.json'}")


if __name__ == "__main__":
    main()
