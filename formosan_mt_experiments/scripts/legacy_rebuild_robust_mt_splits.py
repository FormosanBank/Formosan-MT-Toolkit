#!/usr/bin/env python3
"""
Rebuild source-heldout MT splits for a multilingual Formosan corpus.

This splitter is intentionally stricter than the older row/equivalence splitter:

- Source files are atomic. If a source XML/file is held out, none of its rows are
  used for training.
- Validation/test rows are kept only when their normalized Formosan sentence,
  target sentence, and exact pair are unique to that source file within the same
  language. Held-out rows that would leak exact text are dropped from the output
  CSV by default and counted in the JSON report.
- Optional hard-eval mode chooses validation/test source files by hard, sentence-like
  rows and drops easy held-out rows from headline eval.
- Split targets are applied per language.

The result is a harder test set: source-document heldout, no exact text overlap
with train/validate/test across source files, and diagnostics written to JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


PAIR_SEP = "\u241f"
DEFAULT_OVERLAP_KEYS = ("formosan", "target", "pair")
DEFAULT_HARD_EVAL_EXCLUDED_BUCKETS = (
    "dictionary",
    "learning_vocab",
    "classroom_context",
)
SPLIT_TRAIN = "train"
SPLIT_VAL = "validate"
SPLIT_TEST = "test"
SPLIT_EXCLUDED = "excluded_overlap"


@dataclass(frozen=True)
class SourceGroup:
    source: str
    bucket: str
    raw_rows: int
    clean_rows: int
    short_clean_rows: int
    hard_clean_rows: int
    short_hard_clean_rows: int
    avg_formosan_tokens: float
    avg_target_tokens: float

    @property
    def short_clean_frac(self) -> float:
        if self.clean_rows == 0:
            return 1.0
        return self.short_clean_rows / self.clean_rows

    def selection_rows(self, hard_eval: bool) -> int:
        return self.hard_clean_rows if hard_eval else self.clean_rows

    def selection_short_rows(self, hard_eval: bool) -> int:
        return self.short_hard_clean_rows if hard_eval else self.short_clean_rows

    @property
    def raw_to_clean_extra(self) -> int:
        return max(0, self.raw_rows - self.clean_rows)


def normalize_text(value: object) -> str:
    """NFKC + casefold + whitespace collapse for leakage checks."""
    text = unicodedata.normalize("NFKC", "" if value is None else str(value))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_count(value: object) -> int:
    text = normalize_text(value)
    return 0 if not text else len(text.split())


def is_short_entry(formosan: object, target: object) -> bool:
    """Heuristic for glossary-ish rows; only used to prefer sentence-like eval."""
    return token_count(formosan) <= 2 and token_count(target) <= 3


def source_bucket(source: object) -> str:
    """Coarse source family used to prevent easy-domain headline test splits."""
    s = "" if source is None else str(source)
    if "xue_xi_ci_biao_learning_vocabulary" in s:
        return "learning_vocab"
    if "qing_jing_zu_yu_contextual_indigenous_language" in s:
        return "classroom_context"
    if "Dict" in s or "Dictionary" in s:
        return "dictionary"
    if "tu_hua_gu_shi_pian_picture_story" in s:
        return "picture_story"
    if "hui_ben_ping_tai_picture_book_platform" in s:
        return "picture_book"
    if "zu_yu_duan_wen_indigenous_language_essays" in s:
        return "essays"
    if "yue_du_shu_xie_pian_reading_writing" in s:
        return "reading_writing"
    if "wen_hua_pian_cultural_section" in s:
        return "culture"
    if "jiu_jie_jiao_cai_nine_level_materials" in s:
        return "nine_level"
    if "YouTube" in s:
        return "youtube"
    if "NTU" in s:
        return "ntu"
    if "President" in s or "Apology" in s:
        return "presidential_apology"
    return s.split("/")[0] if s else "unknown"


def parse_csv_values(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def parse_overlap_keys(value: str) -> tuple[str, ...]:
    keys = tuple(k.strip().lower() for k in value.split(",") if k.strip())
    allowed = set(DEFAULT_OVERLAP_KEYS)
    unknown = [k for k in keys if k not in allowed]
    if unknown:
        raise SystemExit(
            f"Unsupported --overlap-keys entries: {unknown}. "
            f"Allowed: {sorted(allowed)}"
        )
    return keys


def infer_target_col(df: pd.DataFrame, target_col: str | None) -> str:
    if target_col:
        if target_col not in df.columns:
            raise SystemExit(f"Missing requested --target-col: {target_col}")
        return target_col
    for candidate in ("english_sentence", "chinese_sentence"):
        if candidate in df.columns:
            return candidate
    raise SystemExit("Could not infer target column; pass --target-col.")


def ensure_required_columns(df: pd.DataFrame, target_col: str) -> None:
    required = {"lang_code", "formosan_sentence", target_col, "source", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"Input CSV is missing required columns: {missing}")


def add_internal_keys(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_source_key"] = out["source"].astype(str).fillna("")
    out["_formosan_key"] = out["formosan_sentence"].map(normalize_text)
    out["_target_key"] = out[target_col].map(normalize_text)
    out["_pair_key"] = out["_formosan_key"] + PAIR_SEP + out["_target_key"]
    out["_formosan_tokens"] = out["formosan_sentence"].map(token_count)
    out["_target_tokens"] = out[target_col].map(token_count)
    out["_short_entry"] = (
        (out["_formosan_tokens"] <= 2) & (out["_target_tokens"] <= 3)
    )
    out["_source_bucket"] = out["_source_key"].map(source_bucket)
    return out


def row_clean_mask(lang_df: pd.DataFrame, overlap_keys: Iterable[str]) -> pd.Series:
    """Rows are clean when all requested text keys occur in only one source file."""
    clean = pd.Series(True, index=lang_df.index)
    source_col = "_source_key"
    key_to_col = {
        "formosan": "_formosan_key",
        "target": "_target_key",
        "pair": "_pair_key",
    }
    for key in overlap_keys:
        col = key_to_col[key]
        source_counts = lang_df.groupby(col, sort=False)[source_col].transform("nunique")
        clean &= source_counts.eq(1)
    return clean


def build_source_groups(
    lang_df: pd.DataFrame,
    clean_mask: pd.Series,
    hard_eval_mask: pd.Series,
) -> list[SourceGroup]:
    groups: list[SourceGroup] = []
    clean_mask = clean_mask.reindex(lang_df.index)
    hard_eval_mask = hard_eval_mask.reindex(lang_df.index)
    for source, group in lang_df.groupby("_source_key", sort=False):
        group_clean = clean_mask.loc[group.index]
        group_hard_clean = group_clean & hard_eval_mask.loc[group.index]
        clean_group = group[group_clean]
        hard_clean_group = group[group_hard_clean]
        clean_rows = int(group_clean.sum())
        hard_clean_rows = int(group_hard_clean.sum())
        if clean_rows:
            short_clean_rows = int(clean_group["_short_entry"].sum())
            avg_formosan_tokens = float(clean_group["_formosan_tokens"].mean())
            avg_target_tokens = float(clean_group["_target_tokens"].mean())
        else:
            short_clean_rows = 0
            avg_formosan_tokens = 0.0
            avg_target_tokens = 0.0
        if hard_clean_rows:
            short_hard_clean_rows = int(hard_clean_group["_short_entry"].sum())
        else:
            short_hard_clean_rows = 0
        bucket = str(group["_source_bucket"].iloc[0]) if len(group) else "unknown"
        groups.append(
            SourceGroup(
                source=str(source),
                bucket=bucket,
                raw_rows=int(len(group)),
                clean_rows=clean_rows,
                short_clean_rows=short_clean_rows,
                hard_clean_rows=hard_clean_rows,
                short_hard_clean_rows=short_hard_clean_rows,
                avg_formosan_tokens=avg_formosan_tokens,
                avg_target_tokens=avg_target_tokens,
            )
        )
    return groups


def source_selection_cost(group: SourceGroup, hard_eval: bool) -> float:
    """Lower is better for heldout selection."""
    selected_rows = group.selection_rows(hard_eval)
    if selected_rows <= 0:
        return float("inf")

    # Prefer sources that have mostly usable heldout rows and are not dominated by
    # very short glossary-like entries. Keep the fit-to-target objective primary.
    heldout_drop_penalty = 0.20 * max(0, group.raw_rows - selected_rows)
    short_penalty = 1.25 * group.selection_short_rows(hard_eval)
    avg_len = (group.avg_formosan_tokens + group.avg_target_tokens) / 2.0
    length_bonus = 0.04 * min(avg_len, 25.0) * selected_rows
    return heldout_drop_penalty + short_penalty - length_bonus


def choose_sources_for_clean_rows(
    groups: list[SourceGroup],
    target_rows: int,
    seed: int,
    tolerance: float,
    hard_eval: bool,
    allow_oversized_fallback: bool,
) -> set[str]:
    """Subset-sum style source selection by clean-row count."""
    candidates = [g for g in groups if g.selection_rows(hard_eval) > 0]
    if not candidates or target_rows <= 0:
        return set()

    rng = random.Random(seed)
    max_candidate = max(g.selection_rows(hard_eval) for g in candidates)

    selected_tuple: tuple[int, ...] | None = None
    best_meta: tuple[float, float, int, float] | None = None

    # Start strict, then relax if document sizes make the target unreachable.
    for tol in (tolerance, 0.20, 0.35, 0.55, 0.85, 1.25):
        lower = max(1, math.floor(target_rows * max(0.0, 1.0 - tol)))
        upper = max(1, math.ceil(target_rows * (1.0 + tol)))
        upper = min(max(upper, min(max_candidate, target_rows)), sum(g.clean_rows for g in candidates))

        dp: dict[int, tuple[float, int, float, tuple[int, ...]]] = {
            0: (0.0, 0, 0.0, ())
        }
        order = list(range(len(candidates)))
        rng.shuffle(order)
        for idx in order:
            group = candidates[idx]
            rows = group.selection_rows(hard_eval)
            cost = source_selection_cost(group, hard_eval)
            if rows > upper:
                continue
            # Snapshot avoids using the same source more than once.
            for current_rows, state in list(dp.items()):
                new_rows = current_rows + rows
                if new_rows > upper:
                    continue
                new_state = (
                    state[0] + cost,
                    state[1] + group.raw_rows,
                    state[2] + rng.random() * 1e-9,
                    state[3] + (idx,),
                )
                old_state = dp.get(new_rows)
                if old_state is None or new_state[:3] < old_state[:3]:
                    dp[new_rows] = new_state

        feasible = []
        for rows, state in dp.items():
            if rows < lower or rows == 0:
                continue
            deviation = abs(rows - target_rows) / max(1, target_rows)
            per_clean_cost = state[0] / rows
            raw_rows = state[1]
            tie = state[2]
            feasible.append((deviation, per_clean_cost, raw_rows, tie, rows, state[3]))

        if feasible:
            feasible.sort()
            best = feasible[0]
            selected_tuple = best[5]
            best_meta = best[:4]
            break

    if selected_tuple is None and not allow_oversized_fallback:
        return set()

    if selected_tuple is None:
        # Fallback: greedily fill with the lowest-cost sources.
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (
                source_selection_cost(item[1], hard_eval)
                / max(1, item[1].selection_rows(hard_eval)),
                item[1].raw_rows,
                item[0],
            ),
        )
        chosen: list[int] = []
        total = 0
        for idx, group in ranked:
            if total >= target_rows:
                break
            chosen.append(idx)
            total += group.selection_rows(hard_eval)
        selected_tuple = tuple(chosen)
        best_meta = (abs(total - target_rows) / max(1, target_rows), 0.0, total, 0.0)

    _ = best_meta  # Retained for easier debugging if needed.
    return {candidates[idx].source for idx in selected_tuple}


def assign_language_splits(
    lang_df: pd.DataFrame,
    overlap_keys: tuple[str, ...],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    tolerance: float,
    hard_eval: bool,
    global_target_unique_eval_mask: pd.Series,
    hard_eval_excluded_buckets: tuple[str, ...],
    hard_eval_min_formosan_tokens: int,
    hard_eval_min_target_tokens: int,
) -> tuple[pd.Series, dict]:
    clean_mask = row_clean_mask(lang_df, overlap_keys)
    if hard_eval:
        hard_eval_mask = (
            ~lang_df["_source_bucket"].isin(hard_eval_excluded_buckets)
            & lang_df["_formosan_tokens"].ge(hard_eval_min_formosan_tokens)
            & lang_df["_target_tokens"].ge(hard_eval_min_target_tokens)
            & global_target_unique_eval_mask.reindex(lang_df.index).fillna(False)
        )
    else:
        hard_eval_mask = pd.Series(True, index=lang_df.index)
    eval_mask = clean_mask & hard_eval_mask
    groups = build_source_groups(lang_df, clean_mask, hard_eval_mask)
    n_rows = len(lang_df)
    target_val = int(round(n_rows * val_ratio))
    target_test = int(round(n_rows * test_ratio))

    test_sources = choose_sources_for_clean_rows(
        groups,
        target_test,
        seed=seed + 17,
        tolerance=tolerance,
        hard_eval=hard_eval,
        allow_oversized_fallback=not hard_eval,
    )
    remaining_groups = [g for g in groups if g.source not in test_sources]
    val_sources = choose_sources_for_clean_rows(
        remaining_groups,
        target_val,
        seed=seed + 31,
        tolerance=tolerance,
        hard_eval=hard_eval,
        allow_oversized_fallback=not hard_eval,
    )

    split = pd.Series(SPLIT_TRAIN, index=lang_df.index, dtype="object")
    is_test_source = lang_df["_source_key"].isin(test_sources)
    is_val_source = lang_df["_source_key"].isin(val_sources)
    split.loc[is_test_source & eval_mask] = SPLIT_TEST
    split.loc[is_val_source & eval_mask] = SPLIT_VAL
    split.loc[(is_test_source | is_val_source) & ~eval_mask] = SPLIT_EXCLUDED

    counts = split.value_counts().to_dict()
    source_counts = {
        SPLIT_TRAIN: int(lang_df.loc[split.eq(SPLIT_TRAIN), "_source_key"].nunique()),
        SPLIT_VAL: int(lang_df.loc[split.eq(SPLIT_VAL), "_source_key"].nunique()),
        SPLIT_TEST: int(lang_df.loc[split.eq(SPLIT_TEST), "_source_key"].nunique()),
        SPLIT_EXCLUDED: int(
            lang_df.loc[split.eq(SPLIT_EXCLUDED), "_source_key"].nunique()
        ),
    }
    report = {
        "rows": n_rows,
        "target_counts": {
            SPLIT_TRAIN: int(round(n_rows * train_ratio)),
            SPLIT_VAL: target_val,
            SPLIT_TEST: target_test,
        },
        "actual_counts": {k: int(v) for k, v in counts.items()},
        "actual_ratios": {
            k: round(float(v) / n_rows, 6) for k, v in sorted(counts.items())
        },
        "source_counts": source_counts,
        "raw_heldout_sources": {
            SPLIT_VAL: sorted(val_sources),
            SPLIT_TEST: sorted(test_sources),
        },
        "clean_candidate_rows": int(clean_mask.sum()),
        "hard_eval_candidate_rows": int((clean_mask & hard_eval_mask).sum()),
        "excluded_overlap_rows": int(split.eq(SPLIT_EXCLUDED).sum()),
        "test_bucket_counts": {
            str(k): int(v)
            for k, v in lang_df.loc[split.eq(SPLIT_TEST), "_source_bucket"]
            .value_counts()
            .items()
        },
        "validate_bucket_counts": {
            str(k): int(v)
            for k, v in lang_df.loc[split.eq(SPLIT_VAL), "_source_bucket"]
            .value_counts()
            .items()
        },
    }
    return split, report


def overlap_stats(df: pd.DataFrame, target_col: str) -> dict:
    split_col = df["split"].astype(str).str.lower()
    splits = {
        SPLIT_TRAIN: df[split_col.eq(SPLIT_TRAIN)],
        SPLIT_VAL: df[split_col.isin(("valid", "val", SPLIT_VAL))],
        SPLIT_TEST: df[split_col.eq(SPLIT_TEST)],
    }

    stats: dict[str, dict] = {}
    key_cols = {
        "source": "_source_key",
        "formosan": "_formosan_key",
        "target": "_target_key",
        "pair": "_pair_key",
    }
    for lang, lang_df in df.groupby("lang_code", sort=True):
        lang_stats: dict[str, dict] = {}
        lang_splits = {
            name: part[part["lang_code"].eq(lang)] for name, part in splits.items()
        }
        for left_name, right_name in (
            (SPLIT_TRAIN, SPLIT_VAL),
            (SPLIT_TRAIN, SPLIT_TEST),
            (SPLIT_VAL, SPLIT_TEST),
        ):
            left = lang_splits[left_name]
            right = lang_splits[right_name]
            pair_name = f"{left_name}_vs_{right_name}"
            lang_stats[pair_name] = {}
            for key_name, col in key_cols.items():
                left_keys = set(left[col].dropna())
                right_keys = set(right[col].dropna())
                overlap = left_keys & right_keys
                lang_stats[pair_name][key_name] = {
                    "overlap_keys": int(len(overlap)),
                    "left_overlap_rows": int(left[col].isin(overlap).sum()),
                    "right_overlap_rows": int(right[col].isin(overlap).sum()),
                }
        stats[str(lang)] = lang_stats
    _ = target_col
    return stats


def compact_split_counts(df: pd.DataFrame) -> dict:
    counts = (
        df.groupby(["lang_code", "split"], sort=True)
        .size()
        .unstack(fill_value=0)
        .astype(int)
    )
    return {
        str(lang): {str(k): int(v) for k, v in row.items() if int(v) != 0}
        for lang, row in counts.iterrows()
    }


def write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild robust source-heldout splits for Formosan MT CSVs."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input corpus CSV.")
    parser.add_argument("--output", required=True, type=Path, help="Output corpus CSV.")
    parser.add_argument("--target-col", default=None, help="Target sentence column.")
    parser.add_argument("--report-json", type=Path, default=None, help="Diagnostics JSON.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed.")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.025)
    parser.add_argument("--test-ratio", type=float, default=0.075)
    parser.add_argument(
        "--overlap-keys",
        default=",".join(DEFAULT_OVERLAP_KEYS),
        help=(
            "Comma-separated exact-normalized keys to forbid across source files "
            "for validation/test rows. Allowed: formosan,target,pair."
        ),
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.08,
        help="Initial per-language clean-row ratio tolerance for heldout selection.",
    )
    parser.add_argument(
        "--hard-eval",
        action="store_true",
        help=(
            "Build validation/test from hard, sentence-like rows only. Held-out "
            "rows outside the hard-eval mask are dropped unless "
            "--keep-excluded-overlap is set."
        ),
    )
    parser.add_argument(
        "--global-target-unique-eval",
        action="store_true",
        help=(
            "In hard-eval mode, only allow validation/test rows whose normalized "
            "target sentence occurs in exactly one language+source group globally. "
            "This prevents multilingual leakage where another language's train "
            "rows contain the same English/Chinese reference."
        ),
    )
    parser.add_argument(
        "--hard-eval-exclude-buckets",
        default=",".join(DEFAULT_HARD_EVAL_EXCLUDED_BUCKETS),
        help=(
            "Comma-separated source buckets excluded from hard validation/test. "
            "Default excludes dictionary, learning vocabulary, and classroom context."
        ),
    )
    parser.add_argument(
        "--hard-eval-min-formosan-tokens",
        type=int,
        default=4,
        help="Minimum normalized whitespace-token count for Formosan hard eval rows.",
    )
    parser.add_argument(
        "--hard-eval-min-target-tokens",
        type=int,
        default=4,
        help="Minimum normalized whitespace-token count for target hard eval rows.",
    )
    parser.add_argument(
        "--keep-excluded-overlap",
        action="store_true",
        help=(
            "Keep rows marked as excluded_overlap in the output CSV instead of "
            "dropping them. Useful only for debugging."
        ),
    )
    args = parser.parse_args()

    ratios = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratios, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise SystemExit(
            f"Ratios must sum to 1.0, got {ratios:.8f} "
            f"({args.train_ratio}, {args.val_ratio}, {args.test_ratio})."
        )

    overlap_keys = parse_overlap_keys(args.overlap_keys)
    hard_eval_excluded_buckets = parse_csv_values(args.hard_eval_exclude_buckets)
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    target_col = infer_target_col(df, args.target_col)
    ensure_required_columns(df, target_col)
    work = add_internal_keys(df, target_col)
    if args.hard_eval and args.global_target_unique_eval:
        global_target_unique_eval_mask = (
            work.groupby("_target_key", sort=False)["_source_key"]
            .transform("nunique")
            .eq(1)
            & work.groupby("_target_key", sort=False)["lang_code"]
            .transform("nunique")
            .eq(1)
        )
    else:
        global_target_unique_eval_mask = pd.Series(True, index=work.index)

    all_splits = pd.Series(SPLIT_TRAIN, index=work.index, dtype="object")
    lang_reports: dict[str, dict] = {}
    for offset, (lang, lang_df) in enumerate(work.groupby("lang_code", sort=True)):
        split, lang_report = assign_language_splits(
            lang_df=lang_df,
            overlap_keys=overlap_keys,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed + offset * 1009,
            tolerance=args.selection_tolerance,
            hard_eval=args.hard_eval,
            global_target_unique_eval_mask=global_target_unique_eval_mask,
            hard_eval_excluded_buckets=hard_eval_excluded_buckets,
            hard_eval_min_formosan_tokens=args.hard_eval_min_formosan_tokens,
            hard_eval_min_target_tokens=args.hard_eval_min_target_tokens,
        )
        all_splits.loc[lang_df.index] = split
        lang_reports[str(lang)] = lang_report

    assigned = df.copy()
    assigned["split"] = all_splits
    removed_excluded_rows = int(assigned["split"].eq(SPLIT_EXCLUDED).sum())
    if args.keep_excluded_overlap:
        out = assigned
    else:
        out = assigned[~assigned["split"].eq(SPLIT_EXCLUDED)].copy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    verify = add_internal_keys(out, target_col)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "target_col": target_col,
        "seed": args.seed,
        "ratios": {
            SPLIT_TRAIN: args.train_ratio,
            SPLIT_VAL: args.val_ratio,
            SPLIT_TEST: args.test_ratio,
        },
        "overlap_keys_for_clean_eval": list(overlap_keys),
        "hard_eval": bool(args.hard_eval),
        "global_target_unique_eval": bool(args.global_target_unique_eval),
        "hard_eval_excluded_buckets": list(hard_eval_excluded_buckets),
        "hard_eval_min_formosan_tokens": args.hard_eval_min_formosan_tokens,
        "hard_eval_min_target_tokens": args.hard_eval_min_target_tokens,
        "input_rows": int(len(df)),
        "output_rows": int(len(out)),
        "removed_excluded_overlap_rows": removed_excluded_rows
        if not args.keep_excluded_overlap
        else 0,
        "kept_excluded_overlap_rows": removed_excluded_rows
        if args.keep_excluded_overlap
        else 0,
        "split_counts": {str(k): int(v) for k, v in out["split"].value_counts().items()},
        "split_counts_by_language": compact_split_counts(out),
        "languages": lang_reports,
        "overlap_stats": overlap_stats(verify, target_col),
    }

    report_path = args.report_json
    if report_path is None:
        report_path = args.output.with_suffix(args.output.suffix + ".report.json")
    write_report(report_path, report)

    print(f"Wrote CSV: {args.output}")
    print(f"Wrote report: {report_path}")
    if args.keep_excluded_overlap:
        print(f"Kept excluded_overlap rows: {removed_excluded_rows}")
    else:
        print(f"Removed excluded_overlap rows: {removed_excluded_rows}")
    print("Split counts:")
    for split_name, count in out["split"].value_counts().items():
        print(f"  {split_name:16s} {count:8d} ({count / len(out) * 100:5.2f}%)")
    print("Per-language active eval counts:")
    for lang, counts in report["split_counts_by_language"].items():
        print(
            f"  {lang:4s} train={counts.get(SPLIT_TRAIN, 0):6d} "
            f"val={counts.get(SPLIT_VAL, 0):5d} "
            f"test={counts.get(SPLIT_TEST, 0):5d} "
            f"excluded={counts.get(SPLIT_EXCLUDED, 0):5d}"
        )


if __name__ == "__main__":
    main()
