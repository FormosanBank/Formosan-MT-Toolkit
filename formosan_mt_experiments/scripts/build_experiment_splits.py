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
import math
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
        "pivot_origin",
        "pivot_direction",
    ]


@dataclass(frozen=True)
class SourceCandidate:
    source: str
    rows: int
    total_source_rows: int
    short_frac: float
    avg_tokens: float


@dataclass(frozen=True)
class GroupCandidate:
    group_id: int
    rows: int
    short_frac: float
    avg_tokens: float
    synthetic_rows: int


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def one_edit_conflicts(train: pd.DataFrame, eval_df: pd.DataFrame, column: str) -> set[int]:
    """Find train rows within one character edit of evaluation text, per language."""
    conflicts: set[int] = set()
    for lang, eval_lang in eval_df.groupby("lang_code", sort=False):
        train_lang = train[train["lang_code"].eq(lang)]
        if train_lang.empty:
            continue
        eval_values = {str(value) for value in eval_lang[column].fillna("") if str(value)}
        eval_exact = {hash(value) for value in eval_values}
        eval_deletes = {
            hash(value[:pos] + value[pos + 1 :])
            for value in eval_values
            for pos in range(len(value))
        }
        for index, value in train_lang[column].fillna("").astype(str).items():
            if not value:
                continue
            value_hash = hash(value)
            if value_hash in eval_exact or value_hash in eval_deletes:
                conflicts.add(int(index))
                continue
            for pos in range(len(value)):
                deleted_hash = hash(value[:pos] + value[pos + 1 :])
                if deleted_hash in eval_exact or deleted_hash in eval_deletes:
                    conflicts.add(int(index))
                    break
    return conflicts


def leakage_group_ids(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Connected components over exact/skeleton keys so a near-dupe cluster has one split."""
    dsu = DisjointSet(len(df))
    for col in columns:
        owner: dict[str, int] = {}
        for pos, value in enumerate(df[col].fillna("").astype(str).tolist()):
            if not value:
                continue
            existing = owner.get(value)
            if existing is None:
                owner[value] = pos
            else:
                dsu.union(existing, pos)

    root_to_group: dict[int, int] = {}
    group_ids: list[int] = []
    for pos in range(len(df)):
        root = dsu.find(pos)
        group_id = root_to_group.setdefault(root, len(root_to_group))
        group_ids.append(group_id)
    return pd.Series(group_ids, index=df.index, dtype="int64")


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


def split_targets(
    rows_total: int,
    eligible_total: int,
    test_ratio: float,
    val_ratio: float,
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[int, int]:
    if eligible_total <= 0:
        return 0, 0
    eval_ratio = test_ratio + val_ratio
    test_share = test_ratio / eval_ratio if eval_ratio else 1.0
    desired_test = max(1, round(rows_total * test_ratio))
    desired_validate = max(1, round(rows_total * val_ratio))
    desired_eval = desired_test + desired_validate

    if eligible_total <= desired_eval:
        target_test = max(1, round(eligible_total * test_share))
        target_validate = eligible_total - target_test
        if eligible_total >= 2 and target_validate == 0:
            target_test -= 1
            target_validate = 1
        return target_test, target_validate

    target_test = max(desired_test, min_test_rows)
    target_validate = max(desired_validate, min_validate_rows)
    if target_test + target_validate > eligible_total:
        target_test = max(1, round(eligible_total * test_share))
        target_validate = eligible_total - target_test
    return target_test, target_validate


def group_candidates(
    lang_df: pd.DataFrame,
    candidate_mask: pd.Series,
    group_ids: pd.Series,
    assigned_groups: dict[int, str],
) -> list[GroupCandidate]:
    candidates: list[GroupCandidate] = []
    lang_candidate_mask = candidate_mask.reindex(lang_df.index, fill_value=False)
    eligible = lang_df[lang_candidate_mask].copy()
    eligible_group_ids = group_ids.loc[eligible.index]
    for group_id, group in eligible.groupby(eligible_group_ids, sort=False):
        group_key = int(group_id)
        if group_key in assigned_groups:
            continue
        avg_tokens = float((group["_formosan_tokens"] + group["_target_tokens"]).mean() / 2.0)
        candidates.append(
            GroupCandidate(
                group_id=group_key,
                rows=int(len(group)),
                short_frac=float(group["_short_entry"].mean()),
                avg_tokens=avg_tokens,
                synthetic_rows=int(group["_is_synthetic"].sum()),
            )
        )
    return candidates


def choose_groups(
    candidates: list[GroupCandidate],
    target_rows: int,
    seed: int,
    attempts: int,
) -> set[int]:
    if target_rows <= 0 or not candidates:
        return set()

    rng = random.Random(seed)
    best: list[GroupCandidate] = []
    best_cost = (float("inf"), float("inf"), float("inf"))

    ordered = sorted(
        candidates,
        key=lambda c: (c.synthetic_rows > 0, c.short_frac, -c.avg_tokens, -c.rows),
    )
    current: list[GroupCandidate] = []
    current_rows = 0
    for candidate in ordered:
        current.append(candidate)
        current_rows += candidate.rows
        if current_rows >= target_rows:
            break
    if current:
        best = current
        best_cost = (
            abs(sum(g.rows for g in current) - target_rows),
            sum(g.synthetic_rows for g in current),
            sum(g.short_frac * g.rows for g in current),
        )

    for _ in range(attempts):
        shuffled = candidates[:]
        rng.shuffle(shuffled)
        shuffled.sort(
            key=lambda c: 10.0 * (c.synthetic_rows > 0)
            + rng.random()
            + 0.015 * c.short_frac
            - 0.0005 * c.avg_tokens
        )
        selected: list[GroupCandidate] = []
        selected_rows = 0
        for candidate in shuffled:
            selected.append(candidate)
            selected_rows += candidate.rows
            if selected_rows >= target_rows:
                break
        cost = (
            abs(selected_rows - target_rows),
            sum(g.synthetic_rows for g in selected),
            sum(g.short_frac * g.rows for g in selected),
        )
        if selected and cost < best_cost:
            best = selected
            best_cost = cost

    return {c.group_id for c in best}


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
    min_test_rows: int,
    min_validate_rows: int,
) -> tuple[pd.DataFrame, dict]:
    leakage_columns = (
        "_formosan_key",
        "_target_key",
        "_pair_key",
        "_formosan_skeleton",
        "_target_skeleton",
        "_pair_skeleton",
    )
    group_ids = leakage_group_ids(df, leakage_columns)
    raw_candidate_mask = tier_mask(df, tier, min_formosan_tokens, min_target_tokens)
    synthetic_mask = df.get("pivot_origin", pd.Series("original", index=df.index)).fillna("original").eq("synthetic")
    df = df.copy()
    df["_is_synthetic"] = synthetic_mask
    raw_candidate_mask &= ~synthetic_mask
    fallback_rows_added = 0
    broad_fallback_rows_added = 0
    synthetic_fallback_rows_added = 0
    if tier == "in_domain_hard":
        fallback_mask = (
            ~synthetic_mask
            & ~df["_is_lexeme"]
            & df["_formosan_tokens"].ge(2)
            & df["_target_tokens"].ge(2)
        )
        for _lang, lang_df in df.groupby("lang_code", sort=False):
            lang_index = lang_df.index
            required_eval_rows = max(
                min_test_rows + min_validate_rows,
                math.ceil(len(lang_df) * (test_ratio + val_ratio)),
            )
            before = int(raw_candidate_mask.loc[lang_index].sum())
            if before < required_eval_rows:
                raw_candidate_mask.loc[lang_index] |= fallback_mask.loc[lang_index]
            fallback_rows_added += int(raw_candidate_mask.loc[lang_index].sum()) - before

            after_standard_fallback = int(raw_candidate_mask.loc[lang_index].sum())
            if after_standard_fallback < required_eval_rows:
                broad_fallback = (
                    ~synthetic_mask.loc[lang_index]
                    & ~df.loc[lang_index, "_is_lexeme"]
                    & df.loc[lang_index, "_formosan_tokens"].ge(1)
                    & df.loc[lang_index, "_target_tokens"].ge(1)
                )
                raw_candidate_mask.loc[lang_index] |= broad_fallback
            broad_fallback_rows_added += (
                int(raw_candidate_mask.loc[lang_index].sum()) - after_standard_fallback
            )

            after_human_fallback = int(raw_candidate_mask.loc[lang_index].sum())
            if after_human_fallback < required_eval_rows:
                synthetic_fallback = (
                    synthetic_mask.loc[lang_index]
                    & ~df.loc[lang_index, "_is_lexeme"]
                    & df.loc[lang_index, "_formosan_tokens"].ge(1)
                    & df.loc[lang_index, "_target_tokens"].ge(1)
                )
                raw_candidate_mask.loc[lang_index] |= synthetic_fallback
            synthetic_fallback_rows_added += (
                int(raw_candidate_mask.loc[lang_index].sum()) - after_human_fallback
            )
    group_has_lexeme = df.groupby(group_ids)["_is_lexeme"].transform("any")
    candidate_mask = raw_candidate_mask & ~group_has_lexeme
    split = pd.Series("", index=df.index, dtype="object")
    language_reports: dict[str, dict] = {}
    assigned_groups: dict[int, str] = {}

    lang_order = []
    for lang, lang_df in df.groupby("lang_code", sort=True):
        eligible_total = int(candidate_mask.loc[lang_df.index].sum())
        lang_order.append((eligible_total, str(lang), lang_df))

    # Small languages get first claim on scarce leakage groups. Larger languages
    # can absorb incidental group assignments more easily.
    for offset, (eligible_sort_key, lang, lang_df) in enumerate(sorted(lang_order, key=lambda item: (item[0], item[1]))):
        lang_candidate_mask = candidate_mask.reindex(lang_df.index, fill_value=False)
        lang_total = int(len(lang_df))
        human_total = int((~synthetic_mask.loc[lang_df.index]).sum())
        eligible_total = int(lang_candidate_mask.sum())
        target_test, target_val = split_targets(
            rows_total=lang_total,
            eligible_total=eligible_total,
            test_ratio=test_ratio,
            val_ratio=val_ratio,
            min_test_rows=min_test_rows,
            min_validate_rows=min_validate_rows,
        )

        assigned_for_lang = group_ids.loc[lang_df.index].map(assigned_groups)
        current_test = int((lang_candidate_mask & assigned_for_lang.eq("test")).sum())
        current_val = int((lang_candidate_mask & assigned_for_lang.eq("validate")).sum())

        candidates = group_candidates(lang_df, lang_candidate_mask, group_ids, assigned_groups)
        test_groups = choose_groups(
            candidates,
            max(0, target_test - current_test),
            seed=seed + 997 * offset,
            attempts=attempts,
        )
        for group_id in test_groups:
            assigned_groups[group_id] = "test"

        candidates = group_candidates(lang_df, lang_candidate_mask, group_ids, assigned_groups)
        val_groups = choose_groups(
            candidates,
            max(0, target_val - current_val),
            seed=seed + 997 * offset + 17,
            attempts=attempts,
        )
        for group_id in val_groups:
            assigned_groups[group_id] = "validate"

        assigned_for_lang = group_ids.loc[lang_df.index].map(assigned_groups)
        final_test = int((lang_candidate_mask & assigned_for_lang.eq("test")).sum())
        final_val = int((lang_candidate_mask & assigned_for_lang.eq("validate")).sum())
        language_reports[str(lang)] = {
            "rows_total": int(lang_total),
            "human_rows": int(human_total),
            "eligible_rows": int(eligible_total),
            "candidate_groups": int(len(set(group_ids.loc[lang_df.index][lang_candidate_mask]))),
            "target_test_rows": int(target_test),
            "target_validate_rows": int(target_val),
            "test_groups": int(sum(1 for group_id in set(group_ids.loc[lang_df.index]) if assigned_groups.get(int(group_id)) == "test")),
            "validate_groups": int(sum(1 for group_id in set(group_ids.loc[lang_df.index]) if assigned_groups.get(int(group_id)) == "validate")),
            "preclean_test_rows": final_test,
            "preclean_validate_rows": final_val,
            "lexeme_cluster_eval_candidates_removed": int((raw_candidate_mask.loc[lang_df.index] & ~lang_candidate_mask).sum()),
        }

    assigned = group_ids.map(assigned_groups)
    split.loc[candidate_mask & assigned.eq("test")] = "test"
    split.loc[candidate_mask & assigned.eq("validate")] = "validate"
    split.loc[assigned.isna()] = "train"

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
    fuzzy_conflicts = one_edit_conflicts(train, eval_df, "_formosan_skeleton")
    fuzzy_conflicts |= one_edit_conflicts(train, eval_df, "_target_skeleton")
    if fuzzy_conflicts:
        out = out.drop(index=sorted(fuzzy_conflicts))
        train = out[out["split"].eq("train")]
        eval_df = out[out["split"].isin(["validate", "test"])]

    ratio_trimmed_by_language = {}
    if tier == "in_domain_hard":
        trim_indices = []
        for lang, lang_out in out.groupby("lang_code", sort=True):
            counts = lang_out["split"].value_counts()
            actual_test = int(counts.get("test", 0))
            actual_validate = int(counts.get("validate", 0))
            maximum_total = min(
                math.floor(actual_test / test_ratio) if test_ratio else len(lang_out),
                math.floor(actual_validate / val_ratio) if val_ratio else len(lang_out),
            )
            trim_count = max(0, len(lang_out) - maximum_total)
            if not trim_count:
                continue
            train_candidates = lang_out[lang_out["split"].eq("train")].copy()
            train_candidates["_trim_synthetic"] = synthetic_mask.loc[train_candidates.index]
            train_candidates["_trim_tokens"] = (
                train_candidates["_formosan_tokens"] + train_candidates["_target_tokens"]
            )
            selected = train_candidates.sort_values(
                ["_trim_synthetic", "_short_entry", "_trim_tokens"],
                ascending=[False, False, True],
                kind="stable",
            ).head(trim_count)
            if len(selected) != trim_count:
                raise SystemExit(
                    f"Cannot trim enough train rows to satisfy ratios for {lang}: "
                    f"needed={trim_count} available={len(selected)}"
                )
            trim_indices.extend(selected.index.tolist())
            ratio_trimmed_by_language[str(lang)] = int(trim_count)
        if trim_indices:
            out = out.drop(index=trim_indices)
            train = out[out["split"].eq("train")]
            eval_df = out[out["split"].isin(["validate", "test"])]

    ratio_shortfalls = {}
    for lang, lang_out in out.groupby("lang_code", sort=True):
        counts = lang_out["split"].value_counts()
        total = int(len(lang_out))
        actual_test = int(counts.get("test", 0))
        actual_validate = int(counts.get("validate", 0))
        synthetic_eval = int(
            (
                synthetic_mask.loc[lang_out.index]
                & lang_out["split"].isin(["validate", "test"])
            ).sum()
        )
        required_test = math.ceil(total * test_ratio)
        required_validate = math.ceil(total * val_ratio)
        language_reports[str(lang)].update(
            {
                "output_rows": total,
                "final_test_rows": actual_test,
                "final_validate_rows": actual_validate,
                "final_test_fraction": actual_test / max(total, 1),
                "final_validate_fraction": actual_validate / max(total, 1),
                "required_test_rows": required_test,
                "required_validate_rows": required_validate,
                "synthetic_eval_rows": synthetic_eval,
            }
        )
        if tier == "in_domain_hard" and (
            actual_test < required_test or actual_validate < required_validate
        ):
            ratio_shortfalls[str(lang)] = {
                "output_rows": total,
                "test": actual_test,
                "required_test": required_test,
                "validate": actual_validate,
                "required_validate": required_validate,
            }
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
        "synthetic_rows_forced_train": int((synthetic_mask & split.eq("train")).sum()),
        "synthetic_eval_rows": int(
            (synthetic_mask & split.isin(["validate", "test"])).sum()
        ),
        "synthetic_rows_removed_for_leakage": int(
            synthetic_mask.sum()
            - (synthetic_mask & split.eq("train")).sum()
            - (synthetic_mask & split.isin(["validate", "test"])).sum()
        ),
        "human_fallback_candidate_rows_added": fallback_rows_added,
        "human_broad_fallback_candidate_rows_added": broad_fallback_rows_added,
        "synthetic_fallback_candidate_rows_added": synthetic_fallback_rows_added,
        "one_edit_train_rows_removed": len(fuzzy_conflicts),
        "train_rows_trimmed_for_final_ratios": int(sum(ratio_trimmed_by_language.values())),
        "train_rows_trimmed_for_final_ratios_by_language": ratio_trimmed_by_language,
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
        "ratio_shortfalls": ratio_shortfalls,
        "required_final_ratios": {"test": test_ratio, "validate": val_ratio},
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
    if report["tier"] == "in_domain_hard" and report.get("ratio_shortfalls"):
        raise SystemExit(
            "Final per-language evaluation ratio validation failed: "
            f"{json.dumps(report['ratio_shortfalls'], sort_keys=True)}"
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
    parser.add_argument(
        "--min-test-rows",
        type=int,
        default=100,
        help=(
            "Minimum desired test rows per language when enough hard eligible "
            "rows exist. Ratios still win when they request more."
        ),
    )
    parser.add_argument(
        "--min-validate-rows",
        type=int,
        default=25,
        help=(
            "Minimum desired validation rows per language when enough hard eligible "
            "rows exist. Ratios still win when they request more."
        ),
    )
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
        "minimum_eval_rows": {
            "test": args.min_test_rows,
            "validate": args.min_validate_rows,
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
            min_test_rows=args.min_test_rows,
            min_validate_rows=args.min_validate_rows,
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
