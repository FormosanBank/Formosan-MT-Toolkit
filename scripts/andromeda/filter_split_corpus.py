#!/usr/bin/env python3
"""
STEP 4:

Filter and split parallel corpus for Machine Translation training.

This script performs several cleaning operations on a parallel corpus CSV:
1. Applies Moses punctuation normalization (following NLLB approach)
2. Removes non-printable characters and normalizes text
3. Removes exact duplicate sentence pairs
4. Applies fertility heuristic based on length ratios
5. Splits the corpus into train/validation/test sets

Key features:
- Near-duplicate grouping: sentences that are extremely similar are kept
  in the same split to avoid leakage. (≤k token removals, default k=1)
- Group-aware splitting that hits per-corpus quotas while keeping groups intact:
  * Fast path for singleton groups (exact quotas)
  * Greedy FFD packing for large/irregular groups
  * Bitset knapsack only when small enough
- Optional pinning of giant groups to TRAIN so they don't dominate eval.
- Deterministic RNG per corpus for reproducibility.

Examples
--------
python filter_split_corpus.py --input corpus.csv --output corpus_ready.csv
python filter_split_corpus.py --input amis_zh.csv --output amis_zh_ready.csv
python filter_split_corpus.py --input corpus.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sacremoses import MosesPunctNormalizer

ASCII_CTRL_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]')
CJK_RE = re.compile(r'[\u3400-\u9FFF\uF900-\uFAFF]')

def replace_nonprinting(s: str) -> str:
    """
    Fast removal/replacement of non-printable chars:
    - Strip ASCII control characters via regex
    - For the rest, replace any Unicode category starting with 'C'
      (Other: Cc, Cf, Cs, Co, Cn) with a space, except keep ^ and '.
    """
    if not s:
        return ""
    s = ASCII_CTRL_RE.sub(" ", s)
    return ''.join((' ' if (unicodedata.category(ch)[0] == 'C' and ch not in "^'") else ch)
                   for ch in s)

def _tok_count(s: str) -> int:
    s = "" if pd.isna(s) else str(s)
    return len(s.split())

def _detect_lang_cols(df: pd.DataFrame) -> tuple[str, str]:
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [c for c in text_cols if c not in ['source', 'kindOf', 'split', 'nd_group', 'corpus']]
    if len(lang_cols) < 2:
        sys.exit("❌ Need at least two language columns besides 'source/kindOf/split'.")
    return lang_cols[0], lang_cols[1]

def preprocess_text(text: str, mpn: MosesPunctNormalizer, verbose: bool = False) -> str:
    """NLLB-style preprocessing."""
    if pd.isna(text):
        return ""
    text = str(text)
    clean = text
    # Apply Moses substitutions
    for pattern, sub in mpn.substitutions:
        clean = pattern.sub(sub, clean)
    # Remove non-printable
    clean = replace_nonprinting(clean)
    # Unicode NFKC
    clean = unicodedata.normalize("NFKC", clean)
    # Trim whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

G_MPN = None

def _init_worker():
    """Initializer for multiprocessing pool: create/compile Moses once per worker."""
    global G_MPN
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    G_MPN = mpn

def process_chunk(args):
    """Process a chunk of text data with Moses normalization."""
    chunk_data, verbose = args
    mpn = G_MPN
    processed = [preprocess_text(text, mpn, verbose) for text in chunk_data]
    return processed

def clean_corpus_text_parallel(df: pd.DataFrame, verbose: bool = False, n_workers: int = None) -> pd.DataFrame:
    """Clean text in all string columns using NLLB approach with multiprocessing."""
    df = df.copy()
    string_columns = df.select_dtypes(include=['object']).columns

    if n_workers is None:
        n_workers = mp.cpu_count()

    if len(df) < 1000:
        print(f"📝  Dataset too small ({len(df)} rows) - using single-threaded processing")
        return clean_corpus_text(df, verbose)

    print(f"🚀  Using {n_workers} CPU cores for parallel text cleaning")
    for col in string_columns:
        if col not in ['source', 'kindOf']:
            print(f"🧼  Cleaning text in column '{col}' (Moses + NFKC)…")
            data = df[col].tolist()
            min_chunk_size = 100
            max_chunks = n_workers * 2
            chunk_size = max(min_chunk_size, len(data) // max_chunks)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            print(f"📦  Processing {len(data):,} rows in {len(chunks)} chunks…")
            chunk_args = [(chunk, verbose) for chunk in chunks]
            with mp.Pool(n_workers, initializer=_init_worker) as pool:
                print("⏳  Processing chunks…")
                chunk_results = list(tqdm(
                    pool.imap(process_chunk, chunk_args),
                    total=len(chunks),
                    desc=f"Chunks",
                    unit="chunk"
                ))
            results = []
            for cr in chunk_results:
                results.extend(cr)
            print(f"✅  Processed {len(results):,} rows for '{col}'")
            df[col] = results
    return df

def clean_corpus_text(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Single-thread fallback."""
    df = df.copy()
    string_columns = df.select_dtypes(include=['object']).columns
    print("🔧  Initializing Moses punctuation normalizer…")
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    tqdm.pandas()
    for col in string_columns:
        if col not in ['source', 'kindOf']:
            print(f"🧼  Cleaning text in column '{col}' (Moses + NFKC)…")
            df[col] = df[col].progress_apply(lambda x: preprocess_text(x, mpn, verbose))
    return df

def load_data(filepath: Path) -> pd.DataFrame:
    """Load the parallel corpus CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅  Loaded {len(df):,} rows from {filepath}")
        return df
    except Exception as e:
        sys.exit(f"❌  Error loading data: {e}")

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate sentence pairs, retaining one instance."""
    initial_len = len(df)
    text_cols = df.select_dtypes(include=['object']).columns
    # FIX: correct list comprehension (no stray 'c')
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split', 'nd_group', 'corpus']]
    if len(lang_cols) >= 2:
        df_cleaned = df.drop_duplicates(subset=lang_cols[:2], keep='first')
    else:
        df_cleaned = df.drop_duplicates(keep='first')
    duplicates_removed = initial_len - len(df_cleaned)
    print(f"🗑️   Removed {duplicates_removed:,} exact duplicate sentence pairs")
    return df_cleaned

def apply_fertility_heuristic(
    df: pd.DataFrame,
    min_ratio: float = 0.2,
    max_ratio: float = 7.0
) -> pd.DataFrame:
    """Remove sentence pairs with target/source length ratio outside bounds."""
    df_work = df.copy()
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split', 'nd_group', 'corpus']]
    if len(lang_cols) < 2:
        print("⚠️  Warning: Cannot apply fertility heuristic - need at least 2 language columns")
        return df
    src_col, tgt_col = lang_cols[0], lang_cols[1]

    df_work['src_length'] = df_work[src_col].astype(str).apply(len)
    df_work['tgt_length'] = df_work[tgt_col].astype(str).apply(len)
    df_work = df_work[(df_work['src_length'] > 0) & (df_work['tgt_length'] > 0)]
    df_work['length_ratio'] = df_work['tgt_length'] / df_work['src_length']
    condition = (df_work['length_ratio'] >= min_ratio) & (df_work['length_ratio'] <= max_ratio)
    filtered_df = df_work[condition]
    removed = len(df_work) - len(filtered_df)
    print(f"📏  Removed {removed:,} sentence pairs with length ratio outside [{min_ratio:.2f}, {max_ratio:.2f}]")
    filtered_df = filtered_df.drop(columns=['src_length', 'tgt_length', 'length_ratio'])
    return filtered_df

def _rng_for_corpus(random_state: int, corpus_name: str) -> np.random.RandomState:
    # Stable per-corpus RNG
    seed = (random_state * 1000003 + (abs(hash(corpus_name)) % (2**31))) % (2**31 - 1)
    return np.random.RandomState(seed)

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n
    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)

def _tokens_basic(s: str) -> list[str]:
    r"""
    Basic tokenization for grouping:
    - Lowercases and extracts \\w+ runs.
    - For CJK-like strings (no spaces and contains CJK), falls back to char-level tokens.
    """
    s = "" if pd.isna(s) else str(s)
    s_low = s.lower()
    toks = _TOKEN_RE.findall(s_low)
    if (len(toks) <= 1) and (' ' not in s_low) and CJK_RE.search(s_low):
        # char-level fallback for CJK-like strings without spaces
        toks = list(s_low)
    return toks

def _signature_variants(tokens: list[str], k: int = 1, max_combos: int = 60):
    """
    Yield tuple signatures for <= k removals:
    - the full-token tuple (exact match)
    - all leave-r-out tuples for r = 1..k (cap combinations for r>=2)
    """
    n = len(tokens)
    if n == 0:
        return
    # exact signature
    yield tuple(tokens)
    if k <= 0:
        return

    from itertools import combinations
    # r = 1
    if n >= 1 and k >= 1:
        for i in range(n):
            yield tuple(tokens[:i] + tokens[i+1:])
    # r = 2..k
    for r in range(2, k + 1):
        if n < r:
            break
        count = 0
        for idxs in combinations(range(n), r):
            if count >= max_combos:
                break
            keep = [tokens[i] for i in range(n) if i not in idxs]
            if keep:
                yield tuple(keep)
            count += 1

def build_near_duplicate_groups(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    scope: str = "epark",
    k: int = 1,
    max_combos: int = 60,
    nd_match_mode: str = "or",    # "or" or "and"
    nd_min_len: int = 3,          # below this, only exact signatures (no removals)
    verbose: bool = False,
) -> pd.Series:
    """
    Returns a pd.Series 'nd_group' (near-duplicate group id) of length len(df).

    Grouping rule (token-based):
      - Two rows are grouped if their source OR target sequences match after
        removing <=k tokens (mode="or"), or BOTH sides match under that rule
        (mode="and").
      - For very short sequences (token length < nd_min_len), only exact signatures
        are used to avoid creating huge chains from trivial strings.
    """
    t0 = time.perf_counter()
    # choose which subset to group
    work_idx = df.index
    if scope == "epark":
        mask = df["source"].astype(str).str.split("/").str[0].eq("Formosan-ePark")
        work_idx = df.index[mask]
    elif scope == "none":
        if verbose:
            print("🔗 Near-dup grouping disabled (scope=none).")
        return pd.Series([f"row:{i}" for i in range(len(df))], index=df.index, name="nd_group")

    if verbose:
        print(f"🔗 Near-dup scope '{scope}': {len(work_idx):,} rows to group (of {len(df):,}).")

    # local index mapping
    idx_to_pos = {idx: pos for pos, idx in enumerate(work_idx)}
    uf = UnionFind(len(work_idx))

    # Helper to emit signatures for one cell
    def sigs_for(idx: int, col: str):
        text = df.at[idx, col]
        toks = _tokens_basic(text)
        # small optimization: cap token length we consider (e.g., 40)
        if len(toks) > 40:
            toks = toks[:20] + toks[-20:]
        eff_k = k if len(toks) >= nd_min_len else 0
        for sig in _signature_variants(toks, k=eff_k, max_combos=max_combos):
            if sig:
                yield sig

    t_build = time.perf_counter()
    if nd_match_mode == "or":
        # Map signatures to a representative local position; union on collision.
        for col in (src_col, tgt_col):
            sigmap: dict[tuple[str, ...], int] = {}
            for i, idx in enumerate(work_idx):
                if verbose and i and i % 100000 == 0:
                    print(f"    …processed {i:,} rows for side='{col}'")
                for sig in sigs_for(idx, col):
                    lp = idx_to_pos[idx]
                    prev = sigmap.get(sig)
                    if prev is None:
                        sigmap[sig] = lp
                    else:
                        uf.union(lp, prev)
    else:  # "and" mode
        # Build inverted indices for both sides (sig -> set of local positions)
        sig_src: dict[tuple[str, ...], set[int]] = {}
        sig_tgt: dict[tuple[str, ...], set[int]] = {}
        for i, idx in enumerate(work_idx):
            if verbose and i and i % 100000 == 0:
                print(f"    …processed {i:,} rows building inverted indices")
            lp = idx_to_pos[idx]
            for sig in sigs_for(idx, src_col):
                sig_src.setdefault(sig, set()).add(lp)
            for sig in sigs_for(idx, tgt_col):
                sig_tgt.setdefault(sig, set()).add(lp)
        for i, idx in enumerate(work_idx):
            if verbose and i and i % 100000 == 0:
                print(f"    …AND-match pass {i:,}/{len(work_idx):,}")
            lp = idx_to_pos[idx]
            cands_src: set[int] = set()
            cands_tgt: set[int] = set()
            for sig in sigs_for(idx, src_col):
                cands_src |= sig_src.get(sig, set())
            for sig in sigs_for(idx, tgt_col):
                cands_tgt |= sig_tgt.get(sig, set())
            both = cands_src & cands_tgt
            for other in both:
                if other != lp:
                    uf.union(lp, other)

    t_emit = time.perf_counter()
    # Emit group ids: rows outside scope get unique IDs; grouped rows share root
    nd_group = pd.Series([None]*len(df), index=df.index, name="nd_group")
    for idx in df.index:
        if idx not in idx_to_pos:
            nd_group.at[idx] = f"row:{idx}"   # unique group
        else:
            root = uf.find(idx_to_pos[idx])
            nd_group.at[idx] = f"ndg:{root}"

    # Quick summary
    if verbose:
        sub = nd_group.loc[work_idx]
        sizes = sub.value_counts()
        print(f"🔗 Near-dup built in {t_emit - t0:.2f}s "
              f"(build phase {t_emit - t_build:.2f}s). "
              f"Groups: {len(sizes):,}; median={int(sizes.median())}, "
              f"p95={int(sizes.quantile(0.95))}, max={int(sizes.max())}.")
        print("    Top 5 group sizes:", list(sizes.head(5).values))

    return nd_group

# -------------------------
# Group packing to targets
# -------------------------

def _pin_giant_groups_to_train(corpus_df: pd.DataFrame, group_col: str, max_share: float) -> set[str]:
    """Return set of group ids whose size exceeds max_share of corpus length."""
    n = len(corpus_df)
    if n == 0:
        return set()
    giant = set()
    for gid, idxs in corpus_df.groupby(group_col).groups.items():
        if len(idxs) / n > max_share:
            giant.add(gid)
    if giant:
        print(f"  ⚓ Pinning {len(giant)} giant groups to TRAIN (> {max_share*100:.1f}% each)")
    return giant

def _pack_groups_to_target(
    group_sizes: list[tuple[str, int]],
    target: int,
    rng: np.random.RandomState,
    verbose: bool = False,
) -> tuple[set[str], int, str]:
    """
    Choose a subset of groups with total size close to 'target'.

    Strategy (in order):
      1) All-ones fast path: sample exactly 'target' groups.
      2) Greedy First-Fit-Decreasing for large instances.
      3) Bitset knapsack when small enough to be effective.

    Returns: (selected_group_ids, selected_sum, strategy_used)
    """
    if target <= 0 or not group_sizes:
        return set(), 0, "none"

    sizes = [s for _, s in group_sizes]
    max_s, min_s = max(sizes), min(sizes)
    n_items = len(group_sizes)

    # 1) Fast path: all groups are size 1
    if max_s == 1 and min_s == 1:
        k = min(target, n_items)
        chosen = {g for g, _ in group_sizes[:k]}  # items already RNG-shuffled outside
        return chosen, k, "all-ones"

    # 2) Greedy FFD for big or irregular cases
    if n_items > 1200 or target > 15000 or max_s > 500:
        # Sort by size descending; take if it fits; then small overshoot fix
        items = sorted(group_sizes, key=lambda x: x[1], reverse=True)
        taken, ssum = set(), 0
        for g, s in items:
            if ssum + s <= target:
                taken.add(g); ssum += s
            if ssum == target:
                return taken, ssum, "greedy-ffd"
        # If undershoot, add smallest items until closest (even if overshoot)
        if ssum < target:
            for g, s in sorted(items, key=lambda x: x[1]):
                if g in taken:
                    continue
                taken.add(g); ssum += s
                if ssum >= target:
                    break
        return taken, ssum, "greedy-ffd"

    # 3) Bitset knapsack for smaller problems
    # Keep up to M items (smallest first) to limit CPU/memory
    M = min(n_items, 800)  # allow more than 400 now
    items = sorted(group_sizes, key=lambda x: x[1])[:M]
    sizes = [s for _, s in items]
    gids = [g for g, _ in items]

    bits = 1  # bit 0 set
    snaps = [bits]
    for s in sizes:
        snaps.append(bits)
        bits |= (bits << s)
        # Early stop if exact target reachable
        if (bits >> target) & 1:
            break

    # Best sum <= target
    best = None
    max_len = bits.bit_length()
    if target < max_len:
        for j in range(target, -1, -1):
            if (bits >> j) & 1:
                best = j
                break
    if best is None:
        # minimal overshoot
        best = target
        while ((bits >> best) & 1) == 0:
            best += 1

    # backtrack to recover chosen subset
    chosen = set()
    s = best
    for i in range(len(sizes), 0, -1):
        prev = snaps[i-1]
        if s >= sizes[i-1] and ((prev >> (s - sizes[i-1])) & 1):
            chosen.add(gids[i-1])
            s -= sizes[i-1]

    return chosen, best, "bitset"

# -------------------------
# Splitting
# -------------------------

def split_corpus(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42,
    min_eval_tokens: int = 2,
    group_col: str = "nd_group",
    max_group_share: float = 0.08,  # pin huge groups to train
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Split corpus into train/validation/test sets by corpus, keeping groups intact,
    and using appropriate packing to match per-corpus quotas closely.
    """
    if group_col not in df.columns:
        df = df.copy()
        df[group_col] = [f"row:{i}" for i in range(len(df))]

    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        sys.exit(f"❌  Split ratios must sum to 1.0, got {total_ratio:.3f}")
    if 'source' not in df.columns:
        sys.exit("❌  'source' column required for corpus-based splitting.")

    src_col, tgt_col = _detect_lang_cols(df)

    df = df.copy()
    df['corpus'] = df['source'].astype(str).str.split('/').str[0]
    corpora = df['corpus'].unique()
    print(f"📚  Found {len(corpora)} corpora: {', '.join(corpora)}")

    all_splits = []
    split_summary = {'train': 0, 'validate': 0, 'test': 0}

    def _is_short_pair(row) -> bool:
        return min(_tok_count(row[src_col]), _tok_count(row[tgt_col])) <= min_eval_tokens

    for corpus in corpora:
        t_c0 = time.perf_counter()
        corpus_df = df[df['corpus'] == corpus].copy()
        n_rows = len(corpus_df)
        if n_rows == 0:
            continue

        rng = _rng_for_corpus(random_state, corpus)

        target_val  = int(round(val_ratio * n_rows))
        target_test = int(round(test_ratio * n_rows))
        target_train = n_rows - target_val - target_test

        # Keep near-dup groups intact; pin giants to train
        giant_to_train = _pin_giant_groups_to_train(corpus_df, group_col, max_group_share)

        groups = corpus_df.groupby(group_col)
        # prepare items excluding giants; items are RNG-shuffled for deterministic randomness
        items = [(g, len(groups.get_group(g))) for g in groups.groups.keys() if g not in giant_to_train]
        rng.shuffle(items)

        # choose validation groups
        val_sel, val_sum, strat_val = _pack_groups_to_target(items, target_val, rng, verbose)
        remaining = [(g, s) for (g, s) in items if g not in val_sel]

        # choose test groups from remaining
        test_sel, test_sum, strat_test = _pack_groups_to_target(remaining, target_test, rng, verbose)
        remaining_gids = {g for (g, _) in remaining if g not in test_sel}

        # train: everything else + giants
        train_sel = remaining_gids | set(giant_to_train)

        # materialize splits
        def cat_groups(sel: set[str]) -> pd.DataFrame:
            if not sel:
                return corpus_df.iloc[0:0].copy()
            blocks = [groups.get_group(g) for g in sel]
            return pd.concat(blocks, ignore_index=True)

        val_corpus   = cat_groups(val_sel)
        test_corpus  = cat_groups(test_sel)
        train_corpus = cat_groups(train_sel)

        # Tag and accumulate
        train_corpus['split'] = 'train'
        val_corpus['split']   = 'validate'
        test_corpus['split']  = 'test'

        all_splits.append(pd.concat([train_corpus, val_corpus, test_corpus], ignore_index=True))

        t_c1 = time.perf_counter()
        print(f"  📖  {corpus}: {len(train_corpus):,} train, {len(val_corpus):,} val, {len(test_corpus):,} test "
              f"(targets: {target_train}/{target_val}/{target_test}; "
              f"strategies: val={strat_val}, test={strat_test}; "
              f"time={t_c1 - t_c0:.2f}s)")

        split_summary['train']    += len(train_corpus)
        split_summary['validate'] += len(val_corpus)
        split_summary['test']     += len(test_corpus)

    # After the loop:
    result = pd.concat(all_splits, ignore_index=True).reset_index(drop=True)

    # Sanity checks and reports
    leaks = result.groupby("nd_group")["split"].nunique()
    n_leaks = int((leaks > 1).sum())
    assert n_leaks == 0, f"Found {n_leaks} near-dup groups crossing splits!"

    # Per-corpus deviation from targets
    mix = result.groupby(['corpus','split']).size().unstack(fill_value=0)
    mix['total'] = mix.sum(axis=1)
    with pd.option_context('display.max_rows', None, 'display.width', 120):
        exp_val  = (mix['total'] * val_ratio ).round().astype(int)
        exp_test = (mix['total'] * test_ratio).round().astype(int)
        dev = pd.DataFrame({
            'validate': mix.get('validate', pd.Series(0, index=mix.index)),
            'test': mix.get('test', pd.Series(0, index=mix.index)),
            'target_val': exp_val,
            'target_test': exp_test,
            'dev_val': (mix.get('validate', 0) - exp_val).abs(),
            'dev_test': (mix.get('test', 0) - exp_test).abs(),
        })
        print("🔍 Per-corpus deviation from targets (top 10 by |dev_val|,|dev_test|):")
        print(dev.sort_values(['dev_val','dev_test'], ascending=False).head(10))

    # Shuffle rows globally (deterministic)
    result = result.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    print(f"📊  Total split: {split_summary['train']:,} train, {split_summary['validate']:,} validation, {split_summary['test']:,} test")

    return result

def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed DataFrame to CSV."""
    try:
        df.to_csv(output_path, index=False)
        print(f"✅  Saved processed corpus to {output_path} ({len(df):,} rows)")
    except Exception as e:
        sys.exit(f"❌  Error saving data: {e}")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and split parallel corpus for MT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument("--input", type=Path, required=True, help="Path to input CSV file")
    parser.add_argument("--output", type=Path, required=True, help="Path to save processed CSV file")

    # Filtering options
    parser.add_argument("--min-ratio", type=float, default=0.2, help="Minimum acceptable length ratio (target/source)")
    parser.add_argument("--max-ratio", type=float, default=8.0, help="Maximum acceptable length ratio (target/source)")
    parser.add_argument("--no-clean-text", action="store_true", help="Skip text cleaning (Moses, non-printable chars)")
    parser.add_argument("--no-dedup", action="store_true", help="Skip duplicate removal")
    parser.add_argument("--no-fertility", action="store_true", help="Skip fertility heuristic filtering")

    # Splitting options
    parser.add_argument("--train-ratio", type=float, default=0.80, help="Training set ratio")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Validation set ratio")
    parser.add_argument("--test-ratio", type=float, default=0.10, help="Test set ratio")
    parser.add_argument("--no-split", action="store_true", help="Skip train/val/test splitting")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--min-eval-tokens", type=int, default=-1,
        help="If >=0, demote eval pairs with min(source,target) tokens <= this value to train, then backfill to keep quotas. (Disabled in group-aware mode.)"
    )

    # Performance
    parser.add_argument("--workers", type=int, default=None, help="CPU cores to use (default: all)")
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing")

    # Other
    parser.add_argument("--verbose", action="store_true", help="Show detailed logging for grouping/packing")

    # Near-duplicate grouping & packing controls
    parser.add_argument("--near-dup-scope", choices=["none", "epark", "all"], default="epark",
        help="Build near-duplicate groups and keep them in the same split. 'epark' only groups Formosan-ePark; 'all' groups all corpora; 'none' disables.")
    parser.add_argument("--near-dup-k", type=int, default=1,
        help="Max tokens to remove when generating signatures (≤k). 1 covers one-token diffs; higher captures more but is heavier.")
    parser.add_argument("--near-dup-max-combos", type=int, default=60,
        help="Cap on remove-k combinations per row when k>=2 (safety for long sentences).")
    parser.add_argument("--nd-match-mode", choices=["or","and"], default="or",
        help="Near-dup needs to match on src OR tgt (or), or on BOTH (and). 'and' reduces chaining, but is stricter.")
    parser.add_argument("--nd-min-len", type=int, default=3,
        help="Below this token length, only exact signatures are used (prevents huge chains from trivial strings).")
    parser.add_argument("--max-group-share", type=float, default=0.08,
        help="If a near-dup group > this fraction of its corpus, force it to train (prevents giant eval chunks).")

    args = parser.parse_args()

    # Load
    df = load_data(args.input)

    # Clean
    if not args.no_clean_text:
        if args.no_parallel:
            df = clean_corpus_text(df, args.verbose)
        else:
            df = clean_corpus_text_parallel(df, args.verbose, args.workers)

    # Dedup
    if not args.no_dedup:
        df = remove_exact_duplicates(df)

    # Fertility
    if not args.no_fertility:
        df = apply_fertility_heuristic(df, args.min_ratio, args.max_ratio)

    # Near-dup grouping (after cleaning/dedup/fertility), before splitting
    src_col, tgt_col = _detect_lang_cols(df)
    if args.near_dup_scope != "none":
        print(f"🔗  Building near-duplicate groups "
              f"(scope={args.near_dup_scope}, k={args.near_dup_k}, "
              f"mode={args.nd_match_mode}, min_len={args.nd_min_len}) …")
        df["nd_group"] = build_near_duplicate_groups(
            df, src_col, tgt_col,
            scope=args.near_dup_scope,
            k=args.near_dup_k,
            max_combos=args.near_dup_max_combos,
            nd_match_mode=args.nd_match_mode,
            nd_min_len=args.nd_min_len,
            verbose=args.verbose,
        )
    else:
        df["nd_group"] = [f"row:{i}" for i in range(len(df))]

    # Split
    if not args.no_split:
        df = split_corpus(
            df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
            min_eval_tokens=args.min_eval_tokens,
            group_col="nd_group",
            max_group_share=args.max_group_share,
            verbose=args.verbose,
        )

    # Save
    save_data(df, args.output)

if __name__ == "__main__":
    main()
