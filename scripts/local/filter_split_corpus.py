#!/usr/bin/env python3
"""
STEP 4 (Simplified):

Filter and split a parallel corpus for MT training.

Pipeline:
- **Text cleaning**: Moses punctuation normalization + NFKC + whitespace cleanup
- **Extra scrub + fixes**: drop speaker tags; strip short bracketed glosses; remove known artifacts;
  strip *trailing commas*; fix spaces around punctuation; trim stray leading/ending quotes/brackets; squeeze whitespace
- **Presentation scaffolding drop**: remove "next item / sharing ends" stagey lines (CN + Amis) **and obvious CN headers**
- **Safety drops**:
    • drop rows with **Chinese chars in the Formosan/source column**
    • drop rows where **either side is only punctuation/symbols** (or becomes empty)
    • drop **noisy/jabber** CN targets (dialog/ellipsis/interjection spam) unless --keep-cn-jabber
- **Lexeme detection**: Single-word pairs (≤1 token both sides) → train only
- **Exact deduplication**: Remove duplicate pairs
- **Fertility filtering**: Token-ratio outlier removal (0.2-8.0 for sentences)
- **Simple 80/10/10 split**: Random split with no train/test contamination

Examples
--------
python filter_split_corpus.py --input corpus.csv --output corpus_ready.csv
python filter_split_corpus.py --input ami_zh.csv --output ami_zh_processed.csv
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Optional: Moses punctuation normalization (like NLLB preprocessing)
try:
    from sacremoses import MosesPunctNormalizer
    HAVE_SACREMOSES = True
except Exception:
    HAVE_SACREMOSES = False

# ──────────────────────────────────────────────────────────────────────────────
# Basic text cleaning utils
# ──────────────────────────────────────────────────────────────────────────────

ASCII_CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
WHITESPACE_RE = re.compile(r"\s+")
# CJK block + ext + compatibility ideographs
CJK_RE = re.compile(r"[\u3400-\u9FFF\U00020000-\U0002B81F\uF900-\uFAFF]")
# Sentence-ending punctuation (ASCII and Chinese), allow trailing quotes/brackets
SENT_END_RE = re.compile(r'[.!?。！？…]+(?:["”」』）】》]*)$')
MANY_DELIMS_RE = re.compile(r"[\/;|]{1,}|,{2,}")

# Hints in 'source' paths that strongly suggest lexeme/wordlist material
LEXEME_SOURCE_HINTS = (
    "學習詞表", "wordlist", "vocab", "dictionary", "dict", "詞表", "lexicon",
    "VirginiaFeyDictionary", "ILRDF_Dicts", "Dicts", "Dict/", "Dict-",
)

RESERVED_META_COLS = {"source", "kindOf", "split", "nd_group", "corpus", "dialect"}

# ── Extra scrub & small fixes (speaker tags, bracketed glosses, artifacts, commas, spacing, stray brackets/quotes) ──
SPEAKER_TAG_RE = re.compile(r"^[A-Z][：:]\s*")  # leading "A:" / "A：" + spaces
META_GLOSS_RE  = re.compile(r"（[^）]{1,10}）|\([^)]{1,10}\)")  # full/half width () content ≤10 chars
ARTIFACT_RE    = re.compile(r"(全文紀錄|中文紀錄|女子全名)")
TRAILING_COMMA_RE = re.compile(r"[，,]+\s*$")

# Spacing fixes (remove spaces **before** punctuation / around brackets & quotes)
SPACE_BEFORE_ASCII_PUNCT_RE = re.compile(r"\s+([,.;:!?%])")
SPACE_BEFORE_CJK_PUNCT_RE   = re.compile(r"\s+([，。！？；：、）】》」』％])")
SPACE_AFTER_OPEN_BRACKET_RE = re.compile(r"([（(【《「『“])\s+")
SPACE_BEFORE_CLOSE_BRKT_RE  = re.compile(r"\s+([)）】》」』”])")
LEADING_STRAY_CLOSERS_RE    = re.compile(r'^[\)\]\}」』】》”]+')
TRAILING_STRAY_OPENERS_RE   = re.compile(r'[\(\[\{（「『【《“]+$')

# --- CN-side "jabber" detection (dialog/interjection/ellipsis spam) ---
ELLIPSIS_RE = re.compile(r"(…|\.{3,}|。{2,}|！{2,}|？{2,})")
INTERJ_RE = re.compile(r"(哈|啊|喔|哦|嘿|啦|嘛|呢){2,}")
SHORT_QUOTED_RE = re.compile(r'^[「『“\(][^」』”\)]{0,12}[」』”\)]?$')

def zh_looks_jabber(s: str) -> bool:
    """
    True if a Chinese string looks like performative dialog or punctuation spam:
    - high punctuation ratio,
    - repeated ellipses / !! / ??,
    - long runs of interjections,
    - very short quoted gasp/exclamation.
    """
    s = "" if s is None else str(s).strip()
    if not s:
        return True
    punct = sum(1 for ch in s if unicodedata.category(ch)[0] in ("P", "S"))
    ratio = punct / max(len(s), 1)
    if ratio > 0.35:
        return True
    if ELLIPSIS_RE.search(s):
        return True
    if INTERJ_RE.search(s):
        return True
    if SHORT_QUOTED_RE.match(s):
        return True
    return False


def extra_scrub(text: str) -> str:
    """
    One-pass scrub for eval/train, now also doing tiny punctuation tidy-ups:
      - drop speaker tags, short bracketed glosses, known artifacts
      - drop trailing commas (ASCII/Chinese)
      - trim stray leading closers / trailing openers
      - remove spaces before punctuation; tighten around brackets/quotes
      - squeeze whitespace
    """
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    # 1) drop leading speaker tags like "A:" / "B：" (Latin capital + colon)
    text = SPEAKER_TAG_RE.sub("", text)

    # 2) strip short bracketed meta-glosses e.g. （推薦） / (答覆)  etc.
    text = META_GLOSS_RE.sub("", text)

    # 3) kill obvious artifact tokens anywhere
    text = ARTIFACT_RE.sub("", text)

    # 4) drop commas at end of sentence (ASCII/Chinese)
    text = TRAILING_COMMA_RE.sub("", text)

    # 5) trim stray closers/openers at edges
    text = LEADING_STRAY_CLOSERS_RE.sub("", text)
    text = TRAILING_STRAY_OPENERS_RE.sub("", text)

    # 6) spacing around punctuation / brackets / quotes
    text = SPACE_BEFORE_ASCII_PUNCT_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CJK_PUNCT_RE.sub(r"\1", text)
    text = SPACE_AFTER_OPEN_BRACKET_RE.sub(r"\1", text)
    text = SPACE_BEFORE_CLOSE_BRKT_RE.sub(r"\1", text)

    # 7) squeeze leftover whitespace
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def replace_nonprinting(s: str) -> str:
    """
    Remove/replace non-printable chars:
    - Drop ASCII control chars
    - Replace Unicode category 'C*' (Other: Cc, Cf, Cs, Co, Cn) with a space,
      but keep caret ^ and apostrophe ' (they matter in Amis orthography).
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = ASCII_CTRL_RE.sub(" ", s)
    return "".join((" " if (unicodedata.category(ch).startswith("C") and ch not in "^'") else ch) for ch in s)

def normalize_text(text: str, mpn: MosesPunctNormalizer | None) -> str:
    """Moses (if available) + NFKC + whitespace squeeze."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    if mpn is not None:
        # MosesPunctNormalizer has compiled substitutions we can apply
        for pattern, sub in mpn.substitutions:
            text = pattern.sub(sub, text)
    text = replace_nonprinting(text)
    text = unicodedata.normalize("NFKC", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def tok_count(s: str) -> int:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip()
    return 0 if not s else len(s.split())

def _cjk_len(s: str) -> int:
    """Count CJK codepoints (helps classify zh sentences that lack spaces)."""
    s = "" if s is None else str(s)
    return sum(1 for ch in s if CJK_RE.match(ch))

# Only-punctuation checker (P/S categories allowed; whitespace ignored)
def _is_only_punct_or_symbols(s: str) -> bool:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = s.strip()
    if not s:
        return True  # treat empty as bad after cleaning
    for ch in s:
        if ch.isspace():
            continue
        cat = unicodedata.category(ch)
        # Letters (L*), Numbers (N*), Marks (M*) => NOT only punctuation
        if cat[0] in ("L", "N", "M"):
            return False
        # Otherwise P=punct, S=symbol, Z=separators are allowed to continue
        if cat[0] not in ("P", "S", "Z"):
            # Any other weird categories => consider not-only-punct
            return False
    return True

# ──────────────────────────────────────────────────────────────────────────────
# Column detection
# ──────────────────────────────────────────────────────────────────────────────

def detect_lang_cols(df: pd.DataFrame) -> tuple[str, str]:
    """
    Prefer well-known names if present; otherwise fall back to first two object columns
    that are not metadata.
    """
    preferred_pairs = [
        ("ami", "english"),
        ("formosan_sentence", "english_sentence"),
        ("formosan_sentence", "chinese_sentence"),
        ("source_text", "target_text"),
    ]
    cols = {c.lower(): c for c in df.columns}
    for a, b in preferred_pairs:
        if a in cols and b in cols:
            return cols[a], cols[b]

    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in RESERVED_META_COLS]
    if len(obj_cols) < 2:
        sys.exit("❌ Need at least two non-metadata string columns for parallel text.")
    return obj_cols[0], obj_cols[1]

# ──────────────────────────────────────────────────────────────────────────────
# Lexeme / sentence classification & cleaning heuristics
# ──────────────────────────────────────────────────────────────────────────────

def looks_like_lexeme(src: str, tgt: str, source_path: str) -> bool:
    """
    Simple lexeme detection: only single-word pairs (≤1 token both sides) are lexemes.
    Everything else is treated as a sentence and split normally 80/10/10.
    """
    s = src if isinstance(src, str) else ""
    t = tgt if isinstance(tgt, str) else ""

    stoks = tok_count(s)
    ttoks = tok_count(t)

    if stoks <= 1 and ttoks <= 1:
        return True
    return False

def is_listy_sentence_like(s: str) -> bool:
    """
    Returns True if text looks like a "list of variants" rather than a sentence.
    Conservative for Chinese so normal prose isn't dropped.
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)

    if SENT_END_RE.search(s):
        return False
    if CJK_RE.search(s) and len(s) >= 20:
        return False

    delim_hits = s.count("/") + s.count(";") + s.count("|") + s.count(",")
    return (delim_hits >= 3) and (len(s) <= 40)

def expand_lexeme_slash_variants(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    """
    For rows flagged as lexemes:
      - If exactly one side contains slash variants (e.g., "a/b/c"), expand into multiple rows.
      - If BOTH sides contain '/', we leave as-is to avoid combinatorial explosion.
      - For English synonym lists like "X, Y, Z", keep only the first by default.
    """
    rows: List[dict] = []
    for _, r in df.iterrows():
        s = r[src_col]
        t = r[tgt_col]
        is_lex = r.get("row_type", "") == "lexeme"
        if not is_lex:
            rows.append(r.to_dict())
            continue

        s_has = isinstance(s, str) and "/" in s
        t_has = isinstance(t, str) and "/" in t

        if isinstance(t, str) and "," in t and tok_count(t) <= 6:
            t = t.split(",")[0].strip()

        if s_has and not t_has:
            parts = [p.strip() for p in s.split("/") if p.strip()]
            if 1 < len(parts) <= 6:
                for p in parts:
                    rr = r.to_dict()
                    rr[src_col] = p
                    rr[tgt_col] = t
                    rows.append(rr)
                continue

        if t_has and not s_has:
            parts = [p.strip() for p in t.split("/") if p.strip()]
            if 1 < len(parts) <= 6:
                for p in parts:
                    rr = r.to_dict()
                    rr[src_col] = s
                    rr[tgt_col] = p
                    rows.append(rr)
                continue

        r2 = r.to_dict()
        r2[src_col] = s
        r2[tgt_col] = t
        rows.append(r2)

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# Cleaning pipeline (parallel-capable)
# ──────────────────────────────────────────────────────────────────────────────

G_MPN = None

def _init_worker():
    global G_MPN
    if HAVE_SACREMOSES:
        mpn = MosesPunctNormalizer(lang="en")
        mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
        G_MPN = mpn
    else:
        G_MPN = None

def _process_chunk_texts(texts: List[str]) -> List[str]:
    mpn = G_MPN
    return [normalize_text(t, mpn) for t in texts]

def clean_text_columns(df: pd.DataFrame, cols: List[str], workers: int | None) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 1000 or (workers is not None and workers <= 1):
        # single-thread
        mpn = None
        if HAVE_SACREMOSES:
            mpn = MosesPunctNormalizer(lang="en")
            mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
        for c in cols:
            print(f"🧼  Cleaning column '{c}' (Moses+NFKC)...")
            df[c] = [normalize_text(x, mpn) for x in tqdm(df[c].tolist(), desc=f"clean:{c}")]
        return df

    # parallel
    n_workers = workers or mp.cpu_count()
    print(f"🚀  Using {n_workers} CPU cores for text cleaning")
    for c in cols:
        print(f"🧼  Cleaning column '{c}' (Moses+NFKC, parallel)...")
        vals = df[c].tolist()
        # chunk into ~2*n_workers pieces
        n_chunks = max(2 * n_workers, 1)
        size = max(1000, len(vals) // n_chunks)
        chunks = [vals[i : i + size] for i in range(0, len(vals), size)]
        with mp.Pool(n_workers, initializer=_init_worker) as pool:
            out_chunks = list(tqdm(pool.imap(_process_chunk_texts, chunks), total=len(chunks), desc=f"clean:{c}"))
        df[c] = [y for ch in out_chunks for y in ch]
    return df

def remove_exact_duplicates(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    n0 = len(df)
    df2 = df.drop_duplicates(subset=[src_col, tgt_col], keep="first")
    print(f"🗑️  Removed {n0 - len(df2):,} exact duplicate pairs")
    return df2

# ──────────────────────────────────────────────────────────────────────────────
# Fertility (length-ratio) filtering
# ──────────────────────────────────────────────────────────────────────────────

def apply_fertility(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    min_ratio_sent: float,
    max_ratio_sent: float,
    min_ratio_lex: float,
    max_ratio_lex: float,
) -> pd.DataFrame:
    """
    Use token-count ratio primarily, with char-length as a fallback for super-short rows.
    Lexemes have looser bounds by default.
    """
    df = df.copy()
    s_tok = df[src_col].astype(str).map(tok_count)
    t_tok = df[tgt_col].astype(str).map(tok_count)

    tok_ratio = (t_tok.replace(0, np.nan) / s_tok.replace(0, np.nan)).astype(float)
    s_len = df[src_col].astype(str).map(len).replace(0, np.nan)
    t_len = df[tgt_col].astype(str).map(len).replace(0, np.nan)
    ch_ratio = (t_len / s_len).astype(float)

    use_tok = (s_tok >= 2) & (t_tok >= 2)
    ratio = tok_ratio.where(use_tok, ch_ratio)

    is_lex = df.get("row_type", "").eq("lexeme")
    lo = np.where(is_lex, min_ratio_lex, min_ratio_sent)
    hi = np.where(is_lex, max_ratio_lex, max_ratio_sent)

    ok = (ratio >= lo) & (ratio <= hi)
    kept = df[ok.fillna(False)]
    print(f"📏  Fertility filter removed {len(df) - len(kept):,} pairs "
          f"(sent bounds [{min_ratio_sent},{max_ratio_sent}] | "
          f"lex bounds [{min_ratio_lex},{max_ratio_lex}])")
    return kept

# ──────────────────────────────────────────────────────────────────────────────
# Near-duplicate grouping (edit distance 1) — still available but off by default
# ──────────────────────────────────────────────────────────────────────────────

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[rb] < self.r[ra]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def _char_ngrams(s: str, n: int = 3) -> set:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    if len(s) <= n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}

def _edit1(s1: str, s2: str) -> bool:
    if s1 == s2:
        return True
    if abs(len(s1) - len(s2)) > 1:
        return False
    if len(s1) == len(s2):
        diffs = sum(a != b for a, b in zip(s1, s2))
        return diffs == 1
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    i = 0
    hit = False
    for j in range(len(s2)):
        if i < len(s1) and s1[i] == s2[j]:
            i += 1
        elif not hit:
            hit = True
        else:
            return False
    return True

def _compare_bucket(items: List[Tuple[int, str]]) -> List[Tuple[int, int]]:
    pairs = []
    n = len(items)
    if n <= 1:
        return pairs
    index = {}
    for idx, (i, txt) in enumerate(items):
        for ng in _char_ngrams(txt, 3):
            index.setdefault(ng, []).append(idx)
    cand = set()
    for idx, (i, txt) in enumerate(items):
        for ng in _char_ngrams(txt, 3):
            for j in index.get(ng, []):
                if j > idx:
                    cand.add((idx, j))
    for a, b in cand:
        i1, t1 = items[a]
        i2, t2 = items[b]
        if _edit1(t1, t2):
            pairs.append((i1, i2))
    return pairs

def build_near_dups(df: pd.DataFrame, src_col: str, tgt_col: str, workers: int | None) -> pd.Series:
    print("🔗  Grouping near-duplicates (edit distance 1 on src or tgt)…")
    n = len(df)
    uf = UnionFind(n)

    def buckets(col: str):
        bylen = {}
        arr = df[col].astype(str).tolist()
        for i, s in enumerate(arr):
            L = len(s)
            for d in (-1, 0, 1):
                bylen.setdefault(max(0, L + d), []).append((i, s))
        return list(bylen.values())

    for label, bks in (("src", buckets(src_col)), ("tgt", buckets(tgt_col))):
        tasks = [bk for bk in bks if len(bk) > 1]
        if len(df) >= 2000 and (workers is None or workers > 1):
            n_workers = workers or min(mp.cpu_count(), 16)
            with mp.Pool(n_workers) as pool:
                for res in tqdm(pool.imap_unordered(_compare_bucket, tasks), total=len(tasks), desc=f"buckets:{label}"):
                    for i, j in res:
                        uf.union(i, j)
        else:
            for bk in tqdm(tasks, desc=f"buckets:{label}"):
                for i, j in _compare_bucket(bk):
                    uf.union(i, j)

    groups = pd.Series([f"ndg:{uf.find(i)}" for i in range(n)], index=df.index, name="nd_group")
    sizes = groups.value_counts()
    print(f"✅  Near-dup groups: {len(sizes):,} | median={int(sizes.median())} | max={int(sizes.max())}")
    return groups

# ──────────────────────────────────────────────────────────────────────────────
# Presentation scaffolding drop (CN + Amis) + Obvious CN Headers
# ──────────────────────────────────────────────────────────────────────────────

CN_STAGEY_PAT = re.compile(r"(換下一個(?:說)?|我(?:分享|說明)到這裡)\s*$")
AMI_STAGEY_PAT = re.compile(
    r"(?:\bro(?:mato|ma)\b|\bsowal ako\b|\bpisoykay\b|\bpisowal ako\b|\bmahaen ko\b)",
    flags=re.IGNORECASE,
)

# Obvious CN headers: page markers, "人物生平 - -", "YYYY年 - -", etc.
CN_HEADER_PAT1 = re.compile(r"^\s*\d{1,3}\s*(?:页|頁)\b")
CN_HEADER_PAT2 = re.compile(r"人物生平\s*[-—–]\s*[-—–]")
CN_HEADER_PAT3 = re.compile(r"^\s*\d{3,4}\s*年\b.*[-—–]\s*[-—–]")  # e.g., "1918年 - - 回首百年前"
CN_HEADER_PAT4 = re.compile(r"^\s*(?:第\s*)?\d+\s*(?:[课章節讲講篇讲]\b|[）).、])\s*$")  # bare enumerations

def _looks_stagey_or_aside_or_header(s: str, assume_cn: bool) -> bool:
    s = "" if s is None else str(s)
    if not s:
        return False
    if assume_cn:
        if CN_STAGEY_PAT.search(s):
            return True
        # Headers: only consider lines with CJK to avoid killing e.g. filenames
        if _cjk_len(s) > 0 and (
            CN_HEADER_PAT1.search(s) or CN_HEADER_PAT2.search(s) or CN_HEADER_PAT3.search(s) or CN_HEADER_PAT4.search(s)
        ):
            return True
        return False
    return bool(AMI_STAGEY_PAT.search(s))

def drop_stagey_rows(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    # Heuristic: target is Chinese if CJK-rich; source is Amis if not
    cjk_tgt = df[tgt_col].astype(str).map(_cjk_len) > 0
    cjk_src = df[src_col].astype(str).map(_cjk_len) > 0

    mask_drop = (
        df[tgt_col].astype(str).where(cjk_tgt, "").map(lambda s: _looks_stagey_or_aside_or_header(s, True)) |
        df[src_col].astype(str).where(~cjk_src, "").map(lambda s: _looks_stagey_or_aside_or_header(s, False))
    )
    n = int(mask_drop.sum())
    if n:
        print(f"🧹  Dropping {n:,} presentation/asides/headers (e.g., 換下一個/我分享到這裡/人物生平 - -/romato/sowal ako)")
    return df[~mask_drop].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Splitting (per-source), with lexeme routing and cap
# ──────────────────────────────────────────────────────────────────────────────

def extract_source_id(p: str) -> str:
    parts = str(p).split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}/"
    return str(p)

def split_by_source(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    include_lexemes_in_eval: bool,
    max_lexeme_frac_train: float | None = None,
) -> pd.DataFrame:
    """
    Simple 80/10/10 split with no train/test contamination.
    - Lexemes (single-word pairs) stay in train only
    - Everything else split randomly 80/10/10
    """
    train_ratio = 1.0 - val_ratio - test_ratio
    print(f"\n📂  Target split: {train_ratio*100:.0f}% train / {val_ratio*100:.0f}% val / {test_ratio*100:.0f}% test")
    df = df.copy()
    rng = np.random.RandomState(random_state)

    is_lexeme = df.get("row_type", "") == "lexeme"
    lexeme_df = df[is_lexeme].copy()
    sentence_df = df[~is_lexeme].copy()

    print(f"  Lexemes: {len(lexeme_df):,} (all → train)")
    print(f"  Sentences: {len(sentence_df):,} (split {train_ratio*100:.0f}/{val_ratio*100:.0f}/{test_ratio*100:.0f})")

    lexeme_df["split"] = "train"

    n_sent = len(sentence_df)
    indices = rng.permutation(n_sent)

    n_train = int(n_sent * train_ratio)
    n_val = int(n_sent * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    sentence_df["split"] = "train"
    sentence_df.iloc[val_idx, sentence_df.columns.get_loc("split")] = "validate"
    sentence_df.iloc[test_idx, sentence_df.columns.get_loc("split")] = "test"

    df = pd.concat([lexeme_df, sentence_df], ignore_index=True)

    counts = df["split"].value_counts()
    total = len(df)
    print(f"\n📊  Final split: "
          f"{counts.get('train',0):,} train ({counts.get('train',0)/total*100:.1f}%), "
          f"{counts.get('validate',0):,} val ({counts.get('validate',0)/total*100:.1f}%), "
          f"{counts.get('test',0):,} test ({counts.get('test',0)/total*100:.1f}%)")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"✅  Loaded {len(df):,} rows from {path}")
        return df
    except Exception as e:
        sys.exit(f"❌ Error loading data: {e}")

def save_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
        print(f"✅  Saved {len(df):,} rows → {path}")
    except Exception as e:
        sys.exit(f"❌ Error saving data: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter and split parallel corpus for MT training (lexeme-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # I/O
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)

    # Cleaning & heuristics
    ap.add_argument("--no-clean-text", action="store_true", help="Skip Moses/NFKC cleaning")
    ap.add_argument("--workers", type=int, default=None, help="CPU cores for cleaning (default: all)")
    ap.add_argument("--min-sent-tokens", type=int, default=1, help="Minimum tokens on BOTH sides to keep (default: 1)")
    ap.add_argument("--drop-listy-sentences", action="store_true", help="Drop sentence-like rows that resemble lists")

    # New: controls for extra safety drops
    ap.add_argument("--keep-stagey", action="store_true", help="Keep presentation/asides/header lines instead of dropping")
    ap.add_argument("--keep-cjk-in-src", action="store_true", help="Keep rows even if source/Formosan has Chinese chars")
    ap.add_argument("--keep-punct-only", action="store_true", help="Keep rows that are only punctuation/symbols")

    # Slash variant expansion (for lexemes only)
    ap.add_argument("--expand-lexeme-slashes", action="store_true", help="Split 'a/b'→'a' & 'b' for lexeme rows")

    # Fertility (length-ratio)
    ap.add_argument("--min-ratio", type=float, default=0.2, help="Min target/source ratio for SENTENCES")
    ap.add_argument("--max-ratio", type=float, default=8.0, help="Max target/source ratio for SENTENCES")
    ap.add_argument("--lexeme-min-ratio", type=float, default=0.05, help="Min ratio for LEXEMES")
    ap.add_argument("--lexeme-max-ratio", type=float, default=20.0, help="Max ratio for LEXEMES")
    ap.add_argument("--no-fertility", action="store_true", help="Skip fertility filtering")

    # Dedup & near-dups
    ap.add_argument("--no-dedup", action="store_true", help="Skip exact duplicate removal")
    ap.add_argument("--no-near-dup", action="store_true", help="Skip near-duplicate grouping")

    # Splitting
    ap.add_argument("--no-split", action="store_true", help="Do not create train/validate/test splits")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--include-lexemes-in-eval", action="store_true", help="Allow lexemes in val/test (default: False)")
    ap.add_argument("--max-lexeme-frac", type=float, default=None, help="DEPRECATED - not used in simplified split")

    ap.add_argument(
        "--keep-cn-jabber",
        action="store_true",
        help="Keep Chinese targets that look like dialog/interjection/ellipsis spam (default: drop them)"
    )

    args = ap.parse_args()

    print("=" * 80)
    print("🚀  MT Corpus Filtering & Splitting (Lexeme-Aware)")
    print("=" * 80)

    # Load
    df = load_csv(args.input)

    # Detect language columns early (and clean only those)
    src_col, tgt_col = detect_lang_cols(df)
    print(f"🗂️  Language columns: {src_col} ↔ {tgt_col}")

    # Drop rows with "(No Record)" in either column
    print("🧹  Removing rows with '(No Record)' translations...")
    n_before = len(df)
    mask_no_record = (
        df[src_col].astype(str).str.strip().eq("(No Record)") |
        df[tgt_col].astype(str).str.strip().eq("(No Record)")
    )
    df = df[~mask_no_record].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"🗑️  Removed {n_dropped:,} rows with '(No Record)'")

    # Text cleaning
    if not args.no_clean_text:
        df = clean_text_columns(df, [src_col, tgt_col], workers=args.workers)

    # One-time scrub pass (always run, even if --no-clean-text was set)
    print("🧽  Scrubbing meta + fixing punctuation/spacing (tags, short ()-glosses, artifacts, trailing commas, spaces, stray brackets/quotes)…")
    for c in (src_col, tgt_col):
        df[c] = df[c].astype(str).map(extra_scrub)

    # Drop presentation scaffolding (CN + Amis) and obvious CN headers
    if not args.keep_stagey:
        df = drop_stagey_rows(df, src_col, tgt_col)

    # Drop rows where *source/Formosan* contains Chinese chars
    if not args.keep_cjk_in_src:
        m_src_cjk = df[src_col].astype(str).map(_cjk_len) > 0
        n = int(m_src_cjk.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows with Chinese characters in source/Formosan column '{src_col}'")
        df = df[~m_src_cjk].reset_index(drop=True)

    # Drop rows that are only punctuation/symbols or empty on either side
    if not args.keep_punct_only:
        m_bad_src = df[src_col].map(_is_only_punct_or_symbols)
        m_bad_tgt = df[tgt_col].map(_is_only_punct_or_symbols)
        m_bad = m_bad_src | m_bad_tgt
        n = int(m_bad.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows that are empty or only punctuation/symbols on either side")
        df = df[~m_bad].reset_index(drop=True)

    # Drop 'jabber' CN targets (performative dialog / ellipses / interjection spam)
    if not args.keep_cn_jabber:
        # Only consider targets that are actually Chinese (have CJK)
        mask_tgt_is_cn = df[tgt_col].astype(str).map(lambda s: _cjk_len(s) >= 1)
        mask_cn_jabber = mask_tgt_is_cn & df[tgt_col].astype(str).map(zh_looks_jabber)
        n = int(mask_cn_jabber.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows: noisy/performative Chinese targets (dialog/ellipsis/interjections)")
        df = df[~mask_cn_jabber].reset_index(drop=True)

    # Classify row type (lexeme vs sentence)
    print("🔎  Classifying rows (lexeme vs sentence)…")
    src_series = df[src_col].astype(str)
    tgt_series = df[tgt_col].astype(str)
    src_paths = df["source"].astype(str) if "source" in df.columns else pd.Series([""] * len(df))

    kind_col = "kindOf" if "kindOf" in df.columns else None
    row_types: List[str] = []
    for i, (s, t, sp) in enumerate(tqdm(zip(src_series, tgt_series, src_paths), total=len(df), desc="classify")):
        if kind_col and str(df[kind_col].iat[i]).strip().lower() == "lexeme":
            row_types.append("lexeme")
        else:
            row_types.append("lexeme" if looks_like_lexeme(s, t, sp) else "sentence")
    df["row_type"] = row_types

    # Optionally drop "listy" rows from sentences (kept for lexemes)
    if args.drop_listy_sentences:
        mask_listy = df["row_type"].eq("sentence") & (
            df[src_col].map(is_listy_sentence_like) | df[tgt_col].map(is_listy_sentence_like)
        )
        n_drop = int(mask_listy.sum())
        if n_drop:
            print(f"🧹  Dropping {n_drop:,} list-like rows from sentences")
        df = df[~mask_listy].reset_index(drop=True)

    # Enforce minimum tokens for sentences (both sides)
    min_tok = max(0, int(args.min_sent_tokens))
    if min_tok > 0:
        s_tok = df[src_col].map(tok_count)
        t_tok = df[tgt_col].map(tok_count)
        mask_bad_sent = df["row_type"].eq("sentence") & ~((s_tok >= min_tok) & (t_tok >= min_tok))
        n_drop = int(mask_bad_sent.sum())
        if n_drop:
            print(f"🧹  Dropping {n_drop:,} short sentence rows (<{min_tok} toks on either side)")
            df = df[~mask_bad_sent].reset_index(drop=True)

    # Expand slash variants for lexemes (optional)
    if args.expand_lexeme_slashes:
        print("➕  Expanding slash variants for lexeme rows")
        df = expand_lexeme_slash_variants(df, src_col, tgt_col)

    # Deduplicate exact pairs (unchanged)
    if not args.no_dedup:
        df = remove_exact_duplicates(df, src_col, tgt_col)

    # Fertility filtering
    if not args.no_fertility:
        df = apply_fertility(
            df,
            src_col,
            tgt_col,
            min_ratio_sent=args.min_ratio,
            max_ratio_sent=args.max_ratio,
            min_ratio_lex=args.lexeme_min_ratio,
            max_ratio_lex=args.lexeme_max_ratio,
        )

    # Near-duplicate grouping (disabled by default)
    # if not args.no_near_dup:
    #     df["nd_group"] = build_near_dups(df, src_col, tgt_col, workers=args.workers)
    # else:
    #     df["nd_group"] = [f"row:{i}" for i in range(len(df))]

    # Split
    if not args.no_split:
        df = split_by_source(
            df,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
            include_lexemes_in_eval=args.include_lexemes_in_eval,
            max_lexeme_frac_train=float(args.max_lexeme_frac) if args.max_lexeme_frac is not None else None,
        )

    # Save
    print("=" * 80)
    save_csv(df, args.output)
    print("=" * 80)
    print("✅  Pipeline complete!")

if __name__ == "__main__":
    main()
