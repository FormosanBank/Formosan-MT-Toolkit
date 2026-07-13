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
    • drop rows with **Han/Kana/Hangul chars in the Formosan/source column**
    • drop rows where **either side is only punctuation/symbols** (or becomes empty)
    • drop bracket-only grammatical labels and short noisy CN targets unless --keep-cn-jabber
- **Lexeme detection**: preserve XML `row_type=lexeme`, known vocabulary/dictionary paths, and compact one-token gloss rows → train only
- **Exact deduplication**: Remove duplicate pairs
- **Fertility filtering**: language-aware target/source unit-ratio outlier removal (0.2-8.0 for sentences)
- **Equivalence-group 80/10/10 split**:
    - Build groups where any rows sharing the same source OR the same target after skeleton normalization
      are forced into the **same split** (train / val / test).
    - This prevents one-to-many and many-to-one (Formosan ↔ EN/ZH) clusters
      from leaking across splits and inflating BLEU.

Examples
--------
python filter_split_corpus.py --input corpus.csv --output corpus_ready.csv
python filter_split_corpus.py --input ami_zh.csv --output ami_zh_processed.csv
"""

from __future__ import annotations

import argparse
import json
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
NON_FORMOSAN_SOURCE_SCRIPT_RE = re.compile(r"[\u3040-\u30FF\uAC00-\uD7AF]")
# Sentence-ending punctuation (ASCII and Chinese), allow trailing quotes/brackets
SENT_END_RE = re.compile(r'[.!?。！？…]+(?:["”」』）】》]*)$')
MANY_DELIMS_RE = re.compile(r"[\/;|]{1,}|,{2,}")

# Hints in 'source' paths that strongly suggest lexeme/wordlist material
LEXEME_SOURCE_HINTS = (
    "學習詞表", "wordlist", "vocab", "dictionary", "dict", "詞表", "lexicon",
    "VirginiaFeyDictionary", "ILRDF_Dicts", "Dicts", "Dict/", "Dict-",
    "xue_xi_ci_biao_learning_vocabulary",
)

RESERVED_META_COLS = {
    "source",
    "kindOf",
    "split",
    "nd_group",
    "group_id",
    "corpus",
    "dialect",
    "row_type",
    "xml_id",
}

# ── Extra scrub & small fixes (speaker tags, bracketed glosses, artifacts, commas, spacing, stray brackets/quotes) ──
SPEAKER_TAG_RE = re.compile(r"^[A-Z][：:]\s*")  # leading "A:" / "A：" + spaces
META_GLOSS_RE  = re.compile(r"（[^）]{1,10}）|\([^)]{1,10}\)")  # full/half width () content ≤10 chars
ARTIFACT_RE    = re.compile(r"(全文紀錄|中文紀錄|女子全名)")
TRAILING_COMMA_RE = re.compile(r"[，,]+\s*$")
MISSING_TRANSLATION_RE = re.compile(
    r"(\[?\s*translation\s+missing\s*\]?|\(no\s+record\)|no\s+record|無翻譯|缺翻譯)",
    flags=re.IGNORECASE,
)
REDACTION_RE = re.compile(r"\bX{3,}\b|X{4,}")

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
TARGET_BRACKET_META_RE = re.compile(
    r"^\s*[\[\【(（]\s*"
    r"(?:介|虛|虚|名|動|动|形|副|代|助|連|连|量|嘆|叹|語助|语助|語氣|语气|"
    r"疑問|疑问|感嘆|感叹|pos|particle|prep(?:osition)?|noun|verb|adj(?:ective)?|adv(?:erb)?)"
    r"\s*[\]\】)）]\s*$",
    flags=re.IGNORECASE,
)
SHORT_BRACKETED_LABEL_RE = re.compile(r"^\s*[\[\【(（]\s*([^\]\】)）]{1,6})\s*[\]\】)）]\s*$")


class DropReporter:
    """Small audit trail for destructive filtering stages."""

    def __init__(self, sample_limit: int = 100):
        self.sample_limit = max(0, int(sample_limit))
        self.reason_counts: dict[str, int] = {}
        self.samples: list[dict] = []

    def record(
        self,
        reason: str,
        df: pd.DataFrame,
        mask: pd.Series,
        src_col: str,
        tgt_col: str,
    ) -> int:
        mask = mask.reindex(df.index, fill_value=False).astype(bool)
        count = int(mask.sum())
        if count == 0:
            return 0

        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + count
        remaining = self.sample_limit - sum(1 for row in self.samples if row.get("drop_reason") == reason)
        if remaining <= 0:
            return count

        sample_cols = [
            src_col,
            tgt_col,
            "source",
            "kindOf",
            "dialect",
            "row_type",
            "xml_id",
        ]
        available = [col for col in sample_cols if col in df.columns]
        for row in df.loc[mask, available].head(remaining).to_dict("records"):
            row = {"drop_reason": reason, **row}
            self.samples.append(row)
        return count

    def write(
        self,
        report_dir: Path,
        *,
        input_path: Path,
        output_path: Path,
        initial_rows: int,
        final_df: pd.DataFrame,
        src_col: str,
        tgt_col: str,
    ) -> None:
        report_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "source_column": src_col,
            "target_column": tgt_col,
            "initial_rows": int(initial_rows),
            "final_rows": int(len(final_df)),
            "drop_counts": self.reason_counts,
            "row_type_counts": {
                str(k): int(v)
                for k, v in final_df.get("row_type", pd.Series(dtype=object)).value_counts(dropna=False).items()
            },
            "split_counts": {
                str(k): int(v)
                for k, v in final_df.get("split", pd.Series(dtype=object)).value_counts(dropna=False).items()
            },
        }
        (report_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if self.samples:
            pd.DataFrame(self.samples).to_csv(report_dir / "reject_samples.csv", index=False)
        print(f"🧾  Filter report: {report_dir}")

def zh_looks_jabber(s: str) -> bool:
    """
    True if a Chinese string looks like performative dialog or punctuation spam:
    - high punctuation ratio,
    - long runs of interjections,
    - very short quoted gasp/exclamation.

    This deliberately does not drop ordinary sentences or lexeme glosses merely
    because they contain ellipses. Valid targets such as "在...之前" and long
    narrative sentences with "..." are common in the current data.
    """
    s = "" if s is None else str(s).strip()
    if not s:
        return True
    cjk_len = _cjk_len(s)
    punct = sum(1 for ch in s if unicodedata.category(ch)[0] in ("P", "S"))
    ratio = punct / max(len(s), 1)
    if ratio > 0.60 and cjk_len <= 8:
        return True
    if INTERJ_RE.search(s) and cjk_len <= 12:
        return True
    if SHORT_QUOTED_RE.match(s) and cjk_len <= 4:
        return True
    return False


def target_is_meta_only(s: str) -> bool:
    """Drop bracket-only POS/grammar labels that are not translations."""
    s = "" if s is None else str(s).strip()
    if TARGET_BRACKET_META_RE.match(s):
        return True
    short = SHORT_BRACKETED_LABEL_RE.match(s)
    if not short:
        return False
    inner = short.group(1).strip()
    # Current corpora use short bracketed Chinese labels like [主], [介],
    # [虛]. Treat these as grammatical metadata, not MT targets.
    return bool(inner) and _cjk_len(inner) == len(inner)


def source_has_non_formosan_script(s: str) -> bool:
    s = "" if s is None else str(s)
    return _cjk_len(s) > 0 or bool(NON_FORMOSAN_SOURCE_SCRIPT_RE.search(s))


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

def target_unit_count(s: str) -> int:
    """
    Length proxy for target text.

    Chinese targets are mostly unsegmented, so whitespace token count is a bad
    fertility signal. Count CJK characters plus non-CJK tokens instead; use
    whitespace tokens for English/Latin targets.
    """
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    cjk_chars = _cjk_len(s)
    if cjk_chars:
        non_cjk = CJK_RE.sub(" ", s)
        return cjk_chars + tok_count(non_cjk)
    return tok_count(s)

def skeleton_key(s: str) -> str:
    """Punctuation/spacing-insensitive key for split grouping."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    text = unicodedata.normalize("NFKC", s).casefold()
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in ("L", "N", "M"))

def exact_key(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFKC", s).casefold()).strip()

def pair_key(src: pd.Series, tgt: pd.Series, key_fn) -> pd.Series:
    return src.astype(str).map(key_fn) + "\u241f" + tgt.astype(str).map(key_fn)

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
    Fallback lexeme detection when XML row_type is missing.

    Keep this conservative: XML row_type and lexical source paths are stronger
    signals than token count.
    """
    s = src if isinstance(src, str) else ""
    t = tgt if isinstance(tgt, str) else ""
    source_path = source_path if isinstance(source_path, str) else ""

    source_lower = source_path.lower()
    if any(hint.lower() in source_lower for hint in LEXEME_SOURCE_HINTS):
        return True

    stoks = tok_count(s)
    ttoks = target_unit_count(t)

    if stoks <= 2 and target_looks_gloss_list(t):
        return True
    if stoks <= 1 and ttoks <= 30:
        return True
    return False

def classify_row_type(existing: object, src: str, tgt: str, source_path: str) -> str:
    existing_norm = str(existing or "").strip().lower()
    if existing_norm in {"lexeme", "morpheme"}:
        return "lexeme"
    source_lower = str(source_path or "").lower()
    if any(hint.lower() in source_lower for hint in LEXEME_SOURCE_HINTS):
        return "lexeme"
    if existing_norm == "sentence":
        # Preserve sentence rows unless the text is plainly lexical.
        return "lexeme" if looks_like_lexeme(src, tgt, source_path) else "sentence"
    return "lexeme" if looks_like_lexeme(src, tgt, source_path) else "sentence"

def target_looks_gloss_list(s: str) -> bool:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    text = s.strip()
    if not text or SENT_END_RE.search(text):
        return False
    if target_unit_count(text) > 18:
        return False
    return any(delim in text for delim in (";", "；", "/", "、"))

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
    work = df.copy()
    if "row_type" in work.columns:
        priority = {"sentence": 0, "lexeme": 1, "morpheme": 2}
        work["_dedupe_priority"] = (
            work["row_type"].astype(str).str.lower().map(priority).fillna(3).astype(int)
        )
        work["_dedupe_order"] = range(len(work))
        work = work.sort_values(["_dedupe_priority", "_dedupe_order"], kind="stable")
    df2 = work.drop_duplicates(subset=[src_col, tgt_col], keep="first")
    df2 = df2.sort_index(kind="stable")
    df2 = df2.drop(columns=[c for c in ("_dedupe_priority", "_dedupe_order") if c in df2.columns])
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
    reporter: DropReporter | None = None,
) -> pd.DataFrame:
    """
    Use language-aware target/source unit ratios.

    Formosan source is whitespace-tokenized. Chinese targets often are not, so
    target_unit_count uses CJK characters plus non-CJK tokens instead of plain
    whitespace tokens.
    """
    df = df.copy()
    s_tok = df[src_col].astype(str).map(tok_count)
    t_units = df[tgt_col].astype(str).map(target_unit_count)
    ratio = (t_units.replace(0, np.nan) / s_tok.replace(0, np.nan)).astype(float)

    is_lex = df.get("row_type", "").eq("lexeme")
    lo = np.where(is_lex, min_ratio_lex, min_ratio_sent)
    hi = np.where(is_lex, max_ratio_lex, max_ratio_sent)

    ok = (ratio >= lo) & (ratio <= hi)
    drop_mask = ~ok.fillna(False)
    if reporter is not None:
        reporter.record("fertility_ratio", df, drop_mask, src_col, tgt_col)
    kept = df[~drop_mask]
    print(f"📏  Fertility filter removed {len(df) - len(kept):,} pairs "
          f"(target/source unit ratio; sent bounds [{min_ratio_sent},{max_ratio_sent}] | "
          f"lex bounds [{min_ratio_lex},{max_ratio_lex}])")
    return kept

# ──────────────────────────────────────────────────────────────────────────────
# Near-duplicate grouping (edit distance 1) — still available but off by default
# (currently unused for splitting; we instead group by exact source/target equality)
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
# Equivalence groups for one-to-many / many-to-one (exact src/tgt matches)
# ──────────────────────────────────────────────────────────────────────────────

def build_equivalence_groups(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.Series:
    """
    Build equivalence groups so that any rows sharing the same source text OR the same target text
    end up in the same group. Keys are punctuation/spacing-insensitive, so near
    duplicates like "Pinsbkan 開始" vs "Pinsbkan開始" are forced into the same
    split before the later hard-split builder does its corpus-wide pass.
    """
    if df.empty:
        return pd.Series([], index=df.index, dtype=object, name="eq_group")

    tmp = df.reset_index()
    n = len(tmp)
    uf = UnionFind(n)

    # group by exact source string
    src_map: dict[str, int] = {}
    src_values = tmp[src_col].astype(str).map(skeleton_key).tolist()
    for i, s in enumerate(src_values):
        if not s:
            continue
        prev = src_map.get(s)
        if prev is not None:
            uf.union(i, prev)
        else:
            src_map[s] = i

    # group by exact target string
    tgt_map: dict[str, int] = {}
    tgt_values = tmp[tgt_col].astype(str).map(skeleton_key).tolist()
    for i, t in enumerate(tgt_values):
        if not t:
            continue
        prev = tgt_map.get(t)
        if prev is not None:
            uf.union(i, prev)
        else:
            tgt_map[t] = i

    groups_local = [f"eqg:{uf.find(i)}" for i in range(n)]
    res = pd.Series(groups_local, index=tmp["index"].values, name="eq_group")
    return res

# ──────────────────────────────────────────────────────────────────────────────
# Presentation scaffolding drop (CN + Amis) + Obvious CN Headers
# ──────────────────────────────────────────────────────────────────────────────

CN_STAGEY_PAT = re.compile(r"(換下一個(?:說)?|我(?:分享|說明)到這裡)\s*$")
SOURCE_STAGEY_EXACT_PAT = re.compile(
    r"^(?:romato|sowal ako|pisoykay|pisowal ako|mahaen ko)[.!?。！？,， ]*$",
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
    return bool(SOURCE_STAGEY_EXACT_PAT.match(s))

def stagey_or_header_mask(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.Series:
    # Heuristic: target is Chinese if CJK-rich; source scaffolding must be an
    # exact line. Earlier substring matching dropped valid Tayal rows containing
    # words like "roma".
    cjk_tgt = df[tgt_col].astype(str).map(_cjk_len) > 0
    cjk_src = df[src_col].astype(str).map(_cjk_len) > 0

    return (
        df[tgt_col].astype(str).where(cjk_tgt, "").map(lambda s: _looks_stagey_or_aside_or_header(s, True)) |
        df[src_col].astype(str).where(~cjk_src, "").map(lambda s: _looks_stagey_or_aside_or_header(s, False))
    )

def drop_stagey_rows(df: pd.DataFrame, src_col: str, tgt_col: str) -> pd.DataFrame:
    mask_drop = stagey_or_header_mask(df, src_col, tgt_col)
    n = int(mask_drop.sum())
    if n:
        print(f"🧹  Dropping {n:,} presentation/asides/headers (e.g., 換下一個/我分享到這裡/人物生平 - -)")
    return df[~mask_drop].reset_index(drop=True)

# ──────────────────────────────────────────────────────────────────────────────
# Splitting (equivalence group-aware), with lexeme routing
# ──────────────────────────────────────────────────────────────────────────────

def extract_source_id(p: str) -> str:
    parts = str(p).split("/")
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}/"
    return str(p)

def split_by_source(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
    include_lexemes_in_eval: bool,
    max_lexeme_frac_train: float | None = None,
) -> pd.DataFrame:
    """
    Group-aware split:
    - Lexemes (row_type == 'lexeme') → train only.
    - Sentences are split 80/10/10 (or as configured), but **by equivalence group**:
      any rows that share the same source OR the same target (after cleaning) are
      forced into the same split to avoid one-to-many / many-to-one leakage.
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

    # Lexical entries stay in train (dictionary-style material).
    lexeme_df["split"] = "train"

    n_sent = len(sentence_df)
    if n_sent > 0:
        # Build equivalence groups so any one-to-many or many-to-one pairs
        # (same src OR same tgt) land in the same split.
        eq_groups = build_equivalence_groups(sentence_df, src_col, tgt_col)
        sentence_df["group_id"] = eq_groups

        grp_sizes = sentence_df.groupby("group_id").size().reset_index(name="size")
        grp_ids = grp_sizes["group_id"].tolist()
        rng.shuffle(grp_ids)

        total_sent = float(n_sent)
        target_train = int(total_sent * train_ratio)
        target_val = int(total_sent * val_ratio)
        target_test = n_sent - target_train - target_val

        counters = {"train": 0, "validate": 0, "test": 0}
        assignment: dict[str, str] = {}

        size_lookup = dict(zip(grp_sizes["group_id"], grp_sizes["size"]))

        for gid in grp_ids:
            size = int(size_lookup[gid])
            rem_train = target_train - counters["train"]
            rem_val = target_val - counters["validate"]
            rem_test = target_test - counters["test"]

            rem = {
                "train": rem_train,
                "validate": rem_val,
                "test": rem_test,
            }
            # Choose split with the largest remaining capacity
            best_split = max(rem, key=lambda k: rem[k])
            # If all are "full" (remaining <= 0), just send to train
            if rem[best_split] <= 0:
                best_split = "train"

            assignment[gid] = best_split
            counters[best_split] += size

        sentence_df["split"] = sentence_df["group_id"].map(assignment)
    else:
        sentence_df["split"] = []

    df_out = pd.concat([lexeme_df, sentence_df], ignore_index=True)

    counts = df_out["split"].value_counts()
    total = len(df_out)
    print(
        f"\n📊  Final split: "
        f"{counts.get('train',0):,} train ({counts.get('train',0)/total*100:.1f}%), "
        f"{counts.get('validate',0):,} val ({counts.get('validate',0)/total*100:.1f}%), "
        f"{counts.get('test',0):,} test ({counts.get('test',0)/total*100:.1f}%)"
    )
    return df_out

def prune_train_eval_overlaps(
    df: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    reporter: DropReporter | None = None,
) -> pd.DataFrame:
    """Drop train rows that would leak exact/skeleton source, target, or pair keys into eval."""
    if "split" not in df.columns or df.empty:
        return df

    split = df["split"].astype(str).str.lower()
    train_mask = split.eq("train")
    eval_mask = split.isin(["validate", "valid", "val", "test"])
    if not train_mask.any() or not eval_mask.any():
        return df

    src = df[src_col].astype(str)
    tgt = df[tgt_col].astype(str)
    key_columns = {
        "source_exact": src.map(exact_key),
        "target_exact": tgt.map(exact_key),
        "pair_exact": pair_key(src, tgt, exact_key),
        "source_skeleton": src.map(skeleton_key),
        "target_skeleton": tgt.map(skeleton_key),
        "pair_skeleton": pair_key(src, tgt, skeleton_key),
    }

    leak_mask = pd.Series(False, index=df.index)
    leak_counts: dict[str, int] = {}
    for name, keys in key_columns.items():
        eval_keys = set(keys[eval_mask].dropna())
        eval_keys.discard("")
        if not eval_keys:
            continue
        col_leaks = train_mask & keys.isin(eval_keys)
        leak_counts[name] = int(col_leaks.sum())
        leak_mask |= col_leaks

    n = int(leak_mask.sum())
    if n:
        if reporter is not None:
            reporter.record("train_eval_overlap_removed", df, leak_mask, src_col, tgt_col)
        detail = ", ".join(f"{k}={v}" for k, v in sorted(leak_counts.items()) if v)
        print(f"🧯  Dropping {n:,} train rows that overlap validation/test keys ({detail})")
        df = df[~leak_mask].reset_index(drop=True)
    return df

def validate_split_invariants(df: pd.DataFrame, src_col: str, tgt_col: str) -> None:
    if "split" not in df.columns:
        return
    split = df["split"].astype(str).str.lower()
    lex_eval = (
        df.get("row_type", pd.Series("", index=df.index))
        .astype(str)
        .str.lower()
        .isin({"lexeme", "morpheme"})
        & split.isin(["validate", "valid", "val", "test"])
    )
    if lex_eval.any():
        raise SystemExit(f"Lexeme routing validation failed: {int(lex_eval.sum())} lexeme rows in eval splits")

    train_mask = split.eq("train")
    eval_mask = split.isin(["validate", "valid", "val", "test"])
    if not train_mask.any() or not eval_mask.any():
        return
    src = df[src_col].astype(str)
    tgt = df[tgt_col].astype(str)
    for name, keys in {
        "source_exact": src.map(exact_key),
        "target_exact": tgt.map(exact_key),
        "pair_exact": pair_key(src, tgt, exact_key),
        "source_skeleton": src.map(skeleton_key),
        "target_skeleton": tgt.map(skeleton_key),
        "pair_skeleton": pair_key(src, tgt, skeleton_key),
    }.items():
        train_keys = set(keys[train_mask].dropna())
        eval_keys = set(keys[eval_mask].dropna())
        train_keys.discard("")
        eval_keys.discard("")
        overlap = train_keys & eval_keys
        if overlap:
            raise SystemExit(f"Split leakage validation failed for {name}: {len(overlap)} overlapping keys")

# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, low_memory=False)
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
        description="Filter and split parallel corpus for MT training (lexeme-aware, group-aware)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # I/O
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Directory for summary.json and reject_samples.csv; defaults beside output.",
    )
    ap.add_argument("--no-report", action="store_true", help="Do not write filter audit reports")
    ap.add_argument("--audit-samples", type=int, default=100, help="Reject samples to keep per drop reason")

    # Cleaning & heuristics
    ap.add_argument("--no-clean-text", action="store_true", help="Skip Moses/NFKC cleaning")
    ap.add_argument("--workers", type=int, default=None, help="CPU cores for cleaning (default: all)")
    ap.add_argument("--min-sent-tokens", type=int, default=1, help="Minimum tokens on BOTH sides to keep (default: 1)")
    ap.add_argument("--drop-listy-sentences", action="store_true", help="Drop sentence-like rows that resemble lists")

    # New: controls for extra safety drops
    ap.add_argument("--keep-stagey", action="store_true", help="Keep presentation/asides/header lines instead of dropping")
    ap.add_argument("--keep-cjk-in-src", action="store_true", help="Keep rows even if source/Formosan has Han/Kana/Hangul chars")
    ap.add_argument("--keep-punct-only", action="store_true", help="Keep rows that are only punctuation/symbols")
    ap.add_argument("--keep-redactions", action="store_true", help="Keep rows containing XXX/XXXX redaction placeholders")

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
    ap.add_argument("--no-near-dup", action="store_true", help="(Deprecated) unused; kept for CLI compatibility")

    # Splitting
    ap.add_argument("--no-split", action="store_true", help="Do not create train/validate/test splits")
    ap.add_argument("--val-ratio", type=float, default=0.10)
    ap.add_argument("--test-ratio", type=float, default=0.10)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--include-lexemes-in-eval", action="store_true", help="Allow lexemes in val/test (currently ignored; lexemes → train only)")
    ap.add_argument("--max-lexeme-frac", type=float, default=None, help="DEPRECATED - not used in simplified split")

    ap.add_argument(
        "--keep-cn-jabber",
        action="store_true",
        help="Keep Chinese targets that look like dialog/interjection/ellipsis spam (default: drop them)"
    )

    args = ap.parse_args()

    print("=" * 80)
    print("🚀  MT Corpus Filtering & Splitting (Lexeme-Aware, Group-Aware)")
    print("=" * 80)

    # Load
    df = load_csv(args.input)
    initial_rows = len(df)
    reporter = DropReporter(sample_limit=args.audit_samples)

    # Detect language columns early (and clean only those)
    src_col, tgt_col = detect_lang_cols(df)
    print(f"🗂️  Language columns: {src_col} ↔ {tgt_col}")

    # Drop rows with explicit missing-translation markers in either column.
    print("🧹  Removing rows with explicit missing-translation markers...")
    n_before = len(df)
    mask_no_record = (
        df[src_col].astype(str).map(lambda s: bool(MISSING_TRANSLATION_RE.search(s))) |
        df[tgt_col].astype(str).map(lambda s: bool(MISSING_TRANSLATION_RE.search(s)))
    )
    reporter.record("missing_translation_marker", df, mask_no_record, src_col, tgt_col)
    df = df[~mask_no_record].reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"🗑️  Removed {n_dropped:,} rows with missing-translation markers")

    # Text cleaning
    if not args.no_clean_text:
        df = clean_text_columns(df, [src_col, tgt_col], workers=args.workers)

    # One-time scrub pass (always run, even if --no-clean-text was set)
    print("🧽  Scrubbing meta + fixing punctuation/spacing (tags, short ()-glosses, artifacts, trailing commas, spaces, stray brackets/quotes)…")
    for c in (src_col, tgt_col):
        df[c] = df[c].astype(str).map(extra_scrub)

    # Drop presentation scaffolding (CN + Amis) and obvious CN headers
    if not args.keep_stagey:
        mask_stagey = stagey_or_header_mask(df, src_col, tgt_col)
        reporter.record("presentation_or_header", df, mask_stagey, src_col, tgt_col)
        n = int(mask_stagey.sum())
        if n:
            print(f"🧹  Dropping {n:,} presentation/asides/headers (e.g., 換下一個/我分享到這裡/人物生平 - -)")
        df = df[~mask_stagey].reset_index(drop=True)

    # Drop rows where *source/Formosan* contains Chinese/Japanese/Korean script.
    if not args.keep_cjk_in_src:
        m_src_bad_script = df[src_col].astype(str).map(source_has_non_formosan_script)
        reporter.record("non_formosan_script_in_source", df, m_src_bad_script, src_col, tgt_col)
        n = int(m_src_bad_script.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows with CJK/Kana/Hangul script in source/Formosan column '{src_col}'")
        df = df[~m_src_bad_script].reset_index(drop=True)

    # Drop rows that are only punctuation/symbols or empty on either side
    if not args.keep_punct_only:
        m_bad_src = df[src_col].map(_is_only_punct_or_symbols)
        m_bad_tgt = df[tgt_col].map(_is_only_punct_or_symbols)
        m_bad = m_bad_src | m_bad_tgt
        reporter.record("empty_or_punctuation_only", df, m_bad, src_col, tgt_col)
        n = int(m_bad.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows that are empty or only punctuation/symbols on either side")
        df = df[~m_bad].reset_index(drop=True)

    # Drop rows with anonymization placeholders by default. These train models
    # to emit non-language artifacts and can leak into evaluation.
    if not args.keep_redactions:
        mask_redacted = (
            df[src_col].astype(str).map(lambda s: bool(REDACTION_RE.search(s))) |
            df[tgt_col].astype(str).map(lambda s: bool(REDACTION_RE.search(s)))
        )
        reporter.record("redaction_placeholder", df, mask_redacted, src_col, tgt_col)
        n = int(mask_redacted.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows with redaction placeholders (XXX/XXXX)")
        df = df[~mask_redacted].reset_index(drop=True)

    # Drop bracket-only grammatical labels like [介] / [虛]. They are useful
    # dictionary metadata, but not translation targets for MT.
    mask_target_meta = df[tgt_col].astype(str).map(target_is_meta_only)
    reporter.record("target_meta_label_only", df, mask_target_meta, src_col, tgt_col)
    n = int(mask_target_meta.sum())
    if n:
        print(f"🧹  Dropping {n:,} bracket-only grammatical target labels")
    df = df[~mask_target_meta].reset_index(drop=True)

    # Drop 'jabber' CN targets (performative dialog / ellipses / interjection spam)
    if not args.keep_cn_jabber:
        # Only consider targets that are actually Chinese (have CJK)
        mask_tgt_is_cn = df[tgt_col].astype(str).map(lambda s: _cjk_len(s) >= 1)
        mask_cn_jabber = mask_tgt_is_cn & df[tgt_col].astype(str).map(zh_looks_jabber)
        reporter.record("short_noisy_chinese_target", df, mask_cn_jabber, src_col, tgt_col)
        n = int(mask_cn_jabber.sum())
        if n:
            print(f"🧹  Dropping {n:,} rows: short noisy/performative Chinese targets")
        df = df[~mask_cn_jabber].reset_index(drop=True)

    # Classify row type (lexeme vs sentence)
    print("🔎  Classifying rows (lexeme vs sentence)…")
    src_series = df[src_col].astype(str)
    tgt_series = df[tgt_col].astype(str)
    src_paths = df["source"].astype(str) if "source" in df.columns else pd.Series([""] * len(df))
    existing_row_types = (
        df["row_type"].astype(str) if "row_type" in df.columns else pd.Series([""] * len(df))
    )

    row_types: List[str] = []
    for existing, s, t, sp in tqdm(
        zip(existing_row_types, src_series, tgt_series, src_paths),
        total=len(df),
        desc="classify",
    ):
        row_types.append(classify_row_type(existing, s, t, sp))
    df["row_type"] = row_types

    # Optionally drop "listy" rows from sentences (kept for lexemes)
    if args.drop_listy_sentences:
        mask_listy = df["row_type"].eq("sentence") & (
            df[src_col].map(is_listy_sentence_like) | df[tgt_col].map(is_listy_sentence_like)
        )
        n_drop = int(mask_listy.sum())
        if n_drop:
            reporter.record("list_like_sentence", df, mask_listy, src_col, tgt_col)
            print(f"🧹  Dropping {n_drop:,} list-like rows from sentences")
        df = df[~mask_listy].reset_index(drop=True)

    # Enforce minimum tokens for sentences (both sides)
    min_tok = max(0, int(args.min_sent_tokens))
    if min_tok > 0:
        s_tok = df[src_col].map(tok_count)
        t_tok = df[tgt_col].map(target_unit_count)
        mask_bad_sent = df["row_type"].eq("sentence") & ~((s_tok >= min_tok) & (t_tok >= min_tok))
        n_drop = int(mask_bad_sent.sum())
        if n_drop:
            reporter.record("short_sentence", df, mask_bad_sent, src_col, tgt_col)
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
            reporter=reporter,
        )

    # Near-duplicate grouping (disabled for now; we use exact-equivalence groups instead)
    # if not args.no_near_dup:
    #     df["nd_group"] = build_near_dups(df, src_col, tgt_col, workers=args.workers)
    # else:
    #     df["nd_group"] = [f"row:{i}" for i in range(len(df))]

    # Split
    if not args.no_split:
        df = split_by_source(
            df,
            src_col,
            tgt_col,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state,
            include_lexemes_in_eval=args.include_lexemes_in_eval,
            max_lexeme_frac_train=float(args.max_lexeme_frac) if args.max_lexeme_frac is not None else None,
        )
        df = prune_train_eval_overlaps(df, src_col, tgt_col, reporter=reporter)
        validate_split_invariants(df, src_col, tgt_col)

    # Save
    print("=" * 80)
    save_csv(df, args.output)
    if not args.no_report:
        report_dir = args.report_dir or (args.output.parent / "filter_reports" / args.output.stem)
        reporter.write(
            report_dir,
            input_path=args.input,
            output_path=args.output,
            initial_rows=initial_rows,
            final_df=df,
            src_col=src_col,
            tgt_col=tgt_col,
        )
    print("=" * 80)
    print("✅  Pipeline complete!")

if __name__ == "__main__":
    main()
