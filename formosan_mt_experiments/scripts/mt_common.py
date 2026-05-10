#!/usr/bin/env python3
"""Shared helpers for the Formosan MT experiment stack."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "formosan_mt_experiments"

DEFAULT_INPUT = PROJECT_ROOT / "pivot_corpora_final" / "big_corpus_en.csv"
DEFAULT_SETUP_SCRIPT = PROJECT_ROOT / "scripts" / "mt" / "nllb" / "prelims" / "setup_formosan_nllb200.py"
DEFAULT_LEGACY_TRAIN_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "mt"
    / "nllb_multilingual"
    / "training"
    / "train_formosan_multilingual_nllb200.py"
)
DEFAULT_LEGACY_EVAL_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "mt"
    / "nllb_multilingual"
    / "eval"
    / "eval_formosan_multilingual_nllb200.py"
)

PAIR_SEP = "\u241f"

LANG_CODE_TO_CANON = {
    "ami": "amis",
    "bnn": "bunun",
    "ckv": "kavalan",
    "dru": "rukai",
    "pwn": "paiwan",
    "pyu": "puyuma",
    "ssf": "thao",
    "sxr": "saaroa",
    "szy": "sakizaya",
    "tao": "tao",
    "tay": "atayal",
    "trv": "seediq",
    "tsu": "tsou",
    "xnb": "kanakanavu",
    "xsy": "saisiyat",
}

CANON_TO_LANG_CODE = {v: k for k, v in LANG_CODE_TO_CANON.items()}

FORMOSAN_CODES = tuple(sorted(LANG_CODE_TO_CANON))

CODE_TO_LID = {
    "ami": "ami_Latn",
    "bnn": "bnn_Latn",
    "ckv": "ckv_Latn",
    "dru": "dru_Latn",
    "pwn": "pwn_Latn",
    "pyu": "pyu_Latn",
    "ssf": "ssf_Latn",
    "sxr": "sxr_Latn",
    "szy": "szy_Latn",
    "tao": "tao_Latn",
    "tay": "tay_Latn",
    "trv": "trv_Latn",
    "tsu": "tsu_Latn",
    "xnb": "xnb_Latn",
    "xsy": "xsy_Latn",
    "english": "eng_Latn",
    "en": "eng_Latn",
    "eng": "eng_Latn",
    "chinese": "zho_Hant",
    "zh": "zho_Hant",
}

EASY_BUCKETS = ("dictionary", "learning_vocab", "classroom_context")

DEFAULT_DOMAIN_BUCKETS = (
    "dictionary",
    "learning_vocab",
    "classroom_context",
    "picture_story",
    "picture_book",
    "essays",
    "reading_writing",
    "culture",
    "nine_level",
    "youtube",
    "ntu",
    "presidential_apology",
    "Formosan-100_Paiwan_Texts",
    "Formosan-Amis_myths_and_customs",
    "Formosan-ePark",
    "Formosan-gitbook_translations",
    "Formosan-Old_Texts",
    "Formosan-PaiwanStories",
    "Formosan-Rik-Bunun",
    "Formosan-SEALS",
    "Formosan-Wilang-Yutas-Videos",
    "Formosan-Yeddas-Blog",
    "Formosan-Zheng-Data",
    "unknown",
)


def normalize_text(value: object) -> str:
    """NFKC + casefold + whitespace collapse for leakage checks."""
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def token_count(value: object) -> int:
    text = normalize_text(value)
    return 0 if not text else len(text.split())


def safe_tag_value(value: object, default: str = "default", max_len: int = 48) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = default
    return text[:max_len].strip("_") or default


def source_bucket(source: object) -> str:
    """Coarse source family used by splitting, sampling, and reporting."""
    s = "" if pd.isna(source) else str(source)
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


def add_normalized_columns(df: pd.DataFrame, target_col: str = "english_sentence") -> pd.DataFrame:
    out = df.copy()
    out["_source_key"] = out["source"].fillna("").astype(str) if "source" in out else ""
    out["_source_bucket"] = out["_source_key"].map(source_bucket)
    out["_formosan_key"] = out["formosan_sentence"].map(normalize_text)
    out["_target_key"] = out[target_col].map(normalize_text)
    out["_pair_key"] = out["_formosan_key"] + PAIR_SEP + out["_target_key"]
    out["_formosan_tokens"] = out["formosan_sentence"].map(token_count)
    out["_target_tokens"] = out[target_col].map(token_count)
    out["_short_entry"] = (out["_formosan_tokens"] <= 2) & (out["_target_tokens"] <= 3)
    out["_lang_source_key"] = out["lang_code"].astype(str) + PAIR_SEP + out["_source_key"]
    out["_target_group_key"] = out["lang_code"].astype(str) + PAIR_SEP + out["_source_key"] + PAIR_SEP + out["_target_key"]
    return out


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise SystemExit(f"{context} is missing required columns: {missing}")


def read_parallel_csv(path: Path, target_col: str = "english_sentence") -> pd.DataFrame:
    df = pd.read_csv(path)
    require_columns(df, ["lang_code", "formosan_sentence", target_col, "source", "dialect"], str(path))
    df = df.dropna(subset=["lang_code", "formosan_sentence", target_col]).copy()
    df["lang_code"] = df["lang_code"].astype(str).str.strip().str.lower()
    df = df[df["lang_code"].isin(FORMOSAN_CODES)].copy()
    return df.reset_index(drop=True)


def write_json(path: Path, data: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def split_counts(df: pd.DataFrame) -> dict:
    if "split" not in df:
        return {}
    counts = df["split"].value_counts(dropna=False).to_dict()
    return {str(k): int(v) for k, v in counts.items()}


def split_counts_by_language(df: pd.DataFrame) -> dict:
    if "split" not in df:
        return {}
    table = pd.crosstab(df["lang_code"], df["split"])
    return {
        str(lang): {str(split): int(value) for split, value in row.items()}
        for lang, row in table.iterrows()
    }


def bucket_counts(df: pd.DataFrame) -> dict:
    if "source_bucket" in df.columns:
        s = df["source_bucket"]
    elif "_source_bucket" in df.columns:
        s = df["_source_bucket"]
    else:
        return {}
    return {str(k): int(v) for k, v in s.value_counts(dropna=False).items()}


def overlap_stats(train: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    stats = {}
    for name, col in (
        ("formosan", "_formosan_key"),
        ("target", "_target_key"),
        ("pair", "_pair_key"),
        ("source", "_source_key"),
    ):
        train_values = set(train[col].dropna())
        eval_values = set(eval_df[col].dropna())
        overlap = train_values & eval_values
        stats[name] = {
            "train_unique": len(train_values),
            "eval_unique": len(eval_values),
            "overlap_unique": len(overlap),
        }
    return stats


def get_lid(code: str) -> str:
    key = str(code).strip().lower()
    if key in CODE_TO_LID:
        return CODE_TO_LID[key]
    if "_" in str(code):
        return str(code)
    raise KeyError(f"Unknown language code: {code}")


def base_special_tokens() -> list[str]:
    tokens = ["<to_eng>", "<src_eng>", "<dae>", "<mask>"]
    for code in FORMOSAN_CODES:
        tokens.extend([f"<to_{code}>", f"<src_{code}>"])
    for bucket in DEFAULT_DOMAIN_BUCKETS:
        tokens.append(f"<dom_{safe_tag_value(bucket)}>")
    tokens.append("<dialect_default>")
    return sorted(set(tokens))


def special_tokens_from_corpus(
    df: pd.DataFrame,
    max_dialect_tags: int = 200,
    min_dialect_frequency: int = 3,
) -> list[str]:
    tokens = set(base_special_tokens())
    if "source" in df.columns:
        for bucket in sorted(df["source"].map(source_bucket).dropna().unique()):
            tokens.add(f"<dom_{safe_tag_value(bucket)}>")
    if "dialect" in df.columns and max_dialect_tags > 0:
        dialects = df["dialect"].map(lambda x: safe_tag_value(x, "default")).value_counts()
        selected = dialects[dialects >= min_dialect_frequency].head(max_dialect_tags)
        for dialect in selected.index:
            tokens.add(f"<dialect_{dialect}>")
    return sorted(tokens)


def build_prefix(row: Mapping, direction: str) -> str:
    code = str(row.get("lang_code", "")).strip().lower()
    bucket = row.get("source_bucket", row.get("_source_bucket", source_bucket(row.get("source", ""))))
    dialect = row.get("dialect", "default")
    domain_tag = f"<dom_{safe_tag_value(bucket)}>"
    dialect_tag = f"<dialect_{safe_tag_value(dialect)}>"
    if direction == "f2en":
        return f"<to_eng> <src_{code}> {domain_tag} {dialect_tag}"
    if direction == "en2f":
        return f"<to_{code}> <src_eng> {domain_tag} {dialect_tag}"
    if direction == "dae":
        return f"<dae> <src_{code}> {domain_tag} {dialect_tag}"
    raise ValueError(f"Unsupported direction: {direction}")


def with_tagged_columns(
    df: pd.DataFrame,
    direction: str,
    target_col: str = "english_sentence",
    use_tags: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if not use_tags:
        return out
    if "source_bucket" not in out.columns:
        out["source_bucket"] = out["source"].map(source_bucket)
    prefixes = out.apply(lambda row: build_prefix(row, direction), axis=1)
    if direction in {"f2en", "dae"}:
        out["formosan_sentence"] = prefixes + " " + out["formosan_sentence"].fillna("").astype(str)
    elif direction == "en2f":
        out[target_col] = prefixes + " " + out[target_col].fillna("").astype(str)
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return out


def language_sampling_probs(counts: Mapping[str, int], alpha: float) -> dict[str, float]:
    weighted = {k: math.pow(max(v, 1), alpha) for k, v in counts.items()}
    total = sum(weighted.values())
    return {k: v / total for k, v in weighted.items()}


def random_word_permutation(words: list[str], max_distance: int, rng: random.Random) -> list[str]:
    if max_distance <= 0 or len(words) <= 1:
        return words
    scores = [i + rng.uniform(0, max_distance + 1) for i in range(len(words))]
    return [w for _, w in sorted(zip(scores, words), key=lambda x: x[0])]


def corrupt_text(
    text: str,
    rng: random.Random,
    word_dropout: float = 0.10,
    span_mask: float = 0.15,
    shuffle_distance: int = 3,
    mask_token: str = "<mask>",
) -> str:
    words = str(text).split()
    if not words:
        return mask_token

    out: list[str] = []
    i = 0
    while i < len(words):
        r = rng.random()
        if r < word_dropout:
            i += 1
            continue
        if r < word_dropout + span_mask:
            out.append(mask_token)
            i += rng.randint(1, min(3, len(words) - i))
            continue
        out.append(words[i])
        i += 1
    if not out:
        out = [mask_token]
    return " ".join(random_word_permutation(out, shuffle_distance, rng))
