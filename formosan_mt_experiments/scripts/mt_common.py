#!/usr/bin/env python3
"""Shared helpers for the Formosan MT experiment stack."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Callable, Iterable, Mapping

import pandas as pd
from columnar_cache import read_csv_or_columnar

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "formosan_mt_experiments"

DEFAULT_SETUP_SCRIPT = EXPERIMENT_ROOT / "scripts" / "setup_formosan_nllb200.py"

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

TARGET_CONFIGS = {
    "english": {"short": "eng", "tag": "eng", "col": "english_sentence", "lid": "eng_Latn"},
    "chinese": {"short": "zh", "tag": "zh", "col": "chinese_sentence", "lid": "zho_Hant"},
}

DOMAIN_BUCKETS = (
    "dictionary",
    "classroom",
    "narrative",
    "linguistic",
    "education",
    "media",
    "culture",
    "religious",
    "unknown",
)
EASY_BUCKETS = ("dictionary", "classroom")
MT_STANDARD_NAMESPACE = "formosan-mt"
MT_STANDARD_REQUIRED_COLUMNS = (
    "kindOf",
    "standard_namespace",
    "formosan_mt_standard",
    "mt_standard_sha256",
    "mt_normalization_status",
    "mt_normalization_confidence",
    "mt_eval_eligible",
    "mt_standard_profile",
    "mt_standard_profile_sha256",
)

def normalize_text(value: object) -> str:
    """NFKC + casefold + whitespace collapse for leakage checks."""
    text = unicodedata.normalize("NFKC", "" if pd.isna(value) else str(value))
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def skeleton_text(value: object) -> str:
    """Aggressive near-duplicate key: normalized letters/numbers/CJK only."""
    text = normalize_text(value)
    return "".join(ch for ch in text if unicodedata.category(ch)[0] in {"L", "N", "M"})


def token_count(value: object) -> int:
    """Whitespace token count for Formosan/English-like text."""
    text = normalize_text(value)
    return 0 if not text else len(text.split())


def cjk_token_count(value: object) -> int:
    """Character-aware token proxy for Chinese text without whitespace."""
    text = normalize_text(value)
    if not text:
        return 0
    cjk_chars = re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text)
    non_cjk = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", " ", text)
    non_cjk_tokens = [tok for tok in non_cjk.split() if tok]
    return len(cjk_chars) + len(non_cjk_tokens)


def normalize_target_language(target_lang: str | None = None, target_col: str | None = None) -> str:
    if target_lang:
        key = str(target_lang).strip().lower()
        if key in {"en", "eng", "english"}:
            return "english"
        if key in {"zh", "zho", "chinese"}:
            return "chinese"
        raise KeyError(f"Unsupported target language: {target_lang}")
    if target_col == "chinese_sentence":
        return "chinese"
    return "english"


def target_col_for(target_lang: str) -> str:
    return TARGET_CONFIGS[normalize_target_language(target_lang)]["col"]


def target_tag_for(target_lang: str) -> str:
    return TARGET_CONFIGS[normalize_target_language(target_lang)]["tag"]


def target_lid_for(target_lang: str) -> str:
    return TARGET_CONFIGS[normalize_target_language(target_lang)]["lid"]


def target_token_count(value: object, target_lang: str | None = None, target_col: str | None = None) -> int:
    lang = normalize_target_language(target_lang, target_col)
    if lang == "chinese":
        return cjk_token_count(value)
    return token_count(value)


def target_language_from_direction(direction: str, target_lang: str | None = None) -> str:
    direction = str(direction).strip().lower()
    if direction in {"f2en", "en2f"}:
        return "english"
    if direction in {"f2zh", "zh2f"}:
        return "chinese"
    return normalize_target_language(target_lang)


def is_formosan_to_target(direction: str) -> bool:
    direction = str(direction).strip().lower()
    return direction.startswith("f2")


def is_target_to_formosan(direction: str) -> bool:
    direction = str(direction).strip().lower()
    return direction.endswith("2f")


def direction_choices() -> list[str]:
    return ["f2en", "en2f", "f2zh", "zh2f"]


def safe_tag_value(value: object, default: str = "default", max_len: int = 48) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = default
    return text[:max_len].strip("_") or default


def source_bucket(source: object) -> str:
    """Map provenance paths to a fixed, repository-independent domain."""
    lowered = "" if pd.isna(source) else str(source).casefold()
    if re.search(
        r"dict(?:ionary|ionaries)?|lexic(?:on|ons|al)?|glossar|glosbe|"
        r"word.?list|vocab",
        lowered,
    ):
        return "dictionary"
    if re.search(r"contextual|classroom|qing_jing|conversation|dialogue", lowered):
        return "classroom"
    if re.search(r"bible|hymn|church|religio", lowered):
        return "religious"
    if re.search(r"youtube|video|audio|wilang.yutas", lowered):
        return "media"
    if re.search(r"cultur|custom|apolog|president|ceremon|ritual", lowered):
        return "culture"
    if re.search(
        r"story|stories|(?:^|[/_.-])texts?(?:[/_.-]|$)|picture.?book|"
        r"myth|blog|narrat|tale|"
        r"legend|folklore|literary|ode.to|raodong|wakelin|montgomery",
        lowered,
    ):
        return "narrative"
    if re.search(
        r"learning|epark|gitbook|essay|reading|writing|nine.level|教材|"
        r"material|textbook|course|tousvusvutu",
        lowered,
    ):
        return "education"
    if re.search(
        r"grammar|grammatical|syntax|linguist|seals|zheng|acl|elicitat|"
        r"construction|sentence|word.order|negation|relative|causative|voice|"
        r"phonolog|corpus|dissertation|thesis|descriptive.study|dialect|"
        r"relationship|classification|topic.focus|complement|comparative|"
        r"demonstrative|affix|time.reference|social.structure|conjunction",
        lowered,
    ):
        return "linguistic"
    return "unknown"


def source_corpus(source: object) -> str:
    """Return the exact public corpus root or private repository name."""
    value = "" if pd.isna(source) else str(source)
    parts = [part for part in value.replace("\\", "/").split("/") if part]
    lowered = [part.casefold() for part in parts]
    if "corpora" in lowered:
        position = lowered.index("corpora")
        if position + 1 < len(parts):
            return parts[position + 1]
    if parts and ({"final_xml", "xml"} & set(lowered)):
        return parts[0]
    if parts and parts[0].casefold().startswith("formosan-"):
        return parts[0]
    return source_bucket(value)


def add_normalized_columns(
    df: pd.DataFrame,
    target_col: str = "english_sentence",
    target_lang: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    lang = normalize_target_language(target_lang, target_col)
    if "row_type" not in out.columns:
        out["row_type"] = "unknown"
    out["row_type"] = out["row_type"].fillna("unknown").astype(str).str.strip().str.lower()
    out["_source_key"] = out["source"].fillna("").astype(str) if "source" in out else ""
    out["_source_bucket"] = out["_source_key"].map(source_bucket)
    out["_source_corpus"] = out["_source_key"].map(source_corpus)
    out["_formosan_key"] = out["formosan_sentence"].map(normalize_text)
    out["_target_key"] = out[target_col].map(normalize_text)
    out["_pair_key"] = out["_formosan_key"] + PAIR_SEP + out["_target_key"]
    out["_formosan_skeleton"] = out["formosan_sentence"].map(skeleton_text)
    out["_target_skeleton"] = out[target_col].map(skeleton_text)
    out["_pair_skeleton"] = out["_formosan_skeleton"] + PAIR_SEP + out["_target_skeleton"]
    out["_formosan_tokens"] = out["formosan_sentence"].map(token_count)
    out["_target_tokens"] = out[target_col].map(lambda x: target_token_count(x, target_lang=lang))
    out["_short_entry"] = (out["_formosan_tokens"] <= 2) & (out["_target_tokens"] <= 3)
    out["_is_lexeme"] = out["row_type"].isin({"lexeme", "morpheme"})
    out["_lang_source_key"] = out["lang_code"].astype(str) + PAIR_SEP + out["_source_key"]
    out["_target_group_key"] = out["lang_code"].astype(str) + PAIR_SEP + out["_source_key"] + PAIR_SEP + out["_target_key"]
    return out


def evaluation_candidate_mask(
    frame: pd.DataFrame,
    *,
    min_formosan_tokens: int,
    min_target_tokens: int,
) -> pd.Series:
    """Return sentence-quality rows that may be used in dev or test.

    Provenance domains do not determine row quality. Structurally typed
    sentences are eligible regardless of source when they pass QC and the
    configured length requirements.
    """
    required = {
        "row_type",
        "mt_eval_eligible",
        "mt_normalization_confidence",
        "_formosan_tokens",
        "_target_tokens",
    }
    require_columns(frame, required, "evaluation eligibility")
    flags = frame.get("quality_flags", pd.Series("", index=frame.index)).astype(str)
    return (
        frame["row_type"].astype(str).str.casefold().eq("sentence")
        & ~flags.str.contains(
            r"(?:contains_unclear|unknown_row_type)",
            regex=True,
        )
        & bool_series(
            frame["mt_eval_eligible"],
            context="evaluation eligibility:mt_eval_eligible",
        )
        & ~frame["mt_normalization_confidence"].astype(str).eq("ambiguous")
        & frame["_formosan_tokens"].ge(min_formosan_tokens)
        & frame["_target_tokens"].ge(min_target_tokens)
    )


def weighted_apportioned_counts(
    weights: Mapping[str, int],
    capacities: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    """Apportion ``total`` by source weights without exceeding capacities."""
    keys = sorted(set(weights) | set(capacities))
    normalized_weights = {
        key: max(0, int(weights.get(key, 0)))
        for key in keys
    }
    normalized_capacities = {
        key: max(0, int(capacities.get(key, 0)))
        for key in keys
    }
    available = sum(normalized_capacities.values())
    if total < 0 or total > available:
        raise ValueError(
            f"Cannot apportion {total:,} rows from {available:,} eligible rows"
        )
    allocated = {key: 0 for key in keys}
    remaining = total
    active = {
        key
        for key in keys
        if normalized_capacities[key] > 0
    }
    while remaining and active:
        weight_total = sum(normalized_weights[key] for key in active)
        if weight_total <= 0:
            active_weights = {
                key: normalized_capacities[key] - allocated[key]
                for key in active
            }
            weight_total = sum(active_weights.values())
        else:
            active_weights = {
                key: normalized_weights[key]
                for key in active
            }
        quotas = {
            key: remaining * active_weights[key] / weight_total
            for key in active
        }
        saturated = {
            key
            for key in active
            if quotas[key] >= normalized_capacities[key] - allocated[key]
        }
        if saturated:
            for key in saturated:
                amount = normalized_capacities[key] - allocated[key]
                allocated[key] += amount
                remaining -= amount
            active -= saturated
            continue

        floors = {
            key: math.floor(quota)
            for key, quota in quotas.items()
        }
        for key, amount in floors.items():
            allocated[key] += amount
            remaining -= amount
        for key in sorted(
            active,
            key=lambda value: (
                -(quotas[value] - floors[value]),
                -active_weights[value],
                value,
            ),
        ):
            if remaining == 0:
                break
            if allocated[key] >= normalized_capacities[key]:
                continue
            allocated[key] += 1
            remaining -= 1
        break
    if remaining:
        raise RuntimeError("Could not apportion source targets within capacities")
    return allocated


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise SystemExit(f"{context} is missing required columns: {missing}")


def bool_series(values: pd.Series, *, context: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        examples = sorted(set(normalized[invalid]))[:5]
        raise SystemExit(f"{context} has invalid boolean values: {examples}")
    return normalized.isin({"true", "1"})


def mt_standard_contract(df: pd.DataFrame, *, context: str) -> dict[str, str]:
    require_columns(df, MT_STANDARD_REQUIRED_COLUMNS, context)
    if not df["kindOf"].astype(str).str.strip().str.lower().eq("standard").all():
        raise SystemExit(f"{context} contains non-standard rows")
    if not df["standard_namespace"].astype(str).eq(MT_STANDARD_NAMESPACE).all():
        raise SystemExit(f"{context} contains rows outside the Formosan MT namespace")
    if not df["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"{context} contains non-accepted MT-standard rows")
    if not df["formosan_sentence"].astype(str).eq(
        df["formosan_mt_standard"].astype(str)
    ).all():
        raise SystemExit(f"{context} violates the formosan_sentence MT-standard alias")
    bool_series(df["mt_eval_eligible"], context=f"{context}:mt_eval_eligible")
    profile_ids = set(df["mt_standard_profile"].astype(str).str.strip())
    profile_hashes = set(df["mt_standard_profile_sha256"].astype(str).str.strip())
    if len(profile_ids) != 1 or not next(iter(profile_ids), ""):
        raise SystemExit(f"{context} must contain exactly one MT-standard profile ID")
    profile_hash = next(iter(profile_hashes), "")
    if len(profile_hashes) != 1 or not re.fullmatch(r"[0-9a-f]{64}", profile_hash):
        raise SystemExit(f"{context} must contain exactly one valid MT-standard profile hash")
    return {"id": next(iter(profile_ids)), "sha256": profile_hash}


def read_parallel_csv(path: Path, target_col: str = "english_sentence") -> pd.DataFrame:
    # Provenance columns mix empty values and strings. Infer against the whole
    # file so chunk boundaries cannot change dtypes or emit noisy warnings.
    df = read_csv_or_columnar(
        path,
        low_memory=False,
        keep_default_na=False,
        na_filter=False,
    )
    require_columns(
        df,
        ["lang_code", "formosan_sentence", target_col, "source", "dialect"],
        str(path),
    )
    mt_standard_contract(df, context=str(path))
    required_nonempty = (
        df["lang_code"].astype(str).str.strip().ne("")
        & df["formosan_sentence"].astype(str).str.strip().ne("")
        & df[target_col].astype(str).str.strip().ne("")
    )
    df = df[required_nonempty].copy()
    df["lang_code"] = df["lang_code"].astype(str).str.strip().str.lower()
    if "row_type" not in df.columns:
        df["row_type"] = "unknown"
    df["row_type"] = df["row_type"].fillna("unknown").astype(str).str.strip().str.lower()
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
    tokens = []
    for config in TARGET_CONFIGS.values():
        tokens.extend([f"<to_{config['tag']}>", f"<src_{config['tag']}>"])
    for code in FORMOSAN_CODES:
        tokens.extend([f"<to_{code}>", f"<src_{code}>"])
    for bucket in DOMAIN_BUCKETS:
        tokens.append(f"<dom_{safe_tag_value(bucket)}>")
    tokens.append("<dialect_default>")
    return sorted(set(tokens))


def special_tokens_from_corpus(
    df: pd.DataFrame,
    max_dialect_tags: int = 200,
    min_dialect_frequency: int = 3,
) -> list[str]:
    tokens = set(base_special_tokens())
    if "dialect" in df.columns and max_dialect_tags > 0:
        dialects = df["dialect"].map(lambda x: safe_tag_value(x, "default")).value_counts()
        selected = dialects[dialects >= min_dialect_frequency].head(max_dialect_tags)
        for dialect in selected.index:
            tokens.add(f"<dialect_{dialect}>")
    return sorted(tokens)


def build_prefix(row: Mapping, direction: str, target_lang: str | None = None) -> str:
    code = str(row.get("lang_code", "")).strip().lower()
    bucket = row.get("source_bucket", row.get("_source_bucket", source_bucket(row.get("source", ""))))
    bucket = safe_tag_value(bucket, "unknown")
    if bucket not in DOMAIN_BUCKETS:
        bucket = "unknown"
    dialect = row.get("dialect", "default")
    domain_tag = f"<dom_{bucket}>"
    dialect_tag = f"<dialect_{safe_tag_value(dialect)}>"
    if is_formosan_to_target(direction):
        target_tag = target_tag_for(target_language_from_direction(direction, target_lang))
        return f"<to_{target_tag}> <src_{code}> {domain_tag} {dialect_tag}"
    if is_target_to_formosan(direction):
        source_tag = target_tag_for(target_language_from_direction(direction, target_lang))
        return f"<to_{code}> <src_{source_tag}> {domain_tag} {dialect_tag}"
    raise ValueError(f"Unsupported direction: {direction}")


def with_tagged_columns(
    df: pd.DataFrame,
    direction: str,
    target_col: str = "english_sentence",
    target_lang: str | None = None,
    use_tags: bool = True,
    prefix_builder: Callable[[Mapping], str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    if not use_tags and prefix_builder is None:
        return out
    if "source_bucket" not in out.columns:
        out["source_bucket"] = out["source"].map(source_bucket)
    if prefix_builder is None:
        prefixes = out.apply(
            lambda row: build_prefix(
                row,
                direction,
                target_lang=target_lang,
            ),
            axis=1,
        )
    else:
        prefixes = out.apply(prefix_builder, axis=1)
    if is_formosan_to_target(direction):
        out["formosan_sentence"] = (
            prefixes + " " + out["formosan_sentence"].fillna("").astype(str)
        ).str.strip()
    elif is_target_to_formosan(direction):
        out[target_col] = (
            prefixes + " " + out[target_col].fillna("").astype(str)
        ).str.strip()
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return out


def language_sampling_probs(counts: Mapping[str, int], alpha: float) -> dict[str, float]:
    weighted = {k: math.pow(max(v, 1), alpha) for k, v in counts.items()}
    total = sum(weighted.values())
    return {k: v / total for k, v in weighted.items()}
