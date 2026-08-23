"""Corpus contracts and sentence eligibility for DeepL pivoting."""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from corpus_quality import (
    target_gloss_reason,
    target_metadata_reason,
    target_units,
    token_count,
)
from pipeline_common import load_pipeline_config, read_csv_or_columnar
from pivot_types import Direction, LoadedCorpus

PIVOT_POLICY = load_pipeline_config()["pivot"]

BASE_COLUMNS = [
    "row_id",
    "source_record_id",
    "lang_code",
    "formosan_sentence",
    "formosan_mt_standard",
    "standard_namespace",
    "mt_normalization_status",
    "mt_normalization_confidence",
    "mt_eval_eligible",
    "mt_standard_profile",
    "mt_standard_profile_sha256",
    "source",
    "dialect",
    "row_type",
    "xml_unit_context",
]
PROVENANCE_COLUMNS = [
    "pivot_origin",
    "pivot_provider",
    "pivot_direction",
    "pivot_source_lang",
    "pivot_target_lang",
    "pivot_source_text",
    "pivot_cache_key",
    "pivot_detected_source_lang",
]


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def pivot_candidate_reason(
    row: Mapping[str, Any],
    direction: Direction,
) -> str:
    """Return why a row must not be sent to DeepL, or an empty string."""
    row_type = str(row.get("row_type") or "").strip().casefold()
    allowed_types = {
        str(value).strip().casefold()
        for value in PIVOT_POLICY["eligible_row_types"]
    }
    if row_type not in allowed_types:
        return "non_sentence"
    if PIVOT_POLICY["require_mt_eval_eligible"] and not bool_value(
        row.get("mt_eval_eligible")
    ):
        return "mt_ineligible"
    if (
        str(row.get("mt_normalization_confidence") or "")
        .strip()
        .casefold()
        == "ambiguous"
    ):
        return "ambiguous_standardization"

    formosan = str(row.get("formosan_sentence") or "").strip()
    if token_count(formosan) < int(PIVOT_POLICY["min_formosan_tokens"]):
        return "short_formosan"

    source_text = str(row.get(direction.source_text_col) or "").strip()
    source_language = direction.source_language
    if target_units(source_text, source_language) < int(
        PIVOT_POLICY["min_source_units"]
    ):
        return "short_pivot_source"

    gloss_reason = target_gloss_reason(
        source_text,
        translation_kind=row.get("translation_kind", ""),
        target_language=source_language,
    )
    if gloss_reason:
        return gloss_reason
    metadata_reason = target_metadata_reason(source_text, source=formosan)
    if metadata_reason:
        return metadata_reason
    return ""


def read_corpus(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return read_csv_or_columnar(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )


def validate_columns(
    frame: pd.DataFrame,
    path: Path,
    columns: Iterable[str],
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")


def validate_mt_standard_contract(
    frame: pd.DataFrame,
    path: Path,
) -> dict[str, str]:
    validate_columns(frame, path, BASE_COLUMNS)
    if not frame["standard_namespace"].astype(str).eq("formosan-mt").all():
        raise SystemExit(f"{path} contains rows outside the Formosan MT namespace")
    if not frame["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"{path} contains non-accepted MT standardization rows")
    if not frame["formosan_sentence"].astype(str).eq(
        frame["formosan_mt_standard"].astype(str)
    ).all():
        raise SystemExit(
            f"{path} violates the Formosan MT-standard alias contract"
        )
    profile_ids = set(frame["mt_standard_profile"].astype(str).str.strip())
    profile_hashes = set(
        frame["mt_standard_profile_sha256"].astype(str).str.strip()
    )
    if len(profile_ids) != 1 or not next(iter(profile_ids), ""):
        raise SystemExit(
            f"{path} does not identify one MT standardization profile"
        )
    profile_hash = next(iter(profile_hashes), "")
    if len(profile_hashes) != 1 or len(profile_hash) != 64:
        raise SystemExit(
            f"{path} does not identify one valid MT standardization profile hash"
        )
    return {"id": next(iter(profile_ids)), "sha256": profile_hash}


def load_source_corpora(
    directions: Iterable[Direction],
) -> dict[Path, LoadedCorpus]:
    loaded: dict[Path, LoadedCorpus] = {}
    for direction in directions:
        for path in (direction.source_path, direction.original_target_path):
            if path in loaded:
                continue
            frame = read_corpus(path, "pivot source corpus")
            loaded[path] = LoadedCorpus(
                path=path,
                frame=frame,
                profile=validate_mt_standard_contract(frame, path),
            )
    profiles = {tuple(corpus.profile.values()) for corpus in loaded.values()}
    if len(profiles) != 1:
        raise SystemExit(
            "Pivot sources do not share one MT standardization profile"
        )
    return loaded


def normalize_key_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.split()).strip()


def frame_records(
    frame: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> Iterable[dict[str, Any]]:
    """Iterate rows without constructing a pandas Series for each row."""
    names = list(columns) if columns is not None else list(frame.columns)
    for values in frame.loc[:, names].itertuples(index=False, name=None):
        yield dict(zip(names, values, strict=True))


def formosan_key(row: Mapping[str, Any]) -> tuple[str, str]:
    lang_code = str(row.get("lang_code", "") or "").strip().lower()
    formosan = normalize_key_text(row.get("formosan_sentence", ""))
    return lang_code, formosan


def target_formosan_keys(
    frame: pd.DataFrame,
) -> set[tuple[str, str]]:
    languages = frame["lang_code"].astype(str).str.strip().str.lower()
    forms = frame["formosan_sentence"].map(normalize_key_text)
    return {
        (language, formosan)
        for language, formosan in zip(languages, forms, strict=True)
        if language and formosan
    }
