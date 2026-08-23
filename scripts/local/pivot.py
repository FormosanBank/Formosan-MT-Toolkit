#!/usr/bin/env python3
"""
Build DeepL-pivoted multilingual corpora.

The script creates:

1. A Chinese-target pivot corpus:
   original big_corpus_zh.csv rows
   + Formosan/English rows from big_corpus_en.csv with English translated to Chinese.

2. An English-target pivot corpus:
   original big_corpus_en.csv rows
   + Formosan/Chinese rows from big_corpus_zh.csv with Chinese translated to English.

Successful DeepL responses are appended to per-direction JSONL caches immediately.
If a run is interrupted or hits quota, rerun the same command and cached translations
will be reused.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import requests
from corpus_quality import (
    exact_key,
    normalize_text,
    quality_decision,
    target_gloss_reason,
    target_metadata_reason,
    target_units,
    token_count,
)
from dotenv import load_dotenv
from pipeline_common import (
    atomic_write_json,
    content_row_id,
    load_pipeline_config,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from tqdm import tqdm

DEEPL_MAX_TEXTS_PER_REQUEST = 50
DEEPL_MAX_REQUEST_BYTES = 128 * 1024
DEFAULT_SAFE_REQUEST_BYTES = 120 * 1024
PROVIDER = "deepl"
PIPELINE_CONFIG = load_pipeline_config()
PIVOT_POLICY = PIPELINE_CONFIG["pivot"]
MAX_TRAINING_UNITS_PER_SIDE = PIPELINE_CONFIG["cleaning"][
    "max_training_units_per_side"
]

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


class DeepLRuntimeError(RuntimeError):
    """Base class for DeepL runtime errors."""


class DeepLQuotaExceeded(DeepLRuntimeError):
    """Raised when DeepL reports quota exhaustion."""


class DeepLFatalError(DeepLRuntimeError):
    """Raised for non-retryable DeepL API errors."""


@dataclass
class Direction:
    name: str
    source_path: Path
    original_target_path: Path
    source_text_col: str
    target_text_col: str
    source_language: str
    deepl_source_lang: str
    deepl_target_lang: str
    output_filename: str
    cache_filename: str


@dataclass
class DirectionStats:
    direction: str
    source_rows: int = 0
    original_rows: int = 0
    candidate_rows: int = 0
    ineligible_source_rows: int = 0
    candidate_exclusion_counts: dict[str, int] = field(default_factory=dict)
    empty_source_rows: int = 0
    cached_unique_before: int = 0
    missing_unique_before: int = 0
    translated_unique: int = 0
    translated_chars: int = 0
    target_overlap_rows_skipped: int = 0
    target_overlap_unique_skipped: int = 0
    deferred_by_budget_unique: int = 0
    deferred_by_budget_chars: int = 0
    skipped_over_request_limit: int = 0
    stopped_reason: Optional[str] = None
    synthetic_rows_available: int = 0
    synthetic_rows_missing: int = 0
    synthetic_rows_quarantined: int = 0
    synthetic_rows_written: int = 0
    duplicate_rows_skipped: int = 0
    split_overrides: int = 0
    output_rows: int = 0
    errors: int = 0
    cache_path: Optional[str] = None
    read_cache_paths: Optional[list[str]] = None
    cache_conflicts: int = 0
    cache_conflict_path: Optional[str] = None
    cache_conflict_sha256: Optional[str] = None
    output_path: Optional[str] = None
    quarantine_path: Optional[str] = None
    quarantine_sha256: Optional[str] = None


@dataclass
class CharBudget:
    remaining: Optional[int]

    def take(self, amount: int) -> None:
        if self.remaining is not None:
            self.remaining -= amount

    def exhausted(self) -> bool:
        return self.remaining is not None and self.remaining <= 0


@dataclass
class TranslationJob:
    key: str
    text: str
    chars: int


def bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def pivot_candidate_reason(row: Mapping[str, Any], direction: Direction) -> str:
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
    if str(row.get("mt_normalization_confidence") or "").strip().casefold() == "ambiguous":
        return "ambiguous_standardization"

    formosan = str(row.get("formosan_sentence") or "").strip()
    if token_count(formosan) < int(PIVOT_POLICY["min_formosan_tokens"]):
        return "short_formosan"

    source_text = str(row.get(direction.source_text_col) or "").strip()
    source_language = direction.source_language
    if target_units(source_text, source_language) < int(PIVOT_POLICY["min_source_units"]):
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


@dataclass
class OutputBuildResult:
    original_rows: int = 0
    synthetic_rows_available: int = 0
    synthetic_rows_missing: int = 0
    synthetic_rows_quarantined: int = 0
    synthetic_rows_written: int = 0
    target_overlap_rows_skipped: int = 0
    duplicate_rows_skipped: int = 0
    split_overrides: int = 0
    output_rows: int = 0
    incomplete_path: Optional[str] = None
    quarantine_path: Optional[str] = None
    quarantine_sha256: Optional[str] = None


@dataclass
class DeepLKey:
    env_name: str
    auth_key: str
    api_base: str


class DeepLClient:
    def __init__(
        self,
        keys: list[DeepLKey],
        timeout: float,
        max_retries: int,
        retry_backoff: float,
    ) -> None:
        if not keys:
            raise ValueError("DeepLClient requires at least one API key.")
        self.keys = keys
        self.key_index = 0
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.retry_backoff = retry_backoff
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "User-Agent": "FormosanMT-Pivot/1.0",
            }
        )
        self._apply_current_key()

    @property
    def current_key(self) -> DeepLKey:
        return self.keys[self.key_index]

    @property
    def active_env_name(self) -> str:
        return self.current_key.env_name

    def _apply_current_key(self) -> None:
        self.session.headers["Authorization"] = f"DeepL-Auth-Key {self.current_key.auth_key}"

    def _advance_key(self) -> bool:
        if self.key_index + 1 >= len(self.keys):
            return False
        exhausted_name = self.current_key.env_name
        self.key_index += 1
        self._apply_current_key()
        print(
            f"DeepL key {exhausted_name} exhausted; switching to {self.current_key.env_name}.",
            file=sys.stderr,
        )
        return True

    def translate(
        self,
        texts: list[str],
        *,
        source_lang: str,
        target_lang: str,
        split_sentences: str,
        preserve_formatting: bool,
        model_type: Optional[str],
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "text": texts,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "split_sentences": split_sentences,
            "preserve_formatting": preserve_formatting,
        }
        if model_type:
            payload["model_type"] = model_type

        last_quota_error = ""
        while True:
            url = f"{self.current_key.api_base}/v2/translate"
            last_error = ""
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = self.session.post(url, json=payload, timeout=self.timeout)
                except requests.RequestException as exc:
                    last_error = str(exc)
                    if attempt == self.max_retries:
                        raise DeepLRuntimeError(f"DeepL request failed: {last_error}") from exc
                    sleep_for = self.retry_backoff * attempt
                    time.sleep(sleep_for)
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    translations = data.get("translations", [])
                    if len(translations) != len(texts):
                        raise DeepLRuntimeError(
                            "DeepL returned a different number of translations "
                            f"({len(translations)}) than inputs ({len(texts)})."
                        )
                    return translations

                body = _safe_response_text(resp)
                if resp.status_code == 456:
                    last_quota_error = f"{self.current_key.env_name}: {body}"
                    if self._advance_key():
                        break
                    raise DeepLQuotaExceeded(f"All DeepL API keys exhausted. Last error: {last_quota_error}")
                if resp.status_code in {401, 403, 404}:
                    bad_name = self.current_key.env_name
                    print(
                        f"DeepL key {bad_name} is invalid or forbidden (HTTP {resp.status_code}); skipping it.",
                        file=sys.stderr,
                    )
                    if self._advance_key():
                        break
                    raise DeepLFatalError(
                        f"All DeepL API keys failed. Last error from {bad_name}: HTTP {resp.status_code}: {body}"
                    )
                if resp.status_code == 400:
                    raise DeepLFatalError(f"DeepL HTTP 400 using {self.current_key.env_name}: {body}")

                retry_after = resp.headers.get("Retry-After")
                if resp.status_code in {408, 409, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                    if retry_after and retry_after.isdigit():
                        sleep_for = float(retry_after)
                    else:
                        sleep_for = self.retry_backoff * attempt
                    time.sleep(sleep_for)
                    last_error = f"HTTP {resp.status_code}: {body}"
                    continue

                raise DeepLRuntimeError(f"DeepL HTTP {resp.status_code} using {self.current_key.env_name}: {body}")
            else:
                raise DeepLRuntimeError(f"DeepL request failed after retries: {last_error}")

    def usage(self) -> Optional[dict[str, Any]]:
        url = f"{self.current_key.api_base}/v2/usage"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                print(
                    f"Warning: could not read DeepL usage for {self.current_key.env_name}: HTTP {resp.status_code}",
                    file=sys.stderr,
                )
                return None
            return resp.json()
        except requests.RequestException as exc:
            print(f"Warning: could not read DeepL usage for {self.current_key.env_name}: {exc}", file=sys.stderr)
            return None


def _safe_response_text(resp: requests.Response) -> str:
    text = resp.text.strip()
    if len(text) > 500:
        text = text[:500] + "..."
    return text or resp.reason


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "README.md").exists() and (parent / "config" / "corpus_pipeline.json").exists():
            return parent
    raise RuntimeError(f"Could not locate Formosan-MT-Toolkit project root from {start}")


def now_iso() -> str:
    return utc_now()


def read_corpus(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def validate_columns(df: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {missing}")


def validate_mt_standard_contract(df: pd.DataFrame, path: Path) -> dict[str, str]:
    validate_columns(df, path, BASE_COLUMNS)
    if not df["standard_namespace"].astype(str).eq("formosan-mt").all():
        raise SystemExit(f"{path} contains rows outside the Formosan MT namespace")
    if not df["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"{path} contains non-accepted MT standardization rows")
    if not df["formosan_sentence"].astype(str).eq(
        df["formosan_mt_standard"].astype(str)
    ).all():
        raise SystemExit(f"{path} violates the Formosan MT-standard alias contract")
    profile_ids = set(df["mt_standard_profile"].astype(str).str.strip())
    profile_hashes = set(df["mt_standard_profile_sha256"].astype(str).str.strip())
    if len(profile_ids) != 1 or not next(iter(profile_ids), ""):
        raise SystemExit(f"{path} does not identify one MT standardization profile")
    profile_hash = next(iter(profile_hashes), "")
    if len(profile_hashes) != 1 or len(profile_hash) != 64:
        raise SystemExit(f"{path} does not identify one valid MT standardization profile hash")
    return {
        "id": next(iter(profile_ids)),
        "sha256": profile_hash,
    }


def normalize_key_text(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    return " ".join(text.split()).strip()


def formosan_key(row: pd.Series) -> tuple[str, str]:
    lang_code = str(row.get("lang_code", "") or "").strip().lower()
    formosan = normalize_key_text(row.get("formosan_sentence", ""))
    return lang_code, formosan


def choose_deepl_api_base(auth_key: str, override: Optional[str]) -> str:
    if override:
        return override.rstrip("/")
    if auth_key.endswith(":fx"):
        return "https://api-free.deepl.com"
    return "https://api.deepl.com"


def discover_api_key_envs(environ: Optional[Mapping[str, str]] = None) -> list[str]:
    """Return configured DEEPL_API_KEY variables in stable numeric order."""
    source = os.environ if environ is None else environ
    names: list[tuple[int, str]] = []
    for env_name, value in source.items():
        match = re.fullmatch(r"DEEPL_API_KEY(?:_(\d+))?", env_name)
        if match and str(value).strip():
            suffix = int(match.group(1) or 1)
            names.append((suffix, env_name))
    return [env_name for _, env_name in sorted(names, key=lambda item: (item[0], item[1]))]


def parse_api_key_envs(raw: str) -> list[str]:
    if str(raw or "").strip().lower() == "auto":
        return discover_api_key_envs()
    envs = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for env_name in envs:
        if env_name not in seen:
            out.append(env_name)
            seen.add(env_name)
    return out


def load_deepl_keys(env_names: list[str], api_base_override: Optional[str]) -> list[DeepLKey]:
    keys: list[DeepLKey] = []
    for env_name in env_names:
        auth_key = os.getenv(env_name, "").strip()
        if not auth_key:
            continue
        keys.append(
            DeepLKey(
                env_name=env_name,
                auth_key=auth_key,
                api_base=choose_deepl_api_base(auth_key, api_base_override),
            )
        )
    return keys


def read_deepl_usage_for_key(key: DeepLKey, timeout: float) -> Optional[dict[str, Any]]:
    headers = {"Authorization": f"DeepL-Auth-Key {key.auth_key}"}
    try:
        resp = requests.get(f"{key.api_base}/v2/usage", headers=headers, timeout=timeout)
        if resp.status_code != 200:
            print(
                f"Warning: could not read DeepL usage for {key.env_name}: HTTP {resp.status_code}",
                file=sys.stderr,
            )
            return None
        usage = resp.json()
        usage["api_key_env"] = key.env_name
        return usage
    except requests.RequestException as exc:
        print(f"Warning: could not read DeepL usage for {key.env_name}: {exc}", file=sys.stderr)
        return None


def make_cache_key(
    *,
    provider: str,
    source_lang: str,
    target_lang: str,
    text: str,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
) -> str:
    payload = {
        "provider": provider,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "split_sentences": split_sentences,
        "preserve_formatting": preserve_formatting,
        "model_type": model_type or "",
        "text": text,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache

    with cache_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed DeepL cache record {cache_path}:{line_no}: {exc}") from exc
            key = record.get("key")
            translation = record.get("translation")
            if not isinstance(key, str) or not isinstance(translation, str):
                raise RuntimeError(f"Invalid DeepL cache record {cache_path}:{line_no}: missing string key/translation")
            expected_key = make_cache_key(
                provider=str(record.get("provider") or PROVIDER),
                source_lang=str(record.get("source_lang") or ""),
                target_lang=str(record.get("target_lang") or ""),
                text=str(record.get("text") or ""),
                split_sentences=str(record.get("split_sentences") or "0"),
                preserve_formatting=bool(record.get("preserve_formatting", True)),
                model_type=(str(record.get("model_type_requested")) if record.get("model_type_requested") else None),
            )
            if key != expected_key:
                raise RuntimeError(f"DeepL cache key mismatch at {cache_path}:{line_no}; cache may be corrupt")
            existing = cache.get(key)
            if existing is not None and existing.get("translation") != translation:
                raise RuntimeError(f"Conflicting DeepL translations for cache key {key} in {cache_path}")
            cache[key] = record
    return cache


def load_cache_chain(
    cache_paths: Iterable[Path],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Load cache layers in increasing priority order and audit conflicts."""
    merged: dict[str, dict[str, Any]] = {}
    selected_paths: dict[str, Path] = {}
    conflicts: list[dict[str, Any]] = []
    for cache_path in cache_paths:
        for key, record in load_cache(cache_path).items():
            existing = merged.get(key)
            if existing is not None and existing.get("translation") != record.get("translation"):
                conflicts.append(
                    {
                        "cache_key": key,
                        "text": str(record.get("text") or existing.get("text") or ""),
                        "lower_priority_cache": str(selected_paths[key]),
                        "lower_priority_created_at": str(existing.get("created_at") or ""),
                        "lower_priority_translation": str(existing.get("translation") or ""),
                        "higher_priority_cache": str(cache_path),
                        "higher_priority_created_at": str(record.get("created_at") or ""),
                        "selected_translation": str(record.get("translation") or ""),
                        "selection_policy": "later_cache_wins",
                    }
                )
            merged[key] = record
            selected_paths[key] = cache_path
    return merged, conflicts


def write_jsonl_atomic(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def append_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def request_body_size(
    texts: list[str],
    *,
    source_lang: str,
    target_lang: str,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
) -> int:
    payload: dict[str, Any] = {
        "text": texts,
        "source_lang": source_lang,
        "target_lang": target_lang,
        "split_sentences": split_sentences,
        "preserve_formatting": preserve_formatting,
    }
    if model_type:
        payload["model_type"] = model_type
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def batch_jobs(
    jobs: Iterable[TranslationJob],
    *,
    max_texts: int,
    max_request_bytes: int,
    source_lang: str,
    target_lang: str,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
) -> Iterable[list[TranslationJob]]:
    batch: list[TranslationJob] = []
    for job in jobs:
        one_size = request_body_size(
            [job.text],
            source_lang=source_lang,
            target_lang=target_lang,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )
        if one_size > max_request_bytes:
            if batch:
                yield batch
                batch = []
            yield [job]
            continue

        candidate = batch + [job]
        candidate_size = request_body_size(
            [j.text for j in candidate],
            source_lang=source_lang,
            target_lang=target_lang,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )
        if batch and (len(candidate) > max_texts or candidate_size > max_request_bytes):
            yield batch
            batch = [job]
        else:
            batch = candidate

    if batch:
        yield batch


def candidate_jobs(
    df: pd.DataFrame,
    *,
    text_col: str,
    direction: Direction,
    cache: dict[str, dict[str, Any]],
    force: bool,
    target_keys: set[tuple[str, str]],
    skip_target_overlaps: bool,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
    budget: CharBudget,
    stats: DirectionStats,
) -> list[TranslationJob]:
    jobs: list[TranslationJob] = []
    seen: set[str] = set()
    seen_overlap: set[tuple[str, str]] = set()

    for _, row in df.iterrows():
        exclusion = pivot_candidate_reason(row, direction)
        if exclusion:
            stats.ineligible_source_rows += 1
            stats.candidate_exclusion_counts[exclusion] = (
                stats.candidate_exclusion_counts.get(exclusion, 0) + 1
            )
            continue
        stats.candidate_rows += 1
        f_key = formosan_key(row)
        if skip_target_overlaps and f_key[0] and f_key[1] and f_key in target_keys:
            stats.target_overlap_rows_skipped += 1
            seen_overlap.add(f_key)
            continue

        text = str(row.get(text_col, "")).strip()
        if not text:
            stats.empty_source_rows += 1
            continue

        key = make_cache_key(
            provider=PROVIDER,
            source_lang=direction.deepl_source_lang,
            target_lang=direction.deepl_target_lang,
            text=text,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )

        if key in seen:
            continue
        seen.add(key)

        if key in cache and not force:
            continue

        chars = len(text)
        if budget.remaining is not None and chars > budget.remaining:
            stats.deferred_by_budget_unique += 1
            stats.deferred_by_budget_chars += chars
            continue
        jobs.append(TranslationJob(key=key, text=text, chars=chars))
        budget.take(chars)

    if stats.deferred_by_budget_unique:
        stats.stopped_reason = "local character budget exhausted; deferred remaining missing translations"
    stats.target_overlap_unique_skipped = len(seen_overlap)
    return jobs


def translate_direction(
    direction: Direction,
    *,
    args: argparse.Namespace,
    client: Optional[DeepLClient],
    budget: CharBudget,
) -> tuple[dict[str, dict[str, Any]], DirectionStats]:
    stats = DirectionStats(direction=direction.name)
    source_df = read_corpus(direction.source_path, f"{direction.name} source corpus")
    target_df = read_corpus(direction.original_target_path, f"{direction.name} target corpus")
    validate_columns(source_df, direction.source_path, [*BASE_COLUMNS, direction.source_text_col])
    validate_columns(target_df, direction.original_target_path, [*BASE_COLUMNS, direction.target_text_col])
    source_profile = validate_mt_standard_contract(source_df, direction.source_path)
    target_profile = validate_mt_standard_contract(target_df, direction.original_target_path)
    if source_profile != target_profile:
        raise SystemExit(
            f"{direction.name} source and target corpora use different MT standard profiles"
        )
    stats.source_rows = len(source_df)
    stats.original_rows = len(target_df)
    target_keys = target_formosan_keys(target_df)

    read_cache_paths = [cache_dir / direction.cache_filename for cache_dir in getattr(args, "read_cache_dir", [])]
    cache_path = args.cache_dir / direction.cache_filename
    error_path = args.cache_dir / direction.cache_filename.replace(".jsonl", ".errors.jsonl")
    cache, cache_conflicts = load_cache_chain([*read_cache_paths, cache_path])
    stats.cache_path = str(cache_path)
    stats.read_cache_paths = [str(path) for path in read_cache_paths]
    stats.cached_unique_before = len(cache)
    stats.cache_conflicts = len(cache_conflicts)
    conflict_path = args.out_dir / f"pivot_cache_conflicts_{direction.name}.jsonl"
    if cache_conflicts:
        print(
            f"{direction.name}: {len(cache_conflicts):,} cache conflicts; "
            f"using the later, higher-priority cache layer"
        )
        if not args.dry_run:
            write_jsonl_atomic(conflict_path, cache_conflicts)
            stats.cache_conflict_path = str(conflict_path)
            stats.cache_conflict_sha256 = sha256_file(conflict_path)
    elif not args.dry_run:
        conflict_path.unlink(missing_ok=True)

    jobs = candidate_jobs(
        source_df,
        text_col=direction.source_text_col,
        direction=direction,
        cache=cache,
        force=args.force,
        target_keys=target_keys,
        skip_target_overlaps=args.skip_target_overlaps,
        split_sentences=args.split_sentences,
        preserve_formatting=args.preserve_formatting,
        model_type=args.model_type,
        budget=budget,
        stats=stats,
    )
    stats.missing_unique_before = len(jobs)

    planned_chars = sum(job.chars for job in jobs)
    deferred = (
        f", {stats.deferred_by_budget_unique:,} deferred by budget ({stats.deferred_by_budget_chars:,} chars)"
        if stats.deferred_by_budget_unique
        else ""
    )
    print(
        f"{direction.name}: {stats.candidate_rows:,} candidate rows, "
        f"{stats.ineligible_source_rows:,} ineligible rows, "
        f"{stats.target_overlap_rows_skipped:,} target-overlap rows skipped, "
        f"{stats.cached_unique_before:,} cached unique translations, "
        f"{len(jobs):,} missing unique translations, {planned_chars:,} chars planned"
        f"{deferred}"
    )

    if args.dry_run or args.skip_translation:
        return cache, stats

    if client is None:
        raise SystemExit("DeepL client is not configured.")

    pbar = tqdm(total=len(jobs), desc=f"DeepL {direction.name}", unit="text")
    try:
        for batch in batch_jobs(
            jobs,
            max_texts=args.batch_size,
            max_request_bytes=args.max_request_bytes,
            source_lang=direction.deepl_source_lang,
            target_lang=direction.deepl_target_lang,
            split_sentences=args.split_sentences,
            preserve_formatting=args.preserve_formatting,
            model_type=args.model_type,
        ):
            oversized = [
                job
                for job in batch
                if request_body_size(
                    [job.text],
                    source_lang=direction.deepl_source_lang,
                    target_lang=direction.deepl_target_lang,
                    split_sentences=args.split_sentences,
                    preserve_formatting=args.preserve_formatting,
                    model_type=args.model_type,
                )
                > args.max_request_bytes
            ]
            if oversized:
                records = [
                    {
                        "created_at": now_iso(),
                        "direction": direction.name,
                        "error": "single text exceeds configured request byte limit",
                        "key": job.key,
                        "source_lang": direction.deepl_source_lang,
                        "target_lang": direction.deepl_target_lang,
                        "text": job.text,
                        "chars": job.chars,
                    }
                    for job in oversized
                ]
                append_jsonl(error_path, records)
                stats.skipped_over_request_limit += len(oversized)
                stats.errors += len(oversized)
                pbar.update(len(oversized))
                batch = [job for job in batch if job not in oversized]
                if not batch:
                    continue

            texts = [job.text for job in batch]
            try:
                translations = client.translate(
                    texts,
                    source_lang=direction.deepl_source_lang,
                    target_lang=direction.deepl_target_lang,
                    split_sentences=args.split_sentences,
                    preserve_formatting=args.preserve_formatting,
                    model_type=args.model_type,
                )
            except DeepLQuotaExceeded as exc:
                stats.stopped_reason = "DeepL quota exceeded"
                append_jsonl(
                    error_path,
                    [
                        {
                            "created_at": now_iso(),
                            "direction": direction.name,
                            "error": str(exc),
                            "batch_size": len(batch),
                            "batch_chars": sum(job.chars for job in batch),
                        }
                    ],
                )
                break

            if len(translations) != len(batch):
                raise DeepLFatalError(f"DeepL returned {len(translations)} translations for a batch of {len(batch)}")

            records: list[dict[str, Any]] = []
            for job, translated in zip(batch, translations, strict=True):
                translation_text = str(translated.get("text", ""))
                record = {
                    "created_at": now_iso(),
                    "provider": PROVIDER,
                    "direction": direction.name,
                    "key": job.key,
                    "source_lang": direction.deepl_source_lang,
                    "target_lang": direction.deepl_target_lang,
                    "text": job.text,
                    "translation": translation_text,
                    "detected_source_language": translated.get("detected_source_language"),
                    "model_type_used": translated.get("model_type_used"),
                    "api_key_env": client.active_env_name,
                    "chars": job.chars,
                    "split_sentences": args.split_sentences,
                    "preserve_formatting": args.preserve_formatting,
                    "model_type_requested": args.model_type,
                }
                records.append(record)
                cache[job.key] = record

            append_jsonl(cache_path, records)
            batch_chars = sum(job.chars for job in batch)
            stats.translated_unique += len(batch)
            stats.translated_chars += batch_chars
            pbar.update(len(batch))
    finally:
        pbar.close()

    return cache, stats


def target_formosan_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for _, row in df.iterrows():
        key = formosan_key(row)
        if key[0] and key[1]:
            keys.add(key)
    return keys


def output_columns(
    original_df: pd.DataFrame,
    source_df: pd.DataFrame,
    direction: Direction,
) -> list[str]:
    columns = list(original_df.columns)
    for column in source_df.columns:
        if column != direction.source_text_col and column not in columns:
            columns.append(column)
    if direction.target_text_col not in columns:
        insert_at = columns.index("formosan_sentence") + 1
        columns.insert(insert_at, direction.target_text_col)
    for column in PROVENANCE_COLUMNS:
        if column not in columns:
            columns.append(column)
    for column in (
        "target_raw",
        "target_transformations",
        "quality_flags",
        "pair_fingerprint",
        "duplicate_group_size",
    ):
        if column not in columns:
            columns.append(column)
    return columns


def target_profile(direction: Direction) -> tuple[str, str]:
    return ("english", "eng") if direction.target_text_col == "english_sentence" else ("chinese", "zho")


def detected_source_mismatch(record: dict[str, Any], direction: Direction) -> bool:
    detected = str(record.get("detected_source_language") or "").strip().upper()
    if not detected:
        return False
    expected = direction.deepl_source_lang.split("-")[0].upper()
    return detected.split("-")[0] != expected


def synthetic_row(
    source_row: pd.Series,
    record: dict[str, Any],
    direction: Direction,
    cache_key: str,
) -> tuple[dict[str, Any] | None, str]:
    if detected_source_mismatch(record, direction):
        return None, "deepl_detected_source_language_mismatch"
    raw_translation = str(record.get("translation") or "")
    normalized = normalize_text(raw_translation)
    if not normalized.text:
        return None, "empty_pivot_translation"

    row = {str(column): value for column, value in source_row.to_dict().items()}
    row.pop(direction.source_text_col, None)
    row[direction.target_text_col] = normalized.text
    row["target_raw"] = raw_translation
    row["target_transformations"] = "|".join(normalized.transformations)
    row["target_lang"] = target_profile(direction)[1]
    row["translation_index"] = ""
    row["translation_kind"] = "synthetic"
    row["translation_version"] = str(record.get("model_type_used") or "")
    row["split"] = "train"
    row["pivot_origin"] = "synthetic"
    row["pivot_provider"] = PROVIDER
    row["pivot_direction"] = direction.name
    row["pivot_source_lang"] = direction.deepl_source_lang
    row["pivot_target_lang"] = direction.deepl_target_lang
    row["pivot_source_text"] = str(record.get("text") or "")
    row["pivot_cache_key"] = cache_key
    row["pivot_detected_source_lang"] = str(record.get("detected_source_language") or "")
    row["row_id"] = content_row_id(
        source_row.get("row_id", ""),
        direction.name,
        cache_key,
    )
    row["content_sha256"] = sha256_bytes(
        "\u241f".join(
            [
                str(row.get("lang_code") or ""),
                str(row.get("formosan_sentence") or ""),
                normalized.text,
                target_profile(direction)[1],
                str(row.get("row_type") or ""),
            ]
        ).encode("utf-8")
    )
    existing_flags = [value for value in str(row.get("quality_flags") or "").split("|") if value]
    row["quality_flags"] = "|".join(sorted(set([*existing_flags, "synthetic"])))
    row["duplicate_group_size"] = 1
    row["pair_fingerprint"] = sha256_bytes(
        (exact_key(str(row.get("formosan_sentence") or "")) + "\u241f" + exact_key(normalized.text)).encode("utf-8")
    )

    decision = quality_decision(
        pd.Series(row),
        source_column="formosan_sentence",
        target_column=direction.target_text_col,
        target_language=target_profile(direction)[0],
        keep_redactions=False,
        max_units_per_side=MAX_TRAINING_UNITS_PER_SIDE,
    )
    if decision.disposition != "accepted":
        return None, f"pivot_quality:{decision.reason}"
    row["quality_flags"] = "|".join(
        sorted(
            set(
                [
                    *[value for value in str(row.get("quality_flags") or "").split("|") if value],
                    *decision.flags,
                ]
            )
        )
    )
    return row, ""


def write_pivot_output(
    direction: Direction,
    *,
    args: argparse.Namespace,
    cache: dict[str, dict[str, Any]],
) -> OutputBuildResult:
    result = OutputBuildResult()
    original_df = read_corpus(direction.original_target_path, f"{direction.name} original target corpus")
    source_df = read_corpus(direction.source_path, f"{direction.name} source corpus")
    result.original_rows = len(original_df)

    validate_columns(original_df, direction.original_target_path, [*BASE_COLUMNS, direction.target_text_col])
    validate_columns(source_df, direction.source_path, [*BASE_COLUMNS, direction.source_text_col])
    source_profile = validate_mt_standard_contract(source_df, direction.source_path)
    target_profile = validate_mt_standard_contract(original_df, direction.original_target_path)
    if source_profile != target_profile:
        raise SystemExit(
            f"{direction.name} source and target corpora use different MT standard profiles"
        )

    target_keys = target_formosan_keys(original_df)
    output_path = args.out_dir / direction.output_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    incomplete_path = output_path.with_suffix(output_path.suffix + ".incomplete")
    rejection_path = args.out_dir / f"pivot_rejections_{direction.name}.csv"
    output_cols = output_columns(original_df, source_df, direction)
    seen_rows: set[tuple[str, str, str]] = set()
    rejections: list[dict[str, str]] = []

    with tmp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_cols, extrasaction="ignore")
        writer.writeheader()

        for _, row in original_df.iterrows():
            target_text = str(row.get(direction.target_text_col, "")).strip()
            formosan = str(row.get("formosan_sentence", "")).strip()
            lang_code = str(row.get("lang_code", "")).strip()
            if not target_text or not formosan:
                raise RuntimeError(f"Original {direction.name} target corpus contains an empty required row")
            if str(row.get("kindOf") or "standard").strip().lower() != "standard":
                raise RuntimeError(f"Original {direction.name} target corpus contains a non-standard row")
            dedupe_key = (
                lang_code,
                exact_key(formosan),
                exact_key(target_text),
            )
            if args.dedupe and dedupe_key in seen_rows:
                result.duplicate_rows_skipped += 1
                continue
            seen_rows.add(dedupe_key)
            out_row = row.to_dict()
            out_row["pivot_origin"] = str(row.get("pivot_origin") or "original")
            for column in PROVENANCE_COLUMNS:
                out_row.setdefault(column, "")
            writer.writerow(out_row)
            result.output_rows += 1

        result.synthetic_rows_available = 0
        result.synthetic_rows_missing = 0

        for _, row in tqdm(
            source_df.iterrows(),
            total=len(source_df),
            desc=f"write {direction.name}",
            unit="row",
            disable=args.quiet,
        ):
            if pivot_candidate_reason(row, direction):
                continue
            source_text = str(row.get(direction.source_text_col, "")).strip()
            formosan = str(row.get("formosan_sentence", "")).strip()
            lang_code = str(row.get("lang_code", "")).strip()
            if not source_text or not formosan:
                continue
            f_key = formosan_key(row)
            if args.skip_target_overlaps and f_key[0] and f_key[1] and f_key in target_keys:
                result.target_overlap_rows_skipped += 1
                continue

            key = make_cache_key(
                provider=PROVIDER,
                source_lang=direction.deepl_source_lang,
                target_lang=direction.deepl_target_lang,
                text=source_text,
                split_sentences=args.split_sentences,
                preserve_formatting=args.preserve_formatting,
                model_type=args.model_type,
            )
            record = cache.get(key)
            if not record:
                result.synthetic_rows_missing += 1
                continue

            out_row, rejection_reason = synthetic_row(
                row,
                record,
                direction,
                key,
            )
            if out_row is None:
                result.synthetic_rows_quarantined += 1
                rejections.append(
                    {
                        "direction": direction.name,
                        "row_id": str(row.get("row_id") or ""),
                        "source_record_id": str(row.get("source_record_id") or ""),
                        "pivot_cache_key": key,
                        "reason": rejection_reason,
                        "source_text": source_text,
                        "translation": str(record.get("translation") or ""),
                    }
                )
                continue
            target_text = str(out_row[direction.target_text_col])
            dedupe_key = (
                lang_code,
                exact_key(formosan),
                exact_key(target_text),
            )
            if args.dedupe and dedupe_key in seen_rows:
                result.duplicate_rows_skipped += 1
                continue
            seen_rows.add(dedupe_key)
            writer.writerow(out_row)
            result.synthetic_rows_available += 1
            result.synthetic_rows_written += 1
            result.output_rows += 1

    if rejections:
        with rejection_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rejections[0]))
            writer.writeheader()
            writer.writerows(rejections)
        result.quarantine_path = str(rejection_path)
        result.quarantine_sha256 = sha256_file(rejection_path)
    else:
        rejection_path.unlink(missing_ok=True)

    if result.synthetic_rows_missing:
        output_path.unlink(missing_ok=True)
        incomplete_path.unlink(missing_ok=True)
        os.replace(tmp_path, incomplete_path)
        result.incomplete_path = str(incomplete_path)
    else:
        incomplete_path.unlink(missing_ok=True)
        os.replace(tmp_path, output_path)
    return result


def write_manifest(
    *,
    args: argparse.Namespace,
    direction_stats: list[DirectionStats],
    usage: Optional[dict[str, Any]],
    sources: dict[str, str],
) -> Path:
    manifest_path = args.out_dir / "pivot_manifest.json"
    stats_payload = [stats.__dict__ for stats in direction_stats]
    complete = all(
        not stats.stopped_reason
        and stats.errors == 0
        and stats.synthetic_rows_missing == 0
        and stats.deferred_by_budget_unique == 0
        and stats.skipped_over_request_limit == 0
        and bool(stats.output_path)
        and Path(str(stats.output_path)).is_file()
        for stats in direction_stats
    )
    source_records = {
        name: {
            "path": path,
            "sha256": sha256_file(Path(path)),
        }
        for name, path in sources.items()
    }
    source_profiles = {
        tuple(
            validate_mt_standard_contract(
                read_corpus(Path(path), name),
                Path(path),
            ).values()
        )
        for name, path in sources.items()
    }
    if len(source_profiles) != 1:
        raise SystemExit("Pivot sources do not share one MT standardization profile")
    profile_id, profile_hash = next(iter(source_profiles))
    cache_records: dict[str, dict[str, Any]] = {}
    for stats in direction_stats:
        for path_string in [*(stats.read_cache_paths or []), stats.cache_path]:
            if not path_string:
                continue
            path = Path(path_string)
            cache_records[str(path)] = {
                "exists": path.is_file(),
                "sha256": sha256_file(path) if path.is_file() else None,
                "bytes": path.stat().st_size if path.is_file() else 0,
            }
    manifest = {
        "schema_version": 3,
        "pipeline_version": load_pipeline_config()["pipeline_version"],
        "created_at": now_iso(),
        "provider": PROVIDER,
        "mt_standardization": {
            "id": profile_id,
            "sha256": profile_hash,
        },
        "sources": source_records,
        "outputs": {
            "out_dir": str(args.out_dir),
            "cache_dir": str(args.cache_dir),
        },
        "caches": cache_records,
        "settings": {
            "directions": args.directions,
            "splits": args.splits,
            "batch_size": args.batch_size,
            "max_request_bytes": args.max_request_bytes,
            "target_zh": args.target_zh,
            "target_en": args.target_en,
            "source_en": args.source_en,
            "source_zh": args.source_zh,
            "split_sentences": args.split_sentences,
            "preserve_formatting": args.preserve_formatting,
            "model_type": args.model_type,
            "include_provenance": args.include_provenance,
            "skip_target_overlaps": args.skip_target_overlaps,
            "api_key_envs": getattr(args, "api_key_env_names", parse_api_key_envs(args.api_key_env)),
            "dedupe": args.dedupe,
            "dry_run": args.dry_run,
            "skip_translation": args.skip_translation,
            "eligibility_policy": PIVOT_POLICY,
        },
        "deepl_usage_at_start": usage,
        "stats": stats_payload,
        "complete": complete,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def build_directions(args: argparse.Namespace) -> dict[str, Direction]:
    return {
        "en2zh": Direction(
            name="en2zh",
            source_path=args.big_corpus_en,
            original_target_path=args.big_corpus_zh,
            source_text_col="english_sentence",
            target_text_col="chinese_sentence",
            source_language="english",
            deepl_source_lang=args.source_en,
            deepl_target_lang=args.target_zh,
            output_filename="big_corpus_zh_pivot.csv",
            cache_filename="deepl_en_to_zh.jsonl",
        ),
        "zh2en": Direction(
            name="zh2en",
            source_path=args.big_corpus_zh,
            original_target_path=args.big_corpus_en,
            source_text_col="chinese_sentence",
            target_text_col="english_sentence",
            source_language="chinese",
            deepl_source_lang=args.source_zh,
            deepl_target_lang=args.target_en,
            output_filename="big_corpus_en_pivot.csv",
            cache_filename="deepl_zh_to_en.jsonl",
        ),
    }


def parse_directions(raw: str) -> list[str]:
    raw = (raw or "both").strip().lower()
    if raw == "both":
        return ["en2zh", "zh2en"]
    directions = [part.strip() for part in raw.split(",") if part.strip()]
    valid = {"en2zh", "zh2en"}
    bad = set(directions) - valid
    if bad:
        raise SystemExit(f"Unsupported direction(s): {sorted(bad)}. Use both, en2zh, zh2en.")
    return directions


def configure_arg_parser(project_root: Path) -> argparse.ArgumentParser:
    processed = project_root / "processed_corpora"
    ap = argparse.ArgumentParser(
        description="Create DeepL-pivoted Formosan multilingual corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--big-corpus-en", type=Path, default=processed / "big_corpus_en.csv")
    ap.add_argument("--big-corpus-zh", type=Path, default=processed / "big_corpus_zh.csv")
    ap.add_argument("--out-dir", type=Path, default=processed / "pivot")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Defaults to OUT_DIR/cache")
    ap.add_argument(
        "--read-cache-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Read existing DeepL cache records from this directory before the write cache. "
            "Can be repeated. New translations are still written only to --cache-dir."
        ),
    )

    ap.add_argument("--directions", default="both", help="both, en2zh, zh2en, or comma-separated values")
    ap.add_argument(
        "--splits",
        default="all",
        help="Compatibility option. Corpus pipeline v3 requires all rows before its single split.",
    )
    ap.add_argument(
        "--include-target-overlaps",
        dest="skip_target_overlaps",
        action="store_false",
        help=(
            "Also DeepL-translate source rows whose lang_code+formosan_sentence already exists "
            "in the target corpus. Default skips these rows to avoid billing and leakage risk."
        ),
    )
    ap.set_defaults(skip_target_overlaps=True)

    ap.add_argument(
        "--api-key-env",
        default="auto",
        help=(
            "Comma-separated environment variable names for DeepL keys, used in order. "
            "The default 'auto' discovers DEEPL_API_KEY and all DEEPL_API_KEY_N variables."
        ),
    )
    ap.add_argument("--api-base", default=None, help="Override DeepL base URL, e.g. https://api-free.deepl.com")
    ap.add_argument("--source-en", default="EN")
    ap.add_argument("--source-zh", default="ZH")
    ap.add_argument("--target-zh", default="ZH-HANT")
    ap.add_argument("--target-en", default="EN-US")
    ap.add_argument(
        "--model-type",
        default="prefer_quality_optimized",
        choices=["prefer_quality_optimized", "quality_optimized", "latency_optimized", "none"],
    )
    ap.add_argument("--split-sentences", default="0", choices=["0", "1", "nonewlines"])
    ap.add_argument("--no-preserve-formatting", dest="preserve_formatting", action="store_false")
    ap.set_defaults(preserve_formatting=True)

    ap.add_argument("--batch-size", type=int, default=DEEPL_MAX_TEXTS_PER_REQUEST)
    ap.add_argument("--max-request-bytes", type=int, default=DEFAULT_SAFE_REQUEST_BYTES)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument(
        "--max-source-chars",
        type=int,
        default=None,
        help="Local per-run character budget for source text sent to DeepL.",
    )
    ap.add_argument(
        "--respect-usage-limit",
        action="store_true",
        help="Read DeepL usage and cap this run to remaining characters when the API reports a limit.",
    )
    ap.add_argument("--reserve-chars", type=int, default=1000)

    ap.add_argument("--force", action="store_true", help="Retranslate even when a cache entry exists.")
    ap.add_argument("--skip-translation", action="store_true", help="Only build outputs from existing cache.")
    ap.add_argument("--dry-run", action="store_true", help="Plan the run without calling DeepL or writing outputs.")
    ap.add_argument("--no-write-output", action="store_true")
    ap.add_argument("--include-provenance", action="store_true", default=True)
    ap.add_argument("--minimal-schema", dest="include_provenance", action="store_false")
    ap.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Diagnostic only. Production output deduplicates after original rows win.",
    )
    ap.set_defaults(dedupe=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> None:
    project_root = find_project_root(Path(__file__))
    load_dotenv(project_root / ".env")

    parser = configure_arg_parser(project_root)
    args = parser.parse_args()

    args.big_corpus_en = args.big_corpus_en.resolve()
    args.big_corpus_zh = args.big_corpus_zh.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.cache_dir is None:
        args.cache_dir = args.out_dir / "cache"
    args.cache_dir = args.cache_dir.resolve()
    args.read_cache_dir = [path.resolve() for path in args.read_cache_dir]
    args.batch_size = max(1, min(int(args.batch_size), DEEPL_MAX_TEXTS_PER_REQUEST))
    args.max_request_bytes = max(1024, min(int(args.max_request_bytes), DEEPL_MAX_REQUEST_BYTES))
    if args.model_type == "none":
        args.model_type = None

    selected_direction_names = parse_directions(args.directions)
    if args.splits.strip().lower() not in {"all", "*"}:
        raise SystemExit("Corpus pipeline v3 requires --splits all; splitting occurs once after pivoting")
    directions = build_directions(args)

    api_key_env_names = parse_api_key_envs(args.api_key_env)
    args.api_key_env_names = api_key_env_names
    deepl_keys = load_deepl_keys(api_key_env_names, args.api_base)
    client: Optional[DeepLClient] = None
    usage: Optional[dict[str, Any]] = None
    budget = CharBudget(args.max_source_chars)

    if not args.dry_run and not args.skip_translation:
        if not deepl_keys:
            raise SystemExit(
                f"Missing DeepL API key. Checked environment variable(s): {', '.join(api_key_env_names) or '(none)'}."
            )
        client = DeepLClient(
            keys=deepl_keys,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
        print(f"DeepL API keys loaded: {', '.join(k.env_name for k in deepl_keys)}")
        if args.respect_usage_limit:
            usages = [u for key in deepl_keys if (u := read_deepl_usage_for_key(key, args.timeout))]
            usable_total = 0
            usage = {"keys": usages, "reserve_chars_per_key": int(args.reserve_chars)}
            for item in usages:
                if "character_limit" not in item or "character_count" not in item:
                    continue
                remaining = int(item["character_limit"]) - int(item["character_count"]) - int(args.reserve_chars)
                usable_total += max(0, remaining)
            if budget.remaining is None:
                budget.remaining = usable_total
            else:
                budget.remaining = min(budget.remaining, usable_total)
            print(f"DeepL reported usable remaining characters across loaded keys: {budget.remaining:,}")

    all_stats: list[DirectionStats] = []
    caches: dict[str, dict[str, dict[str, Any]]] = {}

    print(f"Project root: {project_root}")
    print(f"Output dir:   {args.out_dir}")
    print(f"Cache dir:    {args.cache_dir}")
    if args.read_cache_dir:
        print("Read caches:  " + ", ".join(str(path) for path in args.read_cache_dir))
    print(f"Directions:   {', '.join(selected_direction_names)}")
    print(f"Splits:       {args.splits}")

    for name in selected_direction_names:
        direction = directions[name]
        cache, stats = translate_direction(
            direction,
            args=args,
            client=client,
            budget=budget,
        )
        caches[name] = cache
        all_stats.append(stats)

    if not args.dry_run and not args.no_write_output:
        for stats in all_stats:
            direction = directions[stats.direction]
            result = write_pivot_output(
                direction,
                args=args,
                cache=caches[stats.direction],
            )
            stats.original_rows = result.original_rows
            stats.synthetic_rows_available = result.synthetic_rows_available
            stats.synthetic_rows_missing = result.synthetic_rows_missing
            stats.synthetic_rows_quarantined = (
                result.synthetic_rows_quarantined
            )
            stats.synthetic_rows_written = result.synthetic_rows_written
            stats.target_overlap_rows_skipped = max(
                stats.target_overlap_rows_skipped,
                result.target_overlap_rows_skipped,
            )
            stats.duplicate_rows_skipped = result.duplicate_rows_skipped
            stats.split_overrides = result.split_overrides
            stats.output_rows = result.output_rows
            stats.output_path = result.incomplete_path or str(args.out_dir / direction.output_filename)
            stats.quarantine_path = result.quarantine_path
            stats.quarantine_sha256 = result.quarantine_sha256

    manifest: Optional[Path] = None
    if not args.dry_run:
        manifest = write_manifest(
            args=args,
            direction_stats=all_stats,
            usage=usage,
            sources={
                "big_corpus_en": str(args.big_corpus_en),
                "big_corpus_zh": str(args.big_corpus_zh),
            },
        )

    print("\nDone.")
    for stats in all_stats:
        print(
            f"{stats.direction}: translated {stats.translated_unique:,} unique texts "
            f"({stats.translated_chars:,} chars); target-overlap rows skipped "
            f"{stats.target_overlap_rows_skipped:,}; synthetic rows written "
            f"{stats.synthetic_rows_written:,}; quality-quarantined "
            f"{stats.synthetic_rows_quarantined:,}; output rows "
            f"{stats.output_rows:,}"
        )
        if stats.stopped_reason:
            print(f"{stats.direction}: stopped early: {stats.stopped_reason}")
    if manifest is not None:
        print(f"Manifest: {manifest}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("complete") is not True:
            raise SystemExit(
                "DeepL pivot is incomplete. Caches were preserved, but finalized "
                f"outputs were not promoted; inspect {manifest} and rerun."
            )
    else:
        print("Dry run: no manifest or output files written.")


if __name__ == "__main__":
    main()
