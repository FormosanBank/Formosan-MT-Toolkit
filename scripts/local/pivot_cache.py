"""Durable, integrity-checked JSONL storage for DeepL responses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional

PROVIDER = "deepl"


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

    with cache_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed DeepL cache record {cache_path}:{line_no}: {exc}"
                ) from exc
            key = record.get("key")
            translation = record.get("translation")
            if not isinstance(key, str) or not isinstance(translation, str):
                raise RuntimeError(
                    f"Invalid DeepL cache record {cache_path}:{line_no}: "
                    "missing string key/translation"
                )
            expected_key = make_cache_key(
                provider=str(record.get("provider") or PROVIDER),
                source_lang=str(record.get("source_lang") or ""),
                target_lang=str(record.get("target_lang") or ""),
                text=str(record.get("text") or ""),
                split_sentences=str(record.get("split_sentences") or "0"),
                preserve_formatting=bool(record.get("preserve_formatting", True)),
                model_type=(
                    str(record.get("model_type_requested"))
                    if record.get("model_type_requested")
                    else None
                ),
            )
            if key != expected_key:
                raise RuntimeError(
                    f"DeepL cache key mismatch at {cache_path}:{line_no}; "
                    "cache may be corrupt"
                )
            existing = cache.get(key)
            if existing is not None and existing.get("translation") != translation:
                raise RuntimeError(
                    f"Conflicting DeepL translations for cache key {key} in {cache_path}"
                )
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
            if (
                existing is not None
                and existing.get("translation") != record.get("translation")
            ):
                conflicts.append(
                    {
                        "cache_key": key,
                        "text": str(
                            record.get("text") or existing.get("text") or ""
                        ),
                        "lower_priority_cache": str(selected_paths[key]),
                        "lower_priority_created_at": str(
                            existing.get("created_at") or ""
                        ),
                        "lower_priority_translation": str(
                            existing.get("translation") or ""
                        ),
                        "higher_priority_cache": str(cache_path),
                        "higher_priority_created_at": str(
                            record.get("created_at") or ""
                        ),
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
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
