"""Data contracts shared by DeepL pivot stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd


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
class LoadedCorpus:
    path: Path
    frame: pd.DataFrame
    profile: dict[str, str]


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
