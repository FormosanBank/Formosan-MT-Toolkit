"""Validated pivot corpus assembly and completion manifests."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
from corpus_quality import exact_key, normalize_text, quality_decision
from pipeline_common import (
    atomic_write_json,
    content_row_id,
    load_pipeline_config,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from pivot_cache import PROVIDER, make_cache_key
from pivot_corpus import (
    BASE_COLUMNS,
    PIVOT_POLICY,
    PROVENANCE_COLUMNS,
    formosan_key,
    frame_records,
    pivot_candidate_reason,
    read_corpus,
    target_formosan_keys,
    validate_columns,
    validate_mt_standard_contract,
)
from pivot_deepl import parse_api_key_envs
from pivot_types import Direction, DirectionStats, LoadedCorpus, OutputBuildResult
from tqdm import tqdm

MAX_TRAINING_UNITS_PER_SIDE = load_pipeline_config()["cleaning"][
    "max_training_units_per_side"
]


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
    source_row: Mapping[str, Any],
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

    row = {str(column): value for column, value in source_row.items()}
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
        row,
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
    original_df: pd.DataFrame | None = None,
    source_df: pd.DataFrame | None = None,
) -> OutputBuildResult:
    result = OutputBuildResult()
    if original_df is None:
        original_df = read_corpus(
            direction.original_target_path,
            f"{direction.name} original target corpus",
        )
    if source_df is None:
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

        for row in frame_records(original_df):
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
            out_row = dict(row)
            out_row["pivot_origin"] = str(row.get("pivot_origin") or "original")
            for column in PROVENANCE_COLUMNS:
                out_row.setdefault(column, "")
            writer.writerow(out_row)
            result.output_rows += 1

        result.synthetic_rows_available = 0
        result.synthetic_rows_missing = 0

        for row in tqdm(
            frame_records(source_df),
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
    loaded_corpora: Mapping[Path, LoadedCorpus] | None = None,
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
    source_profiles = set()
    for name, path_string in sources.items():
        path = Path(path_string)
        loaded = (loaded_corpora or {}).get(path)
        profile = (
            loaded.profile
            if loaded is not None
            else validate_mt_standard_contract(read_corpus(path, name), path)
        )
        source_profiles.add(tuple(profile.values()))
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
        "created_at": utc_now(),
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


