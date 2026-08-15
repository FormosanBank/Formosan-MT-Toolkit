#!/usr/bin/env python3
"""Aggregate cleaned pairwise or pivot corpora without losing provenance."""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd
from corpus_quality import lexical_quality_reason, target_gloss_reason
from pipeline_common import atomic_write_json, sha256_file, stable_json_hash, utc_now

PAIRWISE_RE = re.compile(r"^(?P<lang>[a-z]{3})_(?P<target>en|zh)_processed$")
PIVOT_QUARANTINE_RE = re.compile(
    r"^pivot_rejections_(?:en2zh|zh2en)\.csv$"
)
OUTPUT_NAMES = {
    "big_corpus_en.csv",
    "big_corpus_zh.csv",
    "big_corpus_combined.csv",
}
MIB = 1024**2
GIB = 1024**3
CANONICAL_PREFIX = [
    "row_id",
    "source_record_id",
    "content_sha256",
    "lang_code",
    "formosan_sentence",
]
CANONICAL_SUFFIX = [
    "source",
    "repository",
    "repository_commit",
    "xml_path",
    "corpus_id",
    "xml_id",
    "qc_final_xml_id",
    "xml_element_index",
    "kindOf",
    "standard_namespace",
    "standard_origin",
    "original_before_qc_sha256",
    "standard_before_qc_sha256",
    "standard_after_qc_sha256",
    "qc_transform_id",
    "qc_revision",
    "dialect",
    "row_type",
    "xml_unit_context",
    "formosan_original",
    "formosan_standard",
    "formosan_original_raw",
    "formosan_source_standard",
    "formosan_mt_standard",
    "source_standard_sha256",
    "mt_standard_sha256",
    "mt_normalization_status",
    "mt_normalization_confidence",
    "mt_eval_eligible",
    "mt_normalization_reason",
    "mt_transformations",
    "mt_unresolved_markers",
    "speaker_label",
    "mt_standard_profile",
    "mt_standard_profile_sha256",
    "target_lang",
    "translation_index",
    "translation_kind",
    "translation_version",
    "contains_unclear",
    "formosan_raw",
    "target_raw",
    "formosan_transformations",
    "target_transformations",
    "quality_flags",
    "duplicate_group_size",
    "pair_fingerprint",
    "split",
    "pivot_origin",
    "pivot_provider",
    "pivot_direction",
    "pivot_source_lang",
    "pivot_target_lang",
    "pivot_source_text",
    "pivot_cache_key",
    "pivot_detected_source_lang",
]


def read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        raise SystemExit(f"Cannot read corpus CSV {path}: {exc}") from exc
    if frame.empty:
        raise SystemExit(f"Corpus CSV is empty: {path}")
    return frame


def canonical_order(frame: pd.DataFrame, target_column: str) -> list[str]:
    preferred = [*CANONICAL_PREFIX, target_column, *CANONICAL_SUFFIX]
    present = [column for column in preferred if column in frame.columns]
    extras = [column for column in frame.columns if column not in present]
    return [*present, *extras]


def ensure_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    defaults = {
        "split": "",
        "pivot_origin": "original",
        "pivot_provider": "",
        "pivot_direction": "",
        "pivot_source_lang": "",
        "pivot_target_lang": "",
        "pivot_source_text": "",
        "pivot_cache_key": "",
        "pivot_detected_source_lang": "",
    }
    for column, default in defaults.items():
        if column not in output.columns:
            output[column] = default
    return output


def require_gloss_free(
    frame: pd.DataFrame,
    *,
    target_column: str,
    target_language: str,
    path: Path,
) -> None:
    translation_kinds = frame.get(
        "translation_kind",
        pd.Series("", index=frame.index),
    )
    row_types = frame.get("row_type", pd.Series("", index=frame.index))
    unit_contexts = frame.get(
        "xml_unit_context",
        pd.Series("", index=frame.index),
    )
    contaminated_count = 0
    sample_positions: list[int] = []
    for position, (target, translation_kind, row_type, unit_context) in enumerate(
        zip(
            frame[target_column].astype(str),
            translation_kinds,
            row_types,
            unit_contexts,
            strict=True,
        )
    ):
        reason = target_gloss_reason(
            target,
            translation_kind=translation_kind,
            target_language=target_language,
        )
        reason = reason or lexical_quality_reason(
            target,
            row_type=row_type,
            xml_unit_context=unit_context,
            target_language=target_language,
        )
        if reason:
            contaminated_count += 1
            if len(sample_positions) < 5:
                sample_positions.append(position)
    if contaminated_count:
        sample_frame = frame.iloc[sample_positions]
        sample = (
            sample_frame["source_record_id"].astype(str).tolist()
            if "source_record_id" in sample_frame
            else [str(index) for index in sample_frame.index]
        )
        raise SystemExit(
            f"{path} contains {contaminated_count:,} accepted gloss targets; "
            f"first source_record_id values: {sample}"
        )


def pairwise_frame(path: Path, match: re.Match[str]) -> tuple[str, pd.DataFrame]:
    language = match.group("lang")
    target = match.group("target")
    frame = read_csv(path)
    raw_target = "english" if target == "en" else "chinese"
    target_column = "english_sentence" if target == "en" else "chinese_sentence"
    required = {
        "lang_code",
        "formosan_sentence",
        raw_target,
        "row_id",
        "source_record_id",
        "kindOf",
        "standard_namespace",
        "formosan_mt_standard",
        "mt_normalization_status",
        "mt_eval_eligible",
        "mt_standard_profile",
        "mt_standard_profile_sha256",
        "source",
        "row_type",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is missing required cleaned columns: {missing}")
    if not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit(f"{path} contains non-standard Formosan rows")
    if not frame["standard_namespace"].astype(str).eq("formosan-mt").all():
        raise SystemExit(f"{path} contains rows outside the Formosan MT standard namespace")
    if not frame["formosan_sentence"].astype(str).eq(frame["formosan_mt_standard"].astype(str)).all():
        raise SystemExit(f"{path} has a broken formosan_sentence MT-standard alias")
    languages = set(frame["lang_code"].astype(str).str.lower())
    if languages != {language}:
        raise SystemExit(
            f"{path} language mismatch: filename={language}, rows={sorted(languages)}"
        )
    frame = frame.rename(columns={raw_target: target_column})
    require_gloss_free(
        frame,
        target_column=target_column,
        target_language="english" if target == "en" else "chinese",
        path=path,
    )
    return target, ensure_metadata(frame)


def aggregate_frame(path: Path) -> tuple[str, pd.DataFrame]:
    frame = read_csv(path)
    required = {
        "lang_code",
        "formosan_sentence",
        "formosan_mt_standard",
        "standard_namespace",
        "mt_normalization_status",
        "mt_standard_profile",
        "mt_standard_profile_sha256",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is not a supported aggregate corpus: missing {missing}")
    has_english = "english_sentence" in frame.columns
    has_chinese = "chinese_sentence" in frame.columns
    if has_english == has_chinese:
        raise SystemExit(
            f"{path} must contain exactly one target column, not English={has_english}, Chinese={has_chinese}"
        )
    if not frame["standard_namespace"].astype(str).eq("formosan-mt").all():
        raise SystemExit(f"{path} contains rows outside the Formosan MT standard namespace")
    if not frame["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"{path} contains non-accepted MT-standard rows")
    if not frame["formosan_sentence"].astype(str).eq(
        frame["formosan_mt_standard"].astype(str)
    ).all():
        raise SystemExit(f"{path} has a broken formosan_sentence MT-standard alias")
    target_column = "english_sentence" if has_english else "chinese_sentence"
    require_gloss_free(
        frame,
        target_column=target_column,
        target_language="english" if has_english else "chinese",
        path=path,
    )
    return ("en" if has_english else "zh"), ensure_metadata(frame)


def discover_inputs(directory: Path, output_names: set[str]) -> list[Path]:
    files = [
        path
        for path in sorted(directory.glob("*.csv"))
        if path.name not in output_names
        and path.name != "summary_stats.csv"
        and not PIVOT_QUARANTINE_RE.fullmatch(path.name)
    ]
    if not files:
        raise SystemExit(f"No input corpus CSVs found in {directory}")
    return files


def estimated_output_bytes(paths: list[Path]) -> int:
    """Conservatively estimate CSV output plus a small filesystem reserve."""
    input_bytes = sum(path.stat().st_size for path in paths)
    reserve = min(GIB, max(64 * MIB, input_bytes // 4))
    return int(input_bytes * 1.75) + reserve


def require_output_space(paths: list[Path], output_dir: Path) -> None:
    required = estimated_output_bytes(paths)
    available = shutil.disk_usage(output_dir).free
    if available < required:
        raise SystemExit(
            "Insufficient disk space for corpus aggregation: "
            f"{available / GIB:.1f} GiB available, approximately "
            f"{required / GIB:.1f} GiB required. Existing corpora and DeepL "
            "caches were not modified."
        )


def write_csv_atomic(frame: pd.DataFrame, path: Path) -> None:
    """Write a CSV without leaving a truncated release artifact on failure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete")
    temporary.unlink(missing_ok=True)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_inputs(directory: Path, output_names: set[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    english: list[pd.DataFrame] = []
    chinese: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for path in discover_inputs(directory, output_names):
        pair_match = PAIRWISE_RE.match(path.stem)
        if pair_match:
            target, frame = pairwise_frame(path, pair_match)
            input_type = "pairwise"
        elif path.stem in {"big_corpus_en_pivot", "big_corpus_zh_pivot"}:
            target, frame = aggregate_frame(path)
            input_type = "pivot"
        else:
            raise SystemExit(f"Unsupported CSV in aggregate input directory: {path.name}")
        (english if target == "en" else chinese).append(frame)
        inventory.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "rows": len(frame),
                "target": target,
                "type": input_type,
            }
        )
    return english, chinese, inventory


def write_target(frames: list[pd.DataFrame], path: Path, target_column: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False).fillna("")
    required = {
        "row_id",
        "lang_code",
        "formosan_sentence",
        target_column,
        "source",
        "row_type",
        "pivot_origin",
    }
    missing = sorted(required - set(output.columns))
    if missing:
        raise SystemExit(f"Aggregate output {path} would be missing required columns: {missing}")
    empty = (
        output["lang_code"].astype(str).str.strip().eq("")
        | output["formosan_sentence"].astype(str).str.strip().eq("")
        | output[target_column].astype(str).str.strip().eq("")
    )
    if empty.any():
        raise SystemExit(f"Aggregate input contains {int(empty.sum())} empty required rows for {path.name}")
    if "formosan_mt_standard" not in output or not output["formosan_sentence"].astype(str).eq(
        output["formosan_mt_standard"].astype(str)
    ).all():
        raise SystemExit(f"Aggregate input violates the MT-standard alias contract for {path.name}")
    if "mt_standard_profile_sha256" not in output:
        raise SystemExit(f"Aggregate input has no MT standard profile hash for {path.name}")
    profile_hashes = set(output["mt_standard_profile_sha256"].astype(str))
    if len(profile_hashes) != 1 or len(next(iter(profile_hashes), "")) != 64:
        raise SystemExit(f"Aggregate input mixes or omits MT standard profiles for {path.name}")
    profile_ids = set(output["mt_standard_profile"].astype(str))
    if len(profile_ids) != 1 or not next(iter(profile_ids), ""):
        raise SystemExit(f"Aggregate input mixes or omits MT standard profile IDs for {path.name}")
    if not output["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"Aggregate input contains non-accepted MT-standard rows for {path.name}")
    output = output[canonical_order(output, target_column)]
    write_csv_atomic(output, path)
    return output


def source_join_key(frame: pd.DataFrame) -> pd.Series:
    if "source_record_id" in frame.columns:
        value = frame["source_record_id"].astype(str)
        if value.str.strip().ne("").all():
            return value
    return (
        frame["lang_code"].astype(str)
        + "\u241f"
        + frame["source"].astype(str)
        + "\u241f"
        + frame.get("xml_id", pd.Series([""] * len(frame))).astype(str)
        + "\u241f"
        + frame["formosan_sentence"].astype(str)
    )


def write_combined(chinese: pd.DataFrame, english: pd.DataFrame, path: Path) -> int:
    if chinese.empty:
        return 0
    output = chinese.copy()
    output["_source_join_key"] = source_join_key(output)
    english_lookup: dict[str, str] = {}
    if not english.empty:
        english_work = english.copy()
        english_work["_source_join_key"] = source_join_key(english_work)
        for key, target in zip(
            english_work["_source_join_key"],
            english_work["english_sentence"],
            strict=True,
        ):
            if str(target).strip():
                english_lookup.setdefault(str(key), str(target))
    output["english_sentence"] = [english_lookup.get(str(key), "") for key in output["_source_join_key"]]
    output = output.drop(columns=["_source_join_key"])
    order = canonical_order(output, "chinese_sentence")
    insert_at = order.index("chinese_sentence") + 1
    if "english_sentence" in order:
        order.remove("english_sentence")
    order.insert(insert_at, "english_sentence")
    write_csv_atomic(output[order], path)
    return len(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cleaned pairwise or pivot corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-en-name", default="big_corpus_en.csv")
    parser.add_argument("--output-zh-name", default="big_corpus_zh.csv")
    parser.add_argument("--output-combined-name", default="big_corpus_combined.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {
        args.output_en_name,
        args.output_zh_name,
        args.output_combined_name,
    }
    input_paths = discover_inputs(input_dir, output_names)
    require_output_space(input_paths, output_dir)
    english_inputs, chinese_inputs, inventory = load_inputs(input_dir, output_names)
    if not english_inputs and not chinese_inputs:
        raise SystemExit("No supported corpus inputs were found")

    english_path = output_dir / args.output_en_name
    chinese_path = output_dir / args.output_zh_name
    combined_path = output_dir / args.output_combined_name
    english = write_target(english_inputs, english_path, "english_sentence")
    chinese = write_target(chinese_inputs, chinese_path, "chinese_sentence")
    combined_rows = write_combined(chinese, english, combined_path)
    report = {
        "schema_version": 3,
        "created_at": utc_now(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "inputs": inventory,
        "input_inventory_sha256": stable_json_hash(inventory),
        "mt_standardization": {
            "id": str(
                (english if not english.empty else chinese)[
                    "mt_standard_profile"
                ].iloc[0]
            ),
            "sha256": str(
                (english if not english.empty else chinese)[
                    "mt_standard_profile_sha256"
                ].iloc[0]
            ),
            "namespace": "formosan-mt",
        },
        "outputs": {
            "english": {
                "path": str(english_path),
                "rows": len(english),
                "sha256": sha256_file(english_path) if not english.empty else None,
            },
            "chinese": {
                "path": str(chinese_path),
                "rows": len(chinese),
                "sha256": sha256_file(chinese_path) if not chinese.empty else None,
            },
            "combined": {
                "path": str(combined_path),
                "rows": combined_rows,
                "sha256": sha256_file(combined_path) if combined_rows else None,
            },
        },
        "complete": True,
    }
    atomic_write_json(output_dir / "aggregate_manifest.json", report)
    print(f"Aggregated EN={len(english):,}, ZH={len(chinese):,}, combined={combined_rows:,} rows")


if __name__ == "__main__":
    main()
