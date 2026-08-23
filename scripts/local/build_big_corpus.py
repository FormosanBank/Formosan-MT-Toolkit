#!/usr/bin/env python3
"""Aggregate cleaned pairwise or pivot corpora without losing provenance."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import pandas as pd
from corpus_quality import (
    english_language_quality,
    has_malformed_escaping,
    lexical_quality_reason,
    target_alignment_artifact_reason,
    target_gloss_reason,
    target_metadata_reason,
)
from pipeline_common import (
    atomic_write_json,
    read_csv_or_columnar,
    sha256_file,
    stable_json_hash,
    utc_now,
    write_columnar_cache,
    write_csv_atomic,
)

PIVOT_QUARANTINE_RE = re.compile(
    r"^pivot_rejections_(?:en2zh|zh2en)\.csv$"
)
TARGET_SPECS = {
    "en": {
        "xml_langs": {"en", "eng"},
        "columns": ("english", "english_sentence"),
        "output_column": "english_sentence",
        "quality_language": "english",
    },
    "zh": {
        "xml_langs": {"zh", "zho", "chi", "cmn"},
        "columns": ("chinese", "chinese_sentence"),
        "output_column": "chinese_sentence",
        "quality_language": "chinese",
    },
}
OUTPUT_NAMES = {
    "big_corpus_en.csv",
    "big_corpus_zh.csv",
}
LEGACY_OUTPUT_NAMES = {"big_corpus_combined.csv"}
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
        frame = read_csv_or_columnar(
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
    # Parquet preserves per-file numeric and boolean inference. Aggregation has
    # always treated the canonical CSV contract as text, so normalize both read
    # paths before concatenating files with different inferred schemas.
    return frame.fillna("").astype(str)


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


def require_clean_pairs(
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
    rows = zip(
        frame.get(
            "formosan_sentence",
            pd.Series("", index=frame.index),
        ).astype(str),
        frame[target_column].astype(str),
        translation_kinds,
        row_types,
        unit_contexts,
        strict=True,
    )
    for position, (source, target, translation_kind, row_type, unit_context) in enumerate(
        rows
    ):
        reason = target_gloss_reason(
            target,
            translation_kind=translation_kind,
            target_language=target_language,
        )
        reason = reason or target_metadata_reason(target, source=source)
        reason = reason or lexical_quality_reason(
            target,
            row_type=row_type,
            xml_unit_context=unit_context,
            target_language=target_language,
        )
        if not reason and has_malformed_escaping(str(target)):
            reason = "malformed_target_escaping"
        if not reason and target_language == "english":
            reason, _ = english_language_quality(str(target))
        if not reason:
            reason = target_alignment_artifact_reason(
                str(source),
                str(target),
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
            f"{path} contains {contaminated_count:,} accepted target-quality failures; "
            f"first source_record_id values: {sample}"
        )
    malformed_sources = frame["formosan_sentence"].astype(str).map(
        has_malformed_escaping
    )
    if malformed_sources.any():
        sample_frame = frame.loc[malformed_sources].head(5)
        sample = (
            sample_frame["source_record_id"].astype(str).tolist()
            if "source_record_id" in sample_frame
            else [str(index) for index in sample_frame.index]
        )
        raise SystemExit(
            f"{path} contains {int(malformed_sources.sum()):,} accepted malformed "
            f"Formosan rows; first source_record_id values: {sample}"
        )


def corpus_frame(path: Path) -> tuple[str, pd.DataFrame, str]:
    """Read one corpus using row metadata, never its filename, as authority."""
    frame = read_csv(path).drop(
        columns=["source_bucket", "_source_bucket"],
        errors="ignore",
    )
    required = {
        "lang_code",
        "formosan_sentence",
        "target_lang",
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

    target_values = frame["target_lang"].astype(str).str.strip().str.lower()
    if target_values.eq("").any():
        raise SystemExit(f"{path} contains rows without TRANSL/@xml:lang metadata")
    declared_targets = set(target_values)
    target_matches = {
        target
        for target, spec in TARGET_SPECS.items()
        if declared_targets and declared_targets <= spec["xml_langs"]
    }
    if len(target_matches) != 1:
        raise SystemExit(
            f"{path} has unsupported or mixed target_lang values: "
            f"{sorted(declared_targets)}"
        )
    target = target_matches.pop()
    spec = TARGET_SPECS[target]
    text_columns = [column for column in spec["columns"] if column in frame]
    if len(text_columns) != 1:
        raise SystemExit(
            f"{path} must have exactly one text column for declared target "
            f"{target}: found {text_columns}"
        )
    raw_target = text_columns[0]
    target_column = str(spec["output_column"])
    if not frame["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit(f"{path} contains non-standard Formosan rows")
    if not frame["standard_namespace"].astype(str).eq("formosan-mt").all():
        raise SystemExit(f"{path} contains rows outside the Formosan MT standard namespace")
    if not frame["mt_normalization_status"].astype(str).eq("accepted").all():
        raise SystemExit(f"{path} contains non-accepted MT-standard rows")
    if not frame["formosan_sentence"].astype(str).eq(frame["formosan_mt_standard"].astype(str)).all():
        raise SystemExit(f"{path} has a broken formosan_sentence MT-standard alias")
    source_languages = frame["lang_code"].astype(str).str.strip().str.lower()
    if source_languages.eq("").any():
        raise SystemExit(f"{path} contains rows without TEXT/@xml:lang metadata")
    if raw_target != target_column:
        frame = frame.rename(columns={raw_target: target_column})
    require_clean_pairs(
        frame,
        target_column=target_column,
        target_language=str(spec["quality_language"]),
        path=path,
    )
    input_type = "pairwise" if raw_target != target_column else "aggregate"
    return target, ensure_metadata(frame), input_type


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


def load_inputs(directory: Path, output_names: set[str]) -> tuple[list[pd.DataFrame], list[pd.DataFrame], list[dict]]:
    english: list[pd.DataFrame] = []
    chinese: list[pd.DataFrame] = []
    inventory: list[dict] = []
    for path in discover_inputs(directory, output_names):
        target, frame, input_type = corpus_frame(path)
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
    write_columnar_cache(output, path)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate cleaned pairwise or pivot corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-en-name", default="big_corpus_en.csv")
    parser.add_argument("--output-zh-name", default="big_corpus_zh.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_names = {args.output_en_name, args.output_zh_name, *LEGACY_OUTPUT_NAMES}
    input_paths = discover_inputs(input_dir, output_names)
    require_output_space(input_paths, output_dir)
    english_inputs, chinese_inputs, inventory = load_inputs(input_dir, output_names)
    if not english_inputs and not chinese_inputs:
        raise SystemExit("No supported corpus inputs were found")

    english_path = output_dir / args.output_en_name
    chinese_path = output_dir / args.output_zh_name
    english = write_target(english_inputs, english_path, "english_sentence")
    chinese = write_target(chinese_inputs, chinese_path, "chinese_sentence")
    for legacy_name in LEGACY_OUTPUT_NAMES:
        (output_dir / legacy_name).unlink(missing_ok=True)
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
                "columnar_path": str(english_path.with_suffix(".parquet")),
            },
            "chinese": {
                "path": str(chinese_path),
                "rows": len(chinese),
                "sha256": sha256_file(chinese_path) if not chinese.empty else None,
                "columnar_path": str(chinese_path.with_suffix(".parquet")),
            },
        },
        "complete": True,
    }
    atomic_write_json(output_dir / "aggregate_manifest.json", report)
    print(f"Aggregated EN={len(english):,}, ZH={len(chinese):,} rows")


if __name__ == "__main__":
    main()
