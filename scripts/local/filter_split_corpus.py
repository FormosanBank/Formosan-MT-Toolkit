#!/usr/bin/env python3
"""Conservatively clean and deduplicate one parallel corpus.

This stage never creates train/validate/test assignments. The corpus-wide hard
splitter runs exactly once after aggregation and pivot augmentation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from corpus_quality import (
    apply_quality_rules,
    deduplicate_pairs,
    normalize_dataframe,
    reason_counts,
)
from pipeline_common import atomic_write_json, load_pipeline_config, stable_json_hash, utc_now

RESERVED_COLUMNS = {
    "row_id",
    "source_record_id",
    "content_sha256",
    "source",
    "repository",
    "repository_commit",
    "xml_path",
    "corpus_id",
    "xml_id",
    "xml_element_index",
    "kindOf",
    "standard_origin",
    "original_before_qc_sha256",
    "standard_before_qc_sha256",
    "standard_after_qc_sha256",
    "qc_transform_id",
    "qc_revision",
    "dialect",
    "row_type",
    "formosan_original",
    "formosan_standard",
    "target_lang",
    "translation_index",
    "translation_kind",
    "translation_version",
    "contains_unclear",
}


def read_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            low_memory=False,
            keep_default_na=False,
            na_filter=False,
        )
    except Exception as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    if frame.empty:
        raise SystemExit(f"Input corpus is empty: {path}")
    return frame


def detect_language_columns(frame: pd.DataFrame) -> tuple[str, str, str]:
    columns = {str(column).lower(): str(column) for column in frame.columns}
    target_pairs = (
        ("english", "english"),
        ("chinese", "chinese"),
        ("english_sentence", "english"),
        ("chinese_sentence", "chinese"),
    )
    target_column = ""
    target_language = ""
    for candidate, language in target_pairs:
        if candidate in columns:
            target_column = columns[candidate]
            target_language = language
            break
    if not target_column:
        raise SystemExit("Cannot identify an English or Chinese target column")

    preferred_source = columns.get("formosan_sentence")
    if preferred_source:
        return preferred_source, target_column, target_language
    candidates = [
        str(column) for column in frame.columns if str(column) != target_column and str(column) not in RESERVED_COLUMNS
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "Cannot identify the Formosan source column; expected one non-metadata column "
            f"beside {target_column!r}, found {candidates}"
        )
    return candidates[0], target_column, target_language


def validate_extraction_contract(input_path: Path, frame: pd.DataFrame) -> dict:
    report_path = input_path.with_suffix(".extraction.json")
    if not report_path.is_file():
        raise SystemExit(f"Missing extraction report: {report_path}")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed extraction report {report_path}: {exc}") from exc
    if report.get("complete") is not True:
        raise SystemExit(f"Extraction report is incomplete: {report_path}")
    if int(report.get("rows", -1)) != len(frame):
        raise SystemExit(f"Extraction row count mismatch: report={report.get('rows')} csv={len(frame)}")
    required = {
        "row_id",
        "source_record_id",
        "content_sha256",
        "source",
        "repository",
        "repository_commit",
        "xml_path",
        "xml_id",
        "xml_element_index",
        "kindOf",
        "standard_origin",
        "standard_after_qc_sha256",
        "qc_transform_id",
        "qc_revision",
        "dialect",
        "row_type",
        "formosan_original",
        "formosan_standard",
        "target_lang",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Extracted corpus is missing required provenance columns: {missing}")
    if frame["row_id"].astype(str).duplicated().any():
        raise SystemExit("Extracted row_id values are not unique")
    return report


def write_rejection_ledger(rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean and deduplicate a standard-tier Formosan parallel corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--audit-samples", type=int, default=100)
    parser.add_argument("--keep-redactions", action="store_true")
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Diagnostic only. Production builds always leave this disabled.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Compatibility option; conservative cleaning is deterministic and single-process.",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Compatibility option. This stage never splits, regardless of the flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_pipeline_config()
    frame = read_csv(args.input)
    extraction_report = validate_extraction_contract(args.input, frame)
    source_column, target_column, target_language = detect_language_columns(frame)
    initial_rows = len(frame)

    normalized, transformation_counts = normalize_dataframe(
        frame,
        source_column,
        target_column,
    )
    accepted, rejected, decision_counts = apply_quality_rules(
        normalized,
        source_column=source_column,
        target_column=target_column,
        target_language=target_language,
        keep_redactions=args.keep_redactions,
    )
    if args.no_dedup:
        deduplicated = accepted.iloc[0:0].copy()
        accepted["duplicate_group_size"] = 1
        accepted["pair_fingerprint"] = ""
    else:
        accepted, deduplicated = deduplicate_pairs(
            accepted,
            source_column=source_column,
            target_column=target_column,
        )
    rejected = pd.concat([rejected, deduplicated], ignore_index=True, sort=False)
    accepted = accepted.drop(columns=["disposition", "disposition_reason"], errors="ignore")

    accounted = len(accepted) + len(rejected)
    if accounted != initial_rows:
        raise SystemExit(
            f"Row-conservation failure: input={initial_rows}, accepted={len(accepted)}, rejected={len(rejected)}"
        )
    if not accepted["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Non-standard rows survived cleaning")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(args.output, index=False)
    report_dir = args.report_dir or args.output.parent / "filter_reports" / args.output.stem
    rejection_path = report_dir / "rejected_rows.csv"
    write_rejection_ledger(rejected, rejection_path)

    disposition_counts = Counter(rejected.get("disposition", pd.Series(dtype=str)).astype(str))
    report = {
        "schema_version": 2,
        "pipeline_version": config["pipeline_version"],
        "cleaning_profile": config["cleaning"]["profile"],
        "created_at": utc_now(),
        "input": str(args.input),
        "output": str(args.output),
        "source_column": source_column,
        "target_column": target_column,
        "target_language": target_language,
        "initial_rows": initial_rows,
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "row_conservation": {
            "input": initial_rows,
            "accounted": accounted,
            "difference": initial_rows - accounted,
        },
        "disposition_counts": dict(sorted(disposition_counts.items())),
        "reason_counts": reason_counts(rejected.get("disposition_reason", pd.Series(dtype=str)).astype(str)),
        "decision_counts": dict(sorted(decision_counts.items())),
        "transformation_counts": dict(sorted(transformation_counts.items())),
        "row_type_counts": reason_counts(accepted["row_type"].astype(str)),
        "rejection_ledger": str(rejection_path),
        "rejection_ledger_sha256": stable_json_hash(rejected.fillna("").astype(str).to_dict("records")),
        "input_extraction_report": str(args.input.with_suffix(".extraction.json")),
        "input_extraction_inventory_sha256": extraction_report.get("file_inventory_sha256"),
        "complete": True,
    }
    atomic_write_json(report_dir / "summary.json", report)
    if args.audit_samples > 0 and not rejected.empty:
        rejected.head(args.audit_samples).to_csv(report_dir / "reject_samples.csv", index=False)
    print(f"Cleaned {initial_rows:,} rows -> {len(accepted):,} accepted, {len(rejected):,} rejected/deduplicated")
    print(f"Filter report: {report_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
