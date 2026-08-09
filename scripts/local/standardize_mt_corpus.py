#!/usr/bin/env python3
"""Build an auditable MT-standard inventory from preserved source tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from mt_standardization import (
    DEFAULT_PROFILE_PATH,
    StandardizationContext,
    assert_idempotent,
    load_profile,
    profile_sha256,
    standardize_text,
)
from pipeline_common import atomic_write_json, sha256_file, stable_json_hash, utc_now
from tqdm import tqdm

MT_INVENTORY = "_mt_standard_inventory.jsonl"
MT_MANIFEST = "_mt_standard_manifest.json"
ROW_TYPE = {"S": "sentence", "W": "lexeme", "M": "morpheme"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive Formosan MT-standard text without modifying source XML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--src-lang", required=True)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE_PATH)
    return parser.parse_args()


def load_qc_contract(xml_dir: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = xml_dir / "_qc_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Missing XML preparation manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed XML preparation manifest {manifest_path}: {exc}") from exc
    if manifest.get("complete") is not True or manifest.get("source_immutable") is not True:
        raise SystemExit(f"XML preparation contract is incomplete or mutable: {manifest_path}")
    meta = manifest.get("transform_inventory", {})
    inventory_path = xml_dir / str(meta.get("path") or "")
    if not inventory_path.is_file():
        raise SystemExit(f"Missing source-tier inventory: {inventory_path}")
    if sha256_file(inventory_path) != str(meta.get("sha256") or ""):
        raise SystemExit(f"Source-tier inventory checksum mismatch: {inventory_path}")
    return inventory_path, manifest


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def compact_transformations(transformations: tuple[dict[str, Any], ...]) -> str:
    return json.dumps(transformations, ensure_ascii=False, separators=(",", ":"))


def repository_from_path(xml_path: str) -> str:
    parts = [part for part in xml_path.replace("\\", "/").split("/") if part]
    return parts[0] if parts else "unknown"


def standardize_inventory(
    source_path: Path,
    output_path: Path,
    *,
    language: str,
    profile: dict[str, Any],
    profile_hash: str,
) -> dict[str, Any]:
    with source_path.open(encoding="utf-8") as source:
        total = sum(1 for line in source if line.strip())
    status_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    row_type_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    repository_counts: Counter[str] = Counter()
    report_rows: list[tuple[str, str, str, str]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    written = 0
    try:
        with source_path.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as output:
            for line_number, line in enumerate(tqdm(source, total=total, desc="MT standardize", unit="unit"), 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"Malformed source-tier record {source_path}:{line_number}: {exc}") from exc
                element_tag = str(row.get("element_tag") or "")
                row_type = ROW_TYPE.get(element_tag)
                if row_type is None:
                    raise SystemExit(f"Unsupported XML unit {element_tag!r} at {source_path}:{line_number}")
                xml_path = str(row.get("xml_path") or "")
                repository = repository_from_path(xml_path)
                context = StandardizationContext(
                    language=language,
                    row_type=row_type,
                    repository=repository,
                    xml_path=xml_path,
                )
                result = standardize_text(
                    row.get("formosan_source_standard", ""),
                    context=context,
                    profile=profile,
                    contains_unclear=bool(row.get("contains_unclear_source")),
                )
                assert_idempotent(result, context=context, profile=profile)
                record = {
                    "transform_id": str(row.get("transform_id") or ""),
                    "xml_path": xml_path,
                    "element_tag": element_tag,
                    "xml_id": str(row.get("xml_id") or ""),
                    "source_element_index": row.get("source_element_index"),
                    "final_element_index": row.get("final_element_index"),
                    "final_xml_id": row.get("final_xml_id"),
                    "source_disposition": str(row.get("disposition") or ""),
                    "standard_origin": str(row.get("standard_origin") or "missing"),
                    "provided_standard_present": bool(row.get("provided_standard_present")),
                    "formosan_original_raw": str(row.get("formosan_original_raw") or ""),
                    "formosan_source_standard": str(row.get("formosan_source_standard") or ""),
                    "formosan_mt_standard": result.text,
                    "source_standard_sha256": text_sha256(
                        str(row.get("formosan_source_standard") or "")
                    ),
                    "original_before_qc_sha256": str(row.get("original_before_qc_sha256") or ""),
                    "standard_before_qc_sha256": str(row.get("standard_before_qc_sha256") or ""),
                    "standard_after_qc_sha256": str(row.get("standard_after_qc_sha256") or ""),
                    "mt_standard_sha256": text_sha256(result.text),
                    "contains_unclear_source": bool(row.get("contains_unclear_source")),
                    "mt_normalization_status": result.status,
                    "mt_normalization_confidence": result.confidence,
                    "mt_eval_eligible": result.eval_eligible,
                    "mt_normalization_reason": result.reason,
                    "mt_transformations": compact_transformations(result.transformations),
                    "mt_unresolved_markers": "|".join(result.unresolved_markers),
                    "speaker_label": result.speaker_label,
                    "mt_standard_profile": str(profile["profile_id"]),
                    "mt_standard_profile_sha256": profile_hash,
                }
                output.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
                written += 1
                status_counts[result.status] += 1
                confidence_counts[result.confidence] += 1
                origin_counts[record["standard_origin"]] += 1
                row_type_counts[row_type] += 1
                repository_counts[repository] += 1
                if result.reason:
                    reason_counts[result.reason] += 1
                for transformation in result.transformations:
                    rule_counts[str(transformation["rule"])] += int(transformation["count"])
                report_rows.append(
                    (
                        record["transform_id"],
                        record["mt_standard_sha256"],
                        result.status,
                        result.confidence,
                    )
                )
        if written != total:
            raise SystemExit(f"MT standardization row-conservation failure: input={total}, output={written}")
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "records": written,
        "sha256": sha256_file(output_path),
        "inventory_digest": stable_json_hash(report_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "standard_origin_counts": dict(sorted(origin_counts.items())),
        "row_type_counts": dict(sorted(row_type_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "repository_counts": dict(sorted(repository_counts.items())),
    }


def print_summary(manifest: dict[str, Any]) -> None:
    print("\nMT standardization summary")
    print(f"  Profile: {manifest['profile']['id']} ({manifest['profile']['sha256'][:12]})")
    print(f"  Units: {manifest['inventory']['records']:,}")
    print("  Outcomes:")
    for name, count in manifest["inventory"]["status_counts"].items():
        print(f"    {name}: {count:,}")
    print("  Transformations:")
    rules = manifest["inventory"]["rule_counts"]
    if not rules:
        print("    none")
    for name, count in sorted(rules.items(), key=lambda item: (-item[1], item[0])):
        print(f"    {name.replace('_', ' ')}: {count:,}")
    reasons = manifest["inventory"]["reason_counts"]
    if reasons:
        print("  Quarantined or ineligible:")
        for name, count in sorted(reasons.items(), key=lambda item: (-item[1], item[0])):
            print(f"    {name.replace('_', ' ')}: {count:,}")


def main() -> None:
    args = parse_args()
    xml_dir = args.xml_dir.expanduser().resolve()
    profile_path = args.profile.expanduser().resolve()
    inventory_path, qc_manifest = load_qc_contract(xml_dir)
    source_language = str(qc_manifest.get("source_language") or "")
    if source_language != args.src_lang:
        raise SystemExit(f"Source language mismatch: QC={source_language!r}, requested={args.src_lang!r}")
    profile = load_profile(profile_path)
    profile_hash = profile_sha256(profile_path)
    output_path = xml_dir / MT_INVENTORY
    inventory = standardize_inventory(
        inventory_path,
        output_path,
        language=args.src_lang,
        profile=profile,
        profile_hash=profile_hash,
    )
    manifest = {
        "schema_version": 1,
        "pipeline_version": qc_manifest.get("pipeline_version"),
        "created_at": utc_now(),
        "source_language": args.src_lang,
        "xml_dir": str(xml_dir),
        "source_qc_manifest_sha256": sha256_file(xml_dir / "_qc_manifest.json"),
        "source_tier_inventory_sha256": sha256_file(inventory_path),
        "profile": {
            "id": profile["profile_id"],
            "path": str(profile_path),
            "sha256": profile_hash,
        },
        "inventory": {"path": output_path.name, **inventory},
        "complete": True,
    }
    manifest_path = xml_dir / MT_MANIFEST
    atomic_write_json(manifest_path, manifest)
    print_summary(manifest)
    print(f"MT standard inventory: {output_path}")
    print(f"MT standard manifest: {manifest_path}")


if __name__ == "__main__":
    main()
