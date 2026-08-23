"""Load checksum-bound acquisition, QC, and MT-standard inventories."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline_common import sha256_file


def load_fetch_inventory(xml_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = xml_dir / "_fetch_manifest.json"
    inventory_path = xml_dir / "_fetch_inventory.jsonl"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise SystemExit(
            f"Missing immutable fetch manifest/inventory under {xml_dir}; "
            "rerun fetch_xml.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed fetch manifest {manifest_path}: {exc}") from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"Fetch manifest is incomplete: {manifest_path}")
    expected_hash = str(manifest.get("inventory_sha256") or "")
    actual_hash = sha256_file(inventory_path)
    if expected_hash != actual_hash:
        raise SystemExit(
            f"Fetch inventory hash mismatch: expected {expected_hash}, "
            f"found {actual_hash}"
        )

    inventory: dict[str, dict[str, str]] = {}
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Malformed fetch inventory "
                    f"{inventory_path}:{line_number}: {exc}"
                ) from exc
            if record.get("status") != "kept":
                continue
            destination = str(record.get("destination") or "")
            if not destination:
                raise SystemExit(
                    f"Kept fetch record has no destination at line {line_number}"
                )
            inventory[destination] = {
                "repository": str(record.get("repository") or ""),
                "repository_commit": str(record.get("commit_sha") or ""),
                "source_path": str(record.get("source_path") or ""),
                "sha256": str(record.get("sha256") or ""),
            }
    return inventory


def load_qc_inventory(
    xml_dir: Path,
) -> tuple[dict[tuple[str, str, int, str], dict[str, str]], dict]:
    manifest_path = xml_dir / "_qc_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing pinned QC manifest under {xml_dir}; rerun clean_xml.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed QC manifest {manifest_path}: {exc}") from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"QC manifest is incomplete: {manifest_path}")
    inventory_meta = manifest.get("transform_inventory", {})
    inventory_path = xml_dir / str(inventory_meta.get("path") or "")
    if not inventory_path.is_file():
        raise SystemExit(f"Missing QC transform inventory: {inventory_path}")
    actual_hash = sha256_file(inventory_path)
    expected_hash = str(inventory_meta.get("sha256") or "")
    if actual_hash != expected_hash:
        raise SystemExit(
            f"QC transform inventory hash mismatch: expected {expected_hash}, "
            f"found {actual_hash}"
        )
    records: dict[tuple[str, str, int, str], dict[str, str]] = {}
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Malformed QC transform inventory "
                    f"{inventory_path}:{line_number}: {exc}"
                ) from exc
            if row.get("disposition") != "retained":
                continue
            key = (
                str(row.get("xml_path") or ""),
                str(row.get("element_tag") or ""),
                int(row.get("final_element_index")),
                str(row.get("final_xml_id") or row.get("xml_id") or ""),
            )
            if key in records:
                raise SystemExit(
                    f"Duplicate QC transform locator at "
                    f"{inventory_path}:{line_number}: {key}"
                )
            records[key] = {
                name: str(row.get(name) or "")
                for name in (
                    "transform_id",
                    "xml_id",
                    "standard_origin",
                    "original_before_qc_sha256",
                    "standard_before_qc_sha256",
                    "standard_after_qc_sha256",
                )
            }
    qc_revision = str(manifest.get("formosanbank_qc", {}).get("revision") or "")
    if len(qc_revision) != 40:
        raise SystemExit(f"QC manifest has no pinned revision: {manifest_path}")
    return records, manifest


def load_mt_inventory(
    xml_dir: Path,
) -> tuple[dict[tuple[str, str, int, str], dict[str, object]], dict]:
    manifest_path = xml_dir / "_mt_standard_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing MT standard manifest under {xml_dir}; "
            "rerun standardize_mt_corpus.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Malformed MT standard manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"MT standard manifest is incomplete: {manifest_path}")
    profile = manifest.get("profile", {})
    profile_id = str(profile.get("id") or "")
    profile_hash = str(profile.get("sha256") or "")
    if not profile_id or len(profile_hash) != 64:
        raise SystemExit(
            f"MT standard manifest has an invalid profile: {manifest_path}"
        )
    inventory_meta = manifest.get("inventory", {})
    inventory_path = xml_dir / str(inventory_meta.get("path") or "")
    if not inventory_path.is_file():
        raise SystemExit(f"Missing MT standard inventory: {inventory_path}")
    expected_hash = str(inventory_meta.get("sha256") or "")
    if sha256_file(inventory_path) != expected_hash:
        raise SystemExit(
            f"MT standard inventory checksum mismatch: {inventory_path}"
        )

    records: dict[tuple[str, str, int, str], dict[str, object]] = {}
    inventory_rows = 0
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            inventory_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Malformed MT standard inventory "
                    f"{inventory_path}:{line_number}: {exc}"
                ) from exc
            if (
                str(row.get("mt_standard_profile") or "") != profile_id
                or str(row.get("mt_standard_profile_sha256") or "")
                != profile_hash
            ):
                raise SystemExit(
                    f"MT standard profile mismatch at "
                    f"{inventory_path}:{line_number}"
                )
            if row.get("source_disposition") != "retained":
                continue
            final_index = row.get("final_element_index")
            if final_index is None:
                raise SystemExit(
                    f"Retained MT standard row has no final locator at "
                    f"line {line_number}"
                )
            key = (
                str(row.get("xml_path") or ""),
                str(row.get("element_tag") or ""),
                int(final_index),
                str(row.get("final_xml_id") or row.get("xml_id") or ""),
            )
            if key in records:
                raise SystemExit(
                    f"Duplicate MT standard locator at "
                    f"{inventory_path}:{line_number}: {key}"
                )
            records[key] = row
    expected_records = int(inventory_meta.get("records", -1))
    if expected_records != inventory_rows:
        raise SystemExit(
            "MT standard inventory count is inconsistent: "
            f"manifest={expected_records}, inventory={inventory_rows}"
        )
    return records, manifest
