#!/usr/bin/env python3
"""Extract provenance-complete pairs using the toolkit MT standard."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from pipeline_common import (
    atomic_write_json,
    content_row_id,
    sha256_bytes,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from tqdm import tqdm

TARGET_MAP: dict[str, set[str]] = {
    "chinese": {"zh", "zho", "chi", "cmn"},
    "english": {"en", "eng"},
}
UNIT_TAGS = {
    "sentences": ("S",),
    "words": ("W",),
    "morphemes": ("M",),
}
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

OUTPUT_COLUMNS = [
    "row_id",
    "source_record_id",
    "content_sha256",
    "lang_code",
    "formosan_sentence",
    "target_text",
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
]


@dataclass(frozen=True)
class ExtractedPair:
    row_id: str
    source_record_id: str
    content_sha256: str
    lang_code: str
    formosan_sentence: str
    target_text: str
    source: str
    repository: str
    repository_commit: str
    xml_path: str
    corpus_id: str
    xml_id: str
    qc_final_xml_id: str
    xml_element_index: int
    kind_of: str
    standard_namespace: str
    standard_origin: str
    original_before_qc_sha256: str
    standard_before_qc_sha256: str
    standard_after_qc_sha256: str
    qc_transform_id: str
    qc_revision: str
    dialect: str
    row_type: str
    formosan_original: str
    formosan_standard: str
    formosan_original_raw: str
    formosan_source_standard: str
    formosan_mt_standard: str
    source_standard_sha256: str
    mt_standard_sha256: str
    mt_normalization_status: str
    mt_normalization_confidence: str
    mt_eval_eligible: bool
    mt_normalization_reason: str
    mt_transformations: str
    mt_unresolved_markers: str
    speaker_label: str
    mt_standard_profile: str
    mt_standard_profile_sha256: str
    target_lang: str
    translation_index: int
    translation_kind: str
    translation_version: str
    contains_unclear: bool

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["kindOf"] = row.pop("kind_of")
        return row


def xml_lang(element: ET.Element) -> str:
    return (element.attrib.get(XML_LANG) or element.attrib.get("xml:lang") or "").strip().lower()


def mixed_text(element: ET.Element) -> str:
    return "".join(element.itertext()).strip()


def direct_form(element: ET.Element, kind_of: str) -> ET.Element | None:
    matches = [form for form in element.findall("FORM") if (form.get("kindOf") or "").strip().lower() == kind_of]
    if len(matches) > 1:
        raise ValueError(f"{element.tag} id={element.get('id', '')!r} has {len(matches)} {kind_of!r} FORM tiers")
    return matches[0] if matches else None


def row_type_for_tag(tag: str) -> str:
    return {"S": "sentence", "W": "lexeme", "M": "morpheme"}[tag]


def wanted_tags(units: set[str]) -> set[str]:
    return {tag for unit in units for tag in UNIT_TAGS[unit]}


def parse_units(raw: str) -> set[str]:
    units = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = sorted(units - set(UNIT_TAGS))
    if unknown:
        raise SystemExit(f"Unknown --units values: {unknown}. Valid values: {sorted(UNIT_TAGS)}")
    return units or {"sentences", "words"}


def load_fetch_inventory(xml_dir: Path) -> dict[str, dict[str, str]]:
    manifest_path = xml_dir / "_fetch_manifest.json"
    inventory_path = xml_dir / "_fetch_inventory.jsonl"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise SystemExit(f"Missing immutable fetch manifest/inventory under {xml_dir}; rerun fetch_xml.py")
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
                raise SystemExit(f"Malformed fetch inventory {inventory_path}:{line_number}: {exc}") from exc
            if record.get("status") != "kept":
                continue
            destination = str(record.get("destination") or "")
            if not destination:
                raise SystemExit(f"Kept fetch record has no destination at line {line_number}")
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
    qc_revision = str(
        manifest.get("formosanbank_qc", {}).get("revision") or ""
    )
    if len(qc_revision) != 40:
        raise SystemExit(f"QC manifest has no pinned revision: {manifest_path}")
    return records, manifest


def load_mt_inventory(
    xml_dir: Path,
) -> tuple[dict[tuple[str, str, int, str], dict[str, object]], dict]:
    manifest_path = xml_dir / "_mt_standard_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(
            f"Missing MT standard manifest under {xml_dir}; rerun standardize_mt_corpus.py"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed MT standard manifest {manifest_path}: {exc}") from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"MT standard manifest is incomplete: {manifest_path}")
    profile = manifest.get("profile", {})
    profile_id = str(profile.get("id") or "")
    profile_hash = str(profile.get("sha256") or "")
    if not profile_id or len(profile_hash) != 64:
        raise SystemExit(f"MT standard manifest has an invalid profile: {manifest_path}")
    inventory_meta = manifest.get("inventory", {})
    inventory_path = xml_dir / str(inventory_meta.get("path") or "")
    if not inventory_path.is_file():
        raise SystemExit(f"Missing MT standard inventory: {inventory_path}")
    expected_hash = str(inventory_meta.get("sha256") or "")
    if sha256_file(inventory_path) != expected_hash:
        raise SystemExit(f"MT standard inventory checksum mismatch: {inventory_path}")

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
                    f"Malformed MT standard inventory {inventory_path}:{line_number}: {exc}"
                ) from exc
            if (
                str(row.get("mt_standard_profile") or "") != profile_id
                or str(row.get("mt_standard_profile_sha256") or "")
                != profile_hash
            ):
                raise SystemExit(
                    f"MT standard profile mismatch at {inventory_path}:{line_number}"
                )
            if row.get("source_disposition") != "retained":
                continue
            final_index = row.get("final_element_index")
            if final_index is None:
                raise SystemExit(f"Retained MT standard row has no final locator at line {line_number}")
            key = (
                str(row.get("xml_path") or ""),
                str(row.get("element_tag") or ""),
                int(final_index),
                str(row.get("final_xml_id") or row.get("xml_id") or ""),
            )
            if key in records:
                raise SystemExit(f"Duplicate MT standard locator at {inventory_path}:{line_number}: {key}")
            records[key] = row
    expected_records = int(inventory_meta.get("records", -1))
    if expected_records != inventory_rows:
        raise SystemExit(
            "MT standard inventory count is inconsistent: "
            f"manifest={expected_records}, inventory={inventory_rows}"
        )
    return records, manifest


def extract_file_targets(
    xml_path: Path,
    *,
    xml_dir: Path,
    provenance: dict[str, str],
    targets: dict[str, set[str]],
    tags: set[str],
    qc_records: dict[tuple[str, str, int, str], dict[str, str]] | None = None,
    mt_records: dict[tuple[str, str, int, str], dict[str, object]] | None = None,
    qc_revision: str = "",
) -> tuple[dict[str, list[ExtractedPair]], dict[str, Counter[str]]]:
    stats = {target: Counter() for target in targets}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    source_language = xml_lang(root)
    dialect = (root.get("dialect") or "UNKNOWN").strip() or "UNKNOWN"
    corpus_id = (root.get("id") or "").strip()
    relative = str(xml_path.relative_to(xml_dir))
    source_path = f"{provenance['repository']}/{provenance['source_path']}"
    pairs = {target: [] for target in targets}

    unit_index = 0
    for element in root.iter():
        if element.tag not in {"S", "W", "M"}:
            continue
        element_index = unit_index
        unit_index += 1
        if element.tag not in tags:
            continue
        for target_stats in stats.values():
            target_stats[f"{element.tag.lower()}_units_seen"] += 1
        final_xml_id = (element.get("id") or "").strip()
        locator = (relative, element.tag, element_index, final_xml_id)
        qc_record = (
            qc_records.get(
                locator
            )
            if qc_records is not None
            else None
        )
        if qc_records is not None and qc_record is None:
            raise ValueError(
                f"{relative}:{element.tag}:{final_xml_id} has no QC transform record"
            )
        qc_record = qc_record or {
            "transform_id": "",
            "standard_origin": "unknown",
            "original_before_qc_sha256": "",
            "standard_before_qc_sha256": "",
            "standard_after_qc_sha256": "",
        }
        mt_record = mt_records.get(locator) if mt_records is not None else None
        if mt_records is not None and mt_record is None:
            raise ValueError(
                f"{relative}:{element.tag}:{final_xml_id} has no MT standard record"
            )
        if mt_record is None:
            raise ValueError(
                "MT standard inventory is required for extraction; run standardize_mt_corpus.py"
            )
        mt_status = str(mt_record.get("mt_normalization_status") or "")
        if mt_status == "ineligible":
            reason = str(mt_record.get("mt_normalization_reason") or "unspecified")
            for target_stats in stats.values():
                target_stats[f"mt_ineligible:{reason}"] += 1
            continue
        standard_text = str(mt_record.get("formosan_mt_standard") or "")
        if not standard_text:
            raise ValueError(
                f"{relative}:{element.tag}:{final_xml_id} has empty model text with status {mt_status!r}"
            )
        source_standard = str(mt_record.get("formosan_source_standard") or "")
        original_text = str(mt_record.get("formosan_original_raw") or "")
        expected_source_hash = hashlib.sha256(
            source_standard.encode("utf-8")
        ).hexdigest()
        expected_mt_hash = hashlib.sha256(standard_text.encode("utf-8")).hexdigest()
        if str(mt_record.get("source_standard_sha256") or "") != expected_source_hash:
            raise ValueError(
                f"{relative}:{element.tag}:{final_xml_id} has a source-standard hash mismatch"
            )
        if str(mt_record.get("mt_standard_sha256") or "") != expected_mt_hash:
            raise ValueError(
                f"{relative}:{element.tag}:{final_xml_id} has an MT-standard hash mismatch"
            )
        contains_unclear = bool(mt_record.get("contains_unclear_source"))
        source_xml_id = qc_record.get("xml_id") or final_xml_id

        target_indices: Counter[str] = Counter()
        for translation in element.findall("TRANSL"):
            target_language = xml_lang(translation)
            matching_targets = [
                target
                for target, target_codes in targets.items()
                if target_language in target_codes
            ]
            if not matching_targets:
                continue
            target_text = mixed_text(translation)
            for target in matching_targets:
                target_index = target_indices[target]
                if not target_text:
                    stats[target]["empty_target"] += 1
                    continue
                row_id = content_row_id(
                    provenance["repository"],
                    provenance["source_path"],
                    element.tag,
                    element_index,
                    source_xml_id,
                    target_language,
                    target_index,
                )
                source_record_id = content_row_id(
                    provenance["repository"],
                    provenance["source_path"],
                    element.tag,
                    element_index,
                    source_xml_id,
                )
                content_hash = sha256_bytes(
                    "\u241f".join(
                        [
                            source_language,
                            standard_text,
                            target_text,
                            target_language,
                            row_type_for_tag(element.tag),
                        ]
                    ).encode("utf-8")
                )
                pairs[target].append(ExtractedPair(
                    row_id=row_id,
                    source_record_id=source_record_id,
                    content_sha256=content_hash,
                    lang_code=source_language,
                    formosan_sentence=standard_text,
                    target_text=target_text,
                    source=source_path,
                    repository=provenance["repository"],
                    repository_commit=provenance["repository_commit"],
                    xml_path=provenance["source_path"],
                    corpus_id=corpus_id,
                    xml_id=source_xml_id,
                    qc_final_xml_id=final_xml_id,
                    xml_element_index=element_index,
                    kind_of="standard",
                    standard_namespace="formosan-mt",
                    standard_origin=str(mt_record.get("standard_origin") or "missing"),
                    original_before_qc_sha256=qc_record[
                        "original_before_qc_sha256"
                    ],
                    standard_before_qc_sha256=qc_record[
                        "standard_before_qc_sha256"
                    ],
                    standard_after_qc_sha256=qc_record[
                        "standard_after_qc_sha256"
                    ],
                    qc_transform_id=qc_record["transform_id"],
                    qc_revision=qc_revision,
                    dialect=dialect,
                    row_type=row_type_for_tag(element.tag),
                    formosan_original=original_text,
                    formosan_standard=source_standard,
                    formosan_original_raw=original_text,
                    formosan_source_standard=source_standard,
                    formosan_mt_standard=standard_text,
                    source_standard_sha256=str(mt_record.get("source_standard_sha256") or ""),
                    mt_standard_sha256=str(mt_record.get("mt_standard_sha256") or ""),
                    mt_normalization_status=mt_status,
                    mt_normalization_confidence=str(
                        mt_record.get("mt_normalization_confidence") or ""
                    ),
                    mt_eval_eligible=bool(mt_record.get("mt_eval_eligible")),
                    mt_normalization_reason=str(mt_record.get("mt_normalization_reason") or ""),
                    mt_transformations=str(mt_record.get("mt_transformations") or "[]"),
                    mt_unresolved_markers=str(mt_record.get("mt_unresolved_markers") or ""),
                    speaker_label=str(mt_record.get("speaker_label") or ""),
                    mt_standard_profile=str(mt_record.get("mt_standard_profile") or ""),
                    mt_standard_profile_sha256=str(
                        mt_record.get("mt_standard_profile_sha256") or ""
                    ),
                    target_lang=target_language,
                    translation_index=target_index,
                    translation_kind=(translation.get("kindOf") or "").strip(),
                    translation_version=(translation.get("ver") or "").strip(),
                    contains_unclear=contains_unclear,
                ))
                target_indices[target] += 1
                stats[target]["pairs"] += 1
    return pairs, stats


def extract_file(
    xml_path: Path,
    *,
    xml_dir: Path,
    provenance: dict[str, str],
    target_codes: set[str],
    tags: set[str],
    qc_records: dict[tuple[str, str, int, str], dict[str, str]] | None = None,
    mt_records: dict[tuple[str, str, int, str], dict[str, object]] | None = None,
    qc_revision: str = "",
) -> tuple[list[ExtractedPair], Counter[str]]:
    pairs, stats = extract_file_targets(
        xml_path,
        xml_dir=xml_dir,
        provenance=provenance,
        targets={"target": target_codes},
        tags=tags,
        qc_records=qc_records,
        mt_records=mt_records,
        qc_revision=qc_revision,
    )
    return pairs["target"], stats["target"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract parallel rows using the toolkit MT standard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml-dir", default="downloaded_xml", type=Path)
    parser.add_argument("--target", choices=TARGET_MAP.keys())
    parser.add_argument("--out", default="corpus.csv", type=Path)
    parser.add_argument(
        "--output",
        action="append",
        default=[],
        metavar="TARGET=PATH",
        help="Extract multiple targets in one traversal; repeat for each output.",
    )
    parser.add_argument(
        "--units",
        default="sentences,words",
        help="Comma-separated XML units: sentences, words, morphemes.",
    )
    args = parser.parse_args()
    if bool(args.target) == bool(args.output):
        raise SystemExit("Choose either --target with --out or one or more --output TARGET=PATH values")
    return args


def target_outputs(args: argparse.Namespace) -> dict[str, Path]:
    if args.target:
        return {args.target: args.out}
    outputs: dict[str, Path] = {}
    for value in args.output:
        target, separator, raw_path = value.partition("=")
        target = target.strip().lower()
        if not separator or target not in TARGET_MAP or not raw_path.strip():
            raise SystemExit(
                f"Invalid --output {value!r}; expected one of "
                f"{sorted(TARGET_MAP)} followed by =PATH"
            )
        if target in outputs:
            raise SystemExit(f"Duplicate --output target: {target}")
        outputs[target] = Path(raw_path).expanduser()
    return outputs


def main() -> None:
    args = parse_args()
    outputs = target_outputs(args)
    xml_dir = args.xml_dir.expanduser().resolve()
    xml_files = sorted(xml_dir.rglob("*.xml"))
    if not xml_files:
        raise SystemExit(f"No XML files found under {xml_dir}")

    fetch_inventory = load_fetch_inventory(xml_dir)
    qc_inventory, qc_manifest = load_qc_inventory(xml_dir)
    mt_inventory, mt_manifest = load_mt_inventory(xml_dir)
    actual_xml = {
        str(path.relative_to(xml_dir))
        for path in xml_files
    }
    expected_xml = set(fetch_inventory)
    missing_xml = sorted(expected_xml - actual_xml)
    unexpected_xml = sorted(actual_xml - expected_xml)
    if missing_xml or unexpected_xml:
        raise SystemExit(
            "Downloaded XML does not match the immutable fetch inventory: "
            f"missing={missing_xml[:10]}, unexpected={unexpected_xml[:10]}"
        )
    qc_revision = str(
        qc_manifest["formosanbank_qc"]["revision"]
    )
    targets = {target: TARGET_MAP[target] for target in outputs}
    tags = wanted_tags(parse_units(args.units))
    all_stats = {target: Counter() for target in outputs}
    file_reports = {target: [] for target in outputs}
    errors: list[str] = []
    extracted = {target: [] for target in outputs}

    for xml_path in tqdm(xml_files, desc="Extract XML", unit="file"):
        relative = str(xml_path.relative_to(xml_dir))
        provenance = fetch_inventory.get(relative)
        if provenance is None:
            errors.append(f"{relative}: no kept record in fetch inventory")
            continue
        try:
            pairs, stats = extract_file_targets(
                xml_path,
                xml_dir=xml_dir,
                provenance=provenance,
                targets=targets,
                tags=tags,
                qc_records=qc_inventory,
                mt_records=mt_inventory,
                qc_revision=qc_revision,
            )
        except (ET.ParseError, ValueError) as exc:
            errors.append(f"{relative}: {exc}")
            continue
        post_qc_hash = sha256_file(xml_path)
        for target in outputs:
            extracted[target].extend(pairs[target])
            all_stats[target].update(stats[target])
            all_stats[target]["files_parsed"] += 1
            all_stats[target][
                "files_with_pairs" if pairs[target] else "files_without_pairs"
            ] += 1
            file_reports[target].append(
                {
                    "path": relative,
                    "post_qc_sha256": post_qc_hash,
                    "pairs": len(pairs[target]),
                    **dict(stats[target]),
                }
            )

    if errors:
        preview = "\n".join(f"  - {error}" for error in errors[:25])
        suffix = f"\n  ... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise SystemExit(f"Extraction failed:\n{preview}{suffix}")
    empty_targets = [target for target, pairs in extracted.items() if not pairs]
    if empty_targets:
        raise SystemExit(
            f"No translation pairs found for {', '.join(empty_targets)} under {xml_dir}"
        )

    shared_report = {
        "schema_version": 3,
        "created_at": utc_now(),
        "xml_dir": str(xml_dir),
        "fetch_inventory_sha256": sha256_file(xml_dir / "_fetch_inventory.jsonl"),
        "qc_manifest_sha256": sha256_file(xml_dir / "_qc_manifest.json"),
        "qc_transform_inventory_sha256": qc_manifest["transform_inventory"]["sha256"],
        "mt_standard_manifest_sha256": sha256_file(xml_dir / "_mt_standard_manifest.json"),
        "mt_standard_inventory_sha256": mt_manifest["inventory"]["sha256"],
        "mt_standard_profile": mt_manifest["profile"],
        "qc_revision": qc_revision,
        "units": sorted(parse_units(args.units)),
        "files_total": len(xml_files),
        "complete": True,
    }
    for target, output in outputs.items():
        output_columns = [
            column.replace("target_text", target) for column in OUTPUT_COLUMNS
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns)
            writer.writeheader()
            for pair in extracted[target]:
                row = pair.to_csv_row()
                row[target] = row.pop("target_text")
                writer.writerow(row)

        report = {
            **shared_report,
            "output": str(output),
            "source_language": extracted[target][0].lang_code,
            "target": target,
            "target_codes": sorted(targets[target]),
            "rows": len(extracted[target]),
            "counts": dict(sorted(all_stats[target].items())),
            "file_inventory_sha256": stable_json_hash(file_reports[target]),
        }
        report_path = output.with_suffix(".extraction.json")
        atomic_write_json(report_path, report)
        print(f"Wrote {len(extracted[target]):,} MT-standard pairs -> {output}")
        print(f"Extraction report: {report_path}")


if __name__ == "__main__":
    main()
