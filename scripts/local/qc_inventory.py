"""Transformation and repair inventories for prepared XML."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from pipeline_common import sha256_file
from xml_repairs import PROVENANCE_ATTR

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
TRANSFORM_INVENTORY = "_qc_transform_inventory.jsonl"
REPAIR_INVENTORY = "_qc_repair_inventory.jsonl"


def xml_lang(element: ET.Element) -> str:
    return (
        element.attrib.get(XML_LANG)
        or element.attrib.get("xml:lang")
        or ""
    ).strip().lower()


def mixed_content_signature(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


def mixed_text(element: ET.Element | None) -> str:
    return "" if element is None else "".join(element.itertext()).strip()


def form_sha256(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    return hashlib.sha256(
        ET.tostring(element, encoding="utf-8")
    ).hexdigest()


def tag_transform_sources(
    corpus_dir: Path,
) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        unit_index = 0
        for element in root.iter():
            if element.tag not in {"S", "W", "M"}:
                continue
            if PROVENANCE_ATTR in element.attrib:
                raise SystemExit(
                    f"Reserved QC provenance attribute already exists: "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            token = hashlib.sha256(
                (
                    f"{relative}\u241f{element.tag}\u241f{unit_index}\u241f"
                    f"{element.get('id', '')}"
                ).encode("utf-8")
            ).hexdigest()[:24]
            originals = [
                form
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower() == "original"
            ]
            standards = [
                form
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower() == "standard"
            ]
            untyped = [
                form
                for form in element.findall("FORM")
                if not (form.get("kindOf") or "").strip()
            ]
            if len(originals) > 1 or len(standards) > 1 or len(untyped) > 1:
                raise SystemExit(
                    f"Duplicate or ambiguous FORM tiers at "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            original = originals[0] if originals else None
            provided_standard = standards[0] if standards else None
            untyped_form = untyped[0] if untyped else None
            provided_text = mixed_text(provided_standard)
            original_text = mixed_text(original)
            untyped_text = mixed_text(untyped_form)
            if provided_text:
                selected = provided_standard
                selected_text = provided_text
                standard_origin = "provided"
            elif original_text:
                selected = original
                selected_text = original_text
                standard_origin = "derived_from_original"
            elif untyped_text:
                selected = untyped_form
                selected_text = untyped_text
                standard_origin = "derived_from_untyped"
                original_text = untyped_text
            else:
                selected = None
                selected_text = ""
                standard_origin = (
                    "untranscribed_audio"
                    if element.tag == "S" and element.findall("AUDIO")
                    else "missing"
                )
            records[token] = {
                "transform_id": token,
                "xml_path": relative,
                "element_tag": element.tag,
                "xml_id": element.get("id", ""),
                "source_element_index": unit_index,
                "standard_origin": standard_origin,
                "provided_standard_present": bool(provided_standard is not None),
                "original_before_qc_sha256": form_sha256(
                    original if original is not None else untyped_form
                ),
                "standard_before_qc_sha256": form_sha256(provided_standard),
                "source_standard_sha256": form_sha256(selected),
                "formosan_original_raw": original_text,
                "formosan_source_standard": selected_text,
                "provided_standard_raw": provided_text,
                "contains_unclear_source": bool(
                    selected is not None
                    and any(child.tag == "UNCLEAR" for child in selected.iter())
                ),
            }
            element.set(PROVENANCE_ATTR, token)
            unit_index += 1
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    return records


def finalize_transform_inventory(
    corpus_dir: Path,
    records: dict[str, dict[str, object]],
    removal_dispositions: dict[str, str] | None = None,
) -> list[dict[str, object]]:
    removal_dispositions = removal_dispositions or {}
    retained: set[str] = set()
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        unit_index = 0
        for element in root.iter():
            if element.tag not in {"S", "W", "M"}:
                continue
            token = element.attrib.pop(PROVENANCE_ATTR, "")
            if not token or token not in records:
                raise SystemExit(
                    f"QC cleaner lost transform provenance at "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            standards = [
                form
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower() == "standard"
            ]
            if len(standards) > 1:
                raise SystemExit(
                    f"Expected at most one standard tier after QC at "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            record = records[token]
            record.update(
                {
                    "final_element_index": unit_index,
                    "final_xml_id": element.get("id", ""),
                    "standard_after_qc_sha256": form_sha256(
                        standards[0] if standards else None
                    ),
                    "disposition": "retained",
                }
            )
            retained.add(token)
            unit_index += 1
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    for token, record in records.items():
        if token not in retained:
            record.update(
                {
                    "final_element_index": None,
                    "final_xml_id": None,
                    "standard_after_qc_sha256": None,
                    "disposition": removal_dispositions.get(
                        token,
                        "removed_by_cleaner",
                    ),
                }
            )
    return sorted(
        records.values(),
        key=lambda row: (
            str(row["xml_path"]),
            int(row["source_element_index"]),
        ),
    )


def remove_transform_tags(corpus_dir: Path) -> None:
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        changed = False
        for element in tree.getroot().iter():
            if PROVENANCE_ATTR in element.attrib:
                element.attrib.pop(PROVENANCE_ATTR)
                changed = True
        if changed:
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)


def write_transform_inventory(
    corpus_dir: Path,
    rows: list[dict[str, object]],
) -> dict[str, object]:
    path = corpus_dir / TRANSFORM_INVENTORY
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    dispositions = Counter(str(row["disposition"]) for row in rows)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "records": len(rows),
        "retained": dispositions["retained"],
        "removed_by_cleaner": dispositions["removed_by_cleaner"],
        "dispositions": dict(sorted(dispositions.items())),
    }


def write_repair_inventory(
    corpus_dir: Path,
    repairs: list[dict[str, object]],
) -> dict[str, object]:
    path = corpus_dir / REPAIR_INVENTORY
    with path.open("w", encoding="utf-8") as handle:
        for row in repairs:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    counts = Counter(str(row["repair"]) for row in repairs)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "records": len(repairs),
        "counts": dict(sorted(counts.items())),
    }


def snapshot_translation_versions(
    corpus_dir: Path,
) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        for parent in root.iter():
            if parent.tag not in {"S", "W", "M"}:
                continue
            token = parent.get(PROVENANCE_ATTR, "")
            if not token:
                raise SystemExit(
                    "Cannot audit TRANSL/@ver without transform "
                    f"provenance at {relative}:{parent.tag}:"
                    f"{parent.get('id', '')}"
                )
            occurrence = 0
            for child in parent:
                if child.tag != "TRANSL":
                    continue
                records[f"{token}:TRANSL:{occurrence}"] = {
                    "xml_path": relative,
                    "xml_id": parent.get("id", ""),
                    "element_tag": parent.tag,
                    "language": xml_lang(child),
                    "ver": (child.get("ver") or "").strip(),
                }
                occurrence += 1
    return records


def classify_translation_version_repairs(
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    if set(before) != set(after):
        raise SystemExit(
            "Translation version completion changed the TRANSL element set"
        )
    repairs: list[dict[str, object]] = []
    for key, before_row in before.items():
        after_row = after[key]
        if before_row["ver"] == after_row["ver"]:
            continue
        if before_row["ver"] or after_row["ver"] != "alt":
            raise SystemExit(
                "Translation version completion made an unexpected "
                f"change at {before_row['xml_path']}:{before_row['xml_id']}"
            )
        repairs.append(
            {
                "repair": "mark_alternate_translation",
                "xml_path": before_row["xml_path"],
                "element_tag": before_row["element_tag"],
                "xml_id": before_row["xml_id"],
                "language": before_row["language"],
                "before": "",
                "after": "alt",
            }
        )
    return repairs


