"""Deterministic FormosanBank XML repairs for MT corpus preparation."""

from __future__ import annotations

import hashlib
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

PROVENANCE_ATTR = "_mt_toolkit_transform_id"
ZERO_WIDTH_CHARACTERS = frozenset("\u200b\u200c\u200d\ufeff")


def _direct_content(
    element: ET.Element,
) -> list[tuple[ET.Element | None, str]]:
    content: list[tuple[ET.Element | None, str]] = []
    if element.text and element.text.strip():
        content.append((None, element.text))
    for child in element:
        if child.tail and child.tail.strip():
            content.append((child, child.tail))
    return content


def _punctuation_only(text: str) -> bool:
    return all(
        character.isspace()
        or unicodedata.category(character)[0] in {"P", "S"}
        for character in text
    )


def _remove_zero_width_characters(
    *,
    element: ET.Element,
    relative: str,
    repairs: list[dict[str, object]],
) -> int:
    repaired = 0
    for field in [
        *element.findall("FORM"),
        *element.findall("TRANSL"),
    ]:
        if not field.text or not any(
            character in field.text
            for character in ZERO_WIDTH_CHARACTERS
        ):
            continue
        before = field.text
        field.text = "".join(
            character
            for character in before
            if character not in ZERO_WIDTH_CHARACTERS
        )
        repairs.append(
            {
                "repair": "remove_zero_width_characters",
                "xml_path": relative,
                "element_tag": element.tag,
                "xml_id": element.get("id", ""),
                "field_tag": field.tag,
                "field_kind": field.get("kindOf", ""),
                "before_sha256": hashlib.sha256(
                    before.encode("utf-8")
                ).hexdigest(),
                "after_sha256": hashlib.sha256(
                    field.text.encode("utf-8")
                ).hexdigest(),
            }
        )
        repaired += 1
    return repaired


def _trim_form_boundaries(
    *,
    element: ET.Element,
    relative: str,
    repairs: list[dict[str, object]],
) -> int:
    repaired = 0
    for form in element.findall("FORM"):
        if len(form) or not form.text:
            continue
        trimmed = form.text.strip()
        if trimmed == form.text:
            continue
        repairs.append(
            {
                "repair": "trim_form_boundary_whitespace",
                "xml_path": relative,
                "element_tag": element.tag,
                "xml_id": element.get("id", ""),
                "form_kind": form.get("kindOf", ""),
                "before_sha256": hashlib.sha256(
                    form.text.encode("utf-8")
                ).hexdigest(),
                "after_sha256": hashlib.sha256(
                    trimmed.encode("utf-8")
                ).hexdigest(),
            }
        )
        form.text = trimmed
        repaired += 1
    return repaired


def _standard_form(element: ET.Element) -> ET.Element | None:
    return next(
        (
            form
            for form in element.findall("FORM")
            if (form.get("kindOf") or "").strip().lower() == "standard"
        ),
        None,
    )


def repair_mt_xml_structure(
    corpus_dir: Path,
) -> tuple[dict[str, int], list[dict[str, object]], dict[str, str]]:
    """Repair deterministic field defects without deleting XML units."""
    stats: Counter[str] = Counter()
    repairs: list[dict[str, object]] = []
    removal_dispositions: dict[str, str] = {}

    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        changed = False
        seen_ids: set[str] = set()
        duplicate_counts: Counter[str] = Counter()
        referenced_ids = {
            value.lstrip("#")
            for element in root.iter()
            for name, value in element.attrib.items()
            if name != "id" and value
        }

        for element in list(root.iter()):
            if element.tag not in {"S", "W", "M"}:
                continue
            if element.get(PROVENANCE_ATTR, "") in removal_dispositions:
                continue
            if element.tag == "S" and any(
                "∅" in "".join(form.itertext())
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower()
                == "standard"
            ):
                stats["null_source_sentences_preserved"] += 1

            if any(
                "*" in "".join(form.itertext())
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower()
                in {"original", "standard"}
            ):
                stats["source_annotation_units_preserved"] += 1

            zero_width = _remove_zero_width_characters(
                element=element,
                relative=relative,
                repairs=repairs,
            )
            if zero_width:
                stats["zero_width_fields_repaired"] += zero_width
                changed = True

            boundary_repairs = _trim_form_boundaries(
                element=element,
                relative=relative,
                repairs=repairs,
            )
            if boundary_repairs:
                stats[
                    "form_boundary_whitespace_trimmed"
                ] += boundary_repairs
                changed = True

            standard = _standard_form(element)
            standard_text = (
                "".join(standard.itertext()).strip()
                if standard is not None
                else ""
            )
            if standard is not None and not standard_text:
                if element.tag == "S":
                    if standard.find("UNCLEAR") is not None:
                        stats["unclear_source_sentences_preserved"] += 1
                    elif element.findall("AUDIO"):
                        stats[
                            "untranscribed_audio_sentences_preserved"
                        ] += 1
                    else:
                        stats[
                            "empty_source_sentences_preserved"
                        ] += 1
                else:
                    stats["empty_source_lexical_units_preserved"] += 1
            elif (
                standard is None
                and element.tag == "S"
                and element.findall("AUDIO")
            ):
                stats["untranscribed_audio_sentences_preserved"] += 1

            for owner, text in _direct_content(element):
                if not _punctuation_only(text):
                    raise SystemExit(
                        "Substantive untyped content inside XML unit at "
                        f"{relative}:{element.tag}:{element.get('id', '')}: "
                        f"{text.strip()!r}"
                    )
                if owner is None:
                    element.text = None
                else:
                    owner.tail = None
                repairs.append(
                    {
                        "repair": "remove_untyped_punctuation",
                        "xml_path": relative,
                        "element_tag": element.tag,
                        "xml_id": element.get("id", ""),
                        "content": text.strip(),
                    }
                )
                stats["untyped_punctuation_removed"] += 1
                changed = True

            xml_id = (element.get("id") or "").strip()
            if not xml_id:
                continue
            duplicate_counts[xml_id] += 1
            if xml_id not in seen_ids:
                seen_ids.add(xml_id)
                continue
            if xml_id in referenced_ids:
                raise SystemExit(
                    f"Cannot disambiguate referenced duplicate id "
                    f"{xml_id!r} in {relative}"
                )
            suffix = duplicate_counts[xml_id]
            candidate = f"{xml_id}__mtdup{suffix}"
            while candidate in seen_ids:
                suffix += 1
                candidate = f"{xml_id}__mtdup{suffix}"
            element.set("id", candidate)
            seen_ids.add(candidate)
            repairs.append(
                {
                    "repair": "disambiguate_duplicate_id",
                    "xml_path": relative,
                    "element_tag": element.tag,
                    "source_xml_id": xml_id,
                    "final_xml_id": candidate,
                    "transform_id": element.get(PROVENANCE_ATTR, ""),
                }
            )
            stats["duplicate_ids_disambiguated"] += 1
            changed = True

        if changed:
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    return dict(stats), repairs, removal_dispositions
