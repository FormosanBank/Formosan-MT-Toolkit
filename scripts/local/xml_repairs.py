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


def _descendant_tokens(element: ET.Element) -> list[str]:
    return [
        child.get(PROVENANCE_ATTR, "")
        for child in element.iter()
        if child.tag in {"S", "W", "M"} and child.get(PROVENANCE_ATTR)
    ]


def _remove_unit(
    *,
    element: ET.Element,
    parent: ET.Element | None,
    relative: str,
    disposition: str,
    repair: str,
    removal_dispositions: dict[str, str],
    repairs: list[dict[str, object]],
    validator_rule: str | None = None,
) -> int:
    if parent is None:
        raise SystemExit(
            f"Cannot remove root {element.tag} unit at {relative}"
        )
    tokens = _descendant_tokens(element)
    parent.remove(element)
    for token in tokens:
        removal_dispositions[token] = disposition
    record: dict[str, object] = {
        "repair": repair,
        "xml_path": relative,
        "element_tag": element.tag,
        "xml_id": element.get("id", ""),
        "transform_ids": tokens,
    }
    if validator_rule:
        record["validator_rule"] = validator_rule
    repairs.append(record)
    return len(tokens)


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
    """Repair deterministic defects while quarantining unusable MT units."""
    stats: Counter[str] = Counter()
    repairs: list[dict[str, object]] = []
    removal_dispositions: dict[str, str] = {}

    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        parent_by_child = {
            child: parent
            for parent in root.iter()
            for child in parent
        }
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
            parent = parent_by_child.get(element)

            if element.tag == "S" and any(
                "∅" in "".join(form.itertext())
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower()
                == "standard"
            ):
                removed = _remove_unit(
                    element=element,
                    parent=parent,
                    relative=relative,
                    disposition="removed_null_source_sentence",
                    repair="remove_null_source_sentence",
                    validator_rule="V120",
                    removal_dispositions=removal_dispositions,
                    repairs=repairs,
                )
                stats["null_source_sentences_removed"] += 1
                stats["null_source_sentence_descendants_removed"] += removed
                changed = True
                continue

            if any(
                "*" in "".join(form.itertext())
                for form in element.findall("FORM")
                if (form.get("kindOf") or "").strip().lower()
                in {"original", "standard"}
            ):
                removed = _remove_unit(
                    element=element,
                    parent=parent,
                    relative=relative,
                    disposition="removed_hard_text_annotation",
                    repair="remove_hard_text_annotation",
                    validator_rule="V129",
                    removal_dispositions=removal_dispositions,
                    repairs=repairs,
                )
                stats["hard_text_annotation_units_removed"] += 1
                stats[
                    "hard_text_annotation_descendants_removed"
                ] += removed
                changed = True
                continue

            zero_width = _remove_zero_width_characters(
                element=element,
                relative=relative,
                repairs=repairs,
            )
            if zero_width:
                stats["zero_width_fields_repaired"] += zero_width
                changed = True

            if element.tag in {"W", "M"} and any(
                character in "".join(form.itertext())
                for form in element.findall("FORM")
                for character in "()/"
            ):
                removed = _remove_unit(
                    element=element,
                    parent=parent,
                    relative=relative,
                    disposition="removed_lexical_annotation",
                    repair="remove_lexical_annotation",
                    validator_rule="V121",
                    removal_dispositions=removal_dispositions,
                    repairs=repairs,
                )
                stats["lexical_annotation_units_removed"] += 1
                stats["lexical_annotation_descendants_removed"] += removed
                changed = True
                continue

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
                    raise SystemExit(
                        f"Empty sentence standard FORM at "
                        f"{relative}:{element.tag}:{element.get('id', '')}"
                    )
                if element.tag == "W" and any(
                    "".join(form.itertext()).strip()
                    for descendant in element.iter("M")
                    for form in descendant.findall("FORM")
                    if (form.get("kindOf") or "").strip().lower()
                    == "standard"
                ):
                    raise SystemExit(
                        "Cannot remove an empty W containing usable morphemes "
                        f"at {relative}:{element.get('id', '')}"
                    )
                removed = _remove_unit(
                    element=element,
                    parent=parent,
                    relative=relative,
                    disposition="removed_empty_source_lexical_unit",
                    repair="remove_empty_source_lexical_unit",
                    removal_dispositions=removal_dispositions,
                    repairs=repairs,
                )
                stats["empty_source_lexical_units_removed"] += 1
                stats[
                    "empty_source_lexical_descendants_removed"
                ] += removed
                changed = True
                continue

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
