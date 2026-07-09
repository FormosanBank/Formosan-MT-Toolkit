#!/usr/bin/env python3
"""Extract MT-ready raw parallel rows from downloaded FormosanBank XML."""

from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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


@dataclass(frozen=True)
class ExtractedPair:
    source_text: str
    target_text: str
    kind_of: str
    row_type: str
    xml_id: str


def list_xml_files(root: Path) -> Iterable[Path]:
    return root.rglob("*.xml")


def xml_lang(elem: ET.Element) -> str:
    return (
        elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        or elem.attrib.get("xml:lang")
        or ""
    ).strip().lower()


def choose_form(elem: ET.Element, kind_preference: str, allow_fallback: bool) -> tuple[str, str] | None:
    forms = elem.findall("FORM")
    for form in forms:
        if (form.get("kindOf") or "").strip().lower() == kind_preference:
            text = (form.text or "").strip()
            if text:
                return text, kind_preference

    if allow_fallback:
        for form in forms:
            text = (form.text or "").strip()
            if text:
                return text, (form.get("kindOf") or "unknown").strip() or "unknown"
    return None


def row_type_for_tag(tag: str) -> str:
    if tag == "S":
        return "sentence"
    if tag == "W":
        return "lexeme"
    return "morpheme"


def wanted_tags(units: set[str]) -> set[str]:
    tags: set[str] = set()
    for unit in units:
        tags.update(UNIT_TAGS[unit])
    return tags


def extract_pairs(
    xml_path: Path,
    target_codes: set[str],
    kind_preference: str,
    units: set[str],
    allow_form_fallback: bool,
) -> list[ExtractedPair]:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    pairs: list[ExtractedPair] = []
    tags = wanted_tags(units)
    for elem in root.iter():
        if elem.tag not in tags:
            continue
        form = choose_form(elem, kind_preference, allow_fallback=allow_form_fallback)
        if form is None:
            continue
        source_text, actual_kind = form
        for transl in elem.findall("TRANSL"):
            target_lang = xml_lang(transl)
            target_text = (transl.text or "").strip()
            if target_lang in target_codes and target_text:
                pairs.append(
                    ExtractedPair(
                        source_text=source_text,
                        target_text=target_text,
                        kind_of=actual_kind,
                        row_type=row_type_for_tag(elem.tag),
                        xml_id=elem.get("id", ""),
                    )
                )
    return pairs


def parse_units(raw: str) -> set[str]:
    units = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = sorted(units - set(UNIT_TAGS))
    if unknown:
        raise SystemExit(f"Unknown --units values: {unknown}. Valid values: {sorted(UNIT_TAGS)}")
    return units or {"sentences", "words"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a raw parallel corpus from FormosanBank XML.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--xml-dir", default="downloaded_xml", type=Path)
    parser.add_argument("--target", required=True, choices=TARGET_MAP.keys())
    parser.add_argument("--out", default="corpus.csv", type=Path)
    parser.add_argument(
        "--original",
        action="store_true",
        help='Prefer FORM kindOf="original" instead of "standard".',
    )
    parser.add_argument(
        "--allow-form-fallback",
        action="store_true",
        help="If the preferred FORM kind is missing, fall back to the first non-empty FORM.",
    )
    parser.add_argument(
        "--units",
        default="sentences,words",
        help="Comma-separated XML unit types to extract: sentences, words, morphemes.",
    )
    args = parser.parse_args()

    kind_pref = "original" if args.original else "standard"
    target_codes = TARGET_MAP[args.target]
    units = parse_units(args.units)

    xml_files = list(list_xml_files(args.xml_dir))
    if not xml_files:
        sys.exit(f"No XML files found under {args.xml_dir}")

    first_src_lang = "src"
    for path in xml_files:
        try:
            first_src_lang = xml_lang(ET.parse(path).getroot()) or "src"
            if first_src_lang:
                break
        except ET.ParseError:
            continue

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_pairs = 0
    row_type_counts: dict[str, int] = {}
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                first_src_lang,
                args.target,
                "source",
                "kindOf",
                "dialect",
                "row_type",
                "xml_id",
            ]
        )

        for xml_path in tqdm(xml_files, desc="XML files", unit="file"):
            source_path = str(xml_path.relative_to(args.xml_dir))
            try:
                root = ET.parse(xml_path).getroot()
                dialect = root.attrib.get("dialect", "UNKNOWN")
            except ET.ParseError:
                dialect = "UNKNOWN"

            pairs = extract_pairs(
                xml_path,
                target_codes,
                kind_pref,
                units,
                allow_form_fallback=args.allow_form_fallback,
            )
            for pair in pairs:
                writer.writerow(
                    [
                        pair.source_text,
                        pair.target_text,
                        source_path,
                        pair.kind_of,
                        dialect,
                        pair.row_type,
                        pair.xml_id,
                    ]
                )
                row_type_counts[pair.row_type] = row_type_counts.get(pair.row_type, 0) + 1
            total_pairs += len(pairs)

    print(f"Wrote {total_pairs:,} raw pairs -> {args.out}")
    print("Row types: " + ", ".join(f"{k}={v:,}" for k, v in sorted(row_type_counts.items())))


if __name__ == "__main__":
    main()
