#!/usr/bin/env python3
"""
STEP 3: 

Build a sentence‑aligned parallel corpus from the XML files harvested by
fetch_xml.py.

Examples
--------
python make_corpus.py --xml-dir downloaded_ami --target chinese --out amis_zh.csv
python make_corpus.py --xml-dir downloaded_en --target english --out amis_en.csv
python make_corpus.py --xml-dir downloaded_xml --target english --out amis_en_orig.csv --original
"""
from __future__ import annotations

import argparse
import csv
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, List, Tuple

from tqdm import tqdm

# ─────────────────────────────  language maps  ───────────────────────────────
TARGET_MAP: dict[str, set[str]] = {
    "chinese": {"zh", "zho", "chi"},
    "english": {"en", "eng"},
}
# ─────────────────────────────────────────────────────────────────────────────


def list_xml_files(root: Path) -> Iterable[Path]:
    """Recursively yield all *.xml files under *root*."""
    return root.rglob("*.xml")


def choose_src_sentence(s_elem: ET.Element, kind_preference: str) -> str | None:
    """
    Return the text of the preferred <FORM> element inside an <S> element.

    Parameters
    ----------
    s_elem : ET.Element
        The <S> element.
    kind_preference : {"standard", "original"}
        Which orthography to prefer.

    Notes
    -----
    Falls back to the very first <FORM> if the preferred kind is missing.
    """
    for form in s_elem.findall("FORM"):
        if (form.get("kindOf") or "").lower() == kind_preference:
            return form.text
    first = s_elem.find("FORM")
    return first.text if first is not None else None


def extract_pairs(
    xml_path: Path,
    target_codes: set[str],
    kind_preference: str,
) -> List[Tuple[str, str]]:
    """Extract (src, tgt) sentence tuples from *xml_path*."""
    pairs: List[Tuple[str, str]] = []
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return pairs  # skip ill‑formed documents

    for s in tree.iter("S"):
        src_sent = choose_src_sentence(s, kind_preference)
        if not src_sent:
            continue
        for transl in s.findall("TRANSL"):
            tgt_lang = transl.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
            if tgt_lang and tgt_lang.lower() in target_codes:
                tgt_sent = transl.text
                if tgt_sent:
                    pairs.append((src_sent.strip(), tgt_sent.strip()))
                break
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", default="downloaded_xml", type=Path)
    parser.add_argument("--target", required=True, choices=TARGET_MAP.keys())
    parser.add_argument("--out", default="corpus.csv", type=Path)
    parser.add_argument(
        "--original",
        action="store_true",
        help='Prefer FORM kindOf="original" (default: "standard")',
    )
    args = parser.parse_args()

    kind_pref = "original" if args.original else "standard"
    target_codes = TARGET_MAP[args.target]

    xml_files = list(list_xml_files(args.xml_dir))
    if not xml_files:
        sys.exit(f"❌  No XML files found under {args.xml_dir}")

    # — determine the source language tag from the first parsable file —
    first_src_lang = "src"
    for p in xml_files:
        try:
            first_src_lang = (
                ET.parse(p)
                .getroot()
                .attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "src")
            )
            if first_src_lang:
                break
        except ET.ParseError:
            continue

    total_pairs = 0
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([first_src_lang, args.target, "source", "kindOf", "dialect"])

        for xml_path in tqdm(xml_files, desc="XML files", unit="file"):
            source_path = str(xml_path.relative_to(args.xml_dir))

            # Extract dialect from XML root element
            dialect = "UNKNOWN"
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                dialect = root.attrib.get("dialect", "UNKNOWN")
            except ET.ParseError:
                pass  # Keep dialect as "UNKNOWN" for unparseable files

            pairs = extract_pairs(xml_path, target_codes, kind_pref)
            for src_sent, tgt_sent in pairs:
                writer.writerow([src_sent, tgt_sent, source_path, kind_pref, dialect])

            total_pairs += len(pairs)

    print(f"✅  Wrote {total_pairs:,} sentence pairs → {args.out}")


if __name__ == "__main__":
    main()
