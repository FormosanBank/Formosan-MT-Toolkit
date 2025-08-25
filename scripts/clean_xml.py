#!/usr/bin/env python3
"""
STEP 3: 

Clean the XML files harvested by fetch_xml.py using the scripts defined in 
the main FormosanBank/FormosanBank repository.

<TEXT xml:lang="…"> == src_lang  
<TRANSL xml:lang="…"> == tgt_lang (optional)

Usage examples
--------------
$ python clean_xml.py --src-lang ami
$ python clean_xml.py --src-lang ami --tgt-lang zho
"""

from __future__ import annotations
import argparse
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import urllib.request


QC_SCRIPTS = {
    "clean_xml.py": "https://raw.githubusercontent.com/FormosanBank/FormosanBank/main/QC/cleaning/clean_xml.py",
    "standardize.py": "https://raw.githubusercontent.com/FormosanBank/FormosanBank/main/QC/utilities/standardize.py",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", required=True, help="Source language (e.g. 'ami')")
    parser.add_argument("--tgt-lang", help="Target language (e.g. 'zho')")
    parser.add_argument("--in-dir", default="../downloaded_xml", help="Directory of raw XML files")
    return parser.parse_args()


def download_qc_scripts(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for filename, url in QC_SCRIPTS.items():
        print(f"Downloading {filename}...")
        out_path = dest / filename
        urllib.request.urlretrieve(url, out_path)


def is_valid_file(path: Path, src_lang: str, tgt_lang: Optional[str]) -> bool:
    try:
        tree = ET.parse(path)
        root = tree.getroot()

        text_lang = root.attrib.get("xml:lang") or root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        if text_lang != src_lang:
            return False

        if tgt_lang:
            for s in root.findall(".//S"):
                for child in s:
                    if child.tag == "TRANSL" and (
                        child.attrib.get("xml:lang") == tgt_lang or
                        child.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == tgt_lang
                    ):
                        return True
            return False
        return True
    except ET.ParseError:
        print(f"[!] Skipping unparseable file: {path}")
        return False


def filter_invalid_files(in_dir: Path, src_lang: str, tgt_lang: Optional[str]):
    xml_files = list(in_dir.rglob("*.xml"))
    print(f"Filtering {len(xml_files)} files...")
    for xml_file in tqdm(xml_files):
        if not is_valid_file(xml_file, src_lang, tgt_lang):
            xml_file.unlink()  # Delete invalid file


def run_qc_scripts(corpus_dir: Path, qc_dir: Path):
    clean_script = qc_dir / "clean_xml.py"
    std_script = qc_dir / "standardize.py"

    print("Running cleaning script...")
    subprocess.run(["python", str(clean_script), "--corpora_path", str(corpus_dir)], check=True)

    print("Running standardization script...")
    subprocess.run(["python", str(std_script), "--corpora_path", str(corpus_dir)], check=True)


def main():
    args = parse_args()
    in_dir = Path(args.in_dir).resolve()
    qc_dir = Path(".formosan_qc_scripts")

    print("⬇️  Downloading QC scripts...")
    download_qc_scripts(qc_dir)

    print("🧹 Filtering XML files...")
    filter_invalid_files(in_dir, args.src_lang, args.tgt_lang)

    print("⚙️  Running QC scripts on in-place XML data...")
    run_qc_scripts(in_dir, qc_dir)

    print("✅ Done!")


if __name__ == "__main__":
    main()
