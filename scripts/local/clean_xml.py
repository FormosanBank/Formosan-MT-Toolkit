#!/usr/bin/env python3
"""
STEP 2: 

Clean the XML files harvested by fetch_xml.py using the official QC scripts 
from the FormosanBank/FormosanBank repository.

This script:
1. Downloads/updates the latest QC scripts from FormosanBank repository
2. Filters XML files to match source language (and optional target language)
3. Runs the official clean_xml.py and standardize.py scripts in-place

The script intelligently checks for updates to the QC scripts using checksums,
only downloading when changes are detected or when forced.

Usage examples
--------------
$ python clean_xml.py --src-lang ami
$ python clean_xml.py --src-lang ami --force-update  # Force download QC scripts
"""

from __future__ import annotations
import argparse
import hashlib
import os
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import urllib.request
import urllib.error


QC_SCRIPTS = {
    "clean_xml.py": "https://raw.githubusercontent.com/FormosanBank/FormosanBank/main/QC/cleaning/clean_xml.py",
    "standardize.py": "https://raw.githubusercontent.com/FormosanBank/FormosanBank/main/QC/utilities/standardize.py",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean XML files using FormosanBank QC scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src-lang", required=True, help="Source language (e.g. 'ami')")
    parser.add_argument("--tgt-lang", help="Target language (e.g. 'zho')")
    parser.add_argument("--in-dir", help="Directory of raw XML files (default: downloaded_{src_lang})")
    parser.add_argument(
        "--force-update", action="store_true", 
        help="Force download of QC scripts even if local versions exist"
    )
    parser.add_argument(
        "--qc-dir", default="scripts/.formosan_qc_scripts", 
        help="Directory to store downloaded QC scripts [default: %(default)s]"
    )
    return parser.parse_args()


def get_file_checksum(file_path: Path) -> str:
    """Calculate SHA256 checksum of a file."""
    if not file_path.exists():
        return ""
    
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def download_with_retry(url: str, max_retries: int = 3) -> bytes:
    """Download content from URL with retry logic."""
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return response.read()
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if attempt == max_retries - 1:
                raise Exception(f"Failed to download after {max_retries} attempts: {e}")
            print(f"⚠️  Attempt {attempt + 1} failed: {e}")
            print(f"🔄  Retrying in 2 seconds...")
            import time
            time.sleep(2)
    
    # This should never be reached, but just in case
    raise Exception("Unexpected error in download_with_retry")


def validate_python_script(content: bytes, filename: str) -> bool:
    """Validate that the downloaded content is a valid Python script."""
    try:
        # Try to decode as UTF-8
        script_text = content.decode('utf-8')
        
        # Basic validation: should contain some expected Python patterns
        if filename == "clean_xml.py":
            expected_patterns = ["def ", "import ", "xml"]
        elif filename == "standardize.py":
            expected_patterns = ["def ", "import "]
        else:
            expected_patterns = ["def ", "import "]
        
        return all(pattern in script_text for pattern in expected_patterns)
    
    except UnicodeDecodeError:
        return False


def download_qc_scripts(dest: Path, force_update: bool = False):
    """
    Download QC scripts from FormosanBank repository, with smart updating.
    
    Parameters
    ----------
    dest : Path
        Destination directory for scripts
    force_update : bool
        If True, always download scripts regardless of existing files
    """
    dest.mkdir(parents=True, exist_ok=True)
    
    updated_count = 0
    skipped_count = 0
    
    for filename, url in QC_SCRIPTS.items():
        out_path = dest / filename
        
        print(f"📡 Checking {filename}...")
        
        try:
            # Download the current version from remote
            remote_content = download_with_retry(url)
            
            # Validate the downloaded content
            if not validate_python_script(remote_content, filename):
                print(f"⚠️  Warning: Downloaded {filename} doesn't appear to be a valid Python script")
                continue
            
            # Calculate checksum of remote content
            remote_checksum = hashlib.sha256(remote_content).hexdigest()
            
            # Check if we need to update
            if not force_update and out_path.exists():
                local_checksum = get_file_checksum(out_path)
                if local_checksum == remote_checksum:
                    print(f"✅ {filename} is up to date (checksum: {local_checksum[:8]}...)")
                    skipped_count += 1
                    continue
                else:
                    print(f"🔄 {filename} has updates available")
                    print(f"   Local:  {local_checksum[:8]}...")
                    print(f"   Remote: {remote_checksum[:8]}...")
            
            # Write the new content
            print(f"⬇️  Downloading {filename}...")
            with open(out_path, 'wb') as f:
                f.write(remote_content)
            
            # Verify the write was successful
            if get_file_checksum(out_path) == remote_checksum:
                print(f"✅ Successfully updated {filename}")
                updated_count += 1
            else:
                raise Exception(f"Checksum verification failed for {filename}")
                
        except Exception as e:
            print(f"❌ Failed to download {filename}: {e}")
            if not out_path.exists():
                print(f"💥 Cannot continue without {filename}")
                raise
            else:
                print(f"⚠️  Using existing version of {filename}")
    
    # Summary
    if updated_count > 0:
        print(f"🎉 Updated {updated_count} QC script(s)")
    if skipped_count > 0:
        print(f"⏭️  Skipped {skipped_count} up-to-date script(s)")
    
    # Make scripts executable
    for filename in QC_SCRIPTS.keys():
        script_path = dest / filename
        if script_path.exists():
            script_path.chmod(0o755)


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
    
    # Set default input directory if none provided
    if args.in_dir is None:
        args.in_dir = f"downloaded_{args.src_lang}"
    
    in_dir = Path(args.in_dir).resolve()
    qc_dir = Path(args.qc_dir).resolve()

    print("🔄 Checking for QC script updates...")
    download_qc_scripts(qc_dir, force_update=args.force_update)

    print("🧹 Filtering XML files...")
    filter_invalid_files(in_dir, args.src_lang, args.tgt_lang)

    print("⚙️  Running QC scripts on in-place XML data...")
    run_qc_scripts(in_dir, qc_dir)

    print("✅ Done!")


if __name__ == "__main__":
    main()
