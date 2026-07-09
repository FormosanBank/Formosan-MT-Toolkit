#!/usr/bin/env python3
"""Clean downloaded FormosanBank XML with the real FormosanBank QC package.

The old version of this script downloaded only two standalone QC scripts. That
breaks with current FormosanBank because utilities such as standardize.py import
shared modules from the QC package. This version prefers a sibling
../FormosanBank checkout and otherwise syncs the minimal package tree needed to
run QC scripts as package-aware code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
QC_ORG = "FormosanBank"
QC_REPO = "FormosanBank"
QC_BRANCH = "main"
SYNC_PREFIXES = ("QC/", "Orthographies/")
DEFAULT_QC_CACHE = PROJECT_ROOT / "scripts" / ".formosan_qc_repo"

LANGUAGE_EQUIVALENTS: dict[str, set[str]] = {
    "zh": {"zh", "zho", "chi", "cmn"},
    "zho": {"zh", "zho", "chi", "cmn"},
    "chi": {"zh", "zho", "chi", "cmn"},
    "cmn": {"zh", "zho", "chi", "cmn"},
    "en": {"en", "eng"},
    "eng": {"en", "eng"},
}

SESSION = requests.Session()
if GITHUB_TOKEN:
    SESSION.headers.update({"Authorization": f"token {GITHUB_TOKEN}"})
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
        )
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean XML files using the FormosanBank QC package.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-lang", required=True, help="Source language, e.g. ami")
    parser.add_argument("--tgt-lang", help="Optional target language filter, e.g. zho or eng")
    parser.add_argument("--in-dir", help="Directory of raw XML files")
    parser.add_argument(
        "--formosanbank-path",
        type=Path,
        default=None,
        help="Path to a full FormosanBank checkout; defaults to sibling ../FormosanBank when present.",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=DEFAULT_QC_CACHE,
        help="Fallback cache for synced QC/ and Orthographies/ trees.",
    )
    parser.add_argument("--force-update", action="store_true", help="Force refresh of fallback synced QC files.")
    parser.add_argument("--skip-update", action="store_true", help="Do not sync fallback QC files.")
    parser.add_argument("--skip-standardize", action="store_true", help="Run clean_xml.py only.")
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_xml.py and validate_text.py after cleaning as informational checks.",
    )
    return parser.parse_args()


def equivalent_lang_codes(lang_code: str | None) -> set[str]:
    if not lang_code:
        return set()
    key = lang_code.strip().lower()
    return LANGUAGE_EQUIVALENTS.get(key, {key})


def checksum(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def has_qc_package_root(path: Path) -> bool:
    return (
        (path / "QC" / "cleaning" / "clean_xml.py").is_file()
        and (path / "QC" / "utilities" / "standardize.py").is_file()
    )


def raw_url(path: str) -> str:
    encoded = "/".join(quote(part) for part in path.split("/"))
    return f"{RAW_BASE}/{QC_ORG}/{QC_REPO}/{QC_BRANCH}/{encoded}"


def github_tree() -> list[dict]:
    url = f"{GITHUB_API}/repos/{QC_ORG}/{QC_REPO}/git/trees/{QC_BRANCH}"
    resp = SESSION.get(url, params={"recursive": "1"}, timeout=30)
    if resp.status_code == 403:
        raise SystemExit(
            "GitHub returned 403 while syncing FormosanBank QC files. "
            "Set a GITHUB_TOKEN with access or pass --formosanbank-path ../FormosanBank."
        )
    resp.raise_for_status()
    return resp.json().get("tree", [])


def download_raw(path: str) -> bytes:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            resp = SESSION.get(raw_url(path), timeout=30)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {path}: {last_error}")


def sync_qc_tree(dest: Path, force_update: bool) -> Path:
    if not GITHUB_TOKEN:
        raise SystemExit(
            "No sibling FormosanBank checkout was found and GITHUB_TOKEN is not set, "
            "so the QC package cannot be synced."
        )

    dest.mkdir(parents=True, exist_ok=True)
    tree = github_tree()
    wanted = [
        item["path"]
        for item in tree
        if item.get("type") == "blob" and item.get("path", "").startswith(SYNC_PREFIXES)
    ]
    if not wanted:
        raise SystemExit("Could not find QC/ or Orthographies/ files in the FormosanBank tree.")

    updated = 0
    skipped = 0
    for rel_path in tqdm(wanted, desc="Sync QC package", unit="file"):
        out_path = dest / rel_path
        if out_path.exists() and not force_update:
            skipped += 1
            continue
        content = download_raw(rel_path)
        if out_path.exists() and checksum(out_path) == hashlib.sha256(content).hexdigest():
            skipped += 1
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(content)
        updated += 1

    print(f"QC package cache: {dest} ({updated} updated, {skipped} skipped)")
    return dest


def resolve_qc_root(args: argparse.Namespace) -> Path:
    candidates: list[Path] = []
    if args.formosanbank_path:
        candidates.append(args.formosanbank_path.expanduser().resolve())
    candidates.append((PROJECT_ROOT.parent / "FormosanBank").resolve())
    candidates.append(args.qc_dir.expanduser().resolve())

    for candidate in candidates:
        if has_qc_package_root(candidate):
            if candidate == args.qc_dir.expanduser().resolve() and args.force_update and not args.skip_update:
                return sync_qc_tree(candidate, force_update=True)
            print(f"Using FormosanBank QC package: {candidate}")
            return candidate

    if args.skip_update:
        raise SystemExit(
            "No usable QC package root found. Pass --formosanbank-path or rerun without --skip-update."
        )
    return sync_qc_tree(args.qc_dir.expanduser().resolve(), force_update=args.force_update)


def xml_lang(elem: ET.Element) -> str:
    return (
        elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang")
        or elem.attrib.get("xml:lang")
        or ""
    ).strip().lower()


def valid_file(path: Path, src_lang: str, tgt_lang: str | None) -> bool:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        print(f"[skip] Unparseable XML: {path}")
        return False

    if xml_lang(root) != src_lang.strip().lower():
        return False
    if not tgt_lang:
        return True

    target_codes = equivalent_lang_codes(tgt_lang)
    return any(xml_lang(transl) in target_codes for transl in root.iter("TRANSL"))


def filter_invalid_files(in_dir: Path, src_lang: str, tgt_lang: str | None) -> None:
    xml_files = list(in_dir.rglob("*.xml"))
    print(f"Filtering {len(xml_files)} XML files in {in_dir}")
    removed = 0
    for xml_file in tqdm(xml_files, desc="Filter XML", unit="file"):
        if not valid_file(xml_file, src_lang, tgt_lang):
            xml_file.unlink()
            removed += 1
    print(f"Removed {removed} non-matching or invalid XML files")


def qc_env(qc_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(qc_root) if not existing else f"{qc_root}{os.pathsep}{existing}"
    return env


def run_command(cmd: list[str], qc_root: Path, check: bool = True) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, cwd=qc_root, env=qc_env(qc_root), check=check)


def normalize_form_tiers_for_standardize(corpus_dir: Path) -> dict[str, int]:
    """Make mixed legacy XML compatible with FormosanBank standardize --copy.

    Some downloaded XML has bare <FORM> tiers with no kindOf at all, or a
    standard tier but no original tier. FormosanBank's standardizer correctly
    expects original tiers because it creates standard from original. For the MT
    build, preserve the existing text by treating the available source FORM as
    original before standardization.
    """
    stats = {
        "files_changed": 0,
        "untyped_promoted": 0,
        "standard_copied": 0,
        "other_copied": 0,
        "parse_errors": 0,
    }

    for xml_file in corpus_dir.rglob("*.xml"):
        try:
            tree = ET.parse(xml_file)
        except ET.ParseError:
            stats["parse_errors"] += 1
            continue

        root = tree.getroot()
        changed = False
        for elem in root.iter():
            forms = elem.findall("FORM")
            if not forms or any((form.get("kindOf") or "").strip().lower() == "original" for form in forms):
                continue

            standard = next(
                (form for form in forms if (form.get("kindOf") or "").strip().lower() == "standard"),
                None,
            )
            untyped = [form for form in forms if not (form.get("kindOf") or "").strip()]

            if len(forms) == 1 and untyped:
                untyped[0].set("kindOf", "original")
                stats["untyped_promoted"] += 1
                changed = True
                continue

            source_form = standard or (untyped[0] if untyped else forms[0])
            original = copy.deepcopy(source_form)
            original.set("kindOf", "original")
            children = list(elem)
            try:
                insert_at = children.index(source_form)
            except ValueError:
                insert_at = 0
            elem.insert(insert_at, original)
            if standard is not None:
                stats["standard_copied"] += 1
            else:
                stats["other_copied"] += 1
            changed = True

        if changed:
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            stats["files_changed"] += 1

    if any(stats[key] for key in ("untyped_promoted", "standard_copied", "other_copied")):
        print(
            "Normalized FORM tiers before standardization: "
            f"{stats['untyped_promoted']} bare FORM -> original, "
            f"{stats['standard_copied']} standard-only copied to original, "
            f"{stats['other_copied']} other FORM copied to original "
            f"across {stats['files_changed']} files"
        )
    if stats["parse_errors"]:
        print(f"Skipped {stats['parse_errors']} unparseable XML files during FORM tier normalization")
    return stats


def run_qc_scripts(corpus_dir: Path, qc_root: Path, *, standardize: bool, validate: bool) -> None:
    clean_script = qc_root / "QC" / "cleaning" / "clean_xml.py"
    std_script = qc_root / "QC" / "utilities" / "standardize.py"

    run_command([sys.executable, str(clean_script), "--corpora_path", str(corpus_dir)], qc_root)

    if standardize:
        normalize_form_tiers_for_standardize(corpus_dir)
        run_command(
            [sys.executable, str(std_script), "--corpora_path", str(corpus_dir), "--copy"],
            qc_root,
        )

    if validate:
        validate_xml = qc_root / "QC" / "validation" / "validate_xml.py"
        validate_text = qc_root / "QC" / "validation" / "validate_text.py"
        if validate_xml.exists():
            run_command(
                [
                    sys.executable,
                    str(validate_xml),
                    "by_path",
                    "--path",
                    str(corpus_dir),
                    "--no-exit-on-hard",
                ],
                qc_root,
                check=False,
            )
        if validate_text.exists():
            run_command(
                [
                    sys.executable,
                    str(validate_text),
                    "by_path",
                    "--path",
                    str(corpus_dir),
                    "--no-exit-on-hard",
                ],
                qc_root,
                check=False,
            )


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir or f"downloaded_{args.src_lang}").resolve()
    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    qc_root = resolve_qc_root(args)
    filter_invalid_files(in_dir, args.src_lang, args.tgt_lang)
    run_qc_scripts(
        in_dir,
        qc_root,
        standardize=not args.skip_standardize,
        validate=args.validate,
    )
    print("Done.")


if __name__ == "__main__":
    main()
