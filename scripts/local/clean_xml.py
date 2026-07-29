#!/usr/bin/env python3
"""Apply pinned FormosanBank QC without ever overwriting standard tiers."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from pipeline_common import atomic_write_json, load_pipeline_config, sha256_file, stable_json_hash, utc_now
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_CONFIG = load_pipeline_config()
FORMOSANBANK_CONFIG = PIPELINE_CONFIG["formosanbank"]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
QC_ORG = str(FORMOSANBANK_CONFIG["org"])
QC_REPO = str(FORMOSANBANK_CONFIG["repo"])
DEFAULT_QC_REVISION = str(FORMOSANBANK_CONFIG["qc_revision"])
SYNC_PREFIXES = ("QC/", "Orthographies/", "dialects.csv")
DEFAULT_QC_CACHE = PROJECT_ROOT / "scripts" / ".formosan_qc_repo"
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
PROVENANCE_ATTR = "_mt_toolkit_transform_id"
TRANSFORM_INVENTORY = "_qc_transform_inventory.jsonl"

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
    SESSION.headers.update({"Authorization": f"Bearer {GITHUB_TOKEN}"})
SESSION.mount(
    "https://",
    HTTPAdapter(
        max_retries=Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "HEAD"]),
            respect_retry_after_header=True,
        )
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean XML with a commit-pinned FormosanBank QC checkout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--src-lang", required=True, help="Source ISO 639-3 code")
    parser.add_argument("--tgt-lang", help="Optional target-language eligibility check")
    parser.add_argument("--in-dir", help="Directory containing downloaded XML")
    parser.add_argument(
        "--formosanbank-path",
        type=Path,
        default=None,
        help="Explicit FormosanBank checkout. Its HEAD must equal --qc-revision.",
    )
    parser.add_argument(
        "--qc-revision",
        default=DEFAULT_QC_REVISION,
        help="Immutable FormosanBank commit containing the QC implementation.",
    )
    parser.add_argument(
        "--qc-dir",
        type=Path,
        default=DEFAULT_QC_CACHE,
        help="Root for commit-addressed FormosanBank QC caches.",
    )
    parser.add_argument("--force-update", action="store_true", help="Re-verify and refresh the pinned QC cache.")
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip FormosanBank hard validators. Intended only for local diagnosis.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Deprecated compatibility flag; validation is now enabled by default.",
    )
    return parser.parse_args()


def equivalent_lang_codes(lang_code: str | None) -> set[str]:
    if not lang_code:
        return set()
    key = lang_code.strip().lower()
    return LANGUAGE_EQUIVALENTS.get(key, {key})


def has_qc_package_root(path: Path) -> bool:
    return (
        (path / "QC" / "cleaning" / "clean_xml.py").is_file()
        and (path / "QC" / "validation" / "validate_xml.py").is_file()
        and (path / "QC" / "validation" / "validate_text.py").is_file()
        and (path / "Orthographies").is_dir()
    )


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def qc_checkout_is_clean(path: Path) -> bool:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain", "--", "QC", "Orthographies", "dialects.csv"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not output.strip()


def raw_url(revision: str, path: str) -> str:
    encoded = "/".join(quote(part) for part in path.split("/"))
    return f"{RAW_BASE}/{QC_ORG}/{QC_REPO}/{revision}/{encoded}"


def github_tree(revision: str) -> list[dict]:
    response = SESSION.get(
        f"{GITHUB_API}/repos/{QC_ORG}/{QC_REPO}/git/trees/{revision}",
        params={"recursive": "1"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("truncated"):
        raise RuntimeError(f"GitHub truncated the FormosanBank tree for {revision}; refusing an incomplete QC snapshot")
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError(f"GitHub returned no tree for FormosanBank revision {revision}")
    return tree


def download_raw(revision: str, path: str) -> bytes:
    response = SESSION.get(raw_url(revision, path), timeout=60)
    response.raise_for_status()
    return response.content


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def sync_qc_tree(cache_root: Path, revision: str, force_update: bool) -> Path:
    if not revision or not all(ch in "0123456789abcdef" for ch in revision.lower()) or len(revision) != 40:
        raise SystemExit("--qc-revision must be a full 40-character commit SHA")

    destination = cache_root.expanduser().resolve() / revision
    destination.mkdir(parents=True, exist_ok=True)
    try:
        tree = github_tree(revision)
    except requests.RequestException as exc:
        raise SystemExit(f"Unable to fetch pinned FormosanBank QC tree {revision}: {exc}") from exc

    wanted = [
        item
        for item in tree
        if item.get("type") == "blob"
        and (
            any(str(item.get("path", "")).startswith(prefix) for prefix in SYNC_PREFIXES[:2])
            or item.get("path") == "dialects.csv"
        )
    ]
    if not wanted:
        raise SystemExit(f"No QC files found at FormosanBank revision {revision}")

    updated = 0
    verified = 0
    inventory: list[dict[str, object]] = []
    for item in tqdm(wanted, desc="Sync pinned QC", unit="file"):
        relative = str(item["path"])
        expected_blob = str(item["sha"])
        output = destination / relative
        content: bytes | None = None
        if output.is_file() and not force_update:
            existing = output.read_bytes()
            if git_blob_sha(existing) == expected_blob:
                content = existing
                verified += 1
        if content is None:
            try:
                content = download_raw(revision, relative)
            except requests.RequestException as exc:
                raise SystemExit(f"Failed to download pinned QC file {relative}: {exc}") from exc
            if git_blob_sha(content) != expected_blob:
                raise SystemExit(f"Git blob checksum mismatch for pinned QC file {relative}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(content)
            updated += 1
        inventory.append(
            {
                "path": relative,
                "git_blob_sha": expected_blob,
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        )

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "repository": f"{QC_ORG}/{QC_REPO}",
        "revision": revision,
        "files": inventory,
        "inventory_sha256": stable_json_hash(inventory),
    }
    atomic_write_json(destination / "QC_SNAPSHOT.json", manifest)
    if not has_qc_package_root(destination):
        raise SystemExit(f"Pinned QC snapshot is incomplete: {destination}")
    print(f"Pinned QC snapshot: {destination} ({updated} updated, {verified} verified)")
    return destination


def resolve_qc_root(args: argparse.Namespace) -> Path:
    revision = args.qc_revision.lower()
    if args.formosanbank_path:
        candidate = args.formosanbank_path.expanduser().resolve()
        if not has_qc_package_root(candidate):
            raise SystemExit(f"Not a usable FormosanBank checkout: {candidate}")
        head = git_head(candidate)
        if head != revision:
            raise SystemExit(f"FormosanBank checkout HEAD is {head or 'unknown'}, expected pinned revision {revision}")
        if not qc_checkout_is_clean(candidate):
            raise SystemExit(f"FormosanBank QC checkout has local QC changes: {candidate}")
        print(f"Using pinned FormosanBank checkout: {candidate}@{revision}")
        return candidate

    sibling = (PROJECT_ROOT.parent / "FormosanBank").resolve()
    if has_qc_package_root(sibling) and git_head(sibling) == revision and qc_checkout_is_clean(sibling):
        print(f"Using pinned FormosanBank checkout: {sibling}@{revision}")
        return sibling

    if has_qc_package_root(sibling):
        print(
            f"Ignoring sibling FormosanBank checkout at {git_head(sibling) or 'unknown'}; the build requires {revision}"
        )
    return sync_qc_tree(args.qc_dir, revision, force_update=args.force_update)


def xml_lang(elem: ET.Element) -> str:
    return (elem.attrib.get(XML_LANG) or elem.attrib.get("xml:lang") or "").strip().lower()


def validate_input_files(directory: Path, src_lang: str, tgt_lang: str | None) -> dict[str, int]:
    files = sorted(directory.rglob("*.xml"))
    if not files:
        raise SystemExit(f"No XML files found under {directory}")
    expected_source = src_lang.strip().lower()
    target_codes = equivalent_lang_codes(tgt_lang)
    counts: Counter[str] = Counter()
    errors: list[str] = []
    for path in files:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append(f"{path}: malformed XML ({exc})")
            counts["parse_errors"] += 1
            continue
        counts["xml_files"] += 1
        if xml_lang(root) != expected_source:
            errors.append(f"{path}: root xml:lang={xml_lang(root)!r}, expected {expected_source!r}")
            counts["source_language_mismatch"] += 1
        if target_codes:
            if any(xml_lang(transl) in target_codes for transl in root.iter("TRANSL")):
                counts["target_eligible_files"] += 1
            else:
                counts["target_ineligible_files"] += 1
    if errors:
        preview = "\n".join(f"  - {message}" for message in errors[:25])
        suffix = f"\n  ... and {len(errors) - 25} more" if len(errors) > 25 else ""
        raise SystemExit(f"Input XML validation failed:\n{preview}{suffix}")
    return dict(counts)


def validate_fetch_contract(directory: Path) -> dict[str, object]:
    manifest_path = directory / "_fetch_manifest.json"
    inventory_path = directory / "_fetch_inventory.jsonl"
    if not manifest_path.is_file() or not inventory_path.is_file():
        raise SystemExit(
            f"Missing fetch manifest/inventory under {directory}"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"Malformed fetch manifest {manifest_path}: {exc}"
        ) from exc
    if manifest.get("complete") is not True:
        raise SystemExit(f"Fetch manifest is incomplete: {manifest_path}")
    inventory_hash = sha256_file(inventory_path)
    if inventory_hash != manifest.get("inventory_sha256"):
        raise SystemExit(
            f"Fetch inventory hash mismatch under {directory}"
        )
    kept: set[str] = set()
    with inventory_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"Malformed fetch inventory "
                    f"{inventory_path}:{line_number}: {exc}"
                ) from exc
            if row.get("status") == "kept":
                kept.add(str(row.get("destination") or ""))
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*.xml")
    }
    if actual != kept:
        raise SystemExit(
            "Fetched XML set does not match its immutable inventory: "
            f"missing={sorted(kept - actual)[:10]}, "
            f"unexpected={sorted(actual - kept)[:10]}"
        )
    return {
        "manifest_sha256": sha256_file(manifest_path),
        "inventory_sha256": inventory_hash,
        "xml_files": len(actual),
    }


def mixed_content_signature(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


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
            if len(originals) > 1 or len(standards) > 1:
                raise SystemExit(
                    f"Duplicate original/standard tier at "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            records[token] = {
                "transform_id": token,
                "xml_path": relative,
                "element_tag": element.tag,
                "xml_id": element.get("id", ""),
                "source_element_index": unit_index,
                "standard_origin": (
                    "provided"
                    if standards
                    else "derived_from_original"
                ),
                "original_before_qc_sha256": form_sha256(
                    originals[0] if originals else None
                ),
                "standard_before_qc_sha256": form_sha256(
                    standards[0] if standards else None
                ),
            }
            element.set(PROVENANCE_ATTR, token)
            unit_index += 1
        tree.write(xml_file, encoding="utf-8", xml_declaration=True)
    return records


def finalize_transform_inventory(
    corpus_dir: Path,
    records: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
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
            if len(standards) != 1:
                raise SystemExit(
                    f"Expected one standard tier after QC at "
                    f"{relative}:{element.tag}:{element.get('id', '')}"
                )
            record = records[token]
            record.update(
                {
                    "final_element_index": unit_index,
                    "standard_after_qc_sha256": form_sha256(standards[0]),
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
                    "standard_after_qc_sha256": None,
                    "disposition": "removed_by_cleaner",
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
    retained = sum(row["disposition"] == "retained" for row in rows)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "records": len(rows),
        "retained": retained,
        "removed_by_cleaner": len(rows) - retained,
    }


def copy_form(source: ET.Element, kind_of: str) -> ET.Element:
    copied = copy.deepcopy(source)
    copied.set("kindOf", kind_of)
    return copied


def ensure_standard_tiers(corpus_dir: Path) -> dict[str, int | str]:
    """Complete missing tiers while preserving every existing standard tier."""
    stats: Counter[str] = Counter()
    existing_before: dict[tuple[str, str, str], str] = {}
    existing_after: dict[tuple[str, str, str], str] = {}

    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        changed = False
        relative = str(xml_file.relative_to(corpus_dir))
        for parent in root.iter():
            if parent.tag not in {"S", "W", "M"}:
                continue
            forms = parent.findall("FORM")
            if not forms:
                continue
            originals = [form for form in forms if (form.get("kindOf") or "").strip().lower() == "original"]
            standards = [form for form in forms if (form.get("kindOf") or "").strip().lower() == "standard"]
            untyped = [form for form in forms if not (form.get("kindOf") or "").strip()]
            element_key = (relative, parent.tag, parent.get("id", ""))

            if len(originals) > 1 or len(standards) > 1:
                raise SystemExit(f"Duplicate original/standard FORM tier at {element_key}")
            if standards:
                existing_before[element_key] = mixed_content_signature(standards[0])
                stats["existing_standard"] += 1

            if not originals:
                if standards:
                    original = copy_form(standards[0], "original")
                    parent.insert(list(parent).index(standards[0]), original)
                    originals = [original]
                    stats["original_copied_from_standard"] += 1
                    changed = True
                elif len(untyped) == 1:
                    untyped[0].set("kindOf", "original")
                    originals = [untyped[0]]
                    stats["untyped_promoted_to_original"] += 1
                    changed = True
                else:
                    raise SystemExit(f"Cannot identify original FORM tier at {element_key}")

            if not standards:
                standard = copy_form(originals[0], "standard")
                insert_at = list(parent).index(originals[0]) + 1
                parent.insert(insert_at, standard)
                standards = [standard]
                stats["standard_copied_from_original"] += 1
                changed = True

            if element_key in existing_before:
                existing_after[element_key] = mixed_content_signature(standards[0])

        if changed:
            tree.write(xml_file, encoding="utf-8", xml_declaration=True)
            stats["files_changed"] += 1

    if existing_before != existing_after:
        changed_keys = sorted(key for key, value in existing_before.items() if existing_after.get(key) != value)
        raise SystemExit(
            "Tier completion modified existing standard tiers: " + ", ".join("/".join(key) for key in changed_keys[:10])
        )
    stats["preserved_standard_digest"] = stable_json_hash(
        [(*key, value) for key, value in sorted(existing_before.items())]
    )
    return dict(stats)


def audit_standard_tiers(corpus_dir: Path) -> dict[str, int | str]:
    stats: Counter[str] = Counter()
    signatures: list[tuple[str, str, str, str]] = []
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        for parent in root.iter():
            if parent.tag not in {"S", "W", "M"}:
                continue
            standards = [
                form for form in parent.findall("FORM") if (form.get("kindOf") or "").strip().lower() == "standard"
            ]
            if len(standards) != 1:
                raise SystemExit(
                    f"Expected exactly one standard FORM at {relative}:{parent.tag}:{parent.get('id', '')}"
                )
            text = "".join(standards[0].itertext()).strip()
            if not text:
                if parent.tag == "S":
                    raise SystemExit(
                        f"Empty sentence standard FORM at "
                        f"{relative}:{parent.tag}:{parent.get('id', '')}"
                    )
                stats[f"empty_{parent.tag.lower()}_standard_tiers"] += 1
            stats[f"{parent.tag.lower()}_standard_tiers"] += 1
            signatures.append((relative, parent.tag, parent.get("id", ""), mixed_content_signature(standards[0])))
    stats["standard_tier_digest"] = stable_json_hash(signatures)
    return dict(stats)


def qc_env(qc_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(qc_root) if not existing else f"{qc_root}{os.pathsep}{existing}"
    return env


def run_command(cmd: list[str], qc_root: Path) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, cwd=qc_root, env=qc_env(qc_root), check=True)


def run_qc_scripts(corpus_dir: Path, qc_root: Path, *, validate: bool) -> dict[str, object]:
    transform_sources = tag_transform_sources(corpus_dir)
    tier_completion = ensure_standard_tiers(corpus_dir)
    clean_script = qc_root / "QC" / "cleaning" / "clean_xml.py"
    try:
        run_command(
            [
                sys.executable,
                str(clean_script),
                "--corpora_path",
                str(corpus_dir),
            ],
            qc_root,
        )
    except BaseException:
        remove_transform_tags(corpus_dir)
        raise
    transform_inventory = write_transform_inventory(
        corpus_dir,
        finalize_transform_inventory(corpus_dir, transform_sources),
    )
    standard_audit = audit_standard_tiers(corpus_dir)

    validators: list[str] = []
    if validate:
        for script_name in ("validate_xml.py", "validate_text.py"):
            validator = qc_root / "QC" / "validation" / script_name
            run_command(
                [
                    sys.executable,
                    str(validator),
                    "by_path",
                    "--path",
                    str(corpus_dir),
                ],
                qc_root,
            )
            validators.append(script_name)
    return {
        "tier_completion": tier_completion,
        "transform_inventory": transform_inventory,
        "standard_audit": standard_audit,
        "validators": validators,
    }


def main() -> None:
    args = parse_args()
    corpus_dir = Path(args.in_dir or f"downloaded_{args.src_lang}").expanduser().resolve()
    if not corpus_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {corpus_dir}")

    fetch_snapshot = validate_fetch_contract(corpus_dir)
    qc_root = resolve_qc_root(args)
    input_counts = validate_input_files(corpus_dir, args.src_lang, args.tgt_lang)
    qc_result = run_qc_scripts(corpus_dir, qc_root, validate=not args.skip_validation)
    manifest = {
        "schema_version": 2,
        "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
        "created_at": utc_now(),
        "source_language": args.src_lang,
        "target_language": args.tgt_lang,
        "corpus_dir": str(corpus_dir),
        "formosanbank_qc": {
            "repository": f"{QC_ORG}/{QC_REPO}",
            "revision": args.qc_revision.lower(),
            "snapshot_path": str(qc_root),
            "snapshot_manifest_sha256": (
                sha256_file(qc_root / "QC_SNAPSHOT.json") if (qc_root / "QC_SNAPSHOT.json").is_file() else None
            ),
        },
        "input": input_counts,
        "fetch_snapshot": fetch_snapshot,
        **qc_result,
        "complete": not args.skip_validation,
    }
    manifest_path = corpus_dir / "_qc_manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(f"QC manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    main()
