"""Resolve and verify the commit-pinned FormosanBank QC implementation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import subprocess
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from pipeline_common import (
    atomic_write_json,
    load_pipeline_config,
    stable_json_hash,
    utc_now,
)
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORMOSANBANK_CONFIG = load_pipeline_config()["formosanbank"]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
QC_ORG = str(FORMOSANBANK_CONFIG["org"])
QC_REPO = str(FORMOSANBANK_CONFIG["repo"])
DEFAULT_QC_REVISION = str(FORMOSANBANK_CONFIG["qc_revision"])
DEFAULT_QC_CACHE = PROJECT_ROOT / "scripts" / ".formosan_qc_repo"
SYNC_PREFIXES = ("QC/", "Orthographies/")
SYNC_FILES = {"dialects.csv", "languages.csv", "standards.csv"}

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
            [
                "git",
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--",
                "QC",
                "Orthographies",
                "dialects.csv",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return not output.strip()


def git_has_revision(path: Path, revision: str) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(path), "cat-file", "-e", f"{revision}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def raw_url(revision: str, path: str) -> str:
    encoded = "/".join(quote(part) for part in path.split("/"))
    return f"{RAW_BASE}/{QC_ORG}/{QC_REPO}/{revision}/{encoded}"


def public_qc_get(url: str, **kwargs) -> requests.Response:
    """Retry public FormosanBank QC without a stale optional token."""
    response = SESSION.get(url, **kwargs)
    if response.status_code == 401 and GITHUB_TOKEN:
        response = requests.get(url, **kwargs)
    return response


def github_tree(revision: str) -> list[dict]:
    response = public_qc_get(
        f"{GITHUB_API}/repos/{QC_ORG}/{QC_REPO}/git/trees/{revision}",
        params={"recursive": "1"},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("truncated"):
        raise RuntimeError(
            f"GitHub truncated the FormosanBank tree for {revision}; "
            "refusing an incomplete QC snapshot"
        )
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError(
            f"GitHub returned no tree for FormosanBank revision {revision}"
        )
    return tree


def download_raw(revision: str, path: str) -> bytes:
    response = public_qc_get(raw_url(revision, path), timeout=60)
    response.raise_for_status()
    return response.content


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _sync_qc_tree_unlocked(
    cache_root: Path,
    revision: str,
    force_update: bool,
) -> Path:
    if (
        not revision
        or not all(character in "0123456789abcdef" for character in revision.lower())
        or len(revision) != 40
    ):
        raise SystemExit("--qc-revision must be a full 40-character commit SHA")

    destination = cache_root.expanduser().resolve() / revision
    destination.mkdir(parents=True, exist_ok=True)
    try:
        tree = github_tree(revision)
    except requests.RequestException as exc:
        raise SystemExit(
            f"Unable to fetch pinned FormosanBank QC tree {revision}: {exc}"
        ) from exc

    wanted = [
        item
        for item in tree
        if item.get("type") == "blob"
        and (
            any(
                str(item.get("path", "")).startswith(prefix)
                for prefix in SYNC_PREFIXES
            )
            or item.get("path") in SYNC_FILES
        )
    ]
    if not wanted:
        raise SystemExit(
            f"No QC files found at FormosanBank revision {revision}"
        )

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
                raise SystemExit(
                    f"Failed to download pinned QC file {relative}: {exc}"
                ) from exc
            if git_blob_sha(content) != expected_blob:
                raise SystemExit(
                    f"Git blob checksum mismatch for pinned QC file {relative}"
                )
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
    print(
        f"Pinned QC snapshot: {destination} "
        f"({updated} updated, {verified} verified)"
    )
    return destination


def sync_qc_tree(cache_root: Path, revision: str, force_update: bool) -> Path:
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / f".{revision}.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _sync_qc_tree_unlocked(cache_root, revision, force_update)


def _sync_qc_tree_from_git_unlocked(
    repository: Path,
    cache_root: Path,
    revision: str,
) -> Path:
    """Materialize the pinned QC subset from an existing Git object store."""
    destination = cache_root.expanduser().resolve() / revision
    destination.mkdir(parents=True, exist_ok=True)
    listing = subprocess.check_output(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            revision,
            "--",
            "QC",
            "Orthographies",
            "dialects.csv",
            "languages.csv",
            "standards.csv",
        ],
        text=True,
    )
    inventory: list[dict[str, object]] = []
    for line in listing.splitlines():
        metadata, relative = line.split("\t", 1)
        _, object_type, expected_blob = metadata.split()
        if object_type != "blob":
            continue
        content = subprocess.check_output(
            ["git", "-C", str(repository), "show", f"{revision}:{relative}"]
        )
        if git_blob_sha(content) != expected_blob:
            raise SystemExit(
                f"Git blob checksum mismatch for local QC file {relative}"
            )
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(content)
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
        "source": f"local_git:{repository}",
        "files": inventory,
        "inventory_sha256": stable_json_hash(inventory),
    }
    atomic_write_json(destination / "QC_SNAPSHOT.json", manifest)
    if not has_qc_package_root(destination):
        raise SystemExit(f"Pinned local QC snapshot is incomplete: {destination}")
    print(
        f"Pinned QC snapshot: {destination} ({len(inventory)} local Git objects)"
    )
    return destination


def sync_qc_tree_from_git(
    repository: Path,
    cache_root: Path,
    revision: str,
) -> Path:
    cache_root = cache_root.expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    with (cache_root / f".{revision}.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _sync_qc_tree_from_git_unlocked(repository, cache_root, revision)


def resolve_qc_root(args: argparse.Namespace) -> Path:
    revision = args.qc_revision.lower()
    if args.formosanbank_path:
        candidate = args.formosanbank_path.expanduser().resolve()
        if not has_qc_package_root(candidate):
            raise SystemExit(f"Not a usable FormosanBank checkout: {candidate}")
        head = git_head(candidate)
        if head != revision:
            raise SystemExit(
                f"FormosanBank checkout HEAD is {head or 'unknown'}, "
                f"expected pinned revision {revision}"
            )
        if not qc_checkout_is_clean(candidate):
            raise SystemExit(
                f"FormosanBank QC checkout has local QC changes: {candidate}"
            )
        print(f"Using pinned FormosanBank checkout: {candidate}@{revision}")
        return candidate

    sibling = (PROJECT_ROOT.parent / "FormosanBank").resolve()
    if (
        has_qc_package_root(sibling)
        and git_head(sibling) == revision
        and qc_checkout_is_clean(sibling)
    ):
        print(f"Using pinned FormosanBank checkout: {sibling}@{revision}")
        return sibling

    if has_qc_package_root(sibling):
        if git_has_revision(sibling, revision):
            return sync_qc_tree_from_git(sibling, args.qc_dir, revision)
        print(
            f"Sibling FormosanBank does not contain pinned revision {revision}; "
            "using GitHub"
        )
    return sync_qc_tree(
        args.qc_dir,
        revision,
        force_update=args.force_update,
    )
