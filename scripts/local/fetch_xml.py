#!/usr/bin/env python3
"""
STEP 1: 

Harvest XML files from FormosanBank repositories.

Two modes:
----------
1. Default mode: Searches through all repos in the FormosanBank org for files in Final_XML/ folders
2. Public release mode (--public): Only looks inside XML/ directories anywhere under
   FormosanBank/FormosanBank/Corpora/ (files may live directly in XML/ or in deeper
   subdirectories such as XML/Yami/...)

The script filters XML files whose <TEXT xml:lang="…"> == src_lang 
(and, optionally, whose <TRANSL xml:lang="…"> == tgt_lang).

NOTE: Please do not set the target lang as currently target language tags
are not standardized across the XML files. Simply set the source language and 
make_corpus.py will take care of the rest after. 

Usage examples
--------------
# Default mode (all repos, Final_XML folders)
$ python fetch_xml.py --src-lang ami
$ python fetch_xml.py --src-lang pwn

# Public release mode (anything at or below an XML/ directory under Corpora/)
$ python fetch_xml.py --src-lang ami --public
$ python fetch_xml.py --src-lang pwn --public --tgt-lang zh
"""
from __future__ import annotations

import argparse
import concurrent.futures as fut
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
import xml.etree.ElementTree as ET
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in project root (two levels up from this script)
project_root = Path(__file__).parent.parent.parent
load_dotenv(project_root / ".env")

# ─────────────────────────────  language maps  ───────────────────────────────
# Map language codes to their equivalent sets (same as make_corpus.py)
LANGUAGE_EQUIVALENTS: dict[str, set[str]] = {
    # Chinese variants
    "zh": {"zh", "zho", "chi", "cmn"},
    "zho": {"zh", "zho", "chi", "cmn"},
    "chi": {"zh", "zho", "chi", "cmn"},
    "cmn": {"zh", "zho", "chi", "cmn"},
    # English variants
    "en": {"en", "eng"},
    "eng": {"en", "eng"},
}


def get_equivalent_lang_codes(lang_code: str) -> set[str]:
    """Get all equivalent language codes for a given language code."""
    return LANGUAGE_EQUIVALENTS.get(lang_code, {lang_code})


def parse_exclude_patterns(values: Iterable[str] | None) -> list[str]:
    """Parse repeatable or comma-separated case-insensitive substring patterns."""
    patterns: list[str] = []
    for value in values or []:
        for part in value.split(","):
            pattern = part.strip().lower()
            if pattern and pattern not in patterns:
                patterns.append(pattern)
    return patterns


DEFAULT_BIBLE_REPO_EXCLUDE_PATTERNS = (
    "bible",
    "taiwan-bible-society",
    "taiwan bible society",
    "taiwan_bible_society",
)

DEFAULT_BIBLE_PATH_EXCLUDE_PATTERNS = (
    "fhl_bible",
    "fhlbible",
    "taiwan-bible-society",
    "taiwan bible society",
    "taiwan_bible_society",
)


def add_patterns(patterns: list[str], defaults: Iterable[str]) -> list[str]:
    out = list(patterns)
    for pattern in defaults:
        if pattern not in out:
            out.append(pattern)
    return out


def matches_exclude_pattern(value: str, patterns: Iterable[str]) -> bool:
    haystack = value.lower()
    return any(pattern in haystack for pattern in patterns)


# ─────────────────────────────  config  ──────────────────────────────────────
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
MAX_WORKERS = 16
REQUEST_TIMEOUT = 10  # seconds
RETRIES = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=(500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "HEAD"]),
)

# single global Session → connection pooling + retries
SESSION = requests.Session()
SESSION.headers.update(HEADERS)
SESSION.mount("https://", HTTPAdapter(max_retries=RETRIES))
# ─────────────────────────────────────────────────────────────────────────────


def get_repos(org: str) -> Iterable[str]:
    """Yield every repository name in the organisation."""
    page = 1
    while True:
        r = SESSION.get(
            f"{GITHUB_API}/orgs/{org}/repos",
            params={"per_page": 100, "page": page},
            timeout=REQUEST_TIMEOUT,
        )

        # Better diagnostics for 403s / rate limits / permission issues
        if r.status_code == 403:
            try:
                data = r.json()
                message = data.get("message", "")
            except Exception:
                message = r.text.strip()

            remaining = r.headers.get("X-RateLimit-Remaining")
            reset = r.headers.get("X-RateLimit-Reset")

            err_lines = [
                f"❌  GitHub returned 403 Forbidden for org '{org}' (page={page}).",
                f"   Message: {message!r}",
            ]
            if remaining is not None:
                err_lines.append(
                    f"   X-RateLimit-Remaining={remaining}, X-RateLimit-Reset={reset}"
                )
            err_lines.append(
                "   This usually means either:\n"
                "     • your GITHUB_TOKEN no longer has access to this org "
                "(scopes / SSO / membership), or\n"
                "     • you've hit a rate limit / abuse detection and must wait.\n"
                "   Try:\n"
                "     curl -H \"Authorization: token $GITHUB_TOKEN\" "
                "https://api.github.com/orgs/formosanbank\n"
                "   to see the full error message."
            )
            sys.exit("\n".join(err_lines))

        r.raise_for_status()
        data = r.json()
        if not data:
            break

        # remove entry with name "Formosan-Wikipedias"
        data = [repo for repo in data if repo["name"] != "Formosan-Wikipedias"]
        yield from (repo["name"] for repo in data)
        page += 1


def get_tree(org: str, repo: str, branch: str):
    """Return the full git tree (recursive) for a repo/branch."""
    r = SESSION.get(
        f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    tree = r.json().get("tree", [])
    print(f"   🌲 {repo}@{branch}: {len(tree)} tree entries")
    return tree


def raw_url(org: str, repo: str, path: str, branch: str) -> str:
    """
    Build a raw.githubusercontent.com URL for a given repo/path/branch.

    The path is split and each segment is percent-encoded so that non-ASCII
    components (e.g., Chinese chars) and spaces are encoded correctly.
    """
    encoded_path = "/".join(quote(part) for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{encoded_path}"


def is_public_release_xml_path(path: str) -> bool:
    """
    Return True for any XML file that lives somewhere under a Corpora/.../XML/
    subtree in the public release repo.

    Accepted examples:
      - Corpora/Foo/XML/file.xml
      - Corpora/Foo/XML/Yami/file.xml
      - Corpora/Foo/Bar/XML/nested/deeper/file.xml
    """
    if not path.lower().endswith(".xml"):
        return False

    parts = path.split("/")
    if len(parts) < 3 or parts[0] != "Corpora":
        return False

    # Match any file directly in XML/ or any deeper descendant of XML/.
    return "XML" in parts[:-1]


def wants_file(xml_bytes: bytes, src_lang: str, tgt_lang: str | None, dialect: str | None):
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        print(f"⚠️  XML parse error in file: {e}")
        return False

    xml_lang = root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "").strip().lower()
    xml_dialect = root.attrib.get("dialect", "").strip().lower()

    src_ok = (xml_lang == src_lang.strip().lower())
    dialect_ok = True if dialect is None else (xml_dialect == dialect.strip().lower())

    if not src_ok or not dialect_ok:
        return False

    if tgt_lang is None:
        return True

    target_codes = get_equivalent_lang_codes(tgt_lang.lower())

    return any(
        elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "").strip().lower()
        in target_codes
        for elem in root.iter("TRANSL")
    )


def download_blob(
    org: str,
    repo: str,
    item: dict,
    src_lang: str,
    tgt_lang: str | None,
    branch: str,
    out_dir: Path,
    dialect: str | None,
):
    """
    Download → filter → save one blob using raw.githubusercontent.com
    (so we don't burn through the REST API core rate limit).

    Returns destination Path or None.
    """
    url = raw_url(org, repo, item["path"], branch)

    # retry loop for transient HTTP issues
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if resp.status_code != 200:
                print(
                    f"⚠️  [{repo}] HTTP {resp.status_code} for {url} "
                    f"(attempt {attempt+1})"
                )
                resp.raise_for_status()
            xml_bytes = resp.content
            break
        except requests.exceptions.RequestException as e:
            print(
                f"⚠️  [{repo}] Error fetching {url} on attempt {attempt+1}: {e}"
            )
            if attempt == 2:
                print(f"❌  [{repo}] Giving up on {url} after 3 attempts")
                return None
            time.sleep(2 ** attempt)
    else:
        return None  # never reached

    if not wants_file(xml_bytes, src_lang, tgt_lang, dialect):
        # Uncomment for super-verbose logging:
        # print(f"   ↷ [{repo}] Skipped {item['path']} (lang filter)")
        return None

    dest = out_dir / repo / item["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(xml_bytes)
    # Uncomment for per-file success logging:
    # print(f"   ✅ [{repo}] Saved {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", required=True, help="e.g. ami, pau, tsu")
    parser.add_argument(
        "--tgt-lang", help="optional target language filter (e.g. zh, zho, en)"
    )
    parser.add_argument("--org", default="formosanbank")
    parser.add_argument("--branch", help="force a branch name for all repos")
    parser.add_argument(
        "--out-dir",
        help="where to store the files (default: downloaded_{src_lang})",
    )
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Remove the output directory before downloading so stale XML cannot survive.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help=(
            "Use public release structure: look anywhere under XML/ directories "
            "inside FormosanBank/FormosanBank/Corpora/"
        ),
    )
    parser.add_argument(
        "--exclude-bible",
        action="store_true",
        help="Exclude Bible/Taiwan Bible Society XML from fetch results.",
    )
    parser.add_argument(
        "--exclude-repo-pattern",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring for repositories to skip before scanning. "
            "Can be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--exclude-path-pattern",
        action="append",
        default=[],
        help=(
            "Case-insensitive substring for XML paths to skip after repo scanning. "
            "Can be repeated or comma-separated."
        ),
    )
    parser.add_argument(
        "--dialect",
        default=None,
        help=("optional dialect filter for if you would like to also filter by the xml:dialect tag"),
    )
    args = parser.parse_args()

    repo_exclude_patterns = parse_exclude_patterns(args.exclude_repo_pattern)
    path_exclude_patterns = parse_exclude_patterns(args.exclude_path_pattern)
    if args.exclude_bible:
        repo_exclude_patterns = add_patterns(
            repo_exclude_patterns,
            DEFAULT_BIBLE_REPO_EXCLUDE_PATTERNS,
        )
        path_exclude_patterns = add_patterns(
            path_exclude_patterns,
            DEFAULT_BIBLE_PATH_EXCLUDE_PATTERNS,
        )

    if not GITHUB_TOKEN:
        sys.exit("❌  Please set the GITHUB_TOKEN environment variable")

    print(
        f"⚙️  Config:\n"
        f"   src_lang   = {args.src_lang}\n"
        f"   dialect    = {args.dialect}\n"
        f"   tgt_lang   = {args.tgt_lang}\n"
        f"   org        = {args.org}\n"
        f"   public     = {args.public}\n"
        f"   exclude_repos = {repo_exclude_patterns or None}\n"
        f"   exclude_paths = {path_exclude_patterns or None}\n"
        f"   branch_arg = {args.branch}\n"
        f"   token_len  = {len(GITHUB_TOKEN)}"
    )

    dialect = args.dialect if args.dialect else None
    # Set default output directory if none provided
    if args.out_dir is None:
        args.out_dir = f"downloaded_{args.src_lang}"

    out_dir = Path(args.out_dir)
    if args.clean_output and out_dir.exists():
        print(f"🧹  Removing stale download directory: {out_dir.resolve()}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"📂  Output directory: {out_dir.resolve()}")

    # Show which language codes will be matched if target language is specified
    if args.tgt_lang:
        equivalent_codes = get_equivalent_lang_codes(args.tgt_lang.lower())
        print(
            f"🔍  Target language '{args.tgt_lang}' will match: "
            f"{sorted(equivalent_codes)}"
        )

    # Determine which repos to process based on --public flag
    if args.public:
        repos = ["FormosanBank"]
        print(
            "🌐  Public release mode: processing any XML file at or below "
            "FormosanBank/FormosanBank/Corpora/**/XML/"
        )
    else:
        repos = list(get_repos(args.org))
        print(f"📦  Found {len(repos)} repos in {args.org}")

    if repo_exclude_patterns:
        excluded_repos = [
            repo for repo in repos if matches_exclude_pattern(repo, repo_exclude_patterns)
        ]
        repos = [
            repo for repo in repos if not matches_exclude_pattern(repo, repo_exclude_patterns)
        ]
        if excluded_repos:
            print(
                f"🚫  Excluded {len(excluded_repos)} repo(s): "
                + ", ".join(sorted(excluded_repos))
            )

    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []

        # ── repo-level progress bar ──────────────────────────────────────────
        for repo in tqdm(repos, desc="Repos", unit="repo"):
            # Determine branch for this repo
            if args.branch is None:
                try:
                    meta = SESSION.get(
                        f"{GITHUB_API}/repos/{args.org}/{repo}",
                        timeout=REQUEST_TIMEOUT,
                    )
                    meta.raise_for_status()
                    branch = meta.json().get("default_branch", "main")
                    print(f"📦  Repo {repo}: using default branch '{branch}'")
                except requests.RequestException as e:
                    print(f"❌  Failed to get default branch for {repo}: {e}")
                    continue
            else:
                branch = args.branch
                print(f"📦  Repo {repo}: using forced branch '{branch}'")

            try:
                tree = get_tree(args.org, repo, branch)
            except requests.HTTPError as e:
                if e.response.status_code == 409:  # empty repo or huge tree
                    print(f"⚠️  Skipping repo {repo} (empty or too large for API)")
                    continue
                print(f"❌  HTTP error fetching tree for {repo}@{branch}: {e}")
                continue
            except requests.RequestException as e:
                print(f"❌  Request error fetching tree for {repo}@{branch}: {e}")
                continue

            # Filter XML files based on mode
            if args.public:
                # Public release: accept XML files directly under XML/ or deeper.
                xml_blobs = [
                    i
                    for i in tree
                    if i["type"] == "blob"
                    and is_public_release_xml_path(i["path"])
                ]
            else:
                # Default: look for Final_XML/*.xml pattern
                xml_blobs = [
                    i
                    for i in tree
                    if i["type"] == "blob"
                    and i["path"].startswith("Final_XML/")
                    and i["path"].lower().endswith(".xml")
                ]

            if path_exclude_patterns:
                before_path_exclude = len(xml_blobs)
                xml_blobs = [
                    i
                    for i in xml_blobs
                    if not matches_exclude_pattern(i["path"], path_exclude_patterns)
                ]
                removed = before_path_exclude - len(xml_blobs)
                if removed:
                    print(f"🚫  Repo {repo}: excluded {removed} XML candidates by path pattern")

            if not xml_blobs:
                print(f"ℹ️  Repo {repo}: no XML blobs found (public={args.public})")
                continue
            else:
                print(f"📄  Repo {repo}: {len(xml_blobs)} XML candidate files")

            for item in xml_blobs:
                futures.append(
                    ex.submit(
                        download_blob,
                        args.org,
                        repo,
                        item,
                        args.src_lang,
                        args.tgt_lang,
                        branch,  # kept for logging / future use
                        out_dir,
                        dialect
                    )
                )

        # ── file-download progress bar ──────────────────────────────────────
        print(f"📊  Total XML download tasks queued: {len(futures)}")
        kept: list[Path] = []
        for f in tqdm(
            fut.as_completed(futures),
            total=len(futures),
            desc="XML files",
            unit="file",
        ):
            res = f.result()
            if res:
                kept.append(res)

    print(f"✅  Downloaded {len(kept)} XML files → {out_dir}")


if __name__ == "__main__":
    main()
