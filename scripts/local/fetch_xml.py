#!/usr/bin/env python3
"""
STEP 1: 

Harvest XML files from FormosanBank repositories.

Two modes:
----------
1. Default mode: Searches through all repos in the FormosanBank org for files in Final_XML/ folders
2. Public release mode (--public): Only looks in FormosanBank/FormosanBank/Corpora/*/XML/ structure

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

# Public release mode (only FormosanBank/FormosanBank/Corpora/*/XML/)
$ python fetch_xml.py --src-lang ami --public
$ python fetch_xml.py --src-lang pwn --public --tgt-lang zh
"""
from __future__ import annotations

import argparse
import concurrent.futures as fut
import os
import sys
import time
from pathlib import Path
from typing import Iterable

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
    "zh": {"zh", "zho", "chi"},
    "zho": {"zh", "zho", "chi"}, 
    "chi": {"zh", "zho", "chi"},
    # English variants
    "en": {"en", "eng"},
    "eng": {"en", "eng"},
}

def get_equivalent_lang_codes(lang_code: str) -> set[str]:
    """Get all equivalent language codes for a given language code."""
    return LANGUAGE_EQUIVALENTS.get(lang_code, {lang_code})
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────  config  ──────────────────────────────────────
GITHUB_API = "https://api.github.com"
HEADERS = {"Authorization": f"token {os.getenv('GITHUB_TOKEN', '')}"}
MAX_WORKERS = 16
REQUEST_TIMEOUT = 10        # seconds
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
        r.raise_for_status()
        data = r.json()
        if not data:
            break

        # remove entry with name "Formosan-Wikipedias"
        data = [repo for repo in data if repo["name"] != "Formosan-Wikipedias"]
        yield from (repo["name"] for repo in data)
        page += 1


def get_tree(org: str, repo: str, branch: str | None = None):
    """Return the full git tree (recursive) for a repo/branch."""
    if branch is None:
        repo_meta = SESSION.get(
            f"{GITHUB_API}/repos/{org}/{repo}", timeout=REQUEST_TIMEOUT
        ).json()
        branch = repo_meta["default_branch"]

    r = SESSION.get(
        f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["tree"]


def raw_url(org: str, repo: str, path: str, branch: str):
    return f"https://raw.githubusercontent.com/{org}/{repo}/{branch}/{path}"


def wants_file(xml_bytes: bytes, src_lang: str, tgt_lang: str | None):
    """Return True iff the XML root matches src_lang (and tgt_lang if given)."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return False

    src_ok = (
        root.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") == src_lang
    )
    if not src_ok:
        return False

    if tgt_lang is None:
        return True

    # Get all equivalent language codes for the target language
    target_codes = get_equivalent_lang_codes(tgt_lang.lower())
    
    return any(
        elem.attrib.get("{http://www.w3.org/XML/1998/namespace}lang", "").lower() in target_codes
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
):
    """Download → filter → save one blob.  Returns destination Path or None."""
    url = raw_url(org, repo, item["path"], branch)

    # retry loop for DNS hiccups that happen before urllib3 retry kicks in
    for attempt in range(3):
        try:
            resp = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            xml_bytes = resp.content
            break
        except (requests.exceptions.RequestException, ET.ParseError):
            if attempt == 2:
                # give up after 3 tries
                return None
            time.sleep(2 ** attempt)
    else:
        return None  # never reached

    if not wants_file(xml_bytes, src_lang, tgt_lang):
        return None

    dest = out_dir / repo / item["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(xml_bytes)
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-lang", required=True, help="e.g. ami, pau, tsu")
    parser.add_argument("--tgt-lang", help="optional target language filter (e.g. zh, zho, en)")
    parser.add_argument("--org", default="formosanbank")
    parser.add_argument("--branch", help="force a branch name for all repos")
    parser.add_argument(
        "--out-dir", help="where to store the files (default: downloaded_{src_lang})"
    )
    parser.add_argument(
        "--public", 
        action="store_true",
        help="Use public release structure: only look in FormosanBank/FormosanBank/Corpora/*/XML/"
    )
    args = parser.parse_args()

    if not os.getenv("GITHUB_TOKEN"):
        sys.exit("❌  Please set the GITHUB_TOKEN environment variable")

    # Set default output directory if none provided
    if args.out_dir is None:
        args.out_dir = f"downloaded_{args.src_lang}"
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Show which language codes will be matched if target language is specified
    if args.tgt_lang:
        equivalent_codes = get_equivalent_lang_codes(args.tgt_lang.lower())
        print(f"🔍  Target language '{args.tgt_lang}' will match: {sorted(equivalent_codes)}")

    # Determine which repos to process based on --public flag
    if args.public:
        repos = ["FormosanBank"]
        print(f"🌐  Public release mode: only processing FormosanBank/FormosanBank/Corpora/*/XML/")
    else:
        repos = list(get_repos(args.org))
        print(f"Found {len(repos)} repos in {args.org}")

    with fut.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []

        # ── repo‑level progress bar ──────────────────────────────────────────
        for repo in tqdm(repos, desc="Repos", unit="repo"):
            try:
                tree = get_tree(args.org, repo, args.branch)
            except requests.HTTPError as e:
                if e.response.status_code == 409:  # empty repo or huge tree
                    print(f"⚠️  Skipping repo {repo} (empty or too large for API)")
                    continue
                raise

            # Filter XML files based on mode
            if args.public:
                # Public release: look for Corpora/*/XML/*.xml pattern
                xml_blobs = [
                    i
                    for i in tree
                    if i["type"] == "blob"
                    and i["path"].startswith("Corpora/")
                    and "/XML/" in i["path"]
                    and i["path"].lower().endswith(".xml")
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


            for item in xml_blobs:
                futures.append(
                    ex.submit(
                        download_blob,
                        args.org,
                        repo,
                        item,
                        args.src_lang,
                        args.tgt_lang,
                        args.branch or "HEAD",
                        out_dir,
                    )
                )

        # ── file‑download progress bar ──────────────────────────────────────
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

    print(f"Downloaded {len(kept)} XML files → {out_dir}")


if __name__ == "__main__":
    main()
