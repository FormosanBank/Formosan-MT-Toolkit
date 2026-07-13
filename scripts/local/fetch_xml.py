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
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

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

LANGUAGE_PATH_HINTS: dict[str, set[str]] = {
    "ami": {"ami", "amis"},
    "bnn": {"bnn", "bunun"},
    "ckv": {"ckv", "kavalan"},
    "dru": {"dru", "rukai"},
    "pwn": {"pwn", "paiwan"},
    "pyu": {"pyu", "puyuma"},
    "ssf": {"ssf", "thao"},
    "sxr": {"sxr", "saaroa"},
    "szy": {"szy", "sakizaya"},
    "tao": {"tao", "yami"},
    "tay": {"tay", "atayal"},
    "trv": {"trv", "seediq", "sedik"},
    "tsu": {"tsu", "tsou"},
    "xnb": {"xnb", "kanakanavu"},
    "xsy": {"xsy", "saisiyat"},
}

PATH_HINT_TO_LANGUAGE_CODES = {
    hint: code
    for code, hints in LANGUAGE_PATH_HINTS.items()
    for hint in hints
}


def get_equivalent_lang_codes(lang_code: str) -> set[str]:
    """Get all equivalent language codes for a given language code."""
    return LANGUAGE_EQUIVALENTS.get(lang_code, {lang_code})


def path_language_hint_codes(path: str) -> set[str]:
    """Return language codes that are explicitly named by path components/tokens."""
    tokens: set[str] = set()
    for component in path.split("/"):
        lowered = component.lower()
        if lowered.endswith(".xml"):
            lowered = lowered[:-4]
        tokens.add(lowered)
        tokens.update(part for part in re.split(r"[^a-z0-9]+", lowered) if part)
    return {
        PATH_HINT_TO_LANGUAGE_CODES[token]
        for token in tokens
        if token in PATH_HINT_TO_LANGUAGE_CODES
    }


def public_path_may_match_src_lang(path: str, src_lang: str) -> bool:
    """Conservative prefilter for public corpus paths before raw XML download."""
    hints = path_language_hint_codes(path)
    if not hints:
        return True
    return src_lang.strip().lower() in hints


def parse_exclude_patterns(values: Iterable[str] | None) -> list[str]:
    """Parse repeatable or comma-separated case-insensitive substring patterns."""
    patterns: list[str] = []
    for value in values or []:
        for part in value.split(","):
            pattern = part.strip().lower()
            if pattern and pattern not in patterns:
                patterns.append(pattern)
    return patterns


EXACT_BIBLE_REPOS = (
    "Formosan-Taiwan-Bible-Society-Bibles",
)


def normalize_repo_name(value: str) -> str:
    """Return the repo-name component from a GitHub repo name/full-name/URL."""
    text = value.strip().rstrip("/")
    if text.startswith("https://github.com/"):
        text = text.removeprefix("https://github.com/")
    if text.startswith("git@github.com:"):
        text = text.removeprefix("git@github.com:")
    if text.endswith(".git"):
        text = text[:-4]
    return text.split("/")[-1].strip().lower()


def parse_exact_repos(values: Iterable[str] | None) -> list[str]:
    repos: list[str] = []
    for value in values or []:
        for part in value.split(","):
            repo = normalize_repo_name(part)
            if repo and repo not in repos:
                repos.append(repo)
    return repos


def add_exact_repos(repos: list[str], defaults: Iterable[str]) -> list[str]:
    out = list(repos)
    for repo in defaults:
        normalized = normalize_repo_name(repo)
        if normalized not in out:
            out.append(normalized)
    return out


def matches_exclude_pattern(value: str, patterns: Iterable[str]) -> bool:
    haystack = value.lower()
    return any(pattern in haystack for pattern in patterns)


def matches_exact_repo(repo: str, excluded_repos: Iterable[str]) -> bool:
    return normalize_repo_name(repo) in set(excluded_repos)


def public_release_corpus_root(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) >= 3 and parts[0] == "Corpora":
        return parts[1]
    return None


def matches_excluded_public_corpus_root(path: str, excluded_repos: Iterable[str]) -> bool:
    corpus_root = public_release_corpus_root(path)
    return bool(corpus_root and matches_exact_repo(corpus_root, excluded_repos))


# ─────────────────────────────  config  ──────────────────────────────────────
GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}
MAX_WORKERS = 4
REQUEST_TIMEOUT = 10  # seconds
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
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
CACHE_DIR = project_root / "corpus_builds" / ".github_metadata_cache"
RAW_XML_CACHE_DIR = project_root / "corpus_builds" / ".github_raw_xml_cache"
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DownloadResult:
    status: str
    path: Path | None = None
    url: str | None = None
    error: str | None = None


def read_metadata_cache(name: str):
    path = CACHE_DIR / name
    try:
        if time.time() - path.stat().st_mtime > CACHE_MAX_AGE_SECONDS:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def write_metadata_cache(name: str, payload) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def get_repos(org: str) -> Iterable[str]:
    """Yield every repository name in the organisation."""
    cache_name = f"repos_{org.lower()}.json"
    cached = read_metadata_cache(cache_name)
    if isinstance(cached, list):
        yield from (str(name) for name in cached)
        return

    repos: list[str] = []
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
        repos.extend(repo["name"] for repo in data)
        page += 1
    write_metadata_cache(cache_name, repos)
    yield from repos


def get_default_branch(org: str, repo: str) -> str:
    cache_name = f"repo_{org.lower()}_{repo.lower()}.json"
    cached = read_metadata_cache(cache_name)
    if isinstance(cached, dict) and cached.get("default_branch"):
        return str(cached["default_branch"])
    meta = SESSION.get(f"{GITHUB_API}/repos/{org}/{repo}", timeout=REQUEST_TIMEOUT)
    meta.raise_for_status()
    branch = str(meta.json().get("default_branch", "main"))
    write_metadata_cache(cache_name, {"default_branch": branch})
    return branch


def get_tree(org: str, repo: str, branch: str):
    """Return the full git tree (recursive) for a repo/branch."""
    cache_name = f"tree_{org.lower()}_{repo.lower()}_{branch.lower()}.json"
    cached = read_metadata_cache(cache_name)
    if isinstance(cached, list):
        print(f"   🌲 {repo}@{branch}: {len(cached)} cached tree entries")
        return cached
    r = SESSION.get(
        f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{branch}",
        params={"recursive": "1"},
        timeout=REQUEST_TIMEOUT,
    )
    r.raise_for_status()
    tree = r.json().get("tree", [])
    write_metadata_cache(cache_name, tree)
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


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return max(0.0, float(value))
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())


def retry_sleep_seconds(
    response: requests.Response | None,
    attempt: int,
    *,
    base_sleep: float,
    max_sleep: float,
) -> float:
    retry_after = retry_after_seconds(response.headers.get("Retry-After") if response else None)
    if retry_after is not None:
        return min(max_sleep, retry_after)
    backoff = base_sleep * (2 ** attempt)
    jitter = random.uniform(0.0, max(base_sleep, 0.1))
    return min(max_sleep, backoff + jitter)


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
    *,
    download_retries: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
) -> DownloadResult:
    """
    Download → filter → save one blob using raw.githubusercontent.com
    (so we don't burn through the REST API core rate limit).

    Returns destination Path or None.
    """
    url = raw_url(org, repo, item["path"], branch)
    cache_path = RAW_XML_CACHE_DIR / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.xml"
    try:
        if time.time() - cache_path.stat().st_mtime <= CACHE_MAX_AGE_SECONDS:
            xml_bytes = cache_path.read_bytes()
        else:
            xml_bytes = b""
    except OSError:
        xml_bytes = b""

    # retry loop for transient HTTP issues
    last_error = ""
    for attempt in range(download_retries if not xml_bytes else 0):
        response: requests.Response | None = None
        try:
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                xml_bytes = response.content
                RAW_XML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                tmp_cache_path = cache_path.with_suffix(".tmp")
                tmp_cache_path.write_bytes(xml_bytes)
                tmp_cache_path.replace(cache_path)
                break
            last_error = f"HTTP {response.status_code}"
            if response.status_code not in TRANSIENT_HTTP_STATUSES:
                print(
                    f"❌  [{repo}] Non-retryable HTTP {response.status_code} for {url}"
                )
                return DownloadResult(status="failed", url=url, error=last_error)
            print(
                f"⚠️  [{repo}] HTTP {response.status_code} for {url} "
                f"(attempt {attempt + 1}/{download_retries})"
            )
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(
                f"⚠️  [{repo}] Error fetching {url} "
                f"on attempt {attempt + 1}/{download_retries}: {e}"
            )
        if attempt == download_retries - 1:
            print(f"❌  [{repo}] Giving up on {url} after {download_retries} attempts")
            return DownloadResult(status="failed", url=url, error=last_error)
        time.sleep(
            retry_sleep_seconds(
                response,
                attempt,
                base_sleep=retry_base_sleep,
                max_sleep=retry_max_sleep,
            )
        )
    else:
        if xml_bytes:
            pass
        else:
            return DownloadResult(status="failed", url=url, error=last_error or "unknown error")

    if not wants_file(xml_bytes, src_lang, tgt_lang, dialect):
        # Uncomment for super-verbose logging:
        # print(f"   ↷ [{repo}] Skipped {item['path']} (lang filter)")
        return DownloadResult(status="skipped", url=url)

    dest = out_dir / repo / item["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(xml_bytes)
    # Uncomment for per-file success logging:
    # print(f"   ✅ [{repo}] Saved {dest}")
    return DownloadResult(status="kept", path=dest, url=url)


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
        "--workers",
        type=int,
        default=MAX_WORKERS,
        help=(
            "Concurrent raw GitHub downloads. Lower values are slower but avoid "
            f"429 rate limiting. Default: {MAX_WORKERS}."
        ),
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=8,
        help="Attempts per XML candidate for transient HTTP failures such as 429.",
    )
    parser.add_argument(
        "--retry-base-sleep",
        type=float,
        default=2.0,
        help="Initial exponential-backoff sleep in seconds for raw XML downloads.",
    )
    parser.add_argument(
        "--retry-max-sleep",
        type=float,
        default=60.0,
        help="Maximum sleep in seconds between raw XML download retries.",
    )
    parser.add_argument(
        "--allow-download-failures",
        action="store_true",
        help="Do not abort if one or more XML candidates could not be downloaded.",
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
        "--no-public-language-path-prefilter",
        action="store_true",
        help=(
            "In --public mode, do not skip XML paths whose directory/file tokens "
            "clearly name a different Formosan language."
        ),
    )
    parser.add_argument(
        "--exclude-bible",
        action="store_true",
        help="Exclude the exact Formosan-Taiwan-Bible-Society-Bibles repo/corpus root.",
    )
    parser.add_argument(
        "--exclude-repo",
        action="append",
        default=[],
        help=(
            "Exact repository name/full-name/URL to skip before scanning. "
            "Can be repeated or comma-separated."
        ),
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
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.download_retries < 1:
        raise SystemExit("--download-retries must be >= 1")
    if args.retry_base_sleep < 0:
        raise SystemExit("--retry-base-sleep must be >= 0")
    if args.retry_max_sleep < 0:
        raise SystemExit("--retry-max-sleep must be >= 0")

    exact_excluded_repos = parse_exact_repos(args.exclude_repo)
    repo_exclude_patterns = parse_exclude_patterns(args.exclude_repo_pattern)
    path_exclude_patterns = parse_exclude_patterns(args.exclude_path_pattern)
    if args.exclude_bible:
        exact_excluded_repos = add_exact_repos(exact_excluded_repos, EXACT_BIBLE_REPOS)

    if not GITHUB_TOKEN:
        sys.exit("❌  Please set the GITHUB_TOKEN environment variable")

    print(
        f"⚙️  Config:\n"
        f"   src_lang   = {args.src_lang}\n"
        f"   dialect    = {args.dialect}\n"
        f"   tgt_lang   = {args.tgt_lang}\n"
        f"   org        = {args.org}\n"
        f"   public     = {args.public}\n"
        f"   public_path_prefilter = {args.public and not args.no_public_language_path_prefilter}\n"
        f"   exclude_repos_exact = {exact_excluded_repos or None}\n"
        f"   exclude_repo_patterns = {repo_exclude_patterns or None}\n"
        f"   exclude_path_patterns = {path_exclude_patterns or None}\n"
        f"   branch_arg = {args.branch}\n"
        f"   workers    = {args.workers}\n"
        f"   retries    = {args.download_retries}\n"
        f"   backoff    = {args.retry_base_sleep}s..{args.retry_max_sleep}s\n"
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

    if exact_excluded_repos:
        excluded_repos = [
            repo for repo in repos if matches_exact_repo(repo, exact_excluded_repos)
        ]
        repos = [repo for repo in repos if not matches_exact_repo(repo, exact_excluded_repos)]
        if excluded_repos:
            print(
                f"🚫  Excluded {len(excluded_repos)} exact repo(s): "
                + ", ".join(sorted(excluded_repos))
            )

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

    with fut.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = []

        # ── repo-level progress bar ──────────────────────────────────────────
        for repo in tqdm(repos, desc="Repos", unit="repo"):
            # Determine branch for this repo
            if args.branch is None:
                try:
                    branch = get_default_branch(args.org, repo)
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
                if not args.no_public_language_path_prefilter:
                    before_language_prefilter = len(xml_blobs)
                    xml_blobs = [
                        i
                        for i in xml_blobs
                        if public_path_may_match_src_lang(i["path"], args.src_lang)
                    ]
                    removed = before_language_prefilter - len(xml_blobs)
                    if removed:
                        print(
                            f"🚫  Repo {repo}: skipped {removed} XML candidates "
                            f"whose public path clearly names another language"
                        )
                if exact_excluded_repos:
                    before_exact_public_exclude = len(xml_blobs)
                    xml_blobs = [
                        i
                        for i in xml_blobs
                        if not matches_excluded_public_corpus_root(
                            i["path"],
                            exact_excluded_repos,
                        )
                    ]
                    removed = before_exact_public_exclude - len(xml_blobs)
                    if removed:
                        print(
                            f"🚫  Repo {repo}: excluded {removed} XML candidates "
                            "from exact public corpus root(s)"
                        )
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
                        dialect,
                        download_retries=args.download_retries,
                        retry_base_sleep=args.retry_base_sleep,
                        retry_max_sleep=args.retry_max_sleep,
                    )
                )

        # ── file-download progress bar ──────────────────────────────────────
        print(f"📊  Total XML download tasks queued: {len(futures)}")
        kept: list[Path] = []
        failures: list[DownloadResult] = []
        for f in tqdm(
            fut.as_completed(futures),
            total=len(futures),
            desc="XML files",
            unit="file",
        ):
            res = f.result()
            if res.status == "kept" and res.path:
                kept.append(res.path)
            elif res.status == "failed":
                failures.append(res)

    print(f"✅  Downloaded {len(kept)} XML files → {out_dir}")
    if failures:
        print(f"❌  Failed to download {len(failures)} XML candidate file(s).")
        for failure in failures[:25]:
            print(f"   - {failure.url}: {failure.error}")
        if len(failures) > 25:
            print(f"   ... and {len(failures) - 25} more")
        if not args.allow_download_failures:
            raise SystemExit(
                "XML fetch incomplete. Re-run the command; completed downloads are "
                "kept unless --clean-output removes them."
            )


if __name__ == "__main__":
    main()
