#!/usr/bin/env python3
"""Fetch an immutable, fully inventoried FormosanBank XML snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from pipeline_common import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
MAX_WORKERS = 4
GRAPHQL_BATCH_THRESHOLD = 20
REQUEST_TIMEOUT = 30
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
CACHE_DIR = PROJECT_ROOT / "corpus_builds" / ".github_metadata_cache"
RAW_XML_CACHE_DIR = PROJECT_ROOT / "corpus_builds" / ".github_raw_xml_cache"
EXACT_BIBLE_REPOS = ("Formosan-Taiwan-Bible-Society-Bibles",)

LANGUAGE_EQUIVALENTS: dict[str, set[str]] = {
    "zh": {"zh", "zho", "chi", "cmn"},
    "zho": {"zh", "zho", "chi", "cmn"},
    "chi": {"zh", "zho", "chi", "cmn"},
    "cmn": {"zh", "zho", "chi", "cmn"},
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
PATH_HINT_TO_LANGUAGE_CODES = {hint: code for code, hints in LANGUAGE_PATH_HINTS.items() for hint in hints}

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
            allowed_methods=frozenset(["GET", "HEAD", "POST"]),
            respect_retry_after_header=True,
        )
    ),
)


@dataclass(frozen=True)
class RepositorySnapshot:
    name: str
    requested_ref: str
    commit_sha: str
    tree_entries: int
    xml_candidates: int
    queued: int
    excluded: int


@dataclass(frozen=True)
class DownloadResult:
    repository: str
    commit_sha: str
    source_path: str
    git_blob_sha: str
    status: str
    bytes: int = 0
    sha256: str = ""
    destination: str = ""
    root_language: str = ""
    dialect: str = ""
    error: str = ""


@dataclass(frozen=True)
class RepositoryRef:
    name: str
    requested_ref: str
    commit_sha: str


def get_equivalent_lang_codes(lang_code: str) -> set[str]:
    return LANGUAGE_EQUIVALENTS.get(lang_code, {lang_code})


def path_language_hint_codes(path: str) -> set[str]:
    tokens: set[str] = set()
    for component in path.split("/"):
        lowered = component.lower().removesuffix(".xml")
        tokens.add(lowered)
        tokens.update(part for part in re.split(r"[^a-z0-9]+", lowered) if part)
    return {PATH_HINT_TO_LANGUAGE_CODES[token] for token in tokens if token in PATH_HINT_TO_LANGUAGE_CODES}


def public_path_may_match_src_lang(path: str, src_lang: str) -> bool:
    hints = path_language_hint_codes(path)
    return not hints or src_lang.strip().lower() in hints


def parse_exclude_patterns(values: Iterable[str] | None) -> list[str]:
    patterns: list[str] = []
    for value in values or []:
        for part in value.split(","):
            pattern = part.strip().lower()
            if pattern and pattern not in patterns:
                patterns.append(pattern)
    return patterns


def normalize_repo_name(value: str) -> str:
    text = value.strip().rstrip("/")
    for prefix in ("https://github.com/", "git@github.com:"):
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    return text.removesuffix(".git").split("/")[-1].strip().lower()


def parse_exact_repos(values: Iterable[str] | None) -> list[str]:
    repos: list[str] = []
    for value in values or []:
        for part in value.split(","):
            repo = normalize_repo_name(part)
            if repo and repo not in repos:
                repos.append(repo)
    return repos


def add_exact_repos(repos: list[str], defaults: Iterable[str]) -> list[str]:
    output = list(repos)
    for repo in defaults:
        normalized = normalize_repo_name(repo)
        if normalized not in output:
            output.append(normalized)
    return output


def matches_exclude_pattern(value: str, patterns: Iterable[str]) -> bool:
    lowered = value.lower()
    return any(pattern in lowered for pattern in patterns)


def matches_exact_repo(repo: str, excluded_repos: Iterable[str]) -> bool:
    return normalize_repo_name(repo) in set(excluded_repos)


def public_release_corpus_root(path: str) -> str | None:
    parts = path.split("/")
    return parts[1] if len(parts) >= 3 and parts[0] == "Corpora" else None


def matches_excluded_public_corpus_root(path: str, excluded_repos: Iterable[str]) -> bool:
    root = public_release_corpus_root(path)
    return bool(root and matches_exact_repo(root, excluded_repos))


def is_public_release_xml_path(path: str) -> bool:
    parts = path.split("/")
    return path.lower().endswith(".xml") and len(parts) >= 3 and parts[0] == "Corpora" and "XML" in parts[:-1]


def is_private_release_xml_path(path: str) -> bool:
    parts = path.split("/")
    return (
        path.lower().endswith(".xml")
        and len(parts) >= 2
        and parts[0] in {"Final_XML", "XML"}
    )


def read_metadata_cache(name: str, *, max_age: float | None = CACHE_MAX_AGE_SECONDS):
    path = CACHE_DIR / name
    try:
        if max_age is not None and time.time() - path.stat().st_mtime > max_age:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None


def write_metadata_cache(name: str, payload) -> None:
    atomic_write_json(CACHE_DIR / name, payload)


def api_get(url: str, *, params: dict[str, object] | None = None) -> requests.Response:
    response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
    if response.status_code == 403:
        message = ""
        try:
            message = str(response.json().get("message", ""))
        except (ValueError, AttributeError):
            message = response.text.strip()
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        reset = response.headers.get("X-RateLimit-Reset", "unknown")
        raise RuntimeError(f"GitHub denied {url}: {message} (rate-limit remaining={remaining}, reset={reset})")
    response.raise_for_status()
    return response


def api_post(url: str, *, json_payload: dict[str, object]) -> requests.Response:
    response = SESSION.post(url, json=json_payload, timeout=REQUEST_TIMEOUT)
    if response.status_code == 403:
        remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
        reset = response.headers.get("X-RateLimit-Reset", "unknown")
        raise RuntimeError(
            f"GitHub denied {url} (rate-limit remaining={remaining}, reset={reset})"
        )
    response.raise_for_status()
    return response


def get_repos(org: str, *, refresh: bool = False) -> list[str]:
    cache_name = f"repos_{org.lower()}.json"
    cached = None if refresh else read_metadata_cache(cache_name)
    if isinstance(cached, list):
        return [str(name) for name in cached]
    repos: list[str] = []
    page = 1
    while True:
        payload = api_get(
            f"{GITHUB_API}/orgs/{org}/repos",
            params={"per_page": 100, "page": page, "type": "all"},
        ).json()
        if not payload:
            break
        repos.extend(str(repo["name"]) for repo in payload)
        page += 1
    write_metadata_cache(cache_name, repos)
    return repos


def get_default_branch(org: str, repo: str, *, refresh: bool = False) -> str:
    cache_name = f"repo_{org.lower()}_{repo.lower()}.json"
    cached = None if refresh else read_metadata_cache(cache_name)
    if isinstance(cached, dict) and cached.get("default_branch"):
        return str(cached["default_branch"])
    payload = api_get(f"{GITHUB_API}/repos/{org}/{repo}").json()
    branch = str(payload.get("default_branch") or "main")
    write_metadata_cache(cache_name, {"default_branch": branch})
    return branch


def resolve_commit(org: str, repo: str, reference: str) -> str:
    payload = api_get(f"{GITHUB_API}/repos/{org}/{repo}/commits/{quote(reference, safe='')}").json()
    commit = str(payload.get("sha") or "")
    if len(commit) != 40:
        raise RuntimeError(f"Could not resolve {org}/{repo}@{reference} to a full commit SHA")
    return commit


def resolve_default_repository_refs(
    org: str,
    selected: list[str],
) -> list[RepositoryRef]:
    """Resolve default branches and commits in a few GraphQL requests."""
    query = """
    query($org: String!, $after: String) {
      organization(login: $org) {
        repositories(first: 100, after: $after, orderBy: {field: NAME, direction: ASC}) {
          nodes { name defaultBranchRef { name target { ... on Commit { oid } } } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }
    """
    wanted = set(selected)
    resolved: dict[str, RepositoryRef] = {}
    cursor: str | None = None
    while True:
        response = api_post(
            f"{GITHUB_API}/graphql",
            json_payload={
                "query": query,
                "variables": {"org": org, "after": cursor},
            },
        )
        payload = response.json()
        errors = payload.get("errors")
        if errors:
            raise RuntimeError(f"GitHub GraphQL repository lookup failed: {errors}")
        organization = payload.get("data", {}).get("organization")
        if not isinstance(organization, dict):
            raise RuntimeError(f"GitHub organization not found: {org}")
        repositories = organization.get("repositories", {})
        for node in repositories.get("nodes") or []:
            name = str(node.get("name") or "")
            if name not in wanted:
                continue
            branch = node.get("defaultBranchRef") or {}
            reference = str(branch.get("name") or "")
            commit = str((branch.get("target") or {}).get("oid") or "")
            if reference and len(commit) == 40:
                resolved[name] = RepositoryRef(name, reference, commit)
        page_info = repositories.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = str(page_info.get("endCursor") or "")
        if not cursor:
            raise RuntimeError("GitHub GraphQL pagination omitted its next cursor")

    missing = sorted(wanted - set(resolved))
    if missing:
        raise RuntimeError(
            "GitHub GraphQL did not resolve default branches for: "
            + ", ".join(missing)
        )
    return [resolved[name] for name in selected]


def _walk_tree(
    org: str,
    repo: str,
    root_sha: str,
    *,
    root_path: str = "",
) -> list[dict]:
    output: list[dict] = []
    pending: deque[tuple[str, str]] = deque([(root_path.strip("/"), root_sha)])
    while pending:
        prefix, tree_sha = pending.popleft()
        payload = api_get(f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{tree_sha}").json()
        if payload.get("truncated"):
            raise RuntimeError(f"Non-recursive GitHub tree unexpectedly truncated for {repo}:{tree_sha}")
        entries = payload.get("tree")
        if not isinstance(entries, list):
            raise RuntimeError(f"Missing Git tree entries for {repo}:{tree_sha}")
        for item in entries:
            relative = str(item.get("path") or "")
            full_path = f"{prefix}/{relative}" if prefix else relative
            entry = dict(item)
            entry["path"] = full_path
            if entry.get("type") == "tree":
                pending.append((full_path, str(entry["sha"])))
            else:
                output.append(entry)
    return output


def get_tree(
    org: str,
    repo: str,
    commit_sha: str,
    *,
    root_path: str = "",
) -> list[dict]:
    normalized_root = root_path.strip("/")
    root_key = hashlib.sha256(normalized_root.encode("utf-8")).hexdigest()[:12]
    cache_name = f"tree_{org.lower()}_{repo.lower()}_{commit_sha}_{root_key}.json"
    cached = read_metadata_cache(cache_name, max_age=None)
    if isinstance(cached, dict) and cached.get("complete") is True and isinstance(cached.get("tree"), list):
        return list(cached["tree"])

    tree_sha = commit_sha
    if normalized_root:
        for component in normalized_root.split("/"):
            payload = api_get(
                f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{tree_sha}"
            ).json()
            entries = payload.get("tree")
            if payload.get("truncated"):
                raise RuntimeError(
                    f"GitHub truncated a non-recursive tree for {repo}:{tree_sha}"
                )
            if not isinstance(entries, list):
                raise RuntimeError(f"Missing Git tree entries for {repo}:{tree_sha}")
            match = next(
                (
                    item
                    for item in entries
                    if item.get("type") == "tree"
                    and str(item.get("path") or "") == component
                ),
                None,
            )
            if match is None:
                write_metadata_cache(
                    cache_name,
                    {"complete": True, "root_path": normalized_root, "tree": []},
                )
                return []
            tree_sha = str(match["sha"])

    payload = api_get(
        f"{GITHUB_API}/repos/{org}/{repo}/git/trees/{tree_sha}",
        params={"recursive": "1"},
    ).json()
    tree = payload.get("tree")
    if not isinstance(tree, list):
        raise RuntimeError(f"GitHub returned no tree for {org}/{repo}@{commit_sha}")
    if payload.get("truncated"):
        tree = _walk_tree(org, repo, tree_sha, root_path=normalized_root)
    elif normalized_root:
        tree = [
            {**item, "path": f"{normalized_root}/{item['path']}"}
            for item in tree
        ]
    write_metadata_cache(
        cache_name,
        {
            "complete": True,
            "root_path": normalized_root,
            "tree": tree,
        },
    )
    return tree


def repository_selection(
    *,
    org: str,
    public: bool,
    branch: str | None,
    discovered: list[str],
    selected: list[str],
    excluded: list[str],
) -> dict[str, object]:
    return {
        "organization": org,
        "public": public,
        "requested_branch": branch,
        "repositories_discovered": sorted(discovered),
        "repositories_selected": sorted(selected),
        "repositories_excluded": sorted(excluded),
    }


def load_or_create_repository_snapshot(
    path: Path,
    *,
    selection: dict[str, object],
    refresh_metadata: bool,
) -> list[RepositoryRef]:
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Repository snapshot is unreadable at {path}: {exc}"
            ) from exc
        if payload.get("complete") is not True:
            raise SystemExit(f"Repository snapshot is incomplete at {path}")
        if payload.get("selection") != selection:
            raise SystemExit(
                f"Repository snapshot selection does not match this fetch: {path}. "
                "Remove it or choose a different --repository-snapshot path."
            )
        records = payload.get("repositories")
        if not isinstance(records, list):
            raise SystemExit(f"Repository snapshot has no repository records: {path}")
        refs = [
            RepositoryRef(
                name=str(record.get("name") or ""),
                requested_ref=str(record.get("requested_ref") or ""),
                commit_sha=str(record.get("commit_sha") or ""),
            )
            for record in records
            if isinstance(record, dict)
        ]
        selected = selection["repositories_selected"]
        if (
            not isinstance(selected, list)
            or len(refs) != len(selected)
            or any(
                not ref.name
                or not ref.requested_ref
                or len(ref.commit_sha) != 40
                for ref in refs
            )
        ):
            raise SystemExit(f"Repository snapshot contains malformed records: {path}")
        return refs

    org = str(selection["organization"])
    branch = selection["requested_branch"]
    selected = [str(name) for name in selection["repositories_selected"]]
    refs: list[RepositoryRef] = []
    errors: list[str] = []
    if not branch and GITHUB_TOKEN and len(selected) >= GRAPHQL_BATCH_THRESHOLD:
        try:
            refs = resolve_default_repository_refs(org, selected)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            errors.append(str(exc))
    else:
        for repo in tqdm(selected, desc="Resolve repository snapshot", unit="repo"):
            try:
                reference = (
                    str(branch)
                    if branch
                    else get_default_branch(org, repo, refresh=refresh_metadata)
                )
                refs.append(
                    RepositoryRef(
                        name=repo,
                        requested_ref=reference,
                        commit_sha=resolve_commit(org, repo, reference),
                    )
                )
            except (requests.RequestException, RuntimeError) as exc:
                errors.append(f"{repo}: {exc}")

    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "selection": selection,
        "repositories": [asdict(ref) for ref in refs],
        "errors": errors,
        "complete": not errors and len(refs) == len(selected),
    }
    atomic_write_json(path, payload)
    if errors:
        raise SystemExit("Repository snapshot failed:\n  - " + "\n  - ".join(errors))
    return refs


def raw_url(org: str, repo: str, path: str, commit_sha: str) -> str:
    encoded_path = "/".join(quote(part) for part in path.split("/"))
    return f"https://raw.githubusercontent.com/{org}/{repo}/{commit_sha}/{encoded_path}"


def retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip()
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
    return min(max_sleep, base_sleep * (2**attempt) + random.uniform(0.0, max(base_sleep, 0.1)))


def git_blob_sha(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def write_blob_cache(cache_path: Path, content: bytes) -> None:
    """Atomically cache a blob without sharing temporary paths across workers."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
        dir=cache_path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(cache_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def classify_xml(
    xml_bytes: bytes,
    src_lang: str,
    tgt_lang: str | None,
    dialect: str | None,
) -> tuple[str, str, str, str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        return "parse_error", "", "", str(exc)
    root_language = (root.attrib.get(XML_LANG) or "").strip().lower()
    root_dialect = (root.attrib.get("dialect") or "").strip()
    if root_language != src_lang.strip().lower():
        return "source_language_mismatch", root_language, root_dialect, ""
    if dialect is not None and root_dialect.lower() != dialect.strip().lower():
        return "dialect_mismatch", root_language, root_dialect, ""
    if tgt_lang:
        target_codes = get_equivalent_lang_codes(tgt_lang.lower())
        if not any(
            (translation.attrib.get(XML_LANG) or "").strip().lower() in target_codes
            for translation in root.iter("TRANSL")
        ):
            return "target_language_mismatch", root_language, root_dialect, ""
    return "kept", root_language, root_dialect, ""


def download_blob_for_languages(
    org: str,
    repo: str,
    item: dict,
    src_langs: tuple[str, ...],
    tgt_lang: str | None,
    commit_sha: str,
    out_dirs: dict[str, Path],
    dialect: str | None,
    *,
    download_retries: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
) -> dict[str, DownloadResult]:
    """Load and parse one blob once, then route it to interested languages."""
    source_path = str(item["path"])
    expected_blob = str(item["sha"])
    url = raw_url(org, repo, source_path, commit_sha)
    cache_path = RAW_XML_CACHE_DIR / f"{expected_blob}.xml"
    xml_bytes = b""
    try:
        cached = cache_path.read_bytes()
        if git_blob_sha(cached) == expected_blob:
            xml_bytes = cached
    except OSError:
        pass

    error = ""
    for attempt in range(download_retries if not xml_bytes else 0):
        response: requests.Response | None = None
        try:
            response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code == 200:
                xml_bytes = response.content
                if git_blob_sha(xml_bytes) != expected_blob:
                    return {
                        lang: DownloadResult(
                            repo,
                            commit_sha,
                            source_path,
                            expected_blob,
                            "checksum_error",
                            error="downloaded bytes do not match Git tree blob SHA",
                        )
                        for lang in src_langs
                    }
                write_blob_cache(cache_path, xml_bytes)
                break
            error = f"HTTP {response.status_code}"
            if response.status_code not in TRANSIENT_HTTP_STATUSES:
                break
        except requests.RequestException as exc:
            error = str(exc)
        if attempt < download_retries - 1:
            time.sleep(
                retry_sleep_seconds(
                    response,
                    attempt,
                    base_sleep=retry_base_sleep,
                    max_sleep=retry_max_sleep,
                )
            )
    if not xml_bytes:
        return {
            lang: DownloadResult(
                repo,
                commit_sha,
                source_path,
                expected_blob,
                "download_error",
                error=error,
            )
            for lang in src_langs
        }

    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        return {
            lang: DownloadResult(
                repo,
                commit_sha,
                source_path,
                expected_blob,
                "parse_error",
                bytes=len(xml_bytes),
                sha256=sha256_bytes(xml_bytes),
                error=str(exc),
            )
            for lang in src_langs
        }

    root_language = (root.attrib.get(XML_LANG) or "").strip().lower()
    root_dialect = (root.attrib.get("dialect") or "").strip()
    digest = sha256_bytes(xml_bytes)
    target_codes = get_equivalent_lang_codes(tgt_lang.lower()) if tgt_lang else set()
    has_target = not target_codes or any(
        (translation.attrib.get(XML_LANG) or "").strip().lower() in target_codes
        for translation in root.iter("TRANSL")
    )
    routed: dict[str, DownloadResult] = {}
    for lang in src_langs:
        if root_language != lang:
            status = "source_language_mismatch"
        elif dialect is not None and root_dialect.lower() != dialect.strip().lower():
            status = "dialect_mismatch"
        elif not has_target:
            status = "target_language_mismatch"
        else:
            status = "kept"

        destination_value = ""
        if status == "kept":
            destination = out_dirs[lang] / repo / source_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(xml_bytes)
            destination_value = str(destination.relative_to(out_dirs[lang]))
        routed[lang] = DownloadResult(
            repo,
            commit_sha,
            source_path,
            expected_blob,
            status,
            bytes=len(xml_bytes),
            sha256=digest,
            destination=destination_value,
            root_language=root_language,
            dialect=root_dialect,
        )
    return routed


def download_blob(
    org: str,
    repo: str,
    item: dict,
    src_lang: str,
    tgt_lang: str | None,
    commit_sha: str,
    out_dir: Path,
    dialect: str | None,
    *,
    download_retries: int,
    retry_base_sleep: float,
    retry_max_sleep: float,
) -> DownloadResult:
    return download_blob_for_languages(
        org,
        repo,
        item,
        (src_lang,),
        tgt_lang,
        commit_sha,
        {src_lang: out_dir},
        dialect,
        download_retries=download_retries,
        retry_base_sleep=retry_base_sleep,
        retry_max_sleep=retry_max_sleep,
    )[src_lang]


def write_inventory(path: Path, rows: list[DownloadResult]) -> tuple[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row.repository.lower(), row.source_path))
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(asdict(row), ensure_ascii=False, sort_keys=True) + "\n")
    return (
        sha256_file(path),
        stable_json_hash([asdict(row) for row in ordered]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch an immutable FormosanBank XML snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--src-lang")
    source_group.add_argument(
        "--src-langs",
        help="Comma-separated source languages to classify in one XML traversal.",
    )
    parser.add_argument("--tgt-lang")
    parser.add_argument("--org", default="formosanbank")
    parser.add_argument("--branch", help="Force one branch/ref for every repository")
    parser.add_argument("--out-dir")
    parser.add_argument(
        "--out-root",
        type=Path,
        help="Multi-language output root; writes downloaded_<code> directories.",
    )
    parser.add_argument("--clean-output", action="store_true")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--download-retries", type=int, default=8)
    parser.add_argument("--retry-base-sleep", type=float, default=2.0)
    parser.add_argument("--retry-max-sleep", type=float, default=60.0)
    parser.add_argument(
        "--allow-download-failures",
        action="store_true",
        help="Diagnostic escape hatch. The fetch manifest remains incomplete.",
    )
    parser.add_argument("--public", action="store_true")
    parser.add_argument(
        "--repository-snapshot",
        type=Path,
        help="Build-scoped repository ref manifest shared by every language fetch.",
    )
    parser.add_argument(
        "--refresh-repository-metadata",
        action="store_true",
        help="Refresh org/default-branch metadata when creating a repository snapshot.",
    )
    parser.add_argument("--no-public-language-path-prefilter", action="store_true")
    parser.add_argument("--exclude-bible", action="store_true")
    parser.add_argument("--exclude-repo", action="append", default=[])
    parser.add_argument("--exclude-repo-pattern", action="append", default=[])
    parser.add_argument("--exclude-path-pattern", action="append", default=[])
    parser.add_argument("--dialect")
    args = parser.parse_args()
    if args.workers < 1 or args.download_retries < 1:
        raise SystemExit("--workers and --download-retries must be >= 1")
    if args.retry_base_sleep < 0 or args.retry_max_sleep < 0:
        raise SystemExit("Retry sleep values must be >= 0")
    if args.src_langs and (args.out_dir or not args.out_root):
        raise SystemExit("--src-langs requires --out-root and cannot use --out-dir")
    return args


def main() -> None:
    args = parse_args()
    src_langs = tuple(
        dict.fromkeys(
            part.strip().lower()
            for part in (args.src_langs or args.src_lang).split(",")
            if part.strip()
        )
    )
    if not src_langs:
        raise SystemExit("No source languages selected")
    exact_excluded = parse_exact_repos(args.exclude_repo)
    if args.exclude_bible:
        exact_excluded = add_exact_repos(exact_excluded, EXACT_BIBLE_REPOS)
    repo_patterns = parse_exclude_patterns(args.exclude_repo_pattern)
    path_patterns = parse_exclude_patterns(args.exclude_path_pattern)
    if args.src_langs:
        out_root = args.out_root.expanduser().resolve()
        out_dirs = {lang: out_root / f"downloaded_{lang}" for lang in src_langs}
    else:
        out_dir = Path(args.out_dir or f"downloaded_{src_langs[0]}").expanduser().resolve()
        out_dirs = {src_langs[0]: out_dir}
    for out_dir in out_dirs.values():
        if args.clean_output and out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if not GITHUB_TOKEN:
        print("GitHub token is not set; public API rate limits apply.")
    snapshot_path = (
        args.repository_snapshot.expanduser().resolve()
        if args.repository_snapshot
        else None
    )
    existing_snapshot: dict[str, object] | None = None
    if snapshot_path and snapshot_path.is_file():
        try:
            existing_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(
                f"Repository snapshot is unreadable at {snapshot_path}: {exc}"
            ) from exc
    if existing_snapshot:
        snapshot_selection = existing_snapshot.get("selection")
        if not isinstance(snapshot_selection, dict):
            raise SystemExit(
                f"Repository snapshot has no selection record: {snapshot_path}"
            )
        repos = [
            str(name)
            for name in snapshot_selection.get("repositories_discovered", [])
        ]
    else:
        repos = (
            ["FormosanBank"]
            if args.public
            else get_repos(args.org, refresh=args.refresh_repository_metadata)
        )
    selected_repos = [
        repo
        for repo in repos
        if not matches_exact_repo(repo, exact_excluded) and not matches_exclude_pattern(repo, repo_patterns)
    ]
    excluded_repo_names = sorted(set(repos) - set(selected_repos))
    if not selected_repos:
        raise SystemExit("No repositories remain after exclusions")

    selection = repository_selection(
        org=args.org,
        public=args.public,
        branch=args.branch,
        discovered=repos,
        selected=selected_repos,
        excluded=excluded_repo_names,
    )
    if snapshot_path:
        repository_refs = load_or_create_repository_snapshot(
            snapshot_path,
            selection=selection,
            refresh_metadata=args.refresh_repository_metadata,
        )
    else:
        repository_refs = []
        for repo in selected_repos:
            reference = args.branch or get_default_branch(args.org, repo)
            repository_refs.append(
                RepositoryRef(
                    name=repo,
                    requested_ref=reference,
                    commit_sha=resolve_commit(args.org, repo, reference),
                )
            )

    results = {lang: [] for lang in src_langs}
    snapshots = {lang: [] for lang in src_langs}
    repository_errors: list[str] = []
    future_jobs: list[futures.Future[dict[str, DownloadResult]]] = []

    with futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        for repository_ref in tqdm(
            repository_refs,
            desc="Repository XML trees",
            unit="repo",
        ):
            repo = repository_ref.name
            try:
                reference = repository_ref.requested_ref
                commit_sha = repository_ref.commit_sha
                if args.public:
                    tree = get_tree(
                        args.org,
                        repo,
                        commit_sha,
                        root_path="Corpora",
                    )
                else:
                    tree = []
                    for private_root in ("Final_XML", "XML"):
                        tree.extend(
                            get_tree(
                                args.org,
                                repo,
                                commit_sha,
                                root_path=private_root,
                            )
                        )
            except (requests.RequestException, RuntimeError) as exc:
                repository_errors.append(f"{repo}: {exc}")
                continue

            if args.public:
                candidates = [
                    item
                    for item in tree
                    if item.get("type") == "blob" and is_public_release_xml_path(str(item.get("path") or ""))
                ]
            else:
                candidates = [
                    item
                    for item in tree
                    if item.get("type") == "blob"
                    and is_private_release_xml_path(
                        str(item.get("path") or "")
                    )
                ]

            queued_counts: Counter[str] = Counter()
            excluded_counts: Counter[str] = Counter()
            for item in candidates:
                path = str(item["path"])
                common_reason = ""
                if args.public and exact_excluded and matches_excluded_public_corpus_root(path, exact_excluded):
                    common_reason = "excluded_public_corpus_root"
                elif path_patterns and matches_exclude_pattern(path, path_patterns):
                    common_reason = "excluded_path_pattern"
                interested: list[str] = []
                for lang in src_langs:
                    reason = common_reason
                    if (
                        not reason
                        and args.public
                        and not args.no_public_language_path_prefilter
                        and not public_path_may_match_src_lang(path, lang)
                    ):
                        reason = "path_language_prefilter"
                    if reason:
                        results[lang].append(
                            DownloadResult(
                                repo,
                                commit_sha,
                                path,
                                str(item["sha"]),
                                reason,
                            )
                        )
                        excluded_counts[lang] += 1
                    else:
                        interested.append(lang)
                        queued_counts[lang] += 1
                if interested:
                    future_jobs.append(
                        executor.submit(
                            download_blob_for_languages,
                            args.org,
                            repo,
                            item,
                            tuple(interested),
                            args.tgt_lang,
                            commit_sha,
                            out_dirs,
                            args.dialect,
                            download_retries=args.download_retries,
                            retry_base_sleep=args.retry_base_sleep,
                            retry_max_sleep=args.retry_max_sleep,
                        )
                    )
            for lang in src_langs:
                snapshots[lang].append(
                    RepositorySnapshot(
                        name=repo,
                        requested_ref=reference,
                        commit_sha=commit_sha,
                        tree_entries=len(tree),
                        xml_candidates=len(candidates),
                        queued=queued_counts[lang],
                        excluded=excluded_counts[lang],
                    )
                )

        for future in tqdm(
            futures.as_completed(future_jobs),
            total=len(future_jobs),
            desc="XML files",
            unit="file",
        ):
            for lang, result in future.result().items():
                results[lang].append(result)
        future_jobs.clear()

    language_failures: dict[str, int] = {}
    for lang in src_langs:
        out_dir = out_dirs[lang]
        inventory_path = out_dir / "_fetch_inventory.jsonl"
        inventory_hash, inventory_records_hash = write_inventory(
            inventory_path,
            results[lang],
        )
        status_counts = Counter(row.status for row in results[lang])
        hard_failures = sum(
            status_counts[status]
            for status in ("download_error", "checksum_error", "parse_error")
        )
        language_failures[lang] = hard_failures
        complete = (
            not repository_errors
            and hard_failures == 0
            and status_counts["kept"] > 0
        )
        manifest = {
            "schema_version": 3,
            "created_at": utc_now(),
            "organization": args.org,
            "source_language": lang,
            "source_languages_fetched_together": list(src_langs),
            "target_language": args.tgt_lang,
            "dialect": args.dialect,
            "public": args.public,
            "requested_branch": args.branch,
            "repository_snapshot": (
                {
                    "path": str(snapshot_path),
                    "sha256": sha256_file(snapshot_path),
                }
                if snapshot_path
                else None
            ),
            "repositories_discovered": sorted(repos),
            "repositories_excluded": excluded_repo_names,
            "repositories": [
                asdict(snapshot)
                for snapshot in sorted(
                    snapshots[lang], key=lambda row: row.name.lower()
                )
            ],
            "repository_errors": repository_errors,
            "status_counts": dict(sorted(status_counts.items())),
            "inventory": inventory_path.name,
            "inventory_sha256": inventory_hash,
            "inventory_records_sha256": inventory_records_hash,
            "complete": complete,
        }
        manifest_path = out_dir / "_fetch_manifest.json"
        atomic_write_json(manifest_path, manifest)
        print(f"{lang}: kept {status_counts['kept']:,} XML files in {out_dir}")
        print(f"Fetch manifest: {manifest_path}")

    if repository_errors:
        raise SystemExit("Repository discovery failed:\n  - " + "\n  - ".join(repository_errors))
    total_failures = sum(language_failures.values())
    if total_failures and not args.allow_download_failures:
        raise SystemExit(
            f"XML fetch incomplete: {total_failures} language-level "
            "download/checksum/parse failures; see the fetch manifests"
        )
    empty = [lang for lang in src_langs if not any(row.status == "kept" for row in results[lang])]
    if empty:
        raise SystemExit(f"No matching XML files were downloaded for: {', '.join(empty)}")


if __name__ == "__main__":
    main()
