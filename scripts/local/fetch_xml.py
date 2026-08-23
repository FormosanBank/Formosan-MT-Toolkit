#!/usr/bin/env python3
"""Fetch an immutable, fully inventoried FormosanBank XML snapshot."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import hashlib
import json
import os
import random
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import requests
from github_snapshot import (
    EXACT_BIBLE_REPOS,
    GITHUB_TOKEN,
    REQUEST_TIMEOUT,
    SESSION,
    RepositoryRef,
    RepositorySnapshot,
    add_exact_repos,
    get_default_branch,
    get_equivalent_lang_codes,
    get_repos,
    get_tree,
    is_private_release_xml_path,
    is_public_release_xml_path,
    load_or_create_repository_snapshot,
    matches_exact_repo,
    matches_exclude_pattern,
    matches_excluded_public_corpus_root,
    parse_exact_repos,
    parse_exclude_patterns,
    repository_selection,
    resolve_commit,
    resolve_default_repository_refs,
)
from pipeline_common import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
    stable_json_hash,
    utc_now,
)
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
MAX_WORKERS = 4
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
RAW_XML_CACHE_DIR = PROJECT_ROOT / "corpus_builds" / ".github_raw_xml_cache"

__all__ = [
    "classify_xml",
    "download_blob_for_languages",
    "get_tree",
    "git_blob_sha",
    "load_or_create_repository_snapshot",
    "repository_selection",
    "resolve_default_repository_refs",
    "write_blob_cache",
]


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
