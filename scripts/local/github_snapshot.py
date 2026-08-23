"""Immutable GitHub repository selection and tree snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from pipeline_common import atomic_write_json, utc_now
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

GITHUB_API = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GRAPHQL_BATCH_THRESHOLD = 20
REQUEST_TIMEOUT = 30
CACHE_MAX_AGE_SECONDS = 6 * 60 * 60
CACHE_DIR = PROJECT_ROOT / "corpus_builds" / ".github_metadata_cache"
EXACT_BIBLE_REPOS = ("Formosan-Taiwan-Bible-Society-Bibles",)

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
class RepositoryRef:
    name: str
    requested_ref: str
    commit_sha: str


def get_equivalent_lang_codes(lang_code: str) -> set[str]:
    return LANGUAGE_EQUIVALENTS.get(lang_code, {lang_code})


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
