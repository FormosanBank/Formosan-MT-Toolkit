"""Release gates, artifact inventory, and provenance for corpus builds."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path

from build_context import (
    EXACT_BIBLE_REPOS,
    PROJECT_ROOT,
    BuildPaths,
    pivot_read_cache_dirs,
    remove_path,
    require_json_manifest,
)
from pipeline_common import (
    PIPELINE_CONFIG_PATH,
    atomic_write_json,
    git_state,
    load_pipeline_config,
    sha256_file,
    utc_now,
)
from stage_cache import cached_sha256

PIPELINE_CONFIG = load_pipeline_config()


def count_csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def file_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def companion_row_count(path: Path) -> int | None:
    """Return a row count only when the companion is bound to this CSV."""
    from columnar_io import cache_paths, verified_artifact

    _, manifest_path = cache_paths(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    rows = manifest.get("rows")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("complete") is not True
        or not isinstance(rows, int)
        or rows < 0
        or verified_artifact(path, manifest.get("canonical_csv")) is None
    ):
        return None
    return rows


def artifact_row_counts(paths: list[Path]) -> dict[tuple[int, int, int, int], int]:
    """Index verified counts by physical file so hard links share metadata."""
    counts: dict[tuple[int, int, int, int], int] = {}
    for path in paths:
        if not path.is_file() or path.suffix.lower() != ".csv":
            continue
        rows = companion_row_count(path)
        if rows is not None:
            counts[file_identity(path)] = rows
    return counts


def artifact_record(
    path: Path,
    *,
    compute_hash: bool,
    root: Path,
    row_counts: dict[tuple[int, int, int, int], int] | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return record
    record["bytes"] = path.stat().st_size
    if path.suffix.lower() == ".csv":
        identity = file_identity(path)
        rows = (row_counts or {}).get(identity)
        if rows is None:
            rows = count_csv_rows(path)
            if row_counts is not None:
                row_counts[identity] = rows
        record["rows"] = rows
    if compute_hash:
        record["sha256"] = cached_sha256(path, root)
    return record


def stage_manifest_record(path: Path, root: Path | None = None) -> dict[str, object]:
    payload = require_json_manifest(path, stage=path.stem)
    return {
        "path": str(path),
        "sha256": cached_sha256(path, root) if root is not None else sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "complete": True,
    }


def count_bible_source_rows(path: Path) -> int:
    if not path.exists() or path.suffix.lower() != ".csv":
        return 0
    count = 0
    exact_repo_components = {repo.lower() for repo in EXACT_BIBLE_REPOS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "source" not in reader.fieldnames:
            return 0
        for row in reader:
            source_parts = [
                part.strip().lower()
                for part in (row.get("source") or "").replace("\\", "/").split("/")
                if part.strip()
            ]
            if any(part in exact_repo_components for part in source_parts):
                count += 1
    return count


def validate_no_bible_sources(paths: BuildPaths, final_corpus_dir: Path) -> None:
    files = [*sorted(paths.processed_dir.glob("*_processed.csv")), *final_corpus_dir.glob("big_corpus*.csv")]
    failures = {
        str(path): rows
        for path in files
        if (rows := count_bible_source_rows(path)) > 0
    }
    if failures:
        details = ", ".join(
            f"{path}: {rows}" for path, rows in sorted(failures.items())
        )
        raise SystemExit(
            "--exclude-bible validation failed; Bible source rows remain: "
            f"{details}"
        )


def build_artifact_manifest(
    args: argparse.Namespace,
    final_corpus_dir: Path,
    paths: BuildPaths,
) -> dict[str, dict[str, object]]:
    if args.dry_run:
        return {}
    artifact_paths = {
        "big_corpus_en": final_corpus_dir / "big_corpus_en.csv",
        "big_corpus_zh": final_corpus_dir / "big_corpus_zh.csv",
        "big_corpus_en_in_domain_hard": final_corpus_dir
        / "big_corpus_en_in_domain_hard.csv",
        "big_corpus_zh_in_domain_hard": final_corpus_dir
        / "big_corpus_zh_in_domain_hard.csv",
    }
    for split_dir in (
        paths.split_root / "splits_en_v1",
        paths.split_root / "splits_zh_v1",
    ):
        if split_dir.exists():
            for path in sorted(split_dir.glob("*.csv")):
                artifact_paths[f"{split_dir.name}/{path.name}"] = path

    row_counts = artifact_row_counts(list(artifact_paths.values()))
    return {
        name: artifact_record(
            path,
            compute_hash=not args.skip_artifact_checksums,
            root=paths.root,
            row_counts=row_counts,
        )
        for name, path in artifact_paths.items()
    }


def package_training_provenance(paths: BuildPaths, final_corpus_dir: Path) -> Path:
    provenance_dir = final_corpus_dir / "provenance"
    remove_path(provenance_dir)
    provenance_dir.mkdir(parents=True)
    sources = {
        "mt_build_manifest.json": paths.manifest_path,
        "source_repository_snapshot.json": paths.source_snapshot_path,
        "corpus_pipeline.json": PIPELINE_CONFIG_PATH,
        "mt_standardization.json": PROJECT_ROOT / "config" / "mt_standardization.json",
        "aggregate_manifest.json": final_corpus_dir / "aggregate_manifest.json",
        "split_en_in_domain_hard.json": paths.split_root
        / "splits_en_v1"
        / "report_in_domain_hard.json",
        "split_zh_in_domain_hard.json": paths.split_root
        / "splits_zh_v1"
        / "report_in_domain_hard.json",
        "validate_en_in_domain_hard.json": paths.split_root
        / "splits_en_v1"
        / "validation_in_domain_hard.json",
        "validate_zh_in_domain_hard.json": paths.split_root
        / "splits_zh_v1"
        / "validation_in_domain_hard.json",
        "exposure_en_in_domain_hard.json": paths.split_root
        / "splits_en_v1"
        / "exposure_in_domain_hard.json",
        "exposure_zh_in_domain_hard.json": paths.split_root
        / "splits_zh_v1"
        / "exposure_in_domain_hard.json",
    }
    for prepared_dir in sorted(paths.root.glob("prepared_*")):
        language = prepared_dir.name.removeprefix("prepared_")
        for source_name, prefix in (
            ("_qc_manifest.json", "xml_preparation"),
            ("_mt_standard_manifest.json", "mt_standard"),
        ):
            source = prepared_dir / source_name
            if source.is_file():
                sources[f"{prefix}_{language}.json"] = source

    pivot_manifest = paths.processed_dir / "pivot" / "pivot_manifest.json"
    if pivot_manifest.is_file():
        sources["pivot_manifest.json"] = pivot_manifest
        pivot_payload = json.loads(pivot_manifest.read_text(encoding="utf-8"))
        for stats in pivot_payload.get("stats", []):
            for path_field, hash_field, label in (
                ("quarantine_path", "quarantine_sha256", "quarantine"),
                ("cache_conflict_path", "cache_conflict_sha256", "cache conflict"),
            ):
                value = str(stats.get(path_field) or "").strip()
                if not value:
                    continue
                artifact_path = Path(value)
                expected_hash = str(stats.get(hash_field) or "").strip()
                if not artifact_path.is_file():
                    raise SystemExit(
                        "Cannot package training provenance; missing pivot "
                        f"{label} ledger {artifact_path}"
                    )
                actual_hash = cached_sha256(artifact_path, paths.root)
                if not expected_hash or actual_hash != expected_hash:
                    raise SystemExit(
                        f"Cannot package training provenance; pivot {label} "
                        f"hash mismatch for {artifact_path}"
                    )
                sources[artifact_path.name] = artifact_path

    missing = [name for name, source in sources.items() if not source.is_file()]
    if missing:
        raise SystemExit(f"Cannot package training provenance; missing {missing}")

    artifacts: dict[str, dict[str, object]] = {}
    for name, source in sources.items():
        destination = provenance_dir / name
        shutil.copy2(source, destination)
        artifacts[name] = artifact_record(
            destination,
            compute_hash=True,
            root=paths.root,
        )
    bundle_manifest = provenance_dir / "bundle_manifest.json"
    atomic_write_json(
        bundle_manifest,
        {
            "schema_version": 1,
            "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
            "corpus_name": json.loads(
                paths.manifest_path.read_text(encoding="utf-8")
            ).get("corpus_name"),
            "artifacts": artifacts,
            "complete": True,
        },
    )
    return bundle_manifest


def dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "python-dotenv",
        "requests",
        "sacrebleu",
        "sentencepiece",
        "tame-mt",
        "torch",
        "transformers",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def write_manifest(
    args: argparse.Namespace,
    language_reports: list[dict[str, object]],
    final_corpus_dir: Path,
    paths: BuildPaths,
) -> None:
    artifacts = build_artifact_manifest(args, final_corpus_dir, paths)
    required_artifacts = {
        "big_corpus_en",
        "big_corpus_zh",
        "big_corpus_en_in_domain_hard",
        "big_corpus_zh_in_domain_hard",
    }
    missing_artifacts = sorted(
        name
        for name in required_artifacts
        if not artifacts.get(name, {}).get("exists")
    )
    stage_paths = {
        "source_repository_snapshot": paths.source_snapshot_path,
        "processed_aggregate": paths.processed_dir / "aggregate_manifest.json",
        "final_aggregate": final_corpus_dir / "aggregate_manifest.json",
        "split_en": paths.split_root
        / "splits_en_v1"
        / "report_in_domain_hard.json",
        "split_zh": paths.split_root
        / "splits_zh_v1"
        / "report_in_domain_hard.json",
        "validate_en": paths.split_root
        / "splits_en_v1"
        / "validation_in_domain_hard.json",
        "validate_zh": paths.split_root
        / "splits_zh_v1"
        / "validation_in_domain_hard.json",
        "exposure_en": paths.split_root
        / "splits_en_v1"
        / "exposure_in_domain_hard.json",
        "exposure_zh": paths.split_root
        / "splits_zh_v1"
        / "exposure_in_domain_hard.json",
    }
    if args.with_pivot:
        stage_paths["pivot"] = paths.processed_dir / "pivot" / "pivot_manifest.json"
    stage_manifests: dict[str, dict[str, object]] = {}
    missing_stages: list[str] = []
    if not args.dry_run:
        for name, path in stage_paths.items():
            if path.is_file():
                try:
                    stage_manifests[name] = stage_manifest_record(path, paths.root)
                except SystemExit:
                    missing_stages.append(name)
            else:
                missing_stages.append(name)

    complete = not missing_artifacts and not missing_stages
    manifest = {
        "schema_version": 3,
        "pipeline_version": PIPELINE_CONFIG["pipeline_version"],
        "created_at": utc_now(),
        "corpus_name": args.corpus_name,
        "command": [sys.executable, *sys.argv],
        "repository": git_state(PROJECT_ROOT),
        "pipeline_config": {
            "path": str(PIPELINE_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PIPELINE_CONFIG_PATH),
        },
        "mt_standardization": {
            "id": PIPELINE_CONFIG["mt_standardization"]["profile_id"],
            "sha256": sha256_file(args.mt_standard_profile),
            "namespace": PIPELINE_CONFIG["mt_standardization"]["namespace"],
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "dependencies": dependency_versions(),
        },
        "languages": language_reports,
        "settings": {
            "public": args.public,
            "qc_revision": args.qc_revision,
            "units": args.units,
            "language_workers": args.language_workers,
            "analysis_workers": args.analysis_workers,
            "incremental_stage_cache": not args.no_stage_cache,
            "hard_split_ratios_by_target": PIPELINE_CONFIG["splits"][
                "ratios_by_target"
            ],
            "hard_split_minimum_eval_rows": {
                "test": args.min_test_rows,
                "validate": args.min_validate_rows,
            },
            "hard_split_character_ngram_jaccard_threshold": args.ngram_jaccard_threshold,
            "hard_split_source_ratio_tolerance": args.source_ratio_tolerance,
            "with_pivot": args.with_pivot,
            "pivot_read_cache_dirs": [
                str(path) for path in pivot_read_cache_dirs(args, paths)
            ],
            "keep_redactions": args.keep_redactions,
            "fresh_downloads": not args.keep_downloaded,
            "source_repository_snapshot": str(paths.source_snapshot_path),
            "mt_standardization_profile": str(args.mt_standard_profile),
            "keep_build_output": args.keep_build_output,
            "allow_dirty_repository": args.allow_dirty_repository,
            "fetch_workers": args.fetch_workers,
            "fetch_download_retries": args.fetch_download_retries,
            "fetch_retry_base_sleep": args.fetch_retry_base_sleep,
            "fetch_retry_max_sleep": args.fetch_retry_max_sleep,
            "allow_download_failures": args.allow_download_failures,
            "exclude_bible": args.exclude_bible,
            "exclude_bible_exact_repos": (
                list(EXACT_BIBLE_REPOS) if args.exclude_bible else []
            ),
            "exclude_repo_patterns": args.exclude_repo_pattern,
            "exclude_path_patterns": args.exclude_path_pattern,
        },
        "outputs": {
            "root": str(paths.root),
            "raw_corpora": str(paths.raw_dir),
            "processed_corpora": str(paths.processed_dir),
            "final_corpus_dir": str(final_corpus_dir),
            "experiment_splits": str(paths.split_root),
        },
        "stage_manifests": stage_manifests,
        "artifacts": artifacts,
        "release_gate": {
            "required_artifacts": sorted(required_artifacts),
            "missing_artifacts": missing_artifacts,
            "missing_or_incomplete_stages": missing_stages,
        },
        "complete": complete,
    }
    if args.dry_run:
        print(f"[plan]  Write build manifest: {paths.manifest_path}")
        return
    atomic_write_json(paths.manifest_path, manifest)
    print(f"Build manifest: {paths.manifest_path}")
    if not complete:
        raise SystemExit(
            f"Corpus build did not pass release gates; inspect {paths.manifest_path}"
        )
    try:
        bundle_manifest = package_training_provenance(paths, final_corpus_dir)
    except (Exception, SystemExit) as exc:
        manifest["complete"] = False
        manifest["release_gate"]["provenance_bundle_error"] = str(exc)
        atomic_write_json(paths.manifest_path, manifest)
        raise
    print(f"Training provenance: {bundle_manifest}")
