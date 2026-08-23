#!/usr/bin/env python3
"""
Build DeepL-pivoted multilingual corpora.

The script creates:

1. A Chinese-target pivot corpus:
   original big_corpus_zh.csv rows
   + Formosan/English rows from big_corpus_en.csv with English translated to Chinese.

2. An English-target pivot corpus:
   original big_corpus_en.csv rows
   + Formosan/Chinese rows from big_corpus_zh.csv with Chinese translated to English.

Successful DeepL responses are appended to per-direction JSONL caches immediately.
If a run is interrupted or hits quota, rerun the same command and cached translations
will be reused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from dotenv import load_dotenv
from pipeline_common import sha256_file, utc_now
from pivot_cache import (
    PROVIDER,
    append_jsonl,
    load_cache,
    load_cache_chain,
    make_cache_key,
    write_jsonl_atomic,
)
from pivot_corpus import (
    BASE_COLUMNS,
    formosan_key,
    frame_records,
    load_source_corpora,
    pivot_candidate_reason,
    read_corpus,
    target_formosan_keys,
    validate_columns,
    validate_mt_standard_contract,
)
from pivot_deepl import (
    DeepLClient,
    DeepLFatalError,
    DeepLKey,
    DeepLQuotaExceeded,
    DeepLRuntimeError,
    batch_jobs,
    choose_deepl_api_base,
    discover_api_key_envs,
    load_deepl_keys,
    parse_api_key_envs,
    read_deepl_usage_for_key,
    request_body_size,
)
from pivot_output import synthetic_row, write_manifest, write_pivot_output
from pivot_types import (
    CharBudget,
    Direction,
    DirectionStats,
    OutputBuildResult,
    TranslationJob,
)
from tqdm import tqdm

__all__ = [
    "CharBudget",
    "DeepLClient",
    "DeepLFatalError",
    "DeepLKey",
    "DeepLQuotaExceeded",
    "DeepLRuntimeError",
    "Direction",
    "DirectionStats",
    "OutputBuildResult",
    "choose_deepl_api_base",
    "discover_api_key_envs",
    "load_cache",
    "load_cache_chain",
    "load_deepl_keys",
    "make_cache_key",
    "parse_api_key_envs",
    "pivot_candidate_reason",
    "synthetic_row",
    "write_pivot_output",
]

DEEPL_MAX_TEXTS_PER_REQUEST = 50
DEEPL_MAX_REQUEST_BYTES = 128 * 1024
DEFAULT_SAFE_REQUEST_BYTES = 120 * 1024


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for parent in [cur, *cur.parents]:
        if (parent / "README.md").exists() and (parent / "config" / "corpus_pipeline.json").exists():
            return parent
    raise RuntimeError(f"Could not locate Formosan-MT-Toolkit project root from {start}")


def now_iso() -> str:
    return utc_now()


def candidate_jobs(
    df: pd.DataFrame,
    *,
    text_col: str,
    direction: Direction,
    cache: dict[str, dict[str, Any]],
    force: bool,
    target_keys: set[tuple[str, str]],
    skip_target_overlaps: bool,
    split_sentences: str,
    preserve_formatting: bool,
    model_type: Optional[str],
    budget: CharBudget,
    stats: DirectionStats,
) -> list[TranslationJob]:
    jobs: list[TranslationJob] = []
    seen: set[str] = set()
    seen_overlap: set[tuple[str, str]] = set()

    candidate_columns = list(
        dict.fromkeys(
            [
                "row_type",
                "mt_eval_eligible",
                "mt_normalization_confidence",
                "formosan_sentence",
                direction.source_text_col,
                "translation_kind",
            ]
        )
    )
    for row in frame_records(df, candidate_columns):
        exclusion = pivot_candidate_reason(row, direction)
        if exclusion:
            stats.ineligible_source_rows += 1
            stats.candidate_exclusion_counts[exclusion] = (
                stats.candidate_exclusion_counts.get(exclusion, 0) + 1
            )
            continue
        stats.candidate_rows += 1
        f_key = formosan_key(row)
        if skip_target_overlaps and f_key[0] and f_key[1] and f_key in target_keys:
            stats.target_overlap_rows_skipped += 1
            seen_overlap.add(f_key)
            continue

        text = str(row.get(text_col, "")).strip()
        if not text:
            stats.empty_source_rows += 1
            continue

        key = make_cache_key(
            provider=PROVIDER,
            source_lang=direction.deepl_source_lang,
            target_lang=direction.deepl_target_lang,
            text=text,
            split_sentences=split_sentences,
            preserve_formatting=preserve_formatting,
            model_type=model_type,
        )

        if key in seen:
            continue
        seen.add(key)

        if key in cache and not force:
            continue

        chars = len(text)
        if budget.remaining is not None and chars > budget.remaining:
            stats.deferred_by_budget_unique += 1
            stats.deferred_by_budget_chars += chars
            continue
        jobs.append(TranslationJob(key=key, text=text, chars=chars))
        budget.take(chars)

    if stats.deferred_by_budget_unique:
        stats.stopped_reason = "local character budget exhausted; deferred remaining missing translations"
    stats.target_overlap_unique_skipped = len(seen_overlap)
    return jobs


def translate_direction(
    direction: Direction,
    *,
    args: argparse.Namespace,
    client: Optional[DeepLClient],
    budget: CharBudget,
    source_df: pd.DataFrame | None = None,
    target_df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[str, Any]], DirectionStats]:
    stats = DirectionStats(direction=direction.name)
    if source_df is None:
        source_df = read_corpus(direction.source_path, f"{direction.name} source corpus")
    if target_df is None:
        target_df = read_corpus(direction.original_target_path, f"{direction.name} target corpus")
    validate_columns(source_df, direction.source_path, [*BASE_COLUMNS, direction.source_text_col])
    validate_columns(target_df, direction.original_target_path, [*BASE_COLUMNS, direction.target_text_col])
    source_profile = validate_mt_standard_contract(source_df, direction.source_path)
    target_profile = validate_mt_standard_contract(target_df, direction.original_target_path)
    if source_profile != target_profile:
        raise SystemExit(
            f"{direction.name} source and target corpora use different MT standard profiles"
        )
    stats.source_rows = len(source_df)
    stats.original_rows = len(target_df)
    target_keys = target_formosan_keys(target_df)

    read_cache_paths = [cache_dir / direction.cache_filename for cache_dir in getattr(args, "read_cache_dir", [])]
    cache_path = args.cache_dir / direction.cache_filename
    error_path = args.cache_dir / direction.cache_filename.replace(".jsonl", ".errors.jsonl")
    cache, cache_conflicts = load_cache_chain([*read_cache_paths, cache_path])
    stats.cache_path = str(cache_path)
    stats.read_cache_paths = [str(path) for path in read_cache_paths]
    stats.cached_unique_before = len(cache)
    stats.cache_conflicts = len(cache_conflicts)
    conflict_path = args.out_dir / f"pivot_cache_conflicts_{direction.name}.jsonl"
    if cache_conflicts:
        print(
            f"{direction.name}: {len(cache_conflicts):,} cache conflicts; "
            f"using the later, higher-priority cache layer"
        )
        if not args.dry_run:
            write_jsonl_atomic(conflict_path, cache_conflicts)
            stats.cache_conflict_path = str(conflict_path)
            stats.cache_conflict_sha256 = sha256_file(conflict_path)
    elif not args.dry_run:
        conflict_path.unlink(missing_ok=True)

    jobs = candidate_jobs(
        source_df,
        text_col=direction.source_text_col,
        direction=direction,
        cache=cache,
        force=args.force,
        target_keys=target_keys,
        skip_target_overlaps=args.skip_target_overlaps,
        split_sentences=args.split_sentences,
        preserve_formatting=args.preserve_formatting,
        model_type=args.model_type,
        budget=budget,
        stats=stats,
    )
    stats.missing_unique_before = len(jobs)

    planned_chars = sum(job.chars for job in jobs)
    deferred = (
        f", {stats.deferred_by_budget_unique:,} deferred by budget ({stats.deferred_by_budget_chars:,} chars)"
        if stats.deferred_by_budget_unique
        else ""
    )
    print(
        f"{direction.name}: {stats.candidate_rows:,} candidate rows, "
        f"{stats.ineligible_source_rows:,} ineligible rows, "
        f"{stats.target_overlap_rows_skipped:,} target-overlap rows skipped, "
        f"{stats.cached_unique_before:,} cached unique translations, "
        f"{len(jobs):,} missing unique translations, {planned_chars:,} chars planned"
        f"{deferred}"
    )

    if args.dry_run or args.skip_translation:
        return cache, stats

    if client is None:
        raise SystemExit("DeepL client is not configured.")

    pbar = tqdm(total=len(jobs), desc=f"DeepL {direction.name}", unit="text")
    try:
        for batch in batch_jobs(
            jobs,
            max_texts=args.batch_size,
            max_request_bytes=args.max_request_bytes,
            source_lang=direction.deepl_source_lang,
            target_lang=direction.deepl_target_lang,
            split_sentences=args.split_sentences,
            preserve_formatting=args.preserve_formatting,
            model_type=args.model_type,
        ):
            oversized = [
                job
                for job in batch
                if request_body_size(
                    [job.text],
                    source_lang=direction.deepl_source_lang,
                    target_lang=direction.deepl_target_lang,
                    split_sentences=args.split_sentences,
                    preserve_formatting=args.preserve_formatting,
                    model_type=args.model_type,
                )
                > args.max_request_bytes
            ]
            if oversized:
                records = [
                    {
                        "created_at": now_iso(),
                        "direction": direction.name,
                        "error": "single text exceeds configured request byte limit",
                        "key": job.key,
                        "source_lang": direction.deepl_source_lang,
                        "target_lang": direction.deepl_target_lang,
                        "text": job.text,
                        "chars": job.chars,
                    }
                    for job in oversized
                ]
                append_jsonl(error_path, records)
                stats.skipped_over_request_limit += len(oversized)
                stats.errors += len(oversized)
                pbar.update(len(oversized))
                batch = [job for job in batch if job not in oversized]
                if not batch:
                    continue

            texts = [job.text for job in batch]
            try:
                translations = client.translate(
                    texts,
                    source_lang=direction.deepl_source_lang,
                    target_lang=direction.deepl_target_lang,
                    split_sentences=args.split_sentences,
                    preserve_formatting=args.preserve_formatting,
                    model_type=args.model_type,
                )
            except DeepLQuotaExceeded as exc:
                stats.stopped_reason = "DeepL quota exceeded"
                append_jsonl(
                    error_path,
                    [
                        {
                            "created_at": now_iso(),
                            "direction": direction.name,
                            "error": str(exc),
                            "batch_size": len(batch),
                            "batch_chars": sum(job.chars for job in batch),
                        }
                    ],
                )
                break

            if len(translations) != len(batch):
                raise DeepLFatalError(f"DeepL returned {len(translations)} translations for a batch of {len(batch)}")

            records: list[dict[str, Any]] = []
            for job, translated in zip(batch, translations, strict=True):
                translation_text = str(translated.get("text", ""))
                record = {
                    "created_at": now_iso(),
                    "provider": PROVIDER,
                    "direction": direction.name,
                    "key": job.key,
                    "source_lang": direction.deepl_source_lang,
                    "target_lang": direction.deepl_target_lang,
                    "text": job.text,
                    "translation": translation_text,
                    "detected_source_language": translated.get("detected_source_language"),
                    "model_type_used": translated.get("model_type_used"),
                    "api_key_env": client.active_env_name,
                    "chars": job.chars,
                    "split_sentences": args.split_sentences,
                    "preserve_formatting": args.preserve_formatting,
                    "model_type_requested": args.model_type,
                }
                records.append(record)
                cache[job.key] = record

            append_jsonl(cache_path, records)
            batch_chars = sum(job.chars for job in batch)
            stats.translated_unique += len(batch)
            stats.translated_chars += batch_chars
            pbar.update(len(batch))
    finally:
        pbar.close()

    return cache, stats


def build_directions(args: argparse.Namespace) -> dict[str, Direction]:
    return {
        "en2zh": Direction(
            name="en2zh",
            source_path=args.big_corpus_en,
            original_target_path=args.big_corpus_zh,
            source_text_col="english_sentence",
            target_text_col="chinese_sentence",
            source_language="english",
            deepl_source_lang=args.source_en,
            deepl_target_lang=args.target_zh,
            output_filename="big_corpus_zh_pivot.csv",
            cache_filename="deepl_en_to_zh.jsonl",
        ),
        "zh2en": Direction(
            name="zh2en",
            source_path=args.big_corpus_zh,
            original_target_path=args.big_corpus_en,
            source_text_col="chinese_sentence",
            target_text_col="english_sentence",
            source_language="chinese",
            deepl_source_lang=args.source_zh,
            deepl_target_lang=args.target_en,
            output_filename="big_corpus_en_pivot.csv",
            cache_filename="deepl_zh_to_en.jsonl",
        ),
    }


def parse_directions(raw: str) -> list[str]:
    raw = (raw or "both").strip().lower()
    if raw == "both":
        return ["en2zh", "zh2en"]
    directions = [part.strip() for part in raw.split(",") if part.strip()]
    valid = {"en2zh", "zh2en"}
    bad = set(directions) - valid
    if bad:
        raise SystemExit(f"Unsupported direction(s): {sorted(bad)}. Use both, en2zh, zh2en.")
    return directions


def configure_arg_parser(project_root: Path) -> argparse.ArgumentParser:
    processed = project_root / "processed_corpora"
    ap = argparse.ArgumentParser(
        description="Create DeepL-pivoted Formosan multilingual corpora.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--big-corpus-en", type=Path, default=processed / "big_corpus_en.csv")
    ap.add_argument("--big-corpus-zh", type=Path, default=processed / "big_corpus_zh.csv")
    ap.add_argument("--out-dir", type=Path, default=processed / "pivot")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Defaults to OUT_DIR/cache")
    ap.add_argument(
        "--read-cache-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Read existing DeepL cache records from this directory before the write cache. "
            "Can be repeated. New translations are still written only to --cache-dir."
        ),
    )

    ap.add_argument("--directions", default="both", help="both, en2zh, zh2en, or comma-separated values")
    ap.add_argument(
        "--splits",
        default="all",
        help="Compatibility option. Corpus pipeline v3 requires all rows before its single split.",
    )
    ap.add_argument(
        "--include-target-overlaps",
        dest="skip_target_overlaps",
        action="store_false",
        help=(
            "Also DeepL-translate source rows whose lang_code+formosan_sentence already exists "
            "in the target corpus. Default skips these rows to avoid billing and leakage risk."
        ),
    )
    ap.set_defaults(skip_target_overlaps=True)

    ap.add_argument(
        "--api-key-env",
        default="auto",
        help=(
            "Comma-separated environment variable names for DeepL keys, used in order. "
            "The default 'auto' discovers DEEPL_API_KEY and all DEEPL_API_KEY_N variables."
        ),
    )
    ap.add_argument("--api-base", default=None, help="Override DeepL base URL, e.g. https://api-free.deepl.com")
    ap.add_argument("--source-en", default="EN")
    ap.add_argument("--source-zh", default="ZH")
    ap.add_argument("--target-zh", default="ZH-HANT")
    ap.add_argument("--target-en", default="EN-US")
    ap.add_argument(
        "--model-type",
        default="prefer_quality_optimized",
        choices=["prefer_quality_optimized", "quality_optimized", "latency_optimized", "none"],
    )
    ap.add_argument("--split-sentences", default="0", choices=["0", "1", "nonewlines"])
    ap.add_argument("--no-preserve-formatting", dest="preserve_formatting", action="store_false")
    ap.set_defaults(preserve_formatting=True)

    ap.add_argument("--batch-size", type=int, default=DEEPL_MAX_TEXTS_PER_REQUEST)
    ap.add_argument("--max-request-bytes", type=int, default=DEFAULT_SAFE_REQUEST_BYTES)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--retry-backoff", type=float, default=2.0)
    ap.add_argument(
        "--max-source-chars",
        type=int,
        default=None,
        help="Local per-run character budget for source text sent to DeepL.",
    )
    ap.add_argument(
        "--respect-usage-limit",
        action="store_true",
        help="Read DeepL usage and cap this run to remaining characters when the API reports a limit.",
    )
    ap.add_argument("--reserve-chars", type=int, default=1000)

    ap.add_argument("--force", action="store_true", help="Retranslate even when a cache entry exists.")
    ap.add_argument("--skip-translation", action="store_true", help="Only build outputs from existing cache.")
    ap.add_argument("--dry-run", action="store_true", help="Plan the run without calling DeepL or writing outputs.")
    ap.add_argument("--no-write-output", action="store_true")
    ap.add_argument("--include-provenance", action="store_true", default=True)
    ap.add_argument("--minimal-schema", dest="include_provenance", action="store_false")
    ap.add_argument(
        "--no-dedupe",
        dest="dedupe",
        action="store_false",
        help="Diagnostic only. Production output deduplicates after original rows win.",
    )
    ap.set_defaults(dedupe=True)
    ap.add_argument("--quiet", action="store_true")
    return ap


def main() -> None:
    project_root = find_project_root(Path(__file__))
    load_dotenv(project_root / ".env")

    parser = configure_arg_parser(project_root)
    args = parser.parse_args()

    args.big_corpus_en = args.big_corpus_en.resolve()
    args.big_corpus_zh = args.big_corpus_zh.resolve()
    args.out_dir = args.out_dir.resolve()
    if args.cache_dir is None:
        args.cache_dir = args.out_dir / "cache"
    args.cache_dir = args.cache_dir.resolve()
    args.read_cache_dir = [path.resolve() for path in args.read_cache_dir]
    args.batch_size = max(1, min(int(args.batch_size), DEEPL_MAX_TEXTS_PER_REQUEST))
    args.max_request_bytes = max(1024, min(int(args.max_request_bytes), DEEPL_MAX_REQUEST_BYTES))
    if args.model_type == "none":
        args.model_type = None

    selected_direction_names = parse_directions(args.directions)
    if args.splits.strip().lower() not in {"all", "*"}:
        raise SystemExit("Corpus pipeline v3 requires --splits all; splitting occurs once after pivoting")
    directions = build_directions(args)
    selected_directions = [directions[name] for name in selected_direction_names]
    loaded_corpora = load_source_corpora(selected_directions)

    api_key_env_names = parse_api_key_envs(args.api_key_env)
    args.api_key_env_names = api_key_env_names
    deepl_keys = load_deepl_keys(api_key_env_names, args.api_base)
    client: Optional[DeepLClient] = None
    usage: Optional[dict[str, Any]] = None
    budget = CharBudget(args.max_source_chars)

    if not args.dry_run and not args.skip_translation:
        if not deepl_keys:
            raise SystemExit(
                f"Missing DeepL API key. Checked environment variable(s): {', '.join(api_key_env_names) or '(none)'}."
            )
        client = DeepLClient(
            keys=deepl_keys,
            timeout=args.timeout,
            max_retries=args.max_retries,
            retry_backoff=args.retry_backoff,
        )
        print(f"DeepL API keys loaded: {', '.join(k.env_name for k in deepl_keys)}")
        if args.respect_usage_limit:
            usages = [u for key in deepl_keys if (u := read_deepl_usage_for_key(key, args.timeout))]
            usable_total = 0
            usage = {"keys": usages, "reserve_chars_per_key": int(args.reserve_chars)}
            for item in usages:
                if "character_limit" not in item or "character_count" not in item:
                    continue
                remaining = int(item["character_limit"]) - int(item["character_count"]) - int(args.reserve_chars)
                usable_total += max(0, remaining)
            if budget.remaining is None:
                budget.remaining = usable_total
            else:
                budget.remaining = min(budget.remaining, usable_total)
            print(f"DeepL reported usable remaining characters across loaded keys: {budget.remaining:,}")

    all_stats: list[DirectionStats] = []
    caches: dict[str, dict[str, dict[str, Any]]] = {}

    print(f"Project root: {project_root}")
    print(f"Output dir:   {args.out_dir}")
    print(f"Cache dir:    {args.cache_dir}")
    if args.read_cache_dir:
        print("Read caches:  " + ", ".join(str(path) for path in args.read_cache_dir))
    print(f"Directions:   {', '.join(selected_direction_names)}")
    print(f"Splits:       {args.splits}")

    for direction in selected_directions:
        name = direction.name
        cache, stats = translate_direction(
            direction,
            args=args,
            client=client,
            budget=budget,
            source_df=loaded_corpora[direction.source_path].frame,
            target_df=loaded_corpora[direction.original_target_path].frame,
        )
        caches[name] = cache
        all_stats.append(stats)

    if not args.dry_run and not args.no_write_output:
        for stats in all_stats:
            direction = directions[stats.direction]
            result = write_pivot_output(
                direction,
                args=args,
                cache=caches[stats.direction],
                original_df=loaded_corpora[direction.original_target_path].frame,
                source_df=loaded_corpora[direction.source_path].frame,
            )
            stats.original_rows = result.original_rows
            stats.synthetic_rows_available = result.synthetic_rows_available
            stats.synthetic_rows_missing = result.synthetic_rows_missing
            stats.synthetic_rows_quarantined = (
                result.synthetic_rows_quarantined
            )
            stats.synthetic_rows_written = result.synthetic_rows_written
            stats.target_overlap_rows_skipped = max(
                stats.target_overlap_rows_skipped,
                result.target_overlap_rows_skipped,
            )
            stats.duplicate_rows_skipped = result.duplicate_rows_skipped
            stats.split_overrides = result.split_overrides
            stats.output_rows = result.output_rows
            stats.output_path = result.incomplete_path or str(args.out_dir / direction.output_filename)
            stats.quarantine_path = result.quarantine_path
            stats.quarantine_sha256 = result.quarantine_sha256

    manifest: Optional[Path] = None
    if not args.dry_run:
        manifest = write_manifest(
            args=args,
            direction_stats=all_stats,
            usage=usage,
            sources={
                "big_corpus_en": str(args.big_corpus_en),
                "big_corpus_zh": str(args.big_corpus_zh),
            },
            loaded_corpora=loaded_corpora,
        )

    print("\nDone.")
    for stats in all_stats:
        print(
            f"{stats.direction}: translated {stats.translated_unique:,} unique texts "
            f"({stats.translated_chars:,} chars); target-overlap rows skipped "
            f"{stats.target_overlap_rows_skipped:,}; synthetic rows written "
            f"{stats.synthetic_rows_written:,}; quality-quarantined "
            f"{stats.synthetic_rows_quarantined:,}; output rows "
            f"{stats.output_rows:,}"
        )
        if stats.stopped_reason:
            print(f"{stats.direction}: stopped early: {stats.stopped_reason}")
    if manifest is not None:
        print(f"Manifest: {manifest}")
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        if manifest_payload.get("complete") is not True:
            raise SystemExit(
                "DeepL pivot is incomplete. Caches were preserved, but finalized "
                f"outputs were not promoted; inspect {manifest} and rerun."
            )
    else:
        print("Dry run: no manifest or output files written.")


if __name__ == "__main__":
    main()
