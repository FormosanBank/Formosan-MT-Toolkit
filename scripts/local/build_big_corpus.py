#!/usr/bin/env python3
"""
Create multilingual aggregate corpora without deduplication.

Supported input layouts:

1. Pairwise processed corpora in a named build's `processed_corpora/`, for example:
   - `ami_en_processed.csv`
   - `ami_zh_processed.csv`

   Expected pairwise headers:
   - `<lang_code>,english,source,kindOf,dialect,row_type,split,...`
   - `<lang_code>,chinese,source,kindOf,dialect,row_type,split,...`

2. Already-aggregated multilingual corpora, for example:
   - `big_corpus_en.csv`
   - `big_corpus_zh.csv`
   - `big_corpus_en_pivot.csv`
   - `big_corpus_zh_pivot.csv`

   Expected aggregate headers:
   - `lang_code,formosan_sentence,english_sentence,source,dialect,row_type,split,...`
   - `lang_code,formosan_sentence,chinese_sentence,source,dialect,row_type,split,...`

Outputs:

1. `big_corpus_en.csv`
   `lang_code | formosan_sentence | english_sentence | source | dialect | row_type | split`

2. `big_corpus_zh.csv`
   `lang_code | formosan_sentence | chinese_sentence | source | dialect | row_type | split`

3. `big_corpus_combined.csv`
   `lang_code | formosan_sentence | chinese_sentence | english_sentence | source | dialect | row_type | split`

The combined corpus is Chinese-anchored because the tokenizer/model setup
script requires `chinese_sentence` and optionally uses `english_sentence` when
it is present.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

Pair = Tuple[str, str, str, str, str, str, str, str]

ENGLISH_COLUMNS = ("english_sentence", "english")
CHINESE_COLUMNS = ("chinese_sentence", "chinese")
AGGREGATE_REQUIRED = ("lang_code", "formosan_sentence")


def normalize_column(name: str) -> str:
    return str(name or "").strip().lower()


def clean_cell(value: object) -> str:
    return str(value or "").strip()


def clean_row_type(value: object) -> str:
    text = clean_cell(value).lower()
    if text in {"lexeme", "morpheme"}:
        return "lexeme"
    if text == "sentence":
        return "sentence"
    return "unknown"


def first_present(columns: Dict[str, str], candidates: Iterable[str]) -> Optional[str]:
    for candidate in candidates:
        actual = columns.get(candidate)
        if actual:
            return actual
    return None


def detect_pairwise_target(csv_file: Path) -> Optional[str]:
    parts = csv_file.stem.split("_")
    if len(parts) < 2:
        return None
    target = parts[1].lower()
    if target not in {"en", "zh"}:
        return None
    return target


def detect_pairwise_lang(csv_file: Path) -> Optional[str]:
    parts = csv_file.stem.split("_")
    if not parts:
        return None
    lang_code = parts[0].strip().lower()
    return lang_code or None


def add_pair(data: Dict[str, List[Pair]], lang_code: str, pair: Pair) -> None:
    data.setdefault(lang_code, []).append(pair)


def ingest_aggregate_rows(
    csv_file: Path,
    reader: csv.DictReader,
    columns: Dict[str, str],
    data_en: Dict[str, List[Pair]],
    data_zh: Dict[str, List[Pair]],
) -> tuple[int, int]:
    lang_col = columns["lang_code"]
    formosan_col = columns["formosan_sentence"]
    english_col = first_present(columns, ENGLISH_COLUMNS)
    chinese_col = first_present(columns, CHINESE_COLUMNS)
    source_col = columns.get("source")
    dialect_col = columns.get("dialect")
    row_type_col = columns.get("row_type")
    split_col = columns.get("split")
    pivot_origin_col = columns.get("pivot_origin")
    pivot_direction_col = columns.get("pivot_direction")

    en_rows = 0
    zh_rows = 0
    skipped_missing = 0

    for row in reader:
        lang_code = clean_cell(row.get(lang_col))
        formosan = clean_cell(row.get(formosan_col))
        if not lang_code or not formosan:
            skipped_missing += 1
            continue

        source = clean_cell(row.get(source_col)) if source_col else ""
        dialect = clean_cell(row.get(dialect_col)) if dialect_col else ""
        row_type = clean_row_type(row.get(row_type_col)) if row_type_col else "unknown"
        split = clean_cell(row.get(split_col)) if split_col else ""
        pivot_origin = clean_cell(row.get(pivot_origin_col)) if pivot_origin_col else "original"
        pivot_direction = clean_cell(row.get(pivot_direction_col)) if pivot_direction_col else ""

        if english_col:
            english = clean_cell(row.get(english_col))
            if english:
                add_pair(data_en, lang_code, (formosan, english, source, dialect, row_type, split, pivot_origin, pivot_direction))
                en_rows += 1
            elif chinese_col is None:
                skipped_missing += 1

        if chinese_col:
            chinese = clean_cell(row.get(chinese_col))
            if chinese:
                add_pair(data_zh, lang_code, (formosan, chinese, source, dialect, row_type, split, pivot_origin, pivot_direction))
                zh_rows += 1
            elif english_col is None:
                skipped_missing += 1

    print(
        f"📖 Reading {csv_file.name} as aggregate corpus..."
        f"  EN rows: {en_rows:,} | ZH rows: {zh_rows:,} | skipped empty: {skipped_missing:,}"
    )
    return en_rows, zh_rows


def ingest_pairwise_rows(
    csv_file: Path,
    reader: csv.DictReader,
    columns: Dict[str, str],
    data_en: Dict[str, List[Pair]],
    data_zh: Dict[str, List[Pair]],
) -> tuple[int, int]:
    lang_code = detect_pairwise_lang(csv_file)
    target_lang = detect_pairwise_target(csv_file)
    if not lang_code or not target_lang:
        print(f"⚠️  Skipping {csv_file.name}: could not infer pairwise language codes.")
        return 0, 0

    formosan_col = columns.get(lang_code)
    if target_lang == "en":
        target_col = first_present(columns, ENGLISH_COLUMNS)
    else:
        target_col = first_present(columns, CHINESE_COLUMNS)

    if not formosan_col or not target_col:
        print(f"⚠️  Skipping {csv_file.name}: pairwise columns do not match expected schema.")
        return 0, 0

    source_col = columns.get("source")
    dialect_col = columns.get("dialect")
    row_type_col = columns.get("row_type")
    split_col = columns.get("split")

    count = 0
    skipped_missing = 0
    for row in reader:
        formosan = clean_cell(row.get(formosan_col))
        target = clean_cell(row.get(target_col))
        if not formosan or not target:
            skipped_missing += 1
            continue

        source = clean_cell(row.get(source_col)) if source_col else ""
        dialect = clean_cell(row.get(dialect_col)) if dialect_col else ""
        row_type = clean_row_type(row.get(row_type_col)) if row_type_col else "unknown"
        split = clean_cell(row.get(split_col)) if split_col else ""
        pair = (formosan, target, source, dialect, row_type, split, "original", "")

        if target_lang == "en":
            add_pair(data_en, lang_code, pair)
        else:
            add_pair(data_zh, lang_code, pair)
        count += 1

    direction = "EN" if target_lang == "en" else "ZH"
    print(
        f"📖 Reading {csv_file.name} as pairwise corpus..."
        f"  {direction} rows: {count:,} | skipped empty: {skipped_missing:,}"
    )
    return (count, 0) if target_lang == "en" else (0, count)


def read_csv_files(
    directory: Path,
    skip_names: set[str],
) -> Tuple[Dict[str, List[Pair]], Dict[str, List[Pair]]]:
    """
    Read all supported CSV files in a directory and return two dicts:
      - data_en[lang_code] -> list of (formosan, english, source, dialect, row_type, split)
      - data_zh[lang_code] -> list of (formosan, chinese, source, dialect, row_type, split)
    """
    data_en: Dict[str, List[Pair]] = {}
    data_zh: Dict[str, List[Pair]] = {}

    for csv_file in sorted(directory.glob("*.csv")):
        if csv_file.name in skip_names:
            continue

        try:
            with open(csv_file, "r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    print(f"⚠️  Skipping {csv_file.name}: empty or headerless CSV.")
                    continue

                columns = {
                    normalize_column(fieldname): fieldname
                    for fieldname in reader.fieldnames
                    if fieldname is not None
                }

                if all(required in columns for required in AGGREGATE_REQUIRED):
                    ingest_aggregate_rows(csv_file, reader, columns, data_en, data_zh)
                    continue

                ingest_pairwise_rows(csv_file, reader, columns, data_en, data_zh)

        except Exception as exc:
            print(f"⚠️  Error reading {csv_file.name}: {exc}")
            continue

    return data_en, data_zh


def write_pair_corpus(
    data: Dict[str, List[Pair]],
    output_file: Path,
    target_colname: str,
) -> None:
    """
    Write a lang-specific pair corpus:
      columns: lang_code | formosan_sentence | {target_colname} | source | dialect | row_type | split
    (No deduping: writes every input row.)
    """
    total = 0
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lang_code", "formosan_sentence", target_colname, "source", "dialect", "row_type", "split", "pivot_origin", "pivot_direction"])

        for lang_code in sorted(data.keys()):
            rows = data[lang_code]
            print(f"🔄 Processing {lang_code} ({target_colname})...")
            for formosan, target, source, dialect, row_type, split, pivot_origin, pivot_direction in rows:
                writer.writerow([lang_code, formosan, target, source, dialect, row_type, split, pivot_origin, pivot_direction])
                total += 1
            print(f"✅ {lang_code}: {len(rows):,} pairs")

    print(f"\n🎉 Wrote {output_file}  |  📊 Total pairs: {total:,}\n")


def match_keys(
    lang_code: str,
    formosan: str,
    source: str,
    dialect: str,
    split: str,
) -> list[tuple[str, tuple[str, ...]]]:
    return [
        ("exact", (lang_code, formosan, source, dialect, split)),
        ("source", (lang_code, formosan, source)),
        ("dialect", (lang_code, formosan, dialect)),
        ("split", (lang_code, formosan, split)),
        ("formosan", (lang_code, formosan)),
    ]


def build_english_lookups(data_en: Dict[str, List[Pair]]) -> dict[str, dict[tuple[str, ...], str]]:
    lookups: dict[str, dict[tuple[str, ...], str]] = {
        "exact": {},
        "source": {},
        "dialect": {},
        "split": {},
        "formosan": {},
    }

    for lang_code in sorted(data_en.keys()):
        for formosan, english, source, dialect, _row_type, split, _pivot_origin, _pivot_direction in data_en[lang_code]:
            if not english:
                continue
            for level, key in match_keys(lang_code, formosan, source, dialect, split):
                lookups[level].setdefault(key, english)

    return lookups


def write_combined_corpus(
    data_zh: Dict[str, List[Pair]],
    data_en: Dict[str, List[Pair]],
    output_file: Path,
) -> None:
    """
    Write a Chinese-anchored combined corpus compatible with setup_formosan_nllb200.py:
      lang_code | formosan_sentence | chinese_sentence | english_sentence | source | dialect | row_type | split
    """
    lookups = build_english_lookups(data_en)
    matched_by = Counter()
    unmatched = 0
    total = 0

    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "lang_code",
                "formosan_sentence",
                "chinese_sentence",
                "english_sentence",
                "source",
                "dialect",
                "row_type",
                "split",
                "pivot_origin",
                "pivot_direction",
            ]
        )

        for lang_code in sorted(data_zh.keys()):
            rows = data_zh[lang_code]
            print(f"🔄 Processing {lang_code} (combined tokenizer corpus)...")
            for formosan, chinese, source, dialect, row_type, split, pivot_origin, pivot_direction in rows:
                english = ""
                for level, key in match_keys(lang_code, formosan, source, dialect, split):
                    english = lookups[level].get(key, "")
                    if english:
                        matched_by[level] += 1
                        break
                if not english:
                    unmatched += 1

                writer.writerow([lang_code, formosan, chinese, english, source, dialect, row_type, split, pivot_origin, pivot_direction])
                total += 1

            print(f"✅ {lang_code}: {len(rows):,} rows")

    print(f"\n🎉 Wrote {output_file}  |  📊 Total rows: {total:,}")
    print(
        "   English matches: "
        + ", ".join(
            f"{level}={matched_by[level]:,}"
            for level in ["exact", "source", "dialect", "split", "formosan"]
        )
        + f", unmatched={unmatched:,}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir.parent

    parser = argparse.ArgumentParser(
        description="Build multilingual aggregate corpora from processed pairwise or aggregate CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=default_root)
    parser.add_argument("--output-dir", type=Path, default=default_root)
    parser.add_argument("--output-en-name", default="big_corpus_en.csv")
    parser.add_argument("--output-zh-name", default="big_corpus_zh.csv")
    parser.add_argument("--output-combined-name", default="big_corpus_combined.csv")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    out_en = output_dir / args.output_en_name
    out_zh = output_dir / args.output_zh_name
    out_combined = output_dir / args.output_combined_name

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    skip_names = {
        "summary_stats.csv",
        out_en.name,
        out_zh.name,
        out_combined.name,
    }

    print("🚀 Building separate corpora for English and Chinese (no dedupe)...\n")
    print(f"📂 Input dir:  {input_dir}")
    print(f"📝 Output dir: {output_dir}\n")

    data_en, data_zh = read_csv_files(input_dir, skip_names)

    if not data_en and not data_zh:
        print("❌ No supported corpus CSV files found!")
        return

    if data_en:
        print(f"📋 EN languages: {', '.join(sorted(data_en.keys()))}")
        write_pair_corpus(data_en, out_en, "english_sentence")
    else:
        print("ℹ️  No English rows found.")

    if data_zh:
        print(f"📋 ZH languages: {', '.join(sorted(data_zh.keys()))}")
        write_pair_corpus(data_zh, out_zh, "chinese_sentence")
        write_combined_corpus(data_zh, data_en, out_combined)
    else:
        print("ℹ️  No Chinese rows found.")


if __name__ == "__main__":
    main()
