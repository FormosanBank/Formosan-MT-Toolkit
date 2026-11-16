#!/usr/bin/env python3
"""Print size tables for a completed corpus build, optionally against another.

The numbers come from the build's own manifests, so the tables are instant and
always describe exactly what the release gates checked.

    python scripts/local/summarize_build.py corpus_builds/public_no_bible
    python scripts/local/summarize_build.py corpus_builds/public_no_bible \\
        --compare /backups/public_no_bible_august --sources
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TARGETS = (("English", "en"), ("Chinese", "zh"))
LANGUAGE_COLUMNS = (
    ("Human pairs", "human_rows"),
    ("Synthetic", "synthetic_rows"),
    ("Lexical", "lexical_rows"),
    ("Eval-eligible", "eligible_human_sentence_rows"),
    ("Train", "train_rows"),
    ("Validate", "validate_rows"),
    ("Test", "test_rows"),
)
SOURCE_COLUMNS = (
    ("Human pairs", "human_input_rows"),
    ("Synthetic", "synthetic_input_rows"),
    ("Eval-eligible", "eligible_human_rows"),
    ("Validate", "validate_rows"),
    ("Test", "test_rows"),
)


def load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    if payload.get("complete") is not True:
        raise SystemExit(f"Build artifact is incomplete: {path}")
    return payload


def load_build(root: Path) -> dict:
    """Read the build manifest and both hard-split reports."""
    return {
        "root": root,
        "manifest": load_json(root / "mt_build_manifest.json"),
        "reports": {
            short: load_json(
                root / "formosan_mt_experiments" / "data" / f"splits_{short}_v1" / "report_in_domain_hard.json"
            )
            for _, short in TARGETS
        },
    }


def language_rows(report: dict) -> dict[str, dict[str, int]]:
    return {
        language: {key: int(values.get(key, 0)) for _, key in LANGUAGE_COLUMNS}
        for language, values in sorted(report["languages"].items())
    }


def source_rows(report: dict) -> dict[str, dict[str, int]]:
    """Sum each source corpus over all languages."""
    totals: dict[str, dict[str, int]] = {}
    for sources in report["source_strata"].values():
        for source, values in sources.items():
            row = totals.setdefault(source, {key: 0 for _, key in SOURCE_COLUMNS})
            for _, key in SOURCE_COLUMNS:
                row[key] += int(values.get(key, 0))
    return dict(sorted(totals.items(), key=lambda item: (-item[1]["human_input_rows"], item[0])))


def cell(value: int, previous: int | None) -> str:
    if previous is None or previous == value:
        return f"{value:,}"
    return f"{value:,} ({value - previous:+,})"


def table(
    label: str,
    columns: tuple[tuple[str, str], ...],
    rows: dict[str, dict[str, int]],
    previous: dict[str, dict[str, int]] | None,
) -> str:
    """Render a Markdown table with a total row and optional signed changes."""
    keys = [key for _, key in columns]
    empty = {key: 0 for key in keys}
    names = list(rows)
    if previous is not None:
        names += [name for name in previous if name not in rows]

    def line(name: str, current: dict[str, int], old: dict[str, int] | None) -> str:
        cells = [cell(current[key], None if old is None else old[key]) for key in keys]
        return f"| {name} | " + " | ".join(cells) + " |"

    def total(values: dict[str, dict[str, int]]) -> dict[str, int]:
        return {key: sum(row[key] for row in values.values()) for key in keys}

    lines = [
        f"| {label} | " + " | ".join(title for title, _ in columns) + " |",
        "|---|" + "---:|" * len(columns),
    ]
    for name in names:
        old = None if previous is None else previous.get(name, empty)
        lines.append(line(f"`{name}`", rows.get(name, empty), old))
    lines.append(line("**Total**", total(rows), None if previous is None else total(previous)))
    return "\n".join(lines)


def describe(build: dict) -> str:
    manifest = build["manifest"]
    settings = manifest["settings"]
    return (
        f"`{build['root'].name}`: built {manifest['created_at'][:10]} at toolkit commit "
        f"`{manifest['repository'].get('commit', 'unknown')[:7]}`, QC `{settings['qc_revision'][:7]}`, "
        f"{'public' if settings['public'] else 'all visible'} sources, "
        f"{'with' if settings['with_pivot'] else 'without'} DeepL pivot"
    )


def summarize(build: dict, baseline: dict | None, *, sources: bool) -> str:
    sections = [describe(build)]
    if baseline is not None:
        sections.append(f"Changes in parentheses are against {describe(baseline)}")
    for title, short in TARGETS:
        report = build["reports"][short]
        old_report = None if baseline is None else baseline["reports"][short]
        sections.append(
            f"## {title}\n\n"
            + table(
                "Language",
                LANGUAGE_COLUMNS,
                language_rows(report),
                None if old_report is None else language_rows(old_report),
            )
        )
        if sources:
            sections.append(
                table(
                    "Source corpus",
                    SOURCE_COLUMNS,
                    source_rows(report),
                    None if old_report is None else source_rows(old_report),
                )
            )
    return "\n\n".join(sections)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("build_root", type=Path)
    parser.add_argument("--compare", type=Path, help="Earlier build root to show changes against.")
    parser.add_argument("--sources", action="store_true", help="Also list every source corpus.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build = load_build(args.build_root.expanduser().resolve())
    baseline = load_build(args.compare.expanduser().resolve()) if args.compare else None
    print(summarize(build, baseline, sources=args.sources))


if __name__ == "__main__":
    main()
