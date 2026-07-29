"""Quiet execution and rule-level reporting for FormosanBank QC."""

from __future__ import annotations

import ast
import csv
import os
import re
import subprocess
import sys
from collections import Counter, deque
from pathlib import Path

from pipeline_common import sha256_file
from tqdm import tqdm

QC_LOG_DIR = "_qc_logs"
TRANSFORMATION_RE = re.compile(
    r"^\s*(?P<input>.+?)\s+→\s+(?P<output>.+?)\s+:\s+"
    r"(?P<count>[\d,]+)\s*$"
)

RULE_LABELS = {
    "complete_missing_dialect": "complete missing dialect",
    "infer_hundred_paiwan_gloss_language_eng": (
        "tag Hundred Paiwan gloss as English"
    ),
    "mark_alternate_translation": (
        "mark repeated translation as alternate"
    ),
    "normalize_dialect_alias": "normalize legacy dialect label",
    "normalize_translation_language_en_to_eng": (
        "normalize translation language en -> eng"
    ),
    "normalize_translation_language_zh_to_zho": (
        "normalize translation language zh -> zho"
    ),
    "disambiguate_duplicate_id": "disambiguate duplicate XML ID",
    "remove_empty_source_lexical_unit": "remove empty source lexical unit",
    "remove_empty_source_sentence": "remove empty source sentence",
    "remove_hard_text_annotation": "remove hard source annotation",
    "remove_invalid_audio_span": "remove invalid audio timestamp span",
    "remove_lexical_annotation": (
        "remove lexical slash/parenthetical annotation"
    ),
    "remove_null_source_sentence": "remove null/elided source sentence",
    "remove_untyped_punctuation": "remove untyped punctuation",
    "remove_zero_width_characters": "remove zero-width characters",
    "trim_form_boundary_whitespace": "trim FORM boundary whitespace",
}


def qc_env(qc_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(qc_root)
        if not existing
        else f"{qc_root}{os.pathsep}{existing}"
    )
    return environment


def parse_cleaner_transformation(line: str) -> dict[str, object] | None:
    match = TRANSFORMATION_RE.match(line)
    if not match:
        return None

    def parse_value(raw: str) -> str:
        try:
            value = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return raw.strip()
        return str(value)

    output = parse_value(match.group("output"))
    return {
        "input": parse_value(match.group("input")),
        "output": "" if output == "<deleted>" else output,
        "count": int(match.group("count").replace(",", "")),
    }


def summarize_csv_column(path: Path, column: str) -> dict[str, int]:
    if not path.is_file():
        return {}
    counts: Counter[str] = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            value = str(row.get(column, "")).strip()
            if value:
                counts[value] += 1
    return dict(sorted(counts.items()))


def summarize_validator_findings(path: Path) -> dict[str, object]:
    rules: Counter[tuple[str, str, str]] = Counter()
    files: set[str] = set()
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                severity = (
                    str(row.get("severity", "")).strip() or "UNKNOWN"
                )
                rule_id = (
                    str(row.get("rule_id", "")).strip() or "UNKNOWN"
                )
                title = str(row.get("title", "")).strip()
                rules[(severity, rule_id, title)] += 1
                source_file = str(row.get("file", "")).strip()
                if source_file:
                    files.add(source_file)
    by_severity: dict[str, dict[str, object]] = {}
    for (severity, rule_id, title), count in sorted(rules.items()):
        section = by_severity.setdefault(
            severity,
            {"records": 0, "rules": {}},
        )
        section["records"] = int(section["records"]) + count
        section["rules"][rule_id] = {
            "title": title,
            "count": count,
        }
    return {
        "records": sum(rules.values()),
        "files_with_findings": len(files),
        "by_severity": by_severity,
    }


def _write_command_log(
    path: Path,
    cmd: list[str],
    *,
    stdout: str,
    stderr: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n")
        if stdout:
            handle.write("\n[stdout]\n")
            handle.write(stdout)
            if not stdout.endswith("\n"):
                handle.write("\n")
        if stderr:
            handle.write("\n[stderr]\n")
            handle.write(stderr)
            if not stderr.endswith("\n"):
                handle.write("\n")


def run_captured_command(
    cmd: list[str],
    qc_root: Path,
    *,
    log_path: Path,
) -> dict[str, object]:
    result = subprocess.run(
        cmd,
        cwd=qc_root,
        env=qc_env(qc_root),
        text=True,
        capture_output=True,
    )
    _write_command_log(
        log_path,
        cmd,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if detail:
            print(detail, file=sys.stderr)
        print(f"Full stage log: {log_path}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return {
        "path": str(Path(log_path.parent.name) / log_path.name),
        "sha256": sha256_file(log_path),
    }


def run_cleaner_command(
    cmd: list[str],
    qc_root: Path,
    *,
    corpus_dir: Path,
    log_path: Path,
) -> dict[str, object]:
    xml_count = sum(1 for _ in corpus_dir.rglob("*.xml"))
    processed = 0
    cleaned = 0
    transformations: list[dict[str, object]] = []
    recent_lines: deque[str] = deque(maxlen=40)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_handle:
        log_handle.write("$ " + " ".join(cmd) + "\n\n")
        environment = qc_env(qc_root)
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(
            cmd,
            cwd=qc_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError(
                "Pinned QC cleaner did not expose its output"
            )
        with tqdm(
            total=xml_count,
            desc="QC clean XML",
            unit="file",
            dynamic_ncols=True,
        ) as progress:
            for line in process.stdout:
                log_handle.write(line)
                recent_lines.append(line.rstrip())
                stripped = line.strip()
                if stripped.startswith("Processing file:"):
                    processed += 1
                    progress.update(1)
                elif stripped.startswith("File cleaned:"):
                    cleaned += 1
                else:
                    transformation = parse_cleaner_transformation(line)
                    if transformation is not None:
                        transformations.append(transformation)
            returncode = process.wait()
            if returncode == 0 and progress.n < xml_count:
                progress.update(xml_count - progress.n)

    if returncode:
        detail = "\n".join(recent_lines).strip()
        if detail:
            print(detail, file=sys.stderr)
        print(f"Full stage log: {log_path}", file=sys.stderr)
        raise subprocess.CalledProcessError(returncode, cmd)

    warning_path = corpus_dir / "cleaner_warnings.csv"
    return {
        "files_scanned": processed,
        "files_cleaned": cleaned,
        "character_transformations": transformations,
        "warning_counts": summarize_csv_column(
            warning_path,
            "rule_id",
        ),
        "warning_inventory": (
            {
                "path": warning_path.name,
                "sha256": sha256_file(warning_path),
            }
            if warning_path.is_file()
            else None
        ),
        "log": {
            "path": str(log_path.relative_to(corpus_dir)),
            "sha256": sha256_file(log_path),
        },
    }


def rule_label(rule: str) -> str:
    return RULE_LABELS.get(rule, rule.replace("_", " "))


def _print_counts(
    title: str,
    counts: dict[str, int],
    *,
    labels: bool = True,
) -> None:
    filtered = {
        rule: int(count)
        for rule, count in counts.items()
        if int(count) > 0
    }
    if not filtered:
        return
    print(f"  {title}:")
    for rule, count in sorted(
        filtered.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        name = rule_label(rule) if labels else rule
        print(f"    {name}: {count:,}")


def print_qc_rule_summary(
    src_lang: str,
    qc_result: dict[str, object],
) -> None:
    tier_completion = dict(qc_result["tier_completion"])
    tier_counts = {
        key: int(value)
        for key, value in tier_completion.items()
        if key
        in {
            "original_copied_from_standard",
            "standard_copied_from_original",
            "untyped_promoted_to_original",
        }
    }
    repair_counts = {
        str(key): int(value)
        for key, value in dict(
            qc_result["repair_inventory"]
        ).get("counts", {}).items()
    }
    cleaned_counts = {
        rule: count
        for rule, count in repair_counts.items()
        if not rule.startswith("remove_")
    }
    removed_counts = {
        rule: count
        for rule, count in repair_counts.items()
        if rule.startswith("remove_")
    }
    transform_inventory = dict(qc_result["transform_inventory"])
    removed_by_cleaner = int(
        transform_inventory.get("removed_by_cleaner", 0)
    )
    if removed_by_cleaner:
        removed_counts["removed_by_pinned_cleaner"] = (
            removed_by_cleaner
        )
    cleaner = dict(qc_result["cleaner"])

    print(f"\nQC rule summary [{src_lang}]")
    print(
        "  XML units: "
        f"{int(transform_inventory['retained']):,} retained / "
        f"{int(transform_inventory['records']):,} accounted"
    )
    _print_counts("Tier completion", tier_counts)
    _print_counts("Cleaned or repaired", cleaned_counts)
    _print_counts("Removed or quarantined", removed_counts)

    field_changes = dict(cleaner["field_changes"])
    print(
        "  Pinned cleaner: "
        f"{int(cleaner['files_cleaned']):,} files / "
        f"{int(field_changes['fields_modified']):,} fields modified"
    )
    _print_counts(
        "Pinned cleaner rules",
        {
            str(key): int(value)
            for key, value in dict(
                field_changes["rule_counts"]
            ).items()
        },
    )

    transformations = list(cleaner.get("character_transformations", []))
    if transformations:
        print("  Character transformations:")
        for row in sorted(
            transformations,
            key=lambda item: (
                -int(item["count"]),
                str(item["input"]),
                str(item["output"]),
            ),
        ):
            output = (
                repr(row["output"])
                if row["output"]
                else "<deleted>"
            )
            print(
                f"    {row['input']!r} -> {output}: "
                f"{int(row['count']):,}"
            )

    _print_counts(
        "Cleaner warnings",
        {
            str(key): int(value)
            for key, value in dict(
                cleaner.get("warning_counts", {})
            ).items()
        },
        labels=False,
    )

    validator_rules: Counter[str] = Counter()
    for validator in list(qc_result["validators"]):
        summary = dict(validator["summary"])
        for severity, section in dict(
            summary.get("by_severity", {})
        ).items():
            for rule_id, details in dict(
                section.get("rules", {})
            ).items():
                title = str(details.get("title", "")).strip()
                label = f"{severity} {rule_id} {title}".strip()
                validator_rules[label] += int(details["count"])
    _print_counts(
        "Validator findings",
        dict(validator_rules),
        labels=False,
    )
    print(
        "  Audit files: "
        f"{qc_result['repair_inventory']['path']}, "
        f"{qc_result['transform_inventory']['path']}, "
        f"{QC_LOG_DIR}/"
    )
