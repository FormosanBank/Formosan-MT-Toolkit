"""Readable terminal and log output for the corpus orchestrator."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import textwrap
import time
from collections import Counter
from pathlib import Path
from typing import Any

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class CommandExecutionError(RuntimeError):
    """A child stage failed after its output was captured."""


def _elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(seconds), 60)
    return f"{minutes}m {remainder:02d}s"


def _failure_tail(path: Path, *, lines: int = 30) -> str:
    if not path.is_file():
        return ""
    text = ANSI_ESCAPE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    clean = [
        line.strip()
        for line in text.replace("\r", "\n").splitlines()
        if line.strip() and not line.lstrip().startswith("$ ")
    ]
    return "\n".join(f"  {line}" for line in clean[-lines:])


def run_logged(
    cmd: list[str],
    *,
    project_root: Path,
    label: str,
    log_path: Path,
    dry_run: bool = False,
    env: dict[str, str] | None = None,
    verbose: bool = False,
    quiet: bool = False,
    append: bool = False,
) -> None:
    """Run one stage while keeping normal terminal output concise."""
    if dry_run:
        if not quiet:
            print(f"[plan]  {label}")
            if verbose:
                print(f"        {shlex.join(cmd)}")
        return

    if not quiet:
        print(f"[stage] {label}", flush=True)
    started = time.monotonic()
    if verbose:
        print(f"        {shlex.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=project_root, env=env, check=True)
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        child_env = os.environ.copy()
        if env:
            child_env.update(env)
        child_env.setdefault("TQDM_DISABLE", "1")
        mode = "a" if append else "w"
        with log_path.open(mode, encoding="utf-8") as log:
            if append and log_path.stat().st_size:
                log.write("\n")
            log.write(f"$ {shlex.join(cmd)}\n")
            log.flush()
            result = subprocess.run(
                cmd,
                cwd=project_root,
                env=child_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            tail = _failure_tail(log_path)
            message = f"{label} failed (exit {result.returncode}). Log: {log_path}"
            if tail:
                message += f"\nLast output:\n{tail}"
            raise CommandExecutionError(message)
    if not quiet:
        print(f"[done]  {label} ({_elapsed(time.monotonic() - started)})", flush=True)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def _rule_name(value: str) -> str:
    return value.replace(":", " ").replace("_", " ")


def _rules(values: dict[str, Any]) -> str:
    if not values:
        return "none"
    ordered = sorted(values.items(), key=lambda item: (-int(item[1]), item[0]))
    return "; ".join(f"{_rule_name(name)}={_count(count)}" for name, count in ordered)


def _detail(label: str, value: str) -> str:
    width = min(120, max(80, shutil.get_terminal_size(fallback=(100, 24)).columns))
    return textwrap.fill(
        f"  {label}: {value}",
        width=width,
        subsequent_indent="    ",
        break_long_words=False,
        break_on_hyphens=False,
    )


def _language_manifests(
    build_root: Path,
    code: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    qc = _load(build_root / f"prepared_{code}" / "_qc_manifest.json")
    mt = _load(build_root / f"prepared_{code}" / "_mt_standard_manifest.json")
    reports = build_root / "processed_corpora" / "filter_reports"
    en = _load(reports / f"{code}_en_processed" / "summary.json")
    zh = _load(reports / f"{code}_zh_processed" / "summary.json")
    return qc, mt, en, zh


def format_language_summary(build_root: Path, report: dict[str, Any]) -> str:
    """Render one compact language row from stage manifests."""
    code = str(report["code"])
    qc, mt, en, zh = _language_manifests(build_root, code)

    statuses = report.get("stage_status", {})
    cached = sum(value == "cached" for value in statuses.values())
    rebuilt = [name for name, value in statuses.items() if value == "rebuilt"]
    planned = [name for name, value in statuses.items() if value == "planned"]
    if planned:
        stage_text = f"planned: {', '.join(planned)}"
    elif rebuilt:
        stage_text = f"rebuilt: {', '.join(rebuilt)}; cached: {cached}"
    else:
        stage_text = f"cached: {cached}/{len(statuses)}"

    inventory = mt.get("inventory", {})
    status_counts = inventory.get("status_counts", {})
    if not statuses:
        stage_text = "stages skipped"
    return _detail(
        code,
        f"XML={_count(qc.get('input', {}).get('xml_files'))}, "
        f"units={_count(inventory.get('records'))}, "
        f"MT accepted={_count(status_counts.get('accepted'))}, "
        f"quarantine={_count(status_counts.get('quarantine'))}, "
        f"ineligible={_count(status_counts.get('ineligible'))} | "
        f"en {_count(en.get('initial_rows'))}->{_count(en.get('accepted_rows'))} | "
        f"zh {_count(zh.get('initial_rows'))}->{_count(zh.get('accepted_rows'))} | "
        f"{stage_text}",
    )


def format_rule_summary(build_root: Path, reports: list[dict[str, Any]]) -> str:
    """Aggregate every action rule once across all selected languages."""
    totals: dict[str, Counter[str]] = {
        "QC repairs": Counter(),
        "MT cleaning": Counter(),
        "MT flags": Counter(),
        "XML unit exclusions": Counter(),
        "English removals": Counter(),
        "English text cleaning": Counter(),
        "Chinese removals": Counter(),
        "Chinese text cleaning": Counter(),
    }
    finding_records: Counter[str] = Counter()
    finding_rules: dict[str, set[str]] = {}
    for report in reports:
        code = str(report["code"])
        qc, mt, en, zh = _language_manifests(build_root, code)
        extraction = _load(
            build_root / "raw_corpora" / f"{code}_en.extraction.json"
        ) or _load(
            build_root / "raw_corpora" / f"{code}_zh.extraction.json"
        )
        inventory = mt.get("inventory", {})
        totals["QC repairs"].update(qc.get("repair_inventory", {}).get("counts", {}))
        totals["MT cleaning"].update(inventory.get("rule_counts", {}))
        totals["MT flags"].update(inventory.get("reason_counts", {}))
        totals["XML unit exclusions"].update(
            {
                name: count
                for name, count in extraction.get("counts", {}).items()
                if str(name).endswith("_excluded")
            }
        )
        totals["English removals"].update(en.get("filter_rule_counts", {}))
        totals["English text cleaning"].update(en.get("transformation_counts", {}))
        totals["Chinese removals"].update(zh.get("filter_rule_counts", {}))
        totals["Chinese text cleaning"].update(zh.get("transformation_counts", {}))
        for validator in qc.get("validators", []):
            by_severity = validator.get("summary", {}).get("by_severity", {})
            for severity, summary in by_severity.items():
                finding_records[severity] += int(summary.get("records", 0))
                finding_rules.setdefault(severity, set()).update(
                    summary.get("rules", {})
                )

    lines = ["\nCleaning rule totals"]
    for label, counts in totals.items():
        if counts:
            lines.append(_detail(label, _rules(dict(counts))))
    if finding_records:
        details = "; ".join(
            f"{severity}={_count(records)} across {_count(len(finding_rules[severity]))} rules"
            for severity, records in sorted(finding_records.items())
        )
        lines.append(_detail("QC findings", f"{details} (diagnostic only)"))
    return "\n".join(lines)


def format_fetch_summary(build_root: Path, language_codes: list[str]) -> str:
    counts: list[str] = []
    total = 0
    for code in language_codes:
        manifest = _load(build_root / f"downloaded_{code}" / "_fetch_manifest.json")
        kept = int(manifest.get("status_counts", {}).get("kept", 0))
        total += kept
        counts.append(f"{code}={_count(kept)}")
    return f"XML routed: {_count(total)} language files ({', '.join(counts)})"


def format_aggregate_summary(manifest_path: Path, label: str) -> str:
    outputs = _load(manifest_path).get("outputs", {})
    return (
        f"{label}: en={_count(outputs.get('english', {}).get('rows'))}, "
        f"zh={_count(outputs.get('chinese', {}).get('rows'))}, "
        f"combined={_count(outputs.get('combined', {}).get('rows'))}"
    )


def format_pivot_summary(manifest_path: Path) -> str:
    stats = _load(manifest_path).get("stats", [])
    parts = []
    for item in stats:
        parts.append(
            f"{item.get('direction')}: eligible={_count(item.get('candidate_rows'))}, "
            f"excluded={_count(item.get('ineligible_source_rows'))}, "
            f"synthetic={_count(item.get('synthetic_rows_written'))}, "
            f"quarantined={_count(item.get('synthetic_rows_quarantined'))}, "
            f"missing={_count(item.get('synthetic_rows_missing'))}"
        )
    return "Pivot: " + " | ".join(parts)


def format_split_summary(split_dir: Path, target: str) -> str:
    report = _load(split_dir / "report_in_domain_hard.json")
    validation = _load(split_dir / "validation_in_domain_hard.json")
    exposure = _load(split_dir / "exposure_in_domain_hard.json")
    counts = report.get("split_counts", {})
    checks = validation.get("split_validation", {})
    exposure_errors = exposure.get("release_gate", {}).get("errors", [])
    passed = bool(checks.get("ok")) and not exposure_errors
    return (
        f"{target} hard split: train={_count(counts.get('train'))}, "
        f"test={_count(counts.get('test'))}, validate={_count(counts.get('validate'))}; "
        f"leakage and exposure gates={'pass' if passed else 'FAIL'}"
    )
