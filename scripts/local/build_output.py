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


def _validator_totals(qc: dict[str, Any]) -> dict[str, tuple[int, int]]:
    totals: dict[str, tuple[int, int]] = {}
    for validator in qc.get("validators", []):
        by_severity = validator.get("summary", {}).get("by_severity", {})
        for severity, summary in by_severity.items():
            records, rules = totals.get(severity, (0, 0))
            totals[severity] = (
                records + int(summary.get("records", 0)),
                rules + len(summary.get("rules", {})),
            )
    return totals


def format_language_summary(build_root: Path, report: dict[str, Any]) -> str:
    """Render cleaning actions from stage manifests, not subprocess text."""
    code = str(report["code"])
    qc = _load(build_root / f"prepared_{code}" / "_qc_manifest.json")
    mt = _load(build_root / f"prepared_{code}" / "_mt_standard_manifest.json")
    reports = build_root / "processed_corpora" / "filter_reports"
    en = _load(reports / f"{code}_en_processed" / "summary.json")
    zh = _load(reports / f"{code}_zh_processed" / "summary.json")

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
    lines = [
        f"{code} ({report['language']}): en={_count(report['processed_en_rows'])}, "
        f"zh={_count(report['processed_zh_rows'])} [{stage_text}]",
    ]
    if qc or mt:
        lines.append(
            _detail(
                "XML",
                f"{_count(qc.get('input', {}).get('xml_files'))} files, "
                f"{_count(inventory.get('records'))} units; "
                f"MT accepted={_count(status_counts.get('accepted'))}, "
                f"quarantined={_count(status_counts.get('quarantine'))}, "
                f"ineligible={_count(status_counts.get('ineligible'))}",
            )
        )
        lines.append(
            _detail(
                "QC repairs",
                _rules(qc.get("repair_inventory", {}).get("counts", {})),
            )
        )
        lines.append(_detail("MT cleaning", _rules(inventory.get("rule_counts", {}))))
        if inventory.get("reason_counts"):
            lines.append(_detail("MT flags", _rules(inventory["reason_counts"])))
        findings = _validator_totals(qc)
        if findings:
            details = "; ".join(
                f"{severity}={_count(records)} across {_count(rule_count)} rules"
                for severity, (records, rule_count) in sorted(findings.items())
            )
            lines.append(_detail("QC findings", f"{details} (diagnostic only)"))

    for target, payload in (("en", en), ("zh", zh)):
        if not payload:
            continue
        lines.append(
            _detail(
                target,
                f"{_count(payload.get('initial_rows'))} input -> "
                f"{_count(payload.get('accepted_rows'))} kept; "
                f"removed: {_rules(payload.get('filter_rule_counts', {}))}",
            )
        )
        if payload.get("transformation_counts"):
            lines.append(
                _detail(
                    f"{target} text cleaning",
                    _rules(payload["transformation_counts"]),
                )
            )
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
            f"{item.get('direction')}: synthetic={_count(item.get('synthetic_rows_written'))}, "
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
