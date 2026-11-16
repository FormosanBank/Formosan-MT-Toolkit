#!/usr/bin/env python3
"""Test whether one system beats another on the same test set.

Both inputs are ``predictions.csv`` files written by ``evaluate_directional.py``
for the same corpus and direction. The report gives each metric's difference,
its confidence interval, and a paired bootstrap p-value for the pooled corpus,
the macro average over languages, and every language.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from mt_common import write_json
from mt_metrics import (
    METRIC_NAMES,
    bleu_tokenizer_for,
    paired_bootstrap_comparison,
    sentence_statistics,
)

REQUIRED_COLUMNS = ["row_id", "lang_code", "direction", "ref", "hyp_default"]


def read_predictions(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, usecols=REQUIRED_COLUMNS, dtype=str, keep_default_na=False)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Cannot read predictions {path}: {exc}") from exc
    if frame.empty or frame["row_id"].duplicated().any():
        raise SystemExit(f"Predictions need unique row_id values: {path}")
    return frame.sort_values("row_id", kind="stable").reset_index(drop=True)


def paired_predictions(baseline_path: Path, system_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both systems in the same row order and refuse mismatched test sets."""
    baseline = read_predictions(baseline_path)
    system = read_predictions(system_path)
    shared = ["row_id", "lang_code", "direction", "ref"]
    if len(baseline) != len(system) or not baseline[shared].equals(system[shared]):
        raise SystemExit(
            "Predictions do not cover the same test rows and references; "
            "compare systems evaluated on one corpus build and direction"
        )
    if baseline["direction"].nunique() != 1:
        raise SystemExit("Predictions must contain exactly one direction")
    return baseline, system


def format_report(report: dict) -> str:
    def row(label: str, values: dict) -> str:
        cells = []
        for name in METRIC_NAMES:
            metric = values[name]
            marker = "*" if metric["p_value"] < 0.05 else ""
            cells.append(
                f"{metric['delta']:+.2f} [{metric['delta_lower']:+.2f}, {metric['delta_upper']:+.2f}] "
                f"p={metric['p_value']:.3f}{marker}"
            )
        return f"| {label} | " + " | ".join(cells) + " |"

    lines = [
        "| Scope | " + " | ".join(f"{name} delta [95% CI]" for name in METRIC_NAMES) + " |",
        "|---|" + "---|" * len(METRIC_NAMES),
        row("All test rows", report["metrics"]),
        row("Macro average over languages", report["macro"]),
        *(row(f"`{language}`", values) for language, values in report["by_stratum"].items()),
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--system", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lowercase-bleu", action="store_true")
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    return args


def main() -> None:
    args = parse_args()
    baseline, system = paired_predictions(args.baseline, args.system)
    direction = str(baseline["direction"].iloc[0])
    references = baseline["ref"].tolist()
    scoring = {"lowercase": args.lowercase_bleu, "bleu_tokenize": bleu_tokenizer_for(direction)}
    report = {
        "schema_version": 1,
        "complete": True,
        "baseline": str(args.baseline),
        "system": str(args.system),
        "direction": direction,
        "stratification": "language",
        **paired_bootstrap_comparison(
            sentence_statistics(baseline["hyp_default"].tolist(), references, **scoring),
            sentence_statistics(system["hyp_default"].tolist(), references, **scoring),
            strata=baseline["lang_code"].tolist(),
            samples=args.samples,
            seed=args.seed,
        ),
    }
    print(f"Delta is system minus baseline over {report['sentences']:,} sentences. Lower TER is better.")
    print(format_report(report))
    print("* p < 0.05, paired bootstrap")
    if args.output_json:
        write_json(args.output_json, report)
        print(f"comparison: {args.output_json}")


if __name__ == "__main__":
    main()
