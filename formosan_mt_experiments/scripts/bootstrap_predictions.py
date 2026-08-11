#!/usr/bin/env python3
"""Add stratified confidence intervals to a completed evaluation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from mt_common import write_json
from mt_metrics import bootstrap_confidence_intervals

REQUIRED_COLUMNS = {"hyp_default", "ref", "lang_code"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if not args.predictions.is_file() or not args.metrics.is_file():
        raise SystemExit("Predictions and metrics files must exist")

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    if metrics.get("complete") is not True:
        raise SystemExit("Headline evaluation is not complete")
    predictions = pd.read_csv(
        args.predictions,
        usecols=lambda name: name in REQUIRED_COLUMNS,
        keep_default_na=False,
    )
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise SystemExit(
            "Predictions are missing columns: "
            + ", ".join(sorted(missing))
        )
    if len(predictions) != int(metrics.get("samples", -1)):
        raise SystemExit("Prediction count does not match metrics report")

    metrics["bootstrap_95_ci"] = {
        "status": "running",
        "samples": args.samples,
        "seed": args.seed,
        "workers": args.workers,
    }
    write_json(args.metrics, metrics)

    intervals = bootstrap_confidence_intervals(
        predictions["hyp_default"].tolist(),
        predictions["ref"].tolist(),
        strata=predictions["lang_code"].tolist(),
        samples=args.samples,
        seed=args.seed,
        workers=args.workers,
        lowercase=bool(metrics.get("lowercase_bleu", False)),
        bleu_tokenize=str(metrics.get("bleu_tokenize") or "13a"),
    )
    intervals["status"] = "complete"
    metrics["bootstrap_95_ci"] = intervals
    write_json(args.metrics, metrics)
    print(f"bootstrap metrics: {args.metrics}")


if __name__ == "__main__":
    main()
