#!/usr/bin/env python3
"""Run fail-closed TAME-MT exposure audits on a finalized hard corpus."""

from __future__ import annotations

import argparse
import importlib.metadata
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
from experiment_config import sha256_file
from mt_common import normalize_target_language, target_col_for, write_json
from tame_mt import CachedSegmentScorer, TameScorer
from tame_mt.config import (
    BinConfig,
    IndexConfig,
    NormalizationConfig,
    PairConfig,
    ScoreConfig,
)
from tame_mt.exposure import summarize_exposures

EXPECTED_TAME_VERSION = "0.2.2"
EVAL_SPLITS = ("test", "validate")


def read_corpus(path: Path, *, target_col: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    required = {
        "row_id",
        "lang_code",
        "formosan_sentence",
        target_col,
        "split",
        "kindOf",
        "row_type",
        "pivot_origin",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Exposure audit input is missing columns: {missing}")
    if frame.empty:
        raise SystemExit("Exposure audit input is empty")
    if frame["row_id"].duplicated().any():
        raise SystemExit("Exposure audit input has duplicate row_id values")
    if not frame["kindOf"].str.casefold().eq("standard").all():
        raise SystemExit("Exposure audit input contains non-standard Formosan rows")
    splits = set(frame["split"].str.casefold())
    if not {"train", *EVAL_SPLITS}.issubset(splits):
        raise SystemExit(f"Exposure audit requires train/test/validate splits, found {sorted(splits)}")
    evaluation = frame[frame["split"].str.casefold().isin(EVAL_SPLITS)]
    synthetic = evaluation["pivot_origin"].str.casefold().eq("synthetic")
    lexical = ~evaluation["row_type"].str.casefold().eq("sentence")
    if synthetic.any() or lexical.any():
        raise SystemExit(
            "Exposure audit requires human sentence-only evaluation rows; "
            f"found synthetic={int(synthetic.sum())}, lexical={int(lexical.sum())}"
        )
    return frame


def exposure_summary(segments, config: ScoreConfig) -> dict[str, object]:
    return asdict(summarize_exposures(list(segments), config))


def per_language_summary(
    frame: pd.DataFrame,
    segments,
    config: ScoreConfig,
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for language in sorted(frame["lang_code"].unique()):
        indexes = [
            index
            for index, value in enumerate(frame["lang_code"])
            if value == language
        ]
        selected = [segments[index] for index in indexes]
        output[str(language)] = {
            "rows": len(selected),
            "exposure": exposure_summary(selected, config),
        }
    return output


def direction_columns(direction: str, target_col: str) -> tuple[str, str]:
    if direction.startswith("f2"):
        return "formosan_sentence", target_col
    return target_col, "formosan_sentence"


def build_tame_config(
    *,
    high_threshold: float,
    pair_k: int,
    batch_size: int,
) -> ScoreConfig:
    return ScoreConfig(
        normalization=NormalizationConfig(
            unicode_form="NFKC",
            lowercase=True,
        ),
        index=IndexConfig(
            mode="native_exact",
            topk=pair_k,
            batch_size=batch_size,
        ),
        bins=BinConfig(
            leak_thresholds=(0.70, 0.85, high_threshold),
        ),
        pair=PairConfig(exact_thresholds=False),
    )


def gate_errors(
    directions: dict[str, dict[str, object]],
    *,
    high_threshold: str,
    max_high_exposure_rate: float,
) -> list[str]:
    errors: list[str] = []
    for direction, payload in directions.items():
        for split, split_payload in payload["by_split"].items():
            exposure = split_payload["exposure"]
            for side in ("source", "target", "pair"):
                side_metrics = exposure[side]
                if side_metrics is None:
                    errors.append(f"{direction}/{split}/{side}: metrics missing")
                    continue
                exact = float(side_metrics["exact_overlap"])
                high = float(side_metrics["at_threshold"][high_threshold])
                if exact != 0.0:
                    errors.append(
                        f"{direction}/{split}/{side}: exact_overlap={exact:.8f}"
                    )
                if high > max_high_exposure_rate:
                    errors.append(
                        f"{direction}/{split}/{side}: exposure@{high_threshold}="
                        f"{high:.8f} > {max_high_exposure_rate:.8f}"
                    )
    return errors


def audit_direction(
    frame: pd.DataFrame,
    *,
    direction: str,
    target_col: str,
    config: ScoreConfig,
) -> dict[str, object]:
    train = frame[frame["split"].str.casefold().eq("train")].reset_index(drop=True)
    evaluation = frame[
        frame["split"].str.casefold().isin(EVAL_SPLITS)
    ].reset_index(drop=True)
    source_col, reference_col = direction_columns(direction, target_col)
    exposures = [None] * len(evaluation)
    tm_results = [None] * len(evaluation)
    language_reports: dict[str, object] = {}
    scorer = TameScorer(config)
    for language in sorted(evaluation["lang_code"].unique()):
        train_indexes = [
            index
            for index, value in enumerate(train["lang_code"])
            if value == language
        ]
        evaluation_indexes = [
            index
            for index, value in enumerate(evaluation["lang_code"])
            if value == language
        ]
        if not train_indexes:
            raise SystemExit(
                f"Exposure audit has evaluation rows but no training rows for {language}"
            )
        language_train = train.iloc[train_indexes].reset_index(drop=True)
        language_evaluation = evaluation.iloc[evaluation_indexes].reset_index(drop=True)
        result = scorer.evaluate_corpus(
            train_src=language_train[source_col].tolist(),
            train_tgt=language_train[reference_col].tolist(),
            test_src=language_evaluation[source_col].tolist(),
            refs=[language_evaluation[reference_col].tolist()],
            hyp=None,
        )
        language_reports[str(language)] = result.report.to_dict()
        for local_index, global_index in enumerate(evaluation_indexes):
            segment = result.exposures[local_index]
            exposures[global_index] = replace(
                segment,
                index=global_index,
                source_nn_index=(
                    train_indexes[segment.source_nn_index]
                    if segment.source_nn_index is not None
                    else None
                ),
                target_nn_index=(
                    train_indexes[segment.target_nn_index]
                    if segment.target_nn_index is not None
                    else None
                ),
                pair_nn_index=(
                    train_indexes[segment.pair_nn_index]
                    if segment.pair_nn_index is not None
                    else None
                ),
            )
            tm_result = result.tm_results[local_index]
            tm_results[global_index] = replace(
                tm_result,
                index=global_index,
                tm_source_index=(
                    train_indexes[tm_result.tm_source_index]
                    if tm_result.tm_source_index is not None
                    else None
                ),
            )
    if any(segment is None for segment in exposures):
        raise RuntimeError("Exposure audit did not score every evaluation row")
    if any(result is None for result in tm_results):
        raise RuntimeError("Exposure audit did not create every TM result")
    complete_exposures = [segment for segment in exposures if segment is not None]
    complete_tm_results = [result for result in tm_results if result is not None]
    cached = CachedSegmentScorer(
        config=config,
        exposures=complete_exposures,
        tm_results=complete_tm_results,
        refs=[evaluation[reference_col].tolist()],
        num_train=len(train),
        artifact_backend={
            "name": "task_conditioned",
            "conditioning_column": "lang_code",
        },
    )
    by_split: dict[str, dict[str, object]] = {}
    for split in EVAL_SPLITS:
        indexes = [
            index
            for index, value in enumerate(evaluation["split"].str.casefold())
            if value == split
        ]
        segments = [complete_exposures[index] for index in indexes]
        split_frame = evaluation.iloc[indexes].reset_index(drop=True)
        by_split[split] = {
            "rows": len(indexes),
            "exposure": exposure_summary(segments, config),
            "by_language": per_language_summary(split_frame, segments, config),
        }
    return {
        "source_column": source_col,
        "reference_column": reference_col,
        "train_rows": len(train),
        "evaluation_rows": len(evaluation),
        "combined_evaluation": {
            "task_conditioning": "lang_code",
            "data": {
                "num_train": len(train),
                "num_test": len(evaluation),
                "num_refs": 1,
            },
            "quality": {
                "system": {
                    metric: None for metric in cached.tm_scores
                },
                "tm": cached.tm_scores,
                "delta_tm": {
                    metric: None for metric in cached.tm_scores
                },
            },
            "exposure": asdict(cached.exposure_summary),
            "by_language": language_reports,
        },
        "by_split": by_split,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit train/evaluation exposure with pinned TAME-MT.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--target-col")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--high-threshold", type=float, default=0.95)
    parser.add_argument("--max-high-exposure-rate", type=float, default=0.0)
    parser.add_argument("--pair-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_lang = normalize_target_language(args.target_lang)
    target_col = args.target_col or target_col_for(target_lang)
    version = importlib.metadata.version("tame-mt")
    if version != EXPECTED_TAME_VERSION:
        raise SystemExit(
            f"Corpus release requires tame-mt=={EXPECTED_TAME_VERSION}, found {version}"
        )
    if not 0.0 <= args.max_high_exposure_rate <= 1.0:
        raise SystemExit("--max-high-exposure-rate must be in [0, 1]")
    if not 0.0 < args.high_threshold <= 1.0:
        raise SystemExit("--high-threshold must be in (0, 1]")

    frame = read_corpus(args.input, target_col=target_col)
    config = build_tame_config(
        high_threshold=args.high_threshold,
        pair_k=args.pair_k,
        batch_size=args.batch_size,
    )
    suffix = "en" if target_lang == "english" else "zh"
    directions = {
        direction: audit_direction(
            frame,
            direction=direction,
            target_col=target_col,
            config=config,
        )
        for direction in (f"f2{suffix}", f"{suffix}2f")
    }
    threshold_key = f"{args.high_threshold:.2f}"
    errors = gate_errors(
        directions,
        high_threshold=threshold_key,
        max_high_exposure_rate=args.max_high_exposure_rate,
    )
    report = {
        "schema_version": 2,
        "tool": "tame-mt",
        "tool_version": version,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": sha256_file(args.input),
            "rows": len(frame),
            "target_language": target_lang,
            "target_column": target_col,
        },
        "configuration": {
            "retrieval": "exact",
            "task_conditioning": "lang_code",
            "normalization": "NFKC case-insensitive",
            "character_ngram_orders": [3, 4, 5],
            "pair_k": args.pair_k,
            "high_threshold": args.high_threshold,
            "max_high_exposure_rate": args.max_high_exposure_rate,
        },
        "directions": directions,
        "release_gate": {
            "requirements": [
                "zero exact source, target, and pair overlap",
                (
                    "source, target, and pair exposure at "
                    f"{threshold_key} <= {args.max_high_exposure_rate}"
                ),
            ],
            "errors": errors,
        },
        "complete": not errors,
    }
    write_json(args.report, report)
    if errors:
        raise SystemExit(
            f"TAME-MT exposure audit failed with {len(errors)} release-gate error(s); "
            f"inspect {args.report}"
        )
    print(f"TAME-MT exposure audit passed: {args.report}")


if __name__ == "__main__":
    main()
