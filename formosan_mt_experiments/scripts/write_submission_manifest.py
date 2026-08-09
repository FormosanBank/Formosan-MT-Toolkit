#!/usr/bin/env python3
"""Write a reproducible manifest for a directional MT Slurm graph."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_code_inventory import build_code_inventory

DIRECTIONS = ("f2en", "en2f", "f2zh", "zh2f")


def read_job_ids(state_dir: Path) -> dict[str, int]:
    jobs: dict[str, int] = {}
    for path in sorted(state_dir.glob("*.id")):
        raw = path.read_text(encoding="utf-8").strip().split(";", 1)[0]
        if not raw.isdigit():
            raise ValueError(f"Invalid Slurm job ID in {path}: {raw!r}")
        jobs[path.stem] = int(raw)
    return jobs


def build_job_graph(
    job_ids: dict[str, int],
    model_family: str = "nllb",
) -> dict[str, Any]:
    required = {"validate_en", "validate_zh"}
    for direction in DIRECTIONS:
        required.update(
            {
                f"train_{direction}",
                f"eval_{direction}_final",
                f"eval_{direction}_best",
            }
        )
    missing = sorted(required - job_ids.keys())
    if missing:
        raise ValueError(f"Submission state is incomplete; missing: {', '.join(missing)}")

    graph: dict[str, Any] = {
        "validate_en": job_ids["validate_en"],
        "validate_zh": job_ids["validate_zh"],
    }
    if model_family == "madlad400":
        graph["setup_shared"] = job_ids.get("setup_madlad400")
    elif model_family == "nllb":
        graph["setup_en"] = job_ids.get("setup_en_spm8192")
        graph["setup_zh"] = job_ids.get("setup_zh_spm8192")
    else:
        raise ValueError(f"Unsupported model family: {model_family}")
    for direction in DIRECTIONS:
        graph[direction] = [
            job_ids[f"train_{direction}"],
            job_ids[f"eval_{direction}_final"],
            job_ids[f"eval_{direction}_best"],
        ]
    return graph


def corpus_record(corpus_dir: Path, short: str) -> dict[str, Any]:
    build_path = corpus_dir / "provenance" / "mt_build_manifest.json"
    validation_path = corpus_dir / "provenance" / f"validate_{short}_in_domain_hard.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))["split_validation"]
    artifact = build["artifacts"][f"big_corpus_{short}_in_domain_hard"]
    train_evaluation = validation["train_evaluation"]

    ratios = validation["ratios_by_language"].values()
    test = sum(int(row["test"]) for row in ratios)
    validate = sum(int(row["validate"]) for row in ratios)
    rows = int(artifact["rows"])
    return {
        "target_lang": "english" if short == "en" else "chinese",
        "path": str(corpus_dir / f"big_corpus_{short}_in_domain_hard.csv"),
        "rows": rows,
        "splits": {"train": rows - test - validate, "test": test, "validate": validate},
        "sha256": artifact["sha256"],
        "synthetic_eval_rows": int(validation["synthetic_eval_rows"]),
        "validation": {
            "ok": bool(validation["ok"]),
            "exact_overlap": sum(train_evaluation["exact_overlap"].values()),
            "skeleton_overlap": sum(
                train_evaluation["skeleton_overlap"].values()
            ),
            "one_edit_conflicts": sum(
                train_evaluation["one_edit_conflicting_rows"].values()
            ),
            "character_ngram_conflicts": sum(
                train_evaluation["character_ngram_conflicting_rows"].values()
            ),
            "document_overlap": int(train_evaluation["document_overlap"]),
            "lexeme_eval_rows": int(validation["lexical_eval_rows"]),
            "per_language_ratio_failures": len(validation["ratio_failures"]),
        },
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    corpus_dir = args.project_data / args.corpus_name
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    model_family = str(
        profile.get("model_family") or "nllb"
    ).strip().lower()
    return {
        "schema_version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": (
            f"{profile['recipe_id']}_{args.corpus_name}_{args.run_stamp}"
        ),
        "status": "submitted",
        "run_stamp": args.run_stamp,
        "corpus_name": args.corpus_name,
        "recipe_id": profile["recipe_id"],
        "model_family": model_family,
        "base_model": profile["base_model"],
        "source_git_commit": args.git_commit,
        "code": build_code_inventory(
            experiment_root=args.experiment_root,
        ),
        "corpora": {
            "english": corpus_record(corpus_dir, "en"),
            "chinese": corpus_record(corpus_dir, "zh"),
        },
        "split_policy": profile["splits"],
        "training": profile["training_defaults"],
        "generation": profile["generation_defaults"],
        "jobs": build_job_graph(
            read_job_ids(args.state_dir),
            model_family=model_family,
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-name", required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--project-data", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote submission manifest: {args.output}")


if __name__ == "__main__":
    main()
