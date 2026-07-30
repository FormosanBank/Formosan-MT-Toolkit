#!/usr/bin/env python3
"""Evaluate one directional checkpoint with realistic metadata controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from experiment_config import (
    DEFAULT_PROFILE,
    load_profile,
    manifest_contains_hash,
    profile_record,
    sha256_file,
    stable_hash,
)
from model_backends import (
    ModelBackend,
    TaskSpec,
    ensure_source_prefix_tokens,
    get_backend,
    normalize_control_metadata,
)
from mt_common import (
    FORMOSAN_CODES,
    cjk_token_count,
    direction_choices,
    is_formosan_to_target,
    normalize_target_language,
    read_parallel_csv,
    source_bucket,
    target_col_for,
    target_language_from_direction,
    token_count,
    with_tagged_columns,
    write_json,
)
from mt_metrics import (
    bootstrap_confidence_intervals,
    score_translations,
)
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM

LOAD_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def read_complete_manifest(path: Path, label: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Malformed {label} {path}: {exc}") from exc
    if value.get("complete") is not True:
        raise SystemExit(f"Incomplete {label}: {path}")
    return value


def validate_evaluation_contract(
    args: argparse.Namespace,
    profile: dict,
) -> dict[str, object]:
    input_hash = sha256_file(args.input)
    corpus = read_complete_manifest(
        args.corpus_manifest,
        "corpus manifest",
    )
    validation = read_complete_manifest(
        args.validation_report,
        "corpus validation report",
    )
    run_contract = read_complete_manifest(
        args.run_contract,
        "training run contract",
    )
    if not manifest_contains_hash(corpus, input_hash):
        raise SystemExit(
            "Evaluation CSV checksum is absent from corpus manifest"
        )
    if validation.get("input_sha256") != input_hash:
        raise SystemExit(
            "Validation report does not match evaluation CSV"
        )
    if (
        run_contract.get("input", {}).get("sha256") != input_hash
        or run_contract.get("recipe_id") != profile["recipe_id"]
        or str(run_contract.get("model_family") or "nllb")
        != profile["model_family"]
        or run_contract.get("profile", {}).get("sha256")
        != profile_record(args.profile)["sha256"]
    ):
        raise SystemExit(
            "Training run contract does not match evaluation inputs"
        )
    checkpoint_metadata_path = (
        args.model / "experiment_metadata.json"
    )
    if not checkpoint_metadata_path.is_file():
        raise SystemExit(
            f"Checkpoint has no experiment metadata: "
            f"{checkpoint_metadata_path}"
        )
    checkpoint_metadata = json.loads(
        checkpoint_metadata_path.read_text(encoding="utf-8")
    )
    if (
        checkpoint_metadata.get("run_contract_sha256")
        != stable_hash(run_contract)
    ):
        raise SystemExit(
            "Checkpoint was not created under the supplied run contract"
        )
    return {
        "input_sha256": input_hash,
        "corpus_manifest_sha256": sha256_file(args.corpus_manifest),
        "validation_report_sha256": sha256_file(
            args.validation_report
        ),
        "run_contract_sha256": stable_hash(run_contract),
        "checkpoint_metadata_sha256": sha256_file(
            checkpoint_metadata_path
        ),
    }


def length_bin(tokens: int) -> str:
    if tokens <= 3:
        return "001_003"
    if tokens <= 8:
        return "004_008"
    if tokens <= 16:
        return "009_016"
    if tokens <= 32:
        return "017_032"
    return "033_plus"


def lexical_units(text: object, is_chinese: bool) -> list[str]:
    if is_chinese:
        return [
            character
            for character in str(text)
            if not character.isspace()
        ]
    return str(text).casefold().split()


def word_oov_rates(
    full: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    direction: str,
    target_col: str,
    target_lang: str,
) -> pd.Series:
    train = full[
        full["split"].astype(str).str.lower().eq("train")
    ]
    column = (
        "formosan_sentence"
        if is_formosan_to_target(direction)
        else target_col
    )
    is_chinese = (
        column == target_col
        and target_lang == "chinese"
    )
    vocabularies: dict[str, set[str]] = {}
    for language, subset in train.groupby("lang_code"):
        vocabulary: set[str] = set()
        for text in subset[column].astype(str):
            vocabulary.update(
                lexical_units(text, is_chinese=is_chinese)
            )
        vocabularies[str(language)] = vocabulary
    rates = []
    for _, row in evaluation.iterrows():
        units = lexical_units(
            row[column],
            is_chinese=is_chinese,
        )
        vocabulary = vocabularies.get(
            str(row["lang_code"]),
            set(),
        )
        rates.append(
            sum(unit not in vocabulary for unit in units)
            / max(len(units), 1)
        )
    return pd.Series(rates, index=evaluation.index)


def formosan_fragmentation(
    tokenizer,
    evaluation: pd.DataFrame,
) -> pd.Series:
    values = []
    for text in evaluation["formosan_sentence"].astype(str):
        words = [word for word in text.split() if word]
        pieces = sum(
            len(tokenizer.tokenize(word))
            for word in words
        )
        values.append(pieces / max(len(words), 1))
    return pd.Series(values, index=evaluation.index)


def metadata_frame(
    evaluation: pd.DataFrame,
    tokenizer,
    *,
    mode: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    return normalize_control_metadata(
        evaluation,
        tokenizer,
        mode=mode,
    )


@torch.no_grad()
def generate(
    tokenizer,
    model,
    texts: list[str],
    *,
    backend: ModelBackend,
    task: TaskSpec,
    device: torch.device,
    args: argparse.Namespace,
    description: str,
) -> list[str]:
    order = np.argsort([-len(text) for text in texts])
    restore = np.argsort(order)
    sorted_texts = [texts[index] for index in order]
    outputs: list[str] = []
    progress = tqdm(
        total=len(texts),
        desc=description,
        unit="example",
        dynamic_ncols=True,
    )
    for start in range(0, len(sorted_texts), args.batch_size):
        batch = sorted_texts[start : start + args.batch_size]
        backend.prepare_source(tokenizer, task)
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_token_type_ids=False,
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }
        generated = model.generate(
            **encoded,
            num_beams=args.beam,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            **backend.generation_kwargs(tokenizer, model, task),
        )
        outputs.extend(
            tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )
        )
        progress.update(len(batch))
    progress.close()
    return [outputs[index] for index in restore]


def generate_mode(
    evaluation: pd.DataFrame,
    tokenizer,
    model,
    *,
    backend: ModelBackend,
    mode: str,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[pd.Series, dict[str, int]]:
    metadata, fallback = metadata_frame(
        evaluation,
        tokenizer,
        mode=mode,
    )
    if args.validate_tags:
        ensure_source_prefix_tokens(
            backend,
            tokenizer,
            metadata,
            args.direction,
            target_lang=args.target_lang,
            use_tags=args.use_tags,
        )
    tagged = with_tagged_columns(
        metadata,
        args.direction,
        target_col=args.target_col,
        target_lang=args.target_lang,
        use_tags=args.use_tags,
        prefix_builder=lambda row: backend.source_prefix(
            row,
            args.direction,
            target_lang=args.target_lang,
            use_tags=args.use_tags,
        ),
    )
    hypotheses = pd.Series("", index=evaluation.index, dtype="object")
    for language, subset in tagged.groupby(
        "lang_code",
        sort=True,
    ):
        if language not in FORMOSAN_CODES:
            continue
        task = backend.task_spec(
            language,
            args.direction,
            target_lang=args.target_lang,
        )
        backend.validate_task(tokenizer, task)
        if is_formosan_to_target(args.direction):
            source = subset["formosan_sentence"].astype(str).tolist()
        else:
            source = subset[args.target_col].astype(str).tolist()
        hypotheses.loc[subset.index] = generate(
            tokenizer,
            model,
            source,
            backend=backend,
            task=task,
            device=device,
            args=args,
            description=f"{language} {args.direction} {mode}",
        )
    return hypotheses, fallback


def group_scores(
    predictions: pd.DataFrame,
    group_column: str,
    *,
    hypothesis_column: str,
    lowercase: bool,
    bleu_tokenize: str,
) -> dict[str, dict[str, object]]:
    if group_column not in predictions:
        return {}
    return {
        str(name): {
            "samples": len(subset),
            **score_translations(
                subset[hypothesis_column].tolist(),
                subset["ref"].tolist(),
                lowercase=lowercase,
                bleu_tokenize=bleu_tokenize,
            ),
        }
        for name, subset in predictions.groupby(
            group_column,
            dropna=False,
        )
        if not subset.empty
    }


def parse_args() -> tuple[argparse.Namespace, dict]:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
    )
    known, _ = preliminary.parse_known_args()
    profile = load_profile(known.profile)
    defaults = profile["generation_defaults"]
    parser = argparse.ArgumentParser(parents=[preliminary])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--run-contract", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--target-lang",
        choices=["english", "chinese"],
        required=True,
    )
    parser.add_argument("--target-col")
    parser.add_argument(
        "--direction",
        choices=direction_choices(),
        required=True,
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "validate"],
    )
    parser.add_argument(
        "--use-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validate-tags",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--max-length",
        type=int,
        default=defaults["max_length"],
    )
    parser.add_argument("--beam", type=int, default=defaults["beam"])
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=defaults["max_new_tokens"],
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=defaults["min_new_tokens"],
    )
    parser.add_argument(
        "--no-repeat-ngram-size",
        type=int,
        default=defaults["no_repeat_ngram_size"],
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=defaults["repetition_penalty"],
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=defaults["length_penalty"],
    )
    parser.add_argument(
        "--metadata-modes",
        default=",".join(defaults["metadata_modes"]),
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=defaults["bootstrap_samples"],
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=defaults["bootstrap_seed"],
    )
    parser.add_argument("--limit-per-lang", type=int, default=0)
    parser.add_argument("--lowercase-bleu", action="store_true")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--load-dtype",
        choices=sorted(LOAD_DTYPES),
        default=profile["training_defaults"].get(
            "load_dtype",
            "fp32",
        ),
    )
    args = parser.parse_args()
    args.profile = known.profile
    return args, profile


def main() -> None:
    args, profile = parse_args()
    args.target_lang = normalize_target_language(
        args.target_lang,
        args.target_col,
    )
    args.target_col = args.target_col or target_col_for(
        args.target_lang
    )
    if (
        target_language_from_direction(
            args.direction,
            args.target_lang,
        )
        != args.target_lang
    ):
        raise SystemExit(
            f"Direction {args.direction} does not target "
            f"{args.target_lang}"
        )
    modes = [
        value.strip().lower()
        for value in args.metadata_modes.split(",")
        if value.strip()
    ]
    if not args.use_tags:
        modes = ["default"]
    if not modes or set(modes) - {"default", "oracle"}:
        raise SystemExit(
            "--metadata-modes must contain default and/or oracle"
        )
    if "default" not in modes:
        raise SystemExit(
            "Headline evaluation requires default metadata mode"
        )
    contract = validate_evaluation_contract(args, profile)

    full = read_parallel_csv(args.input, target_col=args.target_col)
    if "source_bucket" not in full:
        full["source_bucket"] = full["source"].map(source_bucket)
    if (
        "kindOf" not in full
        or not full["kindOf"].astype(str).str.lower().eq("standard").all()
    ):
        raise SystemExit("Evaluation input contains non-standard rows")
    evaluation = full[
        full["split"].astype(str).str.lower().eq(args.split)
    ].copy()
    if evaluation.empty:
        raise SystemExit(f"No rows with split={args.split}")
    if (
        evaluation.get(
            "pivot_origin",
            pd.Series("original", index=evaluation.index),
        )
        .astype(str)
        .eq("synthetic")
        .any()
        or not evaluation["row_type"].astype(str).eq("sentence").all()
    ):
        raise SystemExit(
            "Evaluation rows must be human sentence pairs"
        )
    if args.limit_per_lang > 0:
        evaluation = pd.concat(
            [
                group.sample(
                    min(len(group), args.limit_per_lang),
                    random_state=17,
                )
                for _, group in evaluation.groupby(
                    "lang_code",
                    sort=False,
                )
            ]
        ).sort_index()

    backend = get_backend(profile)
    tokenizer = backend.load_tokenizer(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        dtype=LOAD_DTYPES[args.load_dtype],
        low_cpu_mem_usage=True,
    )
    backend.configure_model(model, tokenizer)
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        raise SystemExit(
            "Tokenizer and model vocabulary sizes differ; "
            "load matching checkpoint artifacts."
        )
    device_name = (
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device).eval()

    evaluation["_src_oov_rate"] = word_oov_rates(
        full,
        evaluation,
        direction=args.direction,
        target_col=args.target_col,
        target_lang=args.target_lang,
    )
    evaluation["_formosan_pieces_per_word"] = (
        formosan_fragmentation(tokenizer, evaluation)
    )
    mode_hypotheses: dict[str, pd.Series] = {}
    metadata_fallbacks: dict[str, dict[str, int]] = {}
    for mode in modes:
        hypotheses, fallback = generate_mode(
            evaluation,
            tokenizer,
            model,
            backend=backend,
            mode=mode,
            device=device,
            args=args,
        )
        mode_hypotheses[mode] = hypotheses
        metadata_fallbacks[mode] = fallback

    if is_formosan_to_target(args.direction):
        source = evaluation["formosan_sentence"].astype(str)
        reference = evaluation[args.target_col].astype(str)
    else:
        source = evaluation[args.target_col].astype(str)
        reference = evaluation["formosan_sentence"].astype(str)
    predictions = pd.DataFrame(
        {
            "row_id": evaluation["row_id"].astype(str),
            "lang_code": evaluation["lang_code"].astype(str),
            "direction": args.direction,
            "eval_tier": evaluation.get("eval_tier", ""),
            "source_bucket": evaluation["source_bucket"].astype(str),
            "source": evaluation["source"].astype(str),
            "dialect": evaluation["dialect"].astype(str),
            "pivot_origin": evaluation.get("pivot_origin", "original"),
            "src": source,
            "ref": reference,
            "hyp": mode_hypotheses["default"],
            "src_oov_rate": evaluation["_src_oov_rate"],
            "formosan_pieces_per_word": evaluation[
                "_formosan_pieces_per_word"
            ],
        },
        index=evaluation.index,
    )
    for mode, hypotheses in mode_hypotheses.items():
        predictions[f"hyp_{mode}"] = hypotheses
    predictions["src_tokens"] = [
        (
            cjk_token_count(text)
            if args.target_lang == "chinese"
            and not is_formosan_to_target(args.direction)
            else token_count(text)
        )
        for text in predictions["src"]
    ]
    predictions["ref_tokens"] = [
        (
            cjk_token_count(text)
            if args.target_lang == "chinese"
            and is_formosan_to_target(args.direction)
            else token_count(text)
        )
        for text in predictions["ref"]
    ]
    predictions["length_bin"] = predictions["src_tokens"].map(
        length_bin
    )
    bleu_tokenize = (
        "zh"
        if args.target_lang == "chinese"
        and is_formosan_to_target(args.direction)
        else "13a"
    )
    mode_metrics = {
        mode: {
            "samples": len(predictions),
            **score_translations(
                predictions[f"hyp_{mode}"].tolist(),
                predictions["ref"].tolist(),
                lowercase=args.lowercase_bleu,
                bleu_tokenize=bleu_tokenize,
            ),
        }
        for mode in modes
    }
    primary = mode_metrics["default"]
    confidence = bootstrap_confidence_intervals(
        predictions["hyp_default"].tolist(),
        predictions["ref"].tolist(),
        strata=predictions["lang_code"].tolist(),
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
        lowercase=args.lowercase_bleu,
        bleu_tokenize=bleu_tokenize,
    )
    metrics = {
        "schema_version": 2,
        "complete": True,
        "profile": profile_record(args.profile),
        "model_family": backend.family,
        "contract": contract,
        "input": str(args.input),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "direction": args.direction,
        "target_lang": args.target_lang,
        "bleu_tokenize": bleu_tokenize,
        "split": args.split,
        "samples": len(predictions),
        "headline_metadata_mode": "default",
        "global": primary,
        "metadata_modes": mode_metrics,
        "metadata_fallbacks": metadata_fallbacks,
        "bootstrap_95_ci": confidence,
        "by_language": group_scores(
            predictions,
            "lang_code",
            hypothesis_column="hyp_default",
            lowercase=args.lowercase_bleu,
            bleu_tokenize=bleu_tokenize,
        ),
        "by_source_bucket": group_scores(
            predictions,
            "source_bucket",
            hypothesis_column="hyp_default",
            lowercase=args.lowercase_bleu,
            bleu_tokenize=bleu_tokenize,
        ),
        "by_dialect": group_scores(
            predictions,
            "dialect",
            hypothesis_column="hyp_default",
            lowercase=args.lowercase_bleu,
            bleu_tokenize=bleu_tokenize,
        ),
        "by_length_bin": group_scores(
            predictions,
            "length_bin",
            hypothesis_column="hyp_default",
            lowercase=args.lowercase_bleu,
            bleu_tokenize=bleu_tokenize,
        ),
    }
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_csv, index=False)
    write_json(args.output_json, metrics)
    print(json.dumps(primary, indent=2))
    print(f"predictions: {args.output_csv}")
    print(f"metrics: {args.output_json}")


if __name__ == "__main__":
    main()
