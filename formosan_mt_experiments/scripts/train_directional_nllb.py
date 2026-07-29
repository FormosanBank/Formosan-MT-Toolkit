#!/usr/bin/env python3
"""Directional multilingual NLLB fine-tuning for Formosan↔target-language experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from experiment_config import (
    DEFAULT_PROFILE,
    dependency_versions,
    git_record,
    load_profile,
    manifest_contains_hash,
    profile_record,
    sha256_file,
    stable_hash,
)
from mt_common import (
    EASY_BUCKETS,
    FORMOSAN_CODES,
    build_prefix,
    direction_choices,
    get_lid,
    is_formosan_to_target,
    language_sampling_probs,
    normalize_target_language,
    read_parallel_csv,
    source_bucket,
    target_col_for,
    target_language_from_direction,
    target_lid_for,
    with_tagged_columns,
    write_json,
)
from mt_metrics import score_translations
from torch.amp import GradScaler, autocast
from tqdm.auto import trange
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    get_constant_schedule_with_warmup,
)


def ensure_lang_token(tokenizer: NllbTokenizer, code: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(code)
    if tid == tokenizer.unk_token_id:
        raise SystemExit(f"Language token {code} maps to <unk>; rebuild/load the matching tokenizer.")
    return int(tid)


def ensure_control_tags(tokenizer: NllbTokenizer, df: pd.DataFrame, direction: str, target_lang: str = "english") -> None:
    needed = set()
    for _, row in df.iterrows():
        needed.update(build_prefix(row, direction, target_lang=target_lang).split())
    bad = []
    for token in sorted(needed):
        tid = tokenizer.convert_tokens_to_ids(token)
        if tid == tokenizer.unk_token_id or tokenizer.convert_ids_to_tokens(tid) != token:
            bad.append(token)
    if bad:
        raise SystemExit(
            "Control tags are not single tokenizer tokens. "
            f"First missing/broken tags: {bad[:30]}. "
            "Run setup_tokenizer_sweep.py or disable --use-tags."
        )


def prepare_data(args, tokenizer: NllbTokenizer) -> tuple[dict, dict, dict]:
    df = read_parallel_csv(args.input, target_col=args.target_col)
    if "split" not in df.columns:
        raise SystemExit("Training CSV must have split values train/validate/test.")
    df["split"] = df["split"].astype(str).str.lower()
    if "source_bucket" not in df.columns:
        df["source_bucket"] = df["source"].map(source_bucket)
    else:
        df["source_bucket"] = (
            df["source_bucket"]
            .astype(str)
            .replace("", "unknown")
        )
    if "kindOf" not in df or not df["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Training CSV must contain only kindOf=standard rows")
    if "row_id" not in df or df["row_id"].astype(str).duplicated().any():
        raise SystemExit("Training CSV must contain unique stable row_id values")
    unknown_splits = sorted(
        set(df["split"]) - {"train", "validate", "test"}
    )
    if unknown_splits:
        raise SystemExit(f"Training CSV has unknown splits: {unknown_splits}")

    if args.use_tags and args.validate_tags:
        ensure_control_tags(tokenizer, df, args.direction, target_lang=args.target_lang)
    df = with_tagged_columns(
        df,
        args.direction,
        target_col=args.target_col,
        target_lang=args.target_lang,
        use_tags=args.use_tags,
    )

    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].isin(["validate", "valid", "val"])].copy()
    if train.empty:
        raise SystemExit("No train rows found.")
    if val.empty:
        raise SystemExit("No human validation rows found")
    pivot_origin = val.get(
        "pivot_origin",
        pd.Series("original", index=val.index),
    ).astype(str)
    if pivot_origin.eq("synthetic").any():
        raise SystemExit("Synthetic rows are forbidden in validation")
    if not val["row_type"].astype(str).eq("sentence").all():
        raise SystemExit("Validation must contain sentence rows only")

    train_by_lang = {}
    val_by_lang = {}
    for lang in sorted(train["lang_code"].unique()):
        if lang not in FORMOSAN_CODES:
            continue
        train_sub = train[train["lang_code"].eq(lang)].copy()
        val_sub = val[val["lang_code"].eq(lang)].copy()
        if is_formosan_to_target(args.direction):
            train_sub["src_text"] = train_sub["formosan_sentence"].astype(str)
            train_sub["tgt_text"] = train_sub[args.target_col].astype(str)
            val_sub["src_text"] = val_sub["formosan_sentence"].astype(str)
            val_sub["tgt_text"] = val_sub[args.target_col].astype(str)
            src_lid, tgt_lid = get_lid(lang), args.target_lid
        else:
            train_sub["src_text"] = train_sub[args.target_col].astype(str)
            train_sub["tgt_text"] = train_sub["formosan_sentence"].astype(str)
            val_sub["src_text"] = val_sub[args.target_col].astype(str)
            val_sub["tgt_text"] = val_sub["formosan_sentence"].astype(str)
            src_lid, tgt_lid = args.target_lid, get_lid(lang)
        train_by_lang[lang] = {
            "df": train_sub.reset_index(drop=True),
            "src_lid": src_lid,
            "tgt_lid": tgt_lid,
        }
        if not val_sub.empty:
            val_by_lang[lang] = {
                "df": val_sub.reset_index(drop=True),
                "src_lid": src_lid,
                "tgt_lid": tgt_lid,
            }

    if not train_by_lang:
        raise SystemExit("No supported Formosan languages found in train rows.")
    return train_by_lang, val_by_lang, {
        "input_rows": int(len(df)),
        "train_rows": int(len(train)),
        "validate_rows": int(len(val)),
        "direction": args.direction,
        "target_lang": args.target_lang,
        "target_col": args.target_col,
        "use_tags": bool(args.use_tags),
        "standard_rows": int(df["kindOf"].astype(str).str.lower().eq("standard").sum()),
        "synthetic_train_rows": int(
            train.get(
                "pivot_origin",
                pd.Series("original", index=train.index),
            )
            .astype(str)
            .eq("synthetic")
            .sum()
        ),
        "synthetic_validate_rows": 0,
        "train_by_language": {k: int(len(v["df"])) for k, v in train_by_lang.items()},
        "validate_by_language": {k: int(len(v["df"])) for k, v in val_by_lang.items()},
    }


def encode_batch(tokenizer, src_texts, tgt_texts, src_lid, tgt_lid, max_length, device):
    tokenizer.src_lang = src_lid
    enc = tokenizer(
        list(src_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    tokenizer.tgt_lang = tgt_lid
    labels = tokenizer(
        text_target=list(tgt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    return {k: v.to(device) for k, v in enc.items()}, labels.to(device)


def row_probabilities(df: pd.DataFrame, easy_source_weight: float) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float64)
    if "source_bucket" in df.columns:
        easy = df["source_bucket"].isin(EASY_BUCKETS).to_numpy()
        weights[easy] = easy_source_weight
    weights = np.maximum(weights, 1e-8)
    return weights / weights.sum()


@torch.no_grad()
def evaluate_loss(model, tokenizer, val_by_lang, args, device) -> dict:
    model.eval()
    lang_losses = {}
    total_loss = 0.0
    total_tokens = 0
    for lang, info in sorted(val_by_lang.items()):
        df = validation_subset(info["df"], lang, args)
        lang_loss = 0.0
        lang_tokens = 0
        for start in range(0, len(df), args.eval_batch_size):
            batch = df.iloc[start : start + args.eval_batch_size]
            enc, labels = encode_batch(
                tokenizer,
                batch["src_text"].tolist(),
                batch["tgt_text"].tolist(),
                info["src_lid"],
                info["tgt_lid"],
                args.max_length,
                device,
            )
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            tokens = int((labels != -100).sum().item())
            lang_loss += float(outputs.loss.item()) * max(tokens, 1)
            lang_tokens += max(tokens, 1)
        if lang_tokens:
            lang_losses[lang] = lang_loss / lang_tokens
            total_loss += lang_loss
            total_tokens += lang_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {
        "mean_token_loss": float(mean_loss),
        "ppl": float(math.exp(mean_loss) if mean_loss < 50 else float("inf")),
        "by_language": {k: float(v) for k, v in lang_losses.items()},
    }


def validation_subset(df: pd.DataFrame, lang: str, args) -> pd.DataFrame:
    """Select one stable validation sample reused by loss and generation metrics."""
    if args.eval_samples > 0 and len(df) > args.eval_samples:
        return df.sample(args.eval_samples, random_state=args.seed + sum(map(ord, lang))).sort_index()
    return df


def validation_sample_manifest(val_by_lang: dict, args) -> dict:
    """Record exactly which human validation rows drive checkpoint selection."""
    rows = {}
    for lang, info in sorted(val_by_lang.items()):
        subset = validation_subset(info["df"], lang, args)
        rows[lang] = [
            {
                "row_id": str(row.get("row_id", index)),
                "source": str(row.get("source", "")),
            }
            for index, row in subset.iterrows()
        ]
    return {
        "seed": args.seed,
        "maximum_rows_per_language": args.eval_samples,
        "selection": "pandas.DataFrame.sample with stable per-language seed, then original index order",
        "rows": rows,
    }


@torch.no_grad()
def evaluate_generation(model, tokenizer, val_by_lang, args, device) -> dict:
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    by_language: dict[str, dict] = {}
    bleu_tokenize = "zh" if args.direction == "f2zh" else "13a"
    for lang, info in sorted(val_by_lang.items()):
        df = validation_subset(info["df"], lang, args)
        lang_hypotheses: list[str] = []
        lang_references = df["tgt_text"].astype(str).tolist()
        forced_bos_token_id = ensure_lang_token(tokenizer, info["tgt_lid"])
        for start in range(0, len(df), args.generation_batch_size):
            batch = df.iloc[start : start + args.generation_batch_size]
            tokenizer.src_lang = info["src_lid"]
            enc = tokenizer(
                batch["src_text"].astype(str).tolist(),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_token_type_ids=False,
            )
            enc = {key: value.to(device) for key, value in enc.items()}
            generated = model.generate(
                **enc,
                num_beams=args.validation_beam,
                max_new_tokens=args.validation_max_new_tokens,
                forced_bos_token_id=forced_bos_token_id,
                decoder_start_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            lang_hypotheses.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        by_language[lang] = {"samples": len(df)} | score_translations(
            lang_hypotheses,
            lang_references,
            bleu_tokenize=bleu_tokenize,
        )
        hypotheses.extend(lang_hypotheses)
        references.extend(lang_references)
    model.train()
    return {
        "samples": len(hypotheses),
        "bleu_tokenize": bleu_tokenize,
        "global": score_translations(hypotheses, references, bleu_tokenize=bleu_tokenize),
        "by_language": by_language,
    }


def metric_value(metrics: dict, name: str) -> float:
    if name == "mean_token_loss":
        return float(metrics[name])
    return float(metrics["generation"]["global"][name])


def metric_improved(current: float, best: float | None, name: str, min_delta: float) -> bool:
    if best is None:
        return True
    if name in {"mean_token_loss", "TER"}:
        return current < best - min_delta
    return current > best + min_delta


def save_checkpoint(model, tokenizer, path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "experiment_metadata.json", metadata)


def save_resume_checkpoint(model, tokenizer, optimizer, scheduler, scaler, path: Path, state: dict) -> None:
    """Overwrite one restart checkpoint so preemption costs at most one eval interval."""
    tmp = path.with_name(f"{path.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    save_checkpoint(model, tokenizer, tmp, {"step": state["step"], "kind": "resume"})
    torch.save(
        {
            **state,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        tmp / "trainer_state.pt",
    )
    shutil.rmtree(path, ignore_errors=True)
    tmp.replace(path)


def restore_training_state(path: Path, optimizer, scheduler, scaler) -> dict:
    state = torch.load(path / "trainer_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state.pop("optimizer"))
    scheduler.load_state_dict(state.pop("scheduler"))
    scaler.load_state_dict(state.pop("scaler"))
    random.setstate(state.pop("python_rng"))
    np.random.set_state(state.pop("numpy_rng"))
    torch.set_rng_state(state.pop("torch_rng"))
    cuda_rng = state.pop("cuda_rng")
    if cuda_rng is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_rng)
    return state


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


def verify_setup_artifacts(
    setup_manifest: dict,
    tokenizer_dir: Path,
    model_dir: Path,
) -> None:
    for section, directory in (
        ("tokenizer", tokenizer_dir),
        ("model", model_dir),
    ):
        records = setup_manifest.get(section, {}).get("files", [])
        if not records:
            raise SystemExit(f"Setup manifest has no {section} artifacts")
        for record in records:
            source_name = Path(str(record.get("path") or "")).name
            path = directory / source_name
            if not path.is_file():
                raise SystemExit(f"Missing setup {section} artifact: {path}")
            actual = sha256_file(path)
            if actual != record.get("sha256"):
                raise SystemExit(
                    f"Setup {section} checksum mismatch for {path}"
                )


def build_run_contract(args, profile: dict) -> dict:
    input_hash = sha256_file(args.input)
    corpus_manifest = read_complete_manifest(
        args.corpus_manifest,
        "corpus build manifest",
    )
    if (
        corpus_manifest.get("pipeline_version")
        != profile["corpus_pipeline_version"]
    ):
        raise SystemExit("Corpus pipeline version does not match the recipe")
    if not manifest_contains_hash(corpus_manifest, input_hash):
        raise SystemExit(
            "Training CSV checksum is absent from the corpus build manifest"
        )
    validation = read_complete_manifest(
        args.validation_report,
        "corpus validation report",
    )
    if validation.get("input_sha256") != input_hash:
        raise SystemExit(
            "Corpus validation report does not match the training CSV"
        )
    setup = read_complete_manifest(
        args.setup_manifest,
        "tokenizer setup manifest",
    )
    expected_profile = profile_record(args.profile)
    if (
        setup.get("recipe_id") != profile["recipe_id"]
        or setup.get("profile", {}).get("sha256")
        != expected_profile["sha256"]
        or setup.get("input", {}).get("sha256") != input_hash
    ):
        raise SystemExit(
            "Tokenizer setup manifest does not match corpus/profile"
        )
    verify_setup_artifacts(
        setup,
        args.tokenizer,
        args.model,
    )
    hyperparameters = {
        key: value
        for key, value in vars(args).items()
        if key
        not in {
            "output_dir",
            "resume_from",
            "corpus_manifest",
            "validation_report",
            "setup_manifest",
            "profile",
        }
    }
    return {
        "schema_version": 2,
        "complete": True,
        "recipe_id": profile["recipe_id"],
        "profile": expected_profile,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": input_hash,
        },
        "corpus_manifest": {
            "path": str(args.corpus_manifest.resolve()),
            "sha256": sha256_file(args.corpus_manifest),
        },
        "validation_report": {
            "path": str(args.validation_report.resolve()),
            "sha256": sha256_file(args.validation_report),
        },
        "setup_manifest": {
            "path": str(args.setup_manifest.resolve()),
            "sha256": sha256_file(args.setup_manifest),
        },
        "repository": git_record(),
        "dependencies": dependency_versions(),
        "hyperparameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in hyperparameters.items()
        },
    }


def write_or_verify_run_contract(
    output_dir: Path,
    contract: dict,
) -> str:
    path = output_dir / "run_contract.json"
    digest = stable_hash(contract)
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if stable_hash(existing) != digest:
            raise SystemExit(
                f"Existing run contract does not match this invocation: {path}"
            )
    else:
        write_json(path, contract)
    return digest


def main() -> None:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_PROFILE,
    )
    known, _ = preliminary.parse_known_args()
    profile = load_profile(known.profile)
    defaults = profile["training_defaults"]
    parser = argparse.ArgumentParser(parents=[preliminary])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--setup-manifest", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--direction", choices=direction_choices(), required=True)
    parser.add_argument("--use-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=defaults["steps"], help="Optimizer update steps.")
    parser.add_argument("--batch-size", type=int, default=defaults["batch_size"])
    parser.add_argument("--grad-accum-steps", type=int, default=defaults["grad_accum_steps"])
    parser.add_argument("--max-length", type=int, default=defaults["max_length"])
    parser.add_argument("--learning-rate", type=float, default=defaults["learning_rate"])
    parser.add_argument("--warmup-steps", type=int, default=defaults["warmup_steps"])
    parser.add_argument("--weight-decay", type=float, default=defaults["weight_decay"])
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=defaults["alpha"], help="Language sampling exponent p(lang) ∝ n^alpha.")
    parser.add_argument("--easy-source-weight", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=defaults["label_smoothing"])
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=defaults["precision"])
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-interval", type=int, default=defaults["save_interval"], help="Checkpoint interval. Use 0 to keep only best/final.")
    parser.add_argument("--eval-interval", type=int, default=defaults["generation_eval_interval"])
    parser.add_argument("--log-interval", type=int, default=defaults["log_interval"])
    parser.add_argument("--eval-samples", type=int, default=defaults["generation_eval_samples_per_language"])
    parser.add_argument("--eval-batch-size", type=int, default=defaults["generation_eval_batch_size"])
    parser.add_argument("--generation-batch-size", type=int, default=defaults["generation_eval_batch_size"])
    parser.add_argument("--validation-beam", type=int, default=defaults["validation_beam"])
    parser.add_argument("--validation-max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--best-metric",
        choices=["chrF2", "BLEU", "TER", "mean_token_loss"],
        default=defaults["best_metric"],
        help="Validation metric used for best checkpoint selection and early stopping.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=defaults["early_stopping_patience"], help="Evaluations without improvement; 0 disables.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=defaults["early_stopping_min_delta"])
    parser.add_argument("--early-stopping-start-step", type=int, default=defaults["early_stopping_start_step"])
    parser.add_argument("--resume-from", default="auto", help="Checkpoint directory, 'auto', or 'none'.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.profile = known.profile
    args.target_lang = normalize_target_language(args.target_lang, args.target_col)
    args.target_col = args.target_col or target_col_for(args.target_lang)
    args.target_lid = target_lid_for(args.target_lang)
    direction_target = target_language_from_direction(args.direction, args.target_lang)
    if direction_target != args.target_lang:
        raise SystemExit(
            f"--direction {args.direction!r} targets {direction_target}, "
            f"but --target-lang is {args.target_lang!r}."
        )

    if args.easy_source_weight is None:
        args.easy_source_weight = float(
            defaults[f"{args.direction}_easy_source_weight"]
        )
    if args.eval_interval <= 0:
        raise SystemExit("--eval-interval must be positive because best-model selection requires validation.")
    if args.eval_samples <= 0:
        raise SystemExit("--eval-samples must be positive; use a bounded, fixed validation sample per language.")
    if args.early_stopping_patience < 0:
        raise SystemExit("--early-stopping-patience cannot be negative.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = build_run_contract(args, profile)
    contract_sha256 = write_or_verify_run_contract(
        args.output_dir,
        contract,
    )

    resume_arg = str(args.resume_from).lower()
    resume_path = args.output_dir / "resume" if resume_arg == "auto" else Path(args.resume_from)
    resume_exists = (resume_path / "trainer_state.pt").is_file()
    if resume_arg not in {"auto", "none"} and not resume_exists:
        raise SystemExit(f"Explicit resume checkpoint is incomplete: {resume_path}")
    if resume_arg == "none" or not resume_exists:
        resume_path = None
    load_path = resume_path or args.model
    tokenizer = NllbTokenizer.from_pretrained(resume_path or args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    if len(tokenizer) != model.get_input_embeddings().num_embeddings:
        if len(tokenizer) > model.get_input_embeddings().num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
        else:
            raise SystemExit("Tokenizer is smaller than model embeddings; load matching artifacts.")
    for lid in sorted(set(get_lid(c) for c in FORMOSAN_CODES) | {args.target_lid}):
        ensure_lang_token(tokenizer, lid)

    train_by_lang, val_by_lang, data_report = prepare_data(args, tokenizer)
    write_json(args.output_dir / "data_report.json", data_report)
    write_json(args.output_dir / "validation_sample_manifest.json", validation_sample_manifest(val_by_lang, args))
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(
        args.output_dir / "run_config.json",
        serializable_args
        | {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "run_contract_sha256": contract_sha256,
        },
    )

    model.to(device).train()
    optimizer = Adafactor(
        model.parameters(),
        lr=args.learning_rate,
        scale_parameter=False,
        relative_step=False,
        warmup_init=False,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    use_amp = args.precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    lang_counts = {lang: len(info["df"]) for lang, info in train_by_lang.items()}
    lang_probs = language_sampling_probs(lang_counts, args.alpha)
    langs = list(lang_probs)
    probs = np.array([lang_probs[lang] for lang in langs], dtype=np.float64)
    row_probs = {lang: row_probabilities(info["df"], args.easy_source_weight) for lang, info in train_by_lang.items()}

    train_log = (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8")
    eval_log = (args.output_dir / "eval_log.jsonl").open("a", encoding="utf-8")
    start_step = 1
    best_value = None
    best_step = None
    bad_evaluations = 0
    if resume_path is not None:
        restored = restore_training_state(resume_path, optimizer, scheduler, scaler)
        if restored.pop("run_contract_sha256", None) != contract_sha256:
            raise SystemExit("Resume checkpoint run contract does not match")
        start_step = int(restored["step"]) + 1
        best_value = restored.get("best_value")
        best_step = restored.get("best_step")
        bad_evaluations = int(restored.get("bad_evaluations", 0))
        print(f"[resume] checkpoint={resume_path} next_step={start_step}")
    running_loss = []
    interval_tokens = 0
    interval_started = time.monotonic()
    actual_step = start_step - 1
    stopped_early = False

    progress = trange(
        start_step,
        args.steps + 1,
        dynamic_ncols=True,
        desc=f"Training {args.direction}",
        disable=not sys.stderr.isatty(),
    )
    for step in progress:
        actual_step = step
        optimizer.zero_grad(set_to_none=True)
        update_loss = 0.0
        last_lang = None
        for _ in range(args.grad_accum_steps):
            lang = str(np.random.choice(langs, p=probs))
            last_lang = lang
            info = train_by_lang[lang]
            df = info["df"]
            sampled = np.random.choice(len(df), size=args.batch_size, replace=True, p=row_probs[lang])
            batch = df.iloc[sampled]
            enc, labels = encode_batch(
                tokenizer,
                batch["src_text"].tolist(),
                batch["tgt_text"].tolist(),
                info["src_lid"],
                info["tgt_lid"],
                args.max_length,
                device,
            )
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    labels=labels,
                )
                if args.label_smoothing > 0:
                    loss = F.cross_entropy(
                        outputs.logits.view(-1, outputs.logits.size(-1)),
                        labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=args.label_smoothing,
                    )
                else:
                    loss = outputs.loss
                loss = loss / args.grad_accum_steps
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            update_loss += float(loss.detach().item())
            interval_tokens += int((labels != -100).sum().item())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).item())
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        running_loss.append(update_loss)
        progress.set_postfix(lang=last_lang, loss=f"{np.mean(running_loss[-100:]):.4f}")

        if step % args.log_interval == 0:
            elapsed = max(time.monotonic() - interval_started, 1e-6)
            record = {
                "step": step,
                "loss": float(np.mean(running_loss)),
                "direction": args.direction,
                "last_lang": last_lang,
                "learning_rate": float(scheduler.get_last_lr()[0]),
                "grad_norm": grad_norm,
                "tokens_per_second": float(interval_tokens / elapsed),
                "updates_per_second": float(args.log_interval / elapsed),
                "elapsed_seconds": float(elapsed),
                "cuda_max_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30) if device.type == "cuda" else 0.0,
            }
            train_log.write(json.dumps(record) + "\n")
            train_log.flush()
            print(json.dumps({"event": "train", **record}), flush=True)
            running_loss.clear()
            interval_tokens = 0
            interval_started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

        if val_by_lang and step % args.eval_interval == 0:
            metrics = evaluate_loss(model, tokenizer, val_by_lang, args, device)
            metrics["generation"] = evaluate_generation(model, tokenizer, val_by_lang, args, device)
            metrics["step"] = step
            current_value = metric_value(metrics, args.best_metric)
            improved = metric_improved(current_value, best_value, args.best_metric, args.early_stopping_min_delta)
            metrics["selection"] = {
                "metric": args.best_metric,
                "value": current_value,
                "improved": improved,
                "best_value_before_eval": best_value,
            }
            eval_log.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            eval_log.flush()
            global_generation = metrics["generation"]["global"]
            print(
                f"[eval] step={step} loss={metrics['mean_token_loss']:.4f} "
                f"BLEU={global_generation['BLEU']:.2f} chrF2={global_generation['chrF2']:.2f} "
                f"TER={global_generation['TER']:.2f} selection={args.best_metric}:{current_value:.4f}",
                flush=True,
            )
            if improved:
                best_value = current_value
                best_step = step
                bad_evaluations = 0
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_dir / "best",
                    {
                        "step": step,
                        "best_metric": args.best_metric,
                        "best_value": best_value,
                        "direction": args.direction,
                        "run_contract_sha256": contract_sha256,
                        "validation": metrics,
                    },
                )
            elif step >= args.early_stopping_start_step:
                bad_evaluations += 1
            save_resume_checkpoint(
                model,
                tokenizer,
                optimizer,
                scheduler,
                scaler,
                args.output_dir / "resume",
                {
                    "step": step,
                    "best_value": best_value,
                    "best_step": best_step,
                    "bad_evaluations": bad_evaluations,
                    "run_contract_sha256": contract_sha256,
                },
            )
            if args.early_stopping_patience > 0 and bad_evaluations >= args.early_stopping_patience:
                stopped_early = True
                print(
                    f"[early-stop] step={step} best_step={best_step} "
                    f"best_{args.best_metric}={best_value}",
                    flush=True,
                )
                break

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(
                model,
                tokenizer,
                args.output_dir / "checkpoints" / f"step-{step:06d}",
                {
                    "step": step,
                    "direction": args.direction,
                    "run_contract_sha256": contract_sha256,
                },
            )

    save_checkpoint(
        model,
        tokenizer,
        args.output_dir / "final",
        {
            "step": actual_step,
            "planned_steps": args.steps,
            "stopped_early": stopped_early,
            "best_step": best_step,
            "best_metric": args.best_metric,
            "best_value": best_value,
            "direction": args.direction,
            "run_contract_sha256": contract_sha256,
        },
    )
    shutil.rmtree(args.output_dir / "resume", ignore_errors=True)
    train_log.close()
    eval_log.close()
    print(f"final: {args.output_dir / 'final'}")
    if best_value is not None:
        print(f"best: {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
