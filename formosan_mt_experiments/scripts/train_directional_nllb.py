#!/usr/bin/env python3
"""Directional multilingual NLLB fine-tuning for Formosan↔target-language experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm.auto import trange
from transformers import Adafactor, AutoModelForSeq2SeqLM, NllbTokenizer, get_constant_schedule_with_warmup

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
    target_lid_for,
    target_language_from_direction,
    with_tagged_columns,
    write_json,
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
    df["source_bucket"] = df["source"].map(source_bucket)

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
        df = info["df"]
        if args.eval_samples > 0 and len(df) > args.eval_samples:
            df = df.sample(args.eval_samples, random_state=args.seed + len(lang))
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


def save_checkpoint(model, tokenizer, path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    write_json(path / "experiment_metadata.json", metadata)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-lang", choices=["english", "chinese"], default="english")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--direction", choices=direction_choices(), required=True)
    parser.add_argument("--use-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=300000, help="Optimizer update steps.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=4000)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5, help="Language sampling exponent p(lang) ∝ n^alpha.")
    parser.add_argument("--easy-source-weight", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--lora-r", type=int, default=0, help="Enable PEFT LoRA with this rank. 0 disables LoRA.")
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,out_proj,fc1,fc2",
        help="Comma-separated module names for PEFT LoRA.",
    )
    parser.add_argument("--train-embeddings-with-lora", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--merge-lora-final", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-interval", type=int, default=10000, help="Checkpoint interval. Use 0 to keep only best/final.")
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
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
        args.easy_source_weight = 0.05 if is_formosan_to_target(args.direction) else 0.15

    peft_imports = None
    if args.lora_r > 0:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
            peft_imports = (LoraConfig, TaskType, get_peft_model)
        except Exception as exc:
            raise SystemExit(
                "LoRA requested but peft is not installed. "
                "Install it with `pip install -r formosan_mt_experiments/requirements-extras.txt` "
                "or run with --lora-r 0."
            ) from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    tokenizer = NllbTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
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

    lora_enabled = args.lora_r > 0
    if lora_enabled:
        LoraConfig, TaskType, get_peft_model = peft_imports
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        model = get_peft_model(model, lora_config)
        if args.train_embeddings_with_lora:
            model.get_input_embeddings().weight.requires_grad_(True)
            output_embeddings = model.get_output_embeddings()
            if output_embeddings is not None:
                output_embeddings.weight.requires_grad_(True)
        model.print_trainable_parameters()

    train_by_lang, val_by_lang, data_report = prepare_data(args, tokenizer)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "data_report.json", data_report)
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    write_json(args.output_dir / "run_config.json", serializable_args | {"started_at": time.strftime("%Y-%m-%d %H:%M:%S")})

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
    best_loss = float("inf")
    running_loss = []

    progress = trange(1, args.steps + 1, dynamic_ncols=True, desc=f"Training {args.direction}")
    for step in progress:
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

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        scheduler.step()

        running_loss.append(update_loss)
        progress.set_postfix(lang=last_lang, loss=f"{np.mean(running_loss[-100:]):.4f}")

        if step % args.log_interval == 0:
            record = {
                "step": step,
                "loss": float(np.mean(running_loss)),
                "direction": args.direction,
                "last_lang": last_lang,
            }
            train_log.write(json.dumps(record) + "\n")
            train_log.flush()
            running_loss.clear()

        if val_by_lang and step % args.eval_interval == 0:
            metrics = evaluate_loss(model, tokenizer, val_by_lang, args, device)
            metrics["step"] = step
            eval_log.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            eval_log.flush()
            print(f"\n[eval] step={step} mean_token_loss={metrics['mean_token_loss']:.4f} ppl={metrics['ppl']:.2f}")
            if metrics["mean_token_loss"] < best_loss:
                best_loss = metrics["mean_token_loss"]
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_dir / "best",
                    {"step": step, "best_mean_token_loss": best_loss, "direction": args.direction},
                )

        if args.save_interval > 0 and step % args.save_interval == 0:
            save_checkpoint(
                model,
                tokenizer,
                args.output_dir / "checkpoints" / f"step-{step:06d}",
                {"step": step, "direction": args.direction},
            )

    final_model = model
    if lora_enabled and args.merge_lora_final:
        final_model = model.merge_and_unload()
    save_checkpoint(final_model, tokenizer, args.output_dir / "final", {"step": args.steps, "direction": args.direction})
    train_log.close()
    eval_log.close()
    print(f"final: {args.output_dir / 'final'}")
    if best_loss < float("inf"):
        print(f"best: {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
