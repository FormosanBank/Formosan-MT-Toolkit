#!/usr/bin/env python3
"""Formosan denoising pre-adaptation for NLLB before MT fine-tuning."""

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
    corrupt_text,
    get_lid,
    language_sampling_probs,
    read_parallel_csv,
    source_bucket,
    write_json,
)
from train_directional_nllb import encode_batch, ensure_control_tags, ensure_lang_token, save_checkpoint


def prepare_data(args, tokenizer: NllbTokenizer) -> tuple[dict, dict, dict]:
    df = read_parallel_csv(args.input)
    if "split" not in df.columns:
        raise SystemExit("DAE input CSV must have split values train/validate/test.")
    df["split"] = df["split"].astype(str).str.lower()
    df["source_bucket"] = df["source"].map(source_bucket)
    if args.use_tags and args.validate_tags:
        ensure_control_tags(tokenizer, df, "dae")
    if args.use_tags:
        df["dae_prefix"] = df.apply(lambda row: build_prefix(row, "dae"), axis=1)
    else:
        df["dae_prefix"] = ""

    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].isin(["validate", "valid", "val"])].copy()
    train_by_lang = {}
    val_by_lang = {}
    for lang in sorted(train["lang_code"].unique()):
        if lang not in FORMOSAN_CODES:
            continue
        lid = get_lid(lang)
        tr = train[train["lang_code"].eq(lang)].reset_index(drop=True)
        va = val[val["lang_code"].eq(lang)].reset_index(drop=True)
        train_by_lang[lang] = {"df": tr, "src_lid": lid, "tgt_lid": lid}
        if not va.empty:
            val_by_lang[lang] = {"df": va, "src_lid": lid, "tgt_lid": lid}
    return train_by_lang, val_by_lang, {
        "input_rows": int(len(df)),
        "train_rows": int(len(train)),
        "validate_rows": int(len(val)),
        "train_by_language": {k: int(len(v["df"])) for k, v in train_by_lang.items()},
        "validate_by_language": {k: int(len(v["df"])) for k, v in val_by_lang.items()},
    }


def row_probabilities(df: pd.DataFrame, easy_source_weight: float) -> np.ndarray:
    weights = np.ones(len(df), dtype=np.float64)
    easy = df["source_bucket"].isin(EASY_BUCKETS).to_numpy()
    weights[easy] = easy_source_weight
    weights = np.maximum(weights, 1e-8)
    return weights / weights.sum()


def make_dae_batch(batch: pd.DataFrame, rng: random.Random, args) -> tuple[list[str], list[str]]:
    clean = batch["formosan_sentence"].fillna("").astype(str).tolist()
    prefixes = batch["dae_prefix"].fillna("").astype(str).tolist()
    corrupted = [
        (prefix + " " + corrupt_text(
            text,
            rng,
            word_dropout=args.word_dropout,
            span_mask=args.span_mask,
            shuffle_distance=args.shuffle_distance,
        )).strip()
        for prefix, text in zip(prefixes, clean)
    ]
    return corrupted, clean


@torch.no_grad()
def evaluate_loss(model, tokenizer, val_by_lang, args, device, rng: random.Random) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    by_lang = {}
    for lang, info in sorted(val_by_lang.items()):
        df = info["df"]
        if args.eval_samples > 0 and len(df) > args.eval_samples:
            df = df.sample(args.eval_samples, random_state=args.seed + len(lang))
        lang_loss = 0.0
        lang_tokens = 0
        for start in range(0, len(df), args.eval_batch_size):
            batch = df.iloc[start : start + args.eval_batch_size]
            src, tgt = make_dae_batch(batch, rng, args)
            enc, labels = encode_batch(
                tokenizer,
                src,
                tgt,
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
            by_lang[lang] = lang_loss / lang_tokens
            total_loss += lang_loss
            total_tokens += lang_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {
        "mean_token_loss": float(mean_loss),
        "ppl": float(math.exp(mean_loss) if mean_loss < 50 else float("inf")),
        "by_language": {k: float(v) for k, v in by_lang.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--use-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--easy-source-weight", type=float, default=0.15)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--word-dropout", type=float, default=0.10)
    parser.add_argument("--span-mask", type=float, default=0.15)
    parser.add_argument("--shuffle-distance", type=int, default=3)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--eval-samples", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

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
    for lid in sorted(set(get_lid(c) for c in FORMOSAN_CODES)):
        ensure_lang_token(tokenizer, lid)

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
    running = []

    progress = trange(1, args.steps + 1, dynamic_ncols=True, desc="DAE pretraining")
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
            src, tgt = make_dae_batch(batch, rng, args)
            enc, labels = encode_batch(
                tokenizer,
                src,
                tgt,
                info["src_lid"],
                info["tgt_lid"],
                args.max_length,
                device,
            )
            with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
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

        running.append(update_loss)
        progress.set_postfix(lang=last_lang, loss=f"{np.mean(running[-100:]):.4f}")
        if step % args.log_interval == 0:
            train_log.write(json.dumps({"step": step, "loss": float(np.mean(running)), "last_lang": last_lang}) + "\n")
            train_log.flush()
            running.clear()
        if val_by_lang and step % args.eval_interval == 0:
            metrics = evaluate_loss(model, tokenizer, val_by_lang, args, device, rng)
            metrics["step"] = step
            eval_log.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            eval_log.flush()
            print(f"\n[eval] step={step} mean_token_loss={metrics['mean_token_loss']:.4f} ppl={metrics['ppl']:.2f}")
            if metrics["mean_token_loss"] < best_loss:
                best_loss = metrics["mean_token_loss"]
                save_checkpoint(model, tokenizer, args.output_dir / "best", {"step": step, "best_mean_token_loss": best_loss, "stage": "dae"})
        if step % args.save_interval == 0:
            save_checkpoint(model, tokenizer, args.output_dir / "checkpoints" / f"step-{step:06d}", {"step": step, "stage": "dae"})

    save_checkpoint(model, tokenizer, args.output_dir / "final", {"step": args.steps, "stage": "dae"})
    train_log.close()
    eval_log.close()
    print(f"final: {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
