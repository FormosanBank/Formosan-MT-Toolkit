#!/usr/bin/env python3
"""Directional multilingual fine-tuning for Formosan MT."""

from __future__ import annotations

import json
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import milmmt_runtime as milmmt
import nllb_runtime as nllb
import numpy as np
import torch
import torch.nn.functional as F
from mt_common import (
    language_sampling_probs,
    write_json,
)
from torch.amp import GradScaler, autocast
from tqdm.auto import trange
from training_cli import parse_training_args
from training_data import (
    collate_encoded_batches,
    encode_batch,
    evaluate_generation,
    evaluate_loss,
    move_encoded_batch,
    prepare_data,
    training_source_texts,
    validation_sample_manifest,
)
from training_state import (
    build_run_contract,
    clone_model_checkpoint,
    metric_improved,
    metric_value,
    restore_training_state,
    save_checkpoint,
    save_resume_checkpoint,
    write_or_verify_run_contract,
)
from transformers import (
    Adafactor,
    get_constant_schedule_with_warmup,
    get_inverse_sqrt_schedule,
)

LOAD_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def runtime_for(profile: dict):
    return {
        "nllb": nllb,
        "milmmt": milmmt,
    }[profile["model_family"]]


def compute_training_loss(
    model,
    encoded: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    model_family: str,
    label_smoothing: float,
) -> torch.Tensor:
    """Compute one training loss without duplicate NLLB cross-entropy work."""
    if model_family == "nllb" and label_smoothing > 0:
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(labels=labels)
        outputs = model(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
        return F.cross_entropy(
            outputs.logits.view(-1, outputs.logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
    return model(
        input_ids=encoded["input_ids"],
        attention_mask=encoded["attention_mask"],
        labels=labels,
    ).loss


def build_physical_batch(
    *,
    args,
    tokenizer,
    runtime,
    train_by_lang: dict,
    langs: list[str],
    probs: np.ndarray,
    pin_memory: bool,
) -> dict:
    """Preserve language draws while combining chunks into a larger GPU batch."""
    encoded_chunks = []
    sampled_languages = []
    metadata_dropout = {"dialect": 0}
    chunk_count = args.batch_size // args.language_sampling_chunk_size
    for _ in range(chunk_count):
        lang = str(np.random.choice(langs, p=probs))
        sampled_languages.append(lang)
        info = train_by_lang[lang]
        df = info["df"]
        sampled = np.random.choice(
            len(df),
            size=args.language_sampling_chunk_size,
            replace=True,
            p=info["row_sampling_probs"],
        )
        batch = df.iloc[sampled]
        source_texts, dropout_counts = training_source_texts(
            batch,
            args=args,
            runtime=runtime,
        )
        for name, count in dropout_counts.items():
            metadata_dropout[name] += count
        encoded_chunks.append(
            encode_batch(
                tokenizer,
                source_texts,
                batch["tgt_text"].tolist(),
                info["task"],
                args.max_length,
                None,
                runtime,
            )
        )
    encoded, labels, token_count = collate_encoded_batches(
        encoded_chunks,
        pad_token_id=int(tokenizer.pad_token_id),
        pin_memory=pin_memory,
    )
    return {
        "encoded": encoded,
        "labels": labels,
        "token_count": token_count,
        "metadata_dropout": metadata_dropout,
        "sampled_languages": sampled_languages,
    }


def build_training_update(**batch_kwargs) -> list[dict]:
    args = batch_kwargs["args"]
    return [build_physical_batch(**batch_kwargs) for _ in range(args.grad_accum_steps)]


def main() -> None:
    args, profile = parse_training_args()
    runtime = runtime_for(profile)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if args.precision == "bf16" and device_name == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("This GPU does not support bf16 training")
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
    checkpoint_contract = {
        "model_family": runtime.MODEL_FAMILY,
        "recipe_id": profile["recipe_id"],
        "mt_standardization": profile["mt_standardization"],
    }
    tokenizer = runtime.load_tokenizer(resume_path or args.tokenizer)
    model = runtime.load_model(
        load_path,
        dtype=LOAD_DTYPES[args.load_dtype],
    )
    runtime.configure_model(model, tokenizer)
    runtime.validate_model_tokenizer(model, tokenizer)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_by_lang, val_by_lang, data_report = prepare_data(
        args,
        tokenizer,
        runtime,
    )
    write_json(args.output_dir / "data_report.json", data_report)
    write_json(args.output_dir / "validation_sample_manifest.json", validation_sample_manifest(val_by_lang, args))
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    model.to(device).train()
    fused_optimizer_active = bool(args.fused_optimizer and device.type == "cuda")
    if runtime.MODEL_FAMILY == "nllb":
        optimizer = Adafactor(
            model.parameters(),
            lr=args.learning_rate,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            weight_decay=args.weight_decay,
        )
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    else:
        optimizer_kwargs = {"fused": True} if fused_optimizer_active else {}
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            **optimizer_kwargs,
        )
        scheduler = get_inverse_sqrt_schedule(
            optimizer,
            num_warmup_steps=args.warmup_steps,
        )
    write_json(
        args.output_dir / "run_config.json",
        serializable_args
        | {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_family": runtime.MODEL_FAMILY,
            "run_contract_sha256": contract_sha256,
            "fused_optimizer_active": fused_optimizer_active,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
    )
    use_amp = args.precision in {"bf16", "fp16"} and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    lang_counts = {lang: len(info["df"]) for lang, info in train_by_lang.items()}
    lang_probs = language_sampling_probs(
        lang_counts,
        args.language_sampling_alpha,
    )
    langs = list(lang_probs)
    probs = np.array([lang_probs[lang] for lang in langs], dtype=np.float64)

    train_log = (args.output_dir / "train_log.jsonl").open("a", encoding="utf-8")
    eval_log = (args.output_dir / "eval_log.jsonl").open("a", encoding="utf-8")
    start_step = 1
    best_value = None
    best_step = None
    bad_evaluations = 0
    resume_step = None
    if resume_path is not None:
        restored = restore_training_state(resume_path, optimizer, scheduler, scaler)
        if restored.pop("run_contract_sha256", None) != contract_sha256:
            raise SystemExit("Resume checkpoint run contract does not match")
        resume_step = int(restored["step"])
        start_step = resume_step + 1
        best_value = restored.get("best_value")
        best_step = restored.get("best_step")
        bad_evaluations = int(restored.get("bad_evaluations", 0))
        print(f"[resume] checkpoint={resume_path} next_step={start_step}")
    interval_loss = torch.zeros((), device=device)
    interval_tokens = 0
    interval_metadata_dropout = {"dialect": 0}
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
    pin_memory = device.type == "cuda" and torch.cuda.is_available()
    batch_kwargs = {
        "args": args,
        "tokenizer": tokenizer,
        "runtime": runtime,
        "train_by_lang": train_by_lang,
        "langs": langs,
        "probs": probs,
        "pin_memory": pin_memory,
    }
    last_lang = None
    last_grad_norm = torch.zeros((), device=device)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mt-batch") as executor:
        current_update = executor.submit(build_training_update, **batch_kwargs)
        for step in progress:
            actual_step = step
            prepared_batches = current_update.result()
            checkpoint_boundary = (
                bool(val_by_lang and step % args.eval_interval == 0)
                or bool(args.save_interval > 0 and step % args.save_interval == 0)
                or step == args.steps
            )
            next_update = (
                executor.submit(build_training_update, **batch_kwargs)
                if step < args.steps and not checkpoint_boundary
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            for prepared in prepared_batches:
                enc, labels = move_encoded_batch(
                    prepared["encoded"],
                    prepared["labels"],
                    device,
                )
                last_lang = prepared["sampled_languages"][-1]
                for name, count in prepared["metadata_dropout"].items():
                    interval_metadata_dropout[name] += count
                interval_tokens += prepared["token_count"]
                with autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    loss = (
                        compute_training_loss(
                            model,
                            enc,
                            labels,
                            model_family=runtime.MODEL_FAMILY,
                            label_smoothing=args.label_smoothing,
                        )
                        / args.grad_accum_steps
                    )
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                interval_loss += loss.detach()

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            last_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm).detach()
            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()

            if step % args.log_interval == 0:
                elapsed = max(time.monotonic() - interval_started, 1e-6)
                mean_loss = float(interval_loss.item() / args.log_interval)
                record = {
                    "step": step,
                    "loss": mean_loss,
                    "direction": args.direction,
                    "last_lang": last_lang,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "grad_norm": float(last_grad_norm.item()),
                    "tokens_per_second": float(interval_tokens / elapsed),
                    "updates_per_second": float(args.log_interval / elapsed),
                    "elapsed_seconds": float(elapsed),
                    "metadata_dropout_rows": dict(interval_metadata_dropout),
                    "cuda_max_memory_gb": float(torch.cuda.max_memory_allocated() / 2**30)
                    if device.type == "cuda"
                    else 0.0,
                }
                train_log.write(json.dumps(record) + "\n")
                train_log.flush()
                print(json.dumps({"event": "train", **record}), flush=True)
                progress.set_postfix(lang=last_lang, loss=f"{mean_loss:.4f}")
                interval_loss.zero_()
                interval_tokens = 0
                interval_metadata_dropout = {"dialect": 0}
                interval_started = time.monotonic()
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()

            if val_by_lang and step % args.eval_interval == 0:
                if next_update is not None:
                    next_update.result()
                metrics = evaluate_loss(
                    model,
                    tokenizer,
                    val_by_lang,
                    args,
                    device,
                    runtime,
                )
                metrics["generation"] = evaluate_generation(
                    model,
                    tokenizer,
                    val_by_lang,
                    args,
                    device,
                    runtime,
                )
                metrics["step"] = step
                current_value = metric_value(metrics, args.best_metric)
                improved = metric_improved(
                    current_value,
                    best_value,
                    args.best_metric,
                    args.early_stopping_min_delta,
                )
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
                    f"TER={global_generation['TER']:.2f} "
                    f"selection={args.best_metric}:{current_value:.4f}",
                    flush=True,
                )
                if improved:
                    best_value = current_value
                    best_step = step
                    bad_evaluations = 0
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
                        **checkpoint_contract,
                        "step": step,
                        "best_value": best_value,
                        "best_step": best_step,
                        "bad_evaluations": bad_evaluations,
                        "run_contract_sha256": contract_sha256,
                    },
                )
                resume_step = step
                if improved:
                    clone_model_checkpoint(
                        args.output_dir / "resume",
                        args.output_dir / "best",
                        {
                            **checkpoint_contract,
                            "step": step,
                            "best_metric": args.best_metric,
                            "best_value": best_value,
                            "direction": args.direction,
                            "run_contract_sha256": contract_sha256,
                            "validation": metrics,
                        },
                    )
                if args.early_stopping_patience > 0 and bad_evaluations >= args.early_stopping_patience:
                    stopped_early = True
                    print(
                        f"[early-stop] step={step} best_step={best_step} best_{args.best_metric}={best_value}",
                        flush=True,
                    )
                    break

            if args.save_interval > 0 and step % args.save_interval == 0:
                if next_update is not None:
                    next_update.result()
                save_checkpoint(
                    model,
                    tokenizer,
                    args.output_dir / "checkpoints" / f"step-{step:06d}",
                    {
                        **checkpoint_contract,
                        "step": step,
                        "direction": args.direction,
                        "run_contract_sha256": contract_sha256,
                    },
                )
            if step < args.steps and not stopped_early:
                current_update = next_update or executor.submit(
                    build_training_update,
                    **batch_kwargs,
                )

    final_metadata = {
        **checkpoint_contract,
        "step": actual_step,
        "planned_steps": args.steps,
        "stopped_early": stopped_early,
        "best_step": best_step,
        "best_metric": args.best_metric,
        "best_value": best_value,
        "direction": args.direction,
        "run_contract_sha256": contract_sha256,
    }
    if resume_step == actual_step and (args.output_dir / "resume" / "trainer_state.pt").is_file():
        clone_model_checkpoint(
            args.output_dir / "resume",
            args.output_dir / "final",
            final_metadata,
        )
    else:
        save_checkpoint(
            model,
            tokenizer,
            args.output_dir / "final",
            final_metadata,
        )
    shutil.rmtree(args.output_dir / "resume", ignore_errors=True)
    train_log.close()
    eval_log.close()
    print(f"final: {args.output_dir / 'final'}")
    if best_value is not None:
        print(f"best: {args.output_dir / 'best'}")


if __name__ == "__main__":
    main()
