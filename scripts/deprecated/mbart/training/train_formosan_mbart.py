#!/usr/bin/env python3
"""
Train mBART-50 on a Formosan <-> (English|Chinese) parallel corpus.

What this does (correctness):
- Sets source special tokens and *always* sets `tokenizer.tgt_lang` before
  calling tokenizer with `text_target=...` (required by MBART-50).
- Uses `text_target=...` so labels are formatted with the correct target
  language prefix; pads in labels are mapped to -100.
- Tiny eval in both directions uses `forced_bos_token_id` for the target lang.
- Trains bidirectionally by randomly flipping direction each step.
- Saves updates directly to input model (overwrites) - perfect for multilingual training.

Examples
--------
# ami <-> english (saves back to input model)
python train_formosan_mbart.py \
  --src-lang amis --tgt-lang english \
  --tokenizer /path/to/formosan_multilingual_mbart_tokenizer \
  --model     /path/to/formosan_multilingual_mbart_model \
  --input ami_en.csv

# paiwan <-> chinese (traditional texts still use zh_CN code)
python train_formosan_mbart.py \
  --src-lang paiwan --tgt-lang chinese \
  --tokenizer /path/to/formosan_multilingual_mbart_tokenizer \
  --model     /path/to/formosan_multilingual_mbart_model \
  --input pwn_zh.csv
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from contextlib import nullcontext

import numpy as np
import pandas as pd
import torch
try:
    # Newer PyTorch API (recommended)
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except Exception:
    # Fallback for older PyTorch
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_NEW = False

from tqdm.auto import trange
from transformers import (
    MBartForConditionalGeneration,
    MBart50Tokenizer,
    Adafactor,
    get_constant_schedule_with_warmup,
)

# ----------------------- language maps (matches your setup) -----------------------
MBART_LANGUAGE_MAP: Dict[str, str] = {
    # Formosan (custom tokens from your setup script)
    "amis": "ami_XX",        # ami
    "bunun": "bnn_XX",       # bnn
    "kavalan": "ckv_XX",     # ckv
    "rukai": "dru_XX",       # dru
    "paiwan": "pwn_XX",      # pwn
    "puyuma": "pyu_XX",      # pyu
    "thao": "ssf_XX",        # ssf
    "saaroa": "sxr_XX",      # sxr
    "sakizaya": "szy_XX",    # szy
    "tao": "tao_XX",         # tao (Yami)
    "atayal": "tay_XX",      # tay
    "seediq": "trv_XX",      # trv
    "tsou": "tsu_XX",        # tsu
    "kanakanavu": "xnb_XX",  # xnb
    "saisiyat": "xsy_XX",    # xsy

    # mBART-50 built-ins
    "english": "en_XX",
    "chinese": "zh_CN",      # (use for both Simplified & Traditional text)
}

FORMOSAN_SET = {
    "amis","bunun","kavalan","rukai","paiwan","puyuma","thao",
    "saaroa","sakizaya","tao","atayal","seediq","tsou","kanakanavu","saisiyat"
}

# ------------------------------- helpers -----------------------------------
def get_mbart_code(name: str) -> str:
    key = name.lower()
    if key not in MBART_LANGUAGE_MAP:
        sys.exit(f"Unsupported language '{name}'. "
                 f"Supported: {', '.join(sorted(MBART_LANGUAGE_MAP))}")
    return MBART_LANGUAGE_MAP[key]

def smart_find_columns(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    override_src: Optional[str],
    override_tgt: Optional[str],
) -> Tuple[str, str]:
    """
    Try to infer the correct column names if not provided.
    Priority:
      1) --src-col / --tgt-col if given
      2) common columns for our corpora: formosan_sentence, chinese_sentence, english_sentence
      3) columns matching language names / aliases
      4) fallback: first two object dtype columns
    """
    if override_src and override_tgt:
        return override_src, override_tgt

    cols = set(df.columns)

    # corpus common
    if src_lang in FORMOSAN_SET and "formosan_sentence" in cols:
        if tgt_lang == "chinese" and "chinese_sentence" in cols:
            return "formosan_sentence", "chinese_sentence"
        if tgt_lang == "english" and "english_sentence" in cols:
            return "formosan_sentence", "english_sentence"

    # language-name-ish
    aliases = {
        "english": {"english", "en", "en_sentence", "english_sentence"},
        "chinese": {"chinese", "zh", "zh_cn", "chinese_sentence"},
        "amis": {"amis", "ami"}, "bunun": {"bunun","bnn"}, "kavalan": {"kavalan","ckv"},
        "rukai": {"rukai","dru"}, "paiwan": {"paiwan","pwn"}, "puyuma": {"puyuma","pyu"},
        "thao": {"thao","ssf"}, "saaroa": {"saaroa","sxr"}, "sakizaya": {"sakizaya","szy"},
        "tao": {"tao","yami"}, "atayal": {"atayal","tay"}, "seediq": {"seediq","trv"},
        "tsou": {"tsou","tsu"}, "kanakanavu": {"kanakanavu","xnb"}, "saisiyat": {"saisiyat","xsy"},
    }
    src_cands = [c for c in df.columns if c.lower() in aliases.get(src_lang, set())]
    tgt_cands = [c for c in df.columns if c.lower() in aliases.get(tgt_lang, set())]
    if src_cands and tgt_cands:
        return src_cands[0], tgt_cands[0]

    # fallback to first two text columns
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(text_cols) >= 2:
        return text_cols[0], text_cols[1]

    sys.exit("Could not infer language columns. Use --src-col and --tgt-col.")

def load_splits(csv_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        tr = df[df["split"].str.lower().eq("train")]
        va = df[df["split"].str.lower().isin(["valid", "val", "validate"])]
        te = df[df["split"].str.lower().eq("test")]
        if len(tr) == 0:
            sys.exit("No 'train' rows found in CSV 'split' column.")
        return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
    else: 
        raise SystemExit("❌ CSV must contain a 'split' column.")

def load_tok_model(tok_dir: str, model_dir: str, device: str):
    tok = MBart50Tokenizer.from_pretrained(tok_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    # ensure embedding size matches tokenizer (HF guidance)
    model.resize_token_embeddings(len(tok))
    model.to(torch.device(device))
    return tok, model

def restore_custom_lang_ids(tokenizer: MBart50Tokenizer, codes: List[str]) -> None:
    # If you added custom LIDs, re-wire after reload
    for c in codes:
        if c not in tokenizer.lang_code_to_id:
            tid = tokenizer.convert_tokens_to_ids(c)
            tokenizer.lang_code_to_id[c] = tid
            tokenizer.id_to_lang_code[tid] = c

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
def _encode_pair_batch_mbart(
    tokenizer: MBart50Tokenizer,
    src_texts: List[str],
    tgt_texts: List[str],
    src_code: str,
    tgt_code: str,
    max_length: int,
    device: torch.device,
):
    # mBART-50 MUST know both src and tgt languages before tokenization
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(src_code)
    else:
        tokenizer.src_lang = src_code

    if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
        tokenizer.set_tgt_lang_special_tokens(tgt_code)
    else:
        tokenizer.tgt_lang = tgt_code
    tokenizer.tgt_lang = tgt_code  # belt-and-suspenders

    batch = tokenizer(
        src_texts,
        text_target=tgt_texts,  # ensures target gets the tgt lang prefix
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    labels[labels == tokenizer.pad_token_id] = -100
    return input_ids, attn_mask, labels


# ------------------------------ training -----------------------------------
def training_step(
    tokenizer: MBart50Tokenizer,
    model: MBartForConditionalGeneration,
    src_texts: List[str],
    tgt_texts: List[str],
    src_code: str,
    tgt_code: str,
    max_length: int,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
):
    # Required by MBART-50: set source & target lang context before tokenization
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(src_code)
    else:
        tokenizer.src_lang = src_code

    if hasattr(tokenizer, "set_tgt_lang_special_tokens"):
        tokenizer.set_tgt_lang_special_tokens(tgt_code)
    else:
        tokenizer.tgt_lang = tgt_code
    tokenizer.tgt_lang = tgt_code  # make extra sure __call__(...) sees it

    # Encode with text_target so labels carry the [tgt_lang] prefix
    batch = tokenizer(
        src_texts,
        text_target=tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    input_ids = batch["input_ids"].to(device)
    attn_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    # Mask pads in labels
    labels[labels == tokenizer.pad_token_id] = -100

    if use_amp:
        if _AMP_NEW:
            with autocast(device_type=device.type):
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
        else:
            with autocast():
                loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
        assert scaler is not None
        scaler.scale(loss).backward()
    else:
        loss = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
        loss.backward()
    return loss.detach()

@torch.no_grad()
def eval_on_val_mbart(
    tokenizer: MBart50Tokenizer,
    model: MBartForConditionalGeneration,
    df_val: pd.DataFrame,
    s_col: str,
    t_col: str,
    src_code: str,
    tgt_code: str,
    max_length: int,
    device: torch.device,
    sample_size: int = 500,
    eval_batch_size: int = 32,
    num_beams: int = 5,
    n_show: int = 3,
    use_amp: bool = False,
):
    if len(df_val) == 0:
        print("[Eval] No validation data available.")
        return

    n = min(sample_size, len(df_val))
    idx = np.random.choice(len(df_val), size=n, replace=False)
    sub = df_val.iloc[idx]

    src_list = sub[s_col].astype(str).tolist()
    tgt_list = sub[t_col].astype(str).tolist()

    amp_ctx = (autocast(device_type=device.type) if (_AMP_NEW and use_amp) else
               (autocast() if (not _AMP_NEW and use_amp) else nullcontext()))

    def mean_loss(src_texts: List[str], tgt_texts: List[str], from_code: str, to_code: str) -> float:
        total, count = 0.0, 0
        for i in range(0, len(src_texts), eval_batch_size):
            bs = src_texts[i:i+eval_batch_size]
            bt = tgt_texts[i:i+eval_batch_size]
            input_ids, attn_mask, labels = _encode_pair_batch_mbart(
                tokenizer, bs, bt, from_code, to_code, max_length, device
            )
            with amp_ctx:
                out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                loss = out.loss
            total += loss.item() * len(bs)
            count += len(bs)
        return total / max(1, count)

    # Losses (both directions)
    fwd_loss = mean_loss(src_list, tgt_list, src_code, tgt_code)  # src->tgt
    bwd_loss = mean_loss(tgt_list, src_list, tgt_code, src_code)  # tgt->src
    mean_bi = 0.5 * (fwd_loss + bwd_loss)
    try:
        ppl = float(np.exp(mean_bi))
    except OverflowError:
        ppl = float("inf")

    print(f"\n[Eval] Sampled {n} val rows | "
          f"src->tgt loss: {fwd_loss:.4f} | tgt->src loss: {bwd_loss:.4f} | "
          f"mean: {mean_bi:.4f} | ppl≈{ppl:.2f}")

    # Show k examples each way
    k = min(n_show, n)

    # src -> tgt
    show_idx = np.random.choice(n, size=k, replace=False)
    show_src = [src_list[i] for i in show_idx]
    show_ref = [tgt_list[i] for i in show_idx]

    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(src_code)
    else:
        tokenizer.src_lang = src_code
    enc = tokenizer(
        show_src, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    forced_tgt = tokenizer.lang_code_to_id[tgt_code]
    outs = model.generate(
        **enc, max_length=max_length, num_beams=num_beams, forced_bos_token_id=forced_tgt
    )
    hyps = tokenizer.batch_decode(outs, skip_special_tokens=True)

    print("\n[Eval] Examples (src->tgt):")
    for s, r, h in zip(show_src, show_ref, hyps):
        print(f"  SRC: {s}")
        print(f"  REF: {r}")
        print(f"  HYP: {h}\n")

    # tgt -> src
    show_idx2 = np.random.choice(n, size=k, replace=False)
    show_src2 = [tgt_list[i] for i in show_idx2]
    show_ref2 = [src_list[i] for i in show_idx2]

    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(tgt_code)
    else:
        tokenizer.src_lang = tgt_code
    enc2 = tokenizer(
        show_src2, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    ).to(device)
    forced_src = tokenizer.lang_code_to_id[src_code]
    outs2 = model.generate(
        **enc2, max_length=max_length, num_beams=num_beams, forced_bos_token_id=forced_src
    )
    hyps2 = tokenizer.batch_decode(outs2, skip_special_tokens=True)

    print("[Eval] Examples (tgt->src):")
    for s, r, h in zip(show_src2, show_ref2, hyps2):
        print(f"  SRC: {s}")
        print(f"  REF: {r}")
        print(f"  HYP: {h}\n")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-lang", required=True, help="e.g. amis, paiwan, tsou, ...")
    ap.add_argument("--tgt-lang", required=True, help="english or chinese")
    ap.add_argument("--tokenizer", required=True, help="path to tokenizer dir from setup")
    ap.add_argument("--model",     required=True, help="path to model dir from setup")
    ap.add_argument("--input", type=Path, required=True, help="CSV with parallel data")

    # Optional explicit column names (overrides auto-detect)
    ap.add_argument("--src-col", default=None)
    ap.add_argument("--tgt-col", default=None)

    # Training hyperparams
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--learning-rate", type=float, default=5e-5)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip-threshold", type=float, default=1.0)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging / saving / eval
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--eval-interval", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=100)
    ap.add_argument("--eval-beams", type=int, default=5)

    # Eval
    ap.add_argument("--eval-sample-size", type=int, default=500,
                help="Random validation rows to evaluate each eval step (cap at len(val)).")
    ap.add_argument("--eval-batch-size", type=int, default=32,
                help="Batch size for loss computation during eval.")

    # Device / precision
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true")

    # Repro
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Output naming - save over input model
    if args.output_dir is None:
        args.output_dir = args.model  # Save back to input model directory
    if args.model_name is None:
        args.model_name = "formosan-multilingual-mbart"
    
    print(f"Will save model updates to: {args.output_dir}")

    # Device
    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    device = torch.device(device)

    # Seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Language codes
    src_code = get_mbart_code(args.src_lang)
    tgt_code = get_mbart_code(args.tgt_lang)
    print(f"Languages: {args.src_lang} -> {src_code} | {args.tgt_lang} -> {tgt_code}")

    # Data
    df_train_all, df_val_all, df_test_all = load_splits(args.input)
    s_col, t_col = smart_find_columns(df_train_all, args.src_lang, args.tgt_lang, args.src_col, args.tgt_col)
    print(f"Using columns: {s_col} -> {t_col}")

    df_train = df_train_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_val   = df_val_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_test  = df_test_all[[s_col, t_col]].dropna().reset_index(drop=True)
    assert len(df_train), "No training data."

    # Tokenizer + model (your customized tokenizer already has extra lang codes & chars)
    tokenizer, model = load_tok_model(args.tokenizer, args.model, str(device))
    # Re-wire custom language ids in case loading lost them
    restore_custom_lang_ids(tokenizer, [src_code, tgt_code])

    # Optim + sched
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=False,
        relative_step=False,
        lr=args.learning_rate,
        clip_threshold=args.clip_threshold,
        weight_decay=args.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)

    scaler = GradScaler(device.type if (args.fp16 and device.type == "cuda") else "cpu",
                        enabled=(args.fp16 and device.type == "cuda"))
    model.train()
    optimizer.zero_grad(set_to_none=True)

    print(f"Starting training on {device} for {args.steps} steps...")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum_steps} | Max len: {args.max_length} | FP16: {bool(scaler.is_enabled())}")

    losses: List[float] = []
    pbar = trange(args.steps, desc="Training", dynamic_ncols=True)

    # Arrays for quick sampling
    train_src = df_train[s_col].astype(str).to_numpy()
    train_tgt = df_train[t_col].astype(str).to_numpy()

    for step in pbar:
        try:
            # Flip direction per step (bidirectional sampling)
            idx = np.random.randint(0, len(df_train), size=args.batch_size)
            if random.random() < 0.5:
                # src -> tgt
                src_texts = df_train[s_col].iloc[idx].astype(str).tolist()
                tgt_texts = df_train[t_col].iloc[idx].astype(str).tolist()
                s_code, t_code = src_code, tgt_code
            else:
                # tgt -> src
                src_texts = df_train[t_col].iloc[idx].astype(str).tolist()
                tgt_texts = df_train[s_col].iloc[idx].astype(str).tolist()
                s_code, t_code = tgt_code, src_code

            loss = training_step(
                tokenizer, model, src_texts, tgt_texts,
                s_code, t_code, args.max_length, device,
                use_amp=scaler.is_enabled(), scaler=scaler
            )
            losses.append(loss.item())

            if (step + 1) % args.grad_accum_steps == 0:
                # Unscale+clip & step
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            if step % args.log_interval == 0 and losses:
                print("\n   ")
                print(f"Step {step} | Loss: {np.mean(losses[-min(len(losses), args.log_interval):]):.4f}")
                recent = losses[-min(len(losses), args.log_interval):]
                pbar.set_postfix(loss=f"{np.mean(recent):.4f}", dir=f"{s_code}->{t_code}")

            if args.eval_interval and step > 0 and step % args.eval_interval == 0 and len(df_val):
                model.eval()
                eval_on_val_mbart(
                    tokenizer, model,
                    df_val, s_col, t_col,
                    src_code, tgt_code,
                    args.max_length, device,
                    sample_size=args.eval_sample_size,
                    eval_batch_size=args.eval_batch_size,
                    num_beams=args.eval_beams,
                    n_show=min(3, args.eval_samples),
                    use_amp=(scaler.is_enabled() and device.type == "cuda"),
                )
                cleanup_cuda()
                model.train()


            if step > 0 and step % args.save_interval == 0:
                # Save directly to model directory (overwrite)
                print(f"\n[Save] Updating model at step {step} -> {args.output_dir}")
                model.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                print(f"[Save] Model updated successfully")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                optimizer.zero_grad(set_to_none=True)
                cleanup_cuda()
                continue
            raise

    # Final save - overwrite input model with trained version
    print(f"\n[Final Save] Saving final model to: {args.output_dir}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"✅ Training complete! Updated model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
