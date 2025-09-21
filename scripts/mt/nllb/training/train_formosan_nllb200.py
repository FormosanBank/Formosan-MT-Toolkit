#!/usr/bin/env python3
"""
Train NLLB-200 (distilled-600M) on a Formosan <-> (English|Chinese) parallel corpus.

Key NLLB differences vs mBART:
- Set tokenizer.src_lang for inputs; DO NOT prefix target labels with a lang token.
- For generation/eval, ALWAYS pass forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lid].
- Your tokenizer/model dirs come from the 'formosan' setup script you just made.
- Works with Transformers >= 4.38.

Examples
--------
# amis <-> english
python train_formosan_nllb.py \
  --src-lang amis --tgt-lang english \
  --tokenizer formosan_multilingual_nllb_tokenizer \
  --model     formosan_multilingual_nllb_model \
  --input ami_en.csv \
  --output-dir runs/ami_en_nllb

# paiwan <-> chinese (Traditional by default; override with --tgt-lid zho_Hans if needed)
python train_formosan_nllb.py \
  --src-lang paiwan --tgt-lang chinese \
  --tokenizer formosan_multilingual_nllb_tokenizer \
  --model     formosan_multilingual_nllb_model \
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

import numpy as np
import pandas as pd
import torch

# AMP (new vs old API)
try:
    from torch.amp import autocast, GradScaler
    _AMP_NEW = True
except Exception:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _AMP_NEW = False

from tqdm.auto import trange
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
    Adafactor,
    get_constant_schedule_with_warmup,
)

# ----------------------- language maps (align with your setup) -----------------------
# NLLB language IDs (LIDs). Formosan ones were added by your tokenizer setup.
NLLB_LANGUAGE_MAP: Dict[str, str] = {
    # Formosan (custom; Latin orthographies unless you chose differently)
    "amis": "ami_Latn",
    "bunun": "bnn_Latn",
    "kavalan": "ckv_Latn",
    "rukai": "dru_Latn",
    "paiwan": "pwn_Latn",
    "puyuma": "pyu_Latn",
    "thao": "ssf_Latn",
    "saaroa": "sxr_Latn",
    "sakizaya": "szy_Latn",
    "tao": "tao_Latn",        # (Yami)
    "atayal": "tay_Latn",
    "seediq": "trv_Latn",
    "tsou": "tsu_Latn",
    "kanakanavu": "xnb_Latn",
    "saisiyat": "xsy_Latn",

    # Built-ins
    "english": "eng_Latn",
    # Default to Traditional Chinese; override with --src-lid/--tgt-lid for Simplified
    "chinese": "zho_Hant",
}

FORMOSAN_SET = {
    "amis","bunun","kavalan","rukai","paiwan","puyuma","thao",
    "saaroa","sakizaya","tao","atayal","seediq","tsou","kanakanavu","saisiyat"
}

# ------------------------------- helpers -----------------------------------
def get_nllb_code(name: str) -> str:
    key = name.lower()
    if key not in NLLB_LANGUAGE_MAP:
        sys.exit(f"Unsupported language '{name}'. "
                 f"Supported: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
    return NLLB_LANGUAGE_MAP[key]

def smart_find_columns(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    override_src: Optional[str],
    override_tgt: Optional[str],
) -> Tuple[str, str]:
    """
    Infer parallel text columns if not specified.

    Priority:
      1) --src-col / --tgt-col if given
      2) our common names: formosan_sentence, chinese_sentence, english_sentence
      3) columns that look like language names/aliases
      4) fallback: first two object dtype columns
    """
    if override_src and override_tgt:
        return override_src, override_tgt

    cols = set(df.columns)

    # common corpus shape
    if src_lang in FORMOSAN_SET and "formosan_sentence" in cols:
        if tgt_lang == "chinese" and "chinese_sentence" in cols:
            return "formosan_sentence", "chinese_sentence"
        if tgt_lang == "english" and "english_sentence" in cols:
            return "formosan_sentence", "english_sentence"

    aliases = {
        "english": {"english", "eng", "en", "en_sentence", "english_sentence"},
        "chinese": {"chinese", "zh", "zho", "zh_hant", "zh_hans", "chinese_sentence"},
        "amis": {"amis","ami"}, "bunun": {"bunun","bnn"}, "kavalan": {"kavalan","ckv"},
        "rukai": {"rukai","dru"}, "paiwan": {"paiwan","pwn"}, "puyuma": {"puyuma","pyu"},
        "thao": {"thao","ssf"}, "saaroa": {"saaroa","sxr"}, "sakizaya": {"sakizaya","szy"},
        "tao": {"tao","yami"}, "atayal": {"atayal","tay"}, "seediq": {"seediq","trv"},
        "tsou": {"tsou","tsu"}, "kanakanavu": {"kanakanavu","xnb"}, "saisiyat": {"saisiyat","xsy"},
    }
    src_cands = [c for c in df.columns if c.lower() in aliases.get(src_lang, set())]
    tgt_cands = [c for c in df.columns if c.lower() in aliases.get(tgt_lang, set())]
    if src_cands and tgt_cands:
        return src_cands[0], tgt_cands[0]

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
    # quick split if none provided
    # df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    # n = len(df)
    # n_val = max(500, int(0.05 * n))
    # n_test = max(500, int(0.05 * n))
    # te = df.iloc[:n_test]
    # va = df.iloc[n_test:n_test + n_val]
    # tr = df.iloc[n_test + n_val:]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def load_tok_model(tok_dir: str, model_dir: str, device: str):
    tok = NllbTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tok))  # idempotent if already matched
    model.to(torch.device(device))
    return tok, model

def restore_custom_lang_ids(tokenizer, codes):
    """
    Newer transformers removed `lang_code_to_id`. We don't need it:
    always resolve with `convert_tokens_to_ids`. This stays a no-op
    on old versions that still have the dict.
    """
    for c in codes:
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Language token {c} resolves to <unk>. "
                f"Is it present in your tokenizer's `additional_special_tokens`?"
            )
            
    # If you want to keep compatibility with older versions:
    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        for c in codes:
            tokenizer.lang_code_to_id[c] = tokenizer.convert_tokens_to_ids(c)

def get_lang_id(tokenizer, code: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(code)
    if tid == tokenizer.unk_token_id:
        raise ValueError(f"{code} -> <unk>; not in vocab/special tokens.")
    return tid

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ------------------------------ training -----------------------------------
def _encode_pair_batch(
    tokenizer: NllbTokenizer,
    src_texts: List[str],
    tgt_texts: List[str],
    src_code: str,
    max_length: int,
    device: torch.device,
):
    """
    Robust NLLB batch encoding:
      - Set src_lang for inputs
      - Manually encode targets (no LID prefix for labels)
      - Disable token_type_ids everywhere to avoid None in pad()
    """
    # defensively coerce to str
    src_texts = ["" if x is None else str(x) for x in src_texts]
    tgt_texts = ["" if x is None else str(x) for x in tgt_texts]

    tokenizer.src_lang = src_code

    # Inputs
    enc = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,               # or "longest"
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,   # <-- important
    )

    # Targets (labels) – no language prefix for NLLB
    lab = tokenizer(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,   # <-- important
    )

    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)

    labels = lab["input_ids"].to(device)
    labels[labels == tokenizer.pad_token_id] = -100

    return input_ids, attn_mask, labels


def training_step(
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    src_texts: List[str],
    tgt_texts: List[str],
    src_code: str,
    max_length: int,
    device: torch.device,
    use_amp: bool,
    scaler: Optional[GradScaler],
):
    input_ids, attn_mask, labels = _encode_pair_batch(
        tokenizer, src_texts, tgt_texts, src_code, max_length, device
    )
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
def tiny_eval(
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    val_pairs: List[Tuple[str, str]],
    src_code: str,
    tgt_code: str,
    max_length: int,
    device: torch.device,
    n_show: int = 3,
    num_beams: int = 4,
):
    """
    Quick sanity eval in both directions:
      - encode with source lang set
      - generate with forced_bos_token_id=tgt_lid
    """
    def gen_dir(src_texts, from_code, to_code):
        tokenizer.src_lang = from_code
        enc = tokenizer(
            src_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length
        ).to(device)
        forced_id = get_lang_id(tokenizer, to_code)
        outs = model.generate(
            **enc,
            max_length=max_length,
            forced_bos_token_id=forced_id,
            num_beams=num_beams,
        )
        return tokenizer.batch_decode(outs, skip_special_tokens=True)

    sample_src = [p[0] for p in val_pairs[:n_show]]
    sample_tgt = [p[1] for p in val_pairs[:n_show]]

    fwd = gen_dir(sample_src, src_code, tgt_code)
    bwd = gen_dir(sample_tgt, tgt_code, src_code)

    print("\n[Eval] Sample forward (src->tgt):")
    for s, o in zip(sample_src, fwd):
        print(f"  SRC: {s}\n  OUT: {o}\n")

    print("[Eval] Sample backward (tgt->src):")
    for s, o in zip(sample_tgt, bwd):
        print(f"  SRC: {s}\n  OUT: {o}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-lang", required=True, help="e.g. amis, paiwan, tsou, ...")
    ap.add_argument("--tgt-lang", required=True, help="english or chinese")

    # Optional: override LIDs (e.g. --tgt-lid zho_Hans)
    ap.add_argument("--src-lid", default=None, help="Override NLLB LID, e.g. ami_Latn")
    ap.add_argument("--tgt-lid", default=None, help="Override NLLB LID, e.g. zho_Hans")

    ap.add_argument("--tokenizer", required=True, help="path to tokenizer dir from setup")
    ap.add_argument("--model",     required=True, help="path to model dir from setup")
    ap.add_argument("--input", type=Path, required=True, help="CSV with parallel data")

    # Optional explicit column names
    ap.add_argument("--src-col", default=None)
    ap.add_argument("--tgt-col", default=None)

    # Training hyperparams
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
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
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--eval-beams", type=int, default=4)

    # Device / precision
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true")

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Output naming
    if args.output_dir is None:
        args.output_dir = f"runs/{args.src_lang}_{args.tgt_lang}_nllb"
    if args.model_name is None:
        args.model_name = f"nllb-{args.src_lang}-{args.tgt_lang}"
    os.makedirs(args.output_dir, exist_ok=True)

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

    # LIDs
    src_code = args.src_lid or get_nllb_code(args.src_lang)
    tgt_code = args.tgt_lid or get_nllb_code(args.tgt_lang)
    print(f"Languages: {args.src_lang} -> {src_code} | {args.tgt_lang} -> {tgt_code}")

    # Data
    df_train_all, df_val_all, df_test_all = load_splits(args.input)
    s_col, t_col = smart_find_columns(df_train_all, args.src_lang, args.tgt_lang, args.src_col, args.tgt_col)
    print(f"Using columns: {s_col} -> {t_col}")

    df_train = df_train_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_val   = df_val_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_test  = df_test_all[[s_col, t_col]].dropna().reset_index(drop=True)
    assert len(df_train), "No training data."

    # Tokenizer + model (already customized by your setup script)
    tokenizer, model = load_tok_model(args.tokenizer, args.model, str(device))
    restore_custom_lang_ids(tokenizer, [src_code, tgt_code])

    # Optim + sched
    optimizer = Adafactor(
        (p for p in model.parameters() if p.requires_grad),
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
                src_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                tgt_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                s_code = src_code
            else:
                # tgt -> src
                src_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                tgt_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                s_code = tgt_code

            loss = training_step(
                tokenizer, model, src_texts, tgt_texts,
                s_code, args.max_length, device,
                use_amp=scaler.is_enabled(), scaler=scaler
            )
            losses.append(loss.item())

            if (step + 1) % args.grad_accum_steps == 0:
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
                recent = losses[-min(len(losses), args.log_interval):]
                pbar.set_postfix(loss=f"{np.mean(recent):.4f}",
                                 dir=f"{'src->tgt' if s_code==src_code else 'tgt->src'}")

            if args.eval_interval and step > 0 and step % args.eval_interval == 0 and len(df_val):
                model.eval()
                pairs = list(zip(df_val[s_col].astype(str).tolist(), df_val[t_col].astype(str).tolist()))
                tiny_eval(
                    tokenizer, model,
                    pairs[:args.eval_samples],
                    src_code, tgt_code,
                    args.max_length, device,
                    n_show=min(3, args.eval_samples),
                    num_beams=args.eval_beams,
                )
                model.train()

            if step > 0 and step % args.save_interval == 0:
                ckpt = os.path.join(args.output_dir, f"{args.model_name}-step-{step}")
                os.makedirs(ckpt, exist_ok=True)
                model.save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)
                with open(os.path.join(ckpt, "args.json"), "w") as f:
                    json.dump(vars(args), f, indent=2)
                print(f"\n[Save] {ckpt}")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                optimizer.zero_grad(set_to_none=True)
                cleanup_cuda()
                continue
            raise

    # Final save
    final_dir = os.path.join(args.output_dir, f"{args.model_name}-final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nDone! Final model at: {final_dir}")

if __name__ == "__main__":
    main()
