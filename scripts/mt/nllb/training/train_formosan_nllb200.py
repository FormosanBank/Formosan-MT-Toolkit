#!/usr/bin/env python3
"""
Train NLLB-200 (distilled-600M) on a Formosan <-> (English|Chinese) parallel corpus.

Bi-directional like Dale:
- Flip direction per step (default 50/50), set tokenizer.src_lang to the *current source* only,
  NEVER prefix labels with a language token.
- For generation/eval, ALWAYS pass forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lid).

Saves to a fresh run directory:
  runs/<src>_<tgt>/<YYYYmmdd-HHMMSS>/
    - checkpoints/step-000000 (initial)
    - checkpoints/step-005000, step-010000, ...
    - final/

References:
- Dale’s NLLB fine-tune write-up (bi-dir batching, Adafactor, warmup).  # see citations in the chat
- HF NLLB docs: set tokenizer.src_lang + forced_bos_token_id for target.  # see citations in the chat

Example: 

python train_formosan_nllb200.py \
  --src-lang amis --tgt-lang english \
  --tokenizer ../prelims/formosan_multilingual_nllb_tokenizer \
  --model     ../prelims/formosan_multilingual_nllb_model \
  --input     ami_en_processed.csv \
  --normalize \
  --steps 20000 --batch-size 8 \
  --save-interval 5000 --eval-interval 5000 --eval-samples 12
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
import time
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

# Optional normalization (like NLLB preprocessing)
try:
    from sacremoses import MosesPunctNormalizer
    HAVE_SACREMOSES = True
except Exception:
    HAVE_SACREMOSES = False

# ----------------------- language maps (align with your setup) -----------------------
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
    # Default to Traditional Chinese (FLORES code)
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

def _normalize_series(s: pd.Series, lang_hint: str) -> pd.Series:
    # Minimal NFKC, plus Moses punctuation normalization (optional)
    s = s.fillna("").astype(str).map(lambda t: t if t is not None else "")
    try:
        import unicodedata
        s = s.map(lambda t: unicodedata.normalize("NFKC", t))
    except Exception:
        pass
    if HAVE_SACREMOSES:
        mpn = MosesPunctNormalizer(lang="en")  # neutral-ish; NLLB preprocessing uses Moses-like normalizer
        s = s.map(lambda t: mpn.normalize(t))
    return s

def load_splits(csv_path: Path,
                make_val: float = 0.05,
                make_test: float = 0.05,
                seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load 'train/valid/test' by 'split' column; else make deterministic 90/5/5."""
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        tr = df[df["split"].str.lower().eq("train")]
        va = df[df["split"].str.lower().isin(["valid", "val", "validate"])]
        te = df[df["split"].str.lower().eq("test")]
        if len(tr) == 0:
            sys.exit("No 'train' rows found in CSV 'split' column.")
        return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

    # Deterministic shuffle and 90/5/5 split
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_val = int(make_val * n)
    n_test = int(make_test * n)
    te = df.iloc[:n_test]
    va = df.iloc[n_test:n_test + n_val]
    tr = df.iloc[n_test + n_val:]
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)

def load_tok_model(tok_dir: str, model_dir: str, device: torch.device):
    tok = NllbTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

    emb_rows = model.get_input_embeddings().num_embeddings
    tok_rows = len(tok)

    if tok_rows > emb_rows:
        # only ever grow to match a larger tokenizer
        model.resize_token_embeddings(tok_rows)
    elif tok_rows < emb_rows:
        raise SystemExit(
            f"❌ Tokenizer vocab ({tok_rows}) < model embeddings ({emb_rows}). "
            f"Refusing to shrink; load the *same tokenizer* used to train."
        )

    model.to(device).eval()
    return tok, model


def restore_custom_lang_ids(tokenizer, codes):
    """
    Newer transformers removed `lang_code_to_id`. We don't need it:
    always resolve with `convert_tokens_to_ids`.
    Keep backward compatibility if the dict is present.
    """
    for c in codes:
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Language token {c} resolves to <unk>. "
                f"Is it present in your tokenizer's `additional_special_tokens`?"
            )
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

def _make_run_dir(base_out: Optional[str], src: str, tgt: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_out) if base_out else Path("runs") / f"{src}_{tgt}" / ts
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root

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
      - Encode targets WITHOUT any language token on labels
      - Explicitly append EOS to each target label sequence
      - Disable token_type_ids everywhere to avoid None in pad()
    """
    # defensively coerce to str
    src_texts = ["" if x is None else str(x) for x in src_texts]
    tgt_texts = ["" if x is None else str(x) for x in tgt_texts]

    tokenizer.src_lang = src_code

    # Inputs (encoder side)
    enc = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    # Targets (decoder labels) – do NOT add language code; DO add EOS
    lab = tokenizer(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
        add_special_tokens=False,
    )

    labels = lab["input_ids"]  # CPU for now
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; EOS is required for seq2seq training.")

    # Put EOS right after the last non-pad token (or at last slot if full)
    with torch.no_grad():
        lengths = (labels != pad_id).sum(dim=1)          # [B]
        L = labels.size(1)
        pos = torch.clamp(lengths, max=L - 1)            # [B]
        rows = torch.arange(labels.size(0), dtype=torch.long)
        labels[rows, pos] = eos_id

    # Move to device + mask pads for loss
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    labels = labels.to(device)
    labels[labels == pad_id] = -100

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
    forced_bos_id: int,
):
    input_ids, attn_mask, labels = _encode_pair_batch(
        tokenizer, src_texts, tgt_texts, src_code, max_length, device
    )
    # build decoder_input_ids: [forced_bos_id] + labels (shifted right)
    dec_in = labels.clone()
    pad = tokenizer.pad_token_id
    dec_in = dec_in.masked_fill(dec_in == -100, pad)      # restore pads for shifting
    bos_col = torch.full((dec_in.size(0), 1), forced_bos_id, device=device, dtype=dec_in.dtype)
    decoder_input_ids = torch.cat([bos_col, dec_in[:, :-1]], dim=1)

    if use_amp:
        with autocast(device_type=device.type):
            loss = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            ).loss
        scaler.scale(loss).backward()
    else:
        loss = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels
        ).loss
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
    batch_size: int = 64,
):
    """
    Quick sanity eval in both directions.
    Prints:
      - sample generations (src->tgt and tgt->src)
      - avg token-level loss (cross-entropy) over the eval batch in both directions
        (pads ignored by setting labels==pad_token_id -> -100)
    """
    import math

    prev_training = model.training
    model.eval()

    def gen_dir(src_texts, from_code, to_code):
        # 1) encode with the correct source language
        tokenizer.src_lang = from_code
        enc = tokenizer(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        # 2) resolve target BOS id
        forced_id = get_lang_id(tokenizer, to_code)

        # 3) generate with BOTH a forced BOS and decoder start token
        #    (covers transformers versions that lean on either field)
        gen_out = model.generate(
            **enc,
            num_beams=1,                      # turn off beams for sanity
            max_new_tokens=48,                # don’t go to 192 on eval
            min_new_tokens=2,
            no_repeat_ngram_size=3,           # stop loops
            repetition_penalty=1.2,           # gentle deterrent
            length_penalty=1.05,              # nudge away from 1-token replies
            early_stopping=True,
            forced_bos_token_id=forced_id,
            decoder_start_token_id=forced_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        return tokenizer.batch_decode(seqs, skip_special_tokens=True)


    def avg_loss_dir(src_texts, tgt_texts, from_code):
        """
        Compute weighted average loss over tokens (ignoring pads) in mini-batches.
        """
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(src_texts), batch_size):
            batch_src = ["" if s is None else str(s) for s in src_texts[i:i + batch_size]]
            batch_tgt = ["" if t is None else str(t) for t in tgt_texts[i:i + batch_size]]

            # Encode inputs with correct source LID
            tokenizer.src_lang = from_code
            enc = tokenizer(
                batch_src,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=True,
                return_token_type_ids=False,
            )

            # Encode targets as labels (no LID token prepended for labels)
            # targets (labels) – do NOT include any language code
            # Targets (labels) – do NOT add special tokens/LID to labels
            # Encode targets as labels (no LID on labels; add EOS explicitly)
            lab = tokenizer(
                batch_tgt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )

            labels = lab["input_ids"]
            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id
            if eos_id is None:
                raise ValueError("Tokenizer has no eos_token_id; EOS is required for seq2seq evaluation.")

            with torch.no_grad():
                lengths = (labels != pad_id).sum(dim=1)
                L = labels.size(1)
                pos = torch.clamp(lengths, max=L - 1)
                rows = torch.arange(labels.size(0), dtype=torch.long)
                labels[rows, pos] = eos_id

            labels = labels.to(device)
            labels[labels == pad_id] = -100


            non_pad = (labels != -100).sum().item()

            enc = {k: v.to(device) for k, v in enc.items()}
            labels = labels.to(device)

            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            loss = outputs.loss  # averaged over non-pad tokens in the batch

            # Weighted by token count to aggregate across batches safely
            total_loss += loss.item() * max(non_pad, 1)
            total_tokens += max(non_pad, 1)

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        return avg_loss, ppl

    # Sample pretty prints
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

    # Compute losses over the full eval subset passed in
    all_src = [p[0] for p in val_pairs]
    all_tgt = [p[1] for p in val_pairs]

    fwd_loss, fwd_ppl = avg_loss_dir(all_src, all_tgt, src_code)
    bwd_loss, bwd_ppl = avg_loss_dir(all_tgt, all_src, tgt_code)

    print(f"[Eval] Avg token loss  (src->tgt): {fwd_loss:.4f} | ppl: {fwd_ppl:.2f}")
    print(f"[Eval] Avg token loss  (tgt->src): {bwd_loss:.4f} | ppl: {bwd_ppl:.2f}")

    if prev_training:
        model.train()



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
    ap.add_argument("--learning-rate", type=float, default=1e-4)   # Dale-style LR
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--clip-threshold", type=float, default=1.0)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # Direction mix (bidirectional training)
    ap.add_argument("--p-src2tgt", type=float, default=0.5,
                    help="Probability of sampling src->tgt (else tgt->src) each step")

    # Logging / saving / eval
    ap.add_argument("--output-dir", default=None,
                    help="Root run directory; defaults to runs/<src>_<tgt>/<timestamp>")
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--eval-interval", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=8)
    ap.add_argument("--eval-beams", type=int, default=4)

    # Device / precision
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true")

    # Preprocessing
    ap.add_argument("--normalize", action="store_true",
                    help="Apply NFKC + Moses punctuation normalization to inputs")

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

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

    if args.normalize:
        df_train_all[s_col] = _normalize_series(df_train_all[s_col], args.src_lang)
        df_train_all[t_col] = _normalize_series(df_train_all[t_col], args.tgt_lang)
        df_val_all[s_col]   = _normalize_series(df_val_all[s_col], args.src_lang)
        df_val_all[t_col]   = _normalize_series(df_val_all[t_col], args.tgt_lang)

    df_train = df_train_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_val   = df_val_all[[s_col, t_col]].dropna().reset_index(drop=True)
    df_test  = df_test_all[[s_col, t_col]].dropna().reset_index(drop=True)
    assert len(df_train), "No training data."

    # Tokenizer + model (already customized by your setup script)
    tokenizer, model = load_tok_model(args.tokenizer, args.model, device)
    restore_custom_lang_ids(tokenizer, [src_code, tgt_code])

    # Run directory (NEW: always a fresh dir)
    run_dir = _make_run_dir(args.output_dir, args.src_lang, args.tgt_lang)
    (run_dir / "final").mkdir(parents=True, exist_ok=True)

    # Save initial checkpoint
    init_ckpt = run_dir / "checkpoints" / "step-000000"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(str(init_ckpt))
    model.save_pretrained(str(init_ckpt))


    # Optim + sched (Dale-style Adafactor + warmup)
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

    print(f"\nRun dir: {run_dir}")
    print(f"Starting training on {device} for {args.steps} steps...")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum_steps} | Max len: {args.max_length} | FP16: {bool(scaler.is_enabled())}")
    print(f"Direction mix p(src->tgt)={args.p_src2tgt:.2f}")

    # Arrays for quick sampling
    train_src = df_train[s_col].astype(str).to_numpy()
    train_tgt = df_train[t_col].astype(str).to_numpy()

    losses: List[float] = []
    pbar = trange(args.steps, desc="Training", dynamic_ncols=True)
    
    # before the loop
    src_lang_id = get_lang_id(tokenizer, src_code)
    tgt_lang_id = get_lang_id(tokenizer, tgt_code)

    for step in pbar:
        try:
            idx = np.random.randint(0, len(df_train), size=args.batch_size)
            if random.random() < args.p_src2tgt:
                # src -> tgt
                src_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                tgt_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                s_code = src_code
                forced_id = tgt_lang_id
                dir_tag = "src->tgt"
            else:
                # tgt -> src
                src_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                tgt_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                s_code = tgt_code
                forced_id = src_lang_id
                dir_tag = "tgt->src"

            loss = training_step(
                tokenizer, model, src_texts, tgt_texts,
                s_code, args.max_length, device,
                use_amp=scaler.is_enabled(), scaler=scaler,
                forced_bos_id=forced_id
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
                pbar.set_postfix(loss=f"{np.mean(recent):.4f}", dir=dir_tag)
                print(f"Step {step} | Loss: {np.mean(recent):.4f} | Direction: {dir_tag}")

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
                ckpt_dir = run_dir / "checkpoints" / f"step-{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[Save] Step {step} -> {ckpt_dir}")
                model.save_pretrained(str(ckpt_dir))
                tokenizer.save_pretrained(str(ckpt_dir))
                with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss": float(np.mean(losses[-args.log_interval:]))}) + "\n")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                optimizer.zero_grad(set_to_none=True)
                cleanup_cuda()
                continue
            raise

    # Final save
    final_dir = run_dir / "final"
    print(f"\n[Final Save] -> {final_dir}")
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"✅ Training complete! Artifacts in: {run_dir}")

if __name__ == "__main__":
    main()
