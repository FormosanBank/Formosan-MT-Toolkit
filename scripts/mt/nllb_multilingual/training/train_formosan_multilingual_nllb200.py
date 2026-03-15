#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilingual fine-tuning for NLLB-200 (distilled-600M) on 15 Formosan <-> Chinese.

This *extends* your single-pair trainer while preserving the core src/target LID logic:

- For each batch we set `tokenizer.src_lang` to the *current source* language only.
- We NEVER prefix labels with any language token (labels are raw tokens + EOS).
- For generation/eval we ALWAYS pass `forced_bos_token_id` (target LID id) and also set
  `decoder_start_token_id` to the same value for safety across Transformers versions.

New in this version
-------------------
• Multilingual training across all listed Formosan languages to Chinese using
  temperature-based sampling across languages: p(lang) ∝ (N_lang)^(1/T).  (Default T=5)
• Language-balanced evaluation: at each eval interval, compute src->zh and zh->src losses
  for every language that has validation data and print per-language + averages.
• Compatible with your single-pair flow: if --multilingual is omitted, behavior is unchanged.

CSV format (multilingual mode)
------------------------------
lang_code,formosan_sentence,chinese_sentence,source,dialect,split

Notes:
- Rows without a Chinese sentence are dropped for this zh-focused run.
- 'split' values: train / valid|val / test (or 90/5/5 auto-split per language if missing).
- 'lang_code' may be 3-letter codes (ami, pwn, ...) or names; we normalize flexibly.

References (sampling + NLLB usage)
----------------------------------
- Temperature-based sampling for multilingual balancing (e.g., T=5 effective):
  Arivazhagan et al., 2019; follow-ups report similar practice.  [cited in code preface]
- Hugging Face NLLB docs for src_lang + forced_bos usage.                       [cited]

Example (multilingual, all 15 -> English):
python train_formosan_multilingual_nllb200.py \
  --multilingual \
  --tgt-lang chinese \
  --tokenizer formosan_multilingual_nllb_tokenizer \
  --model     formosan_multilingual_nllb_model \
  --input     big_corpus_en.csv \
  --temperature 5 \
  --steps 150000 --batch-size 8 \
  --save-interval 5000 --eval-interval 5000 --eval-samples 256 --normalize

Example (single pair unchanged behavior):
python train_formosan_multilingual_nllb200.py \
  --src-lang amis --tgt-lang chinese \
  --tokenizer ../prelims/formosan_multilingual_nllb_tokenizer \
  --model     ../prelims/formosan_multilingual_nllb_model \
  --input     ami_zh_processed.csv \
  --steps 20000 --batch-size 8
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
    # Formosan (Latin orthographies unless you chose differently)
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

# Accept common 3-letter codes or names in CSV 'lang_code'
CSV_LANG_ALIAS: Dict[str, str] = {
    # 3-letter -> canonical key
    "ami":"amis","bnn":"bunun","ckv":"kavalan","dru":"rukai","pwn":"paiwan","pyu":"puyuma",
    "ssf":"thao","sxr":"saaroa","szy":"sakizaya","tao":"tao","tay":"atayal","trv":"seediq",
    "tsu":"tsou","xnb":"kanakanavu","xsy":"saisiyat",
    # also map names to themselves
    "amis":"amis","bunun":"bunun","kavalan":"kavalan","rukai":"rukai","paiwan":"paiwan",
    "puyuma":"puyuma","thao":"thao","saaroa":"saaroa","sakizaya":"sakizaya","tao":"tao",
    "atayal":"atayal","seediq":"seediq","tsou":"tsou","kanakanavu":"kanakanavu","saisiyat":"saisiyat",
}

# ------------------------------- helpers -----------------------------------
def get_nllb_code(name: str) -> str:
    key = name.lower()
    if key not in NLLB_LANGUAGE_MAP:
        sys.exit(f"Unsupported language '{name}'. "
                 f"Supported: {', '.join(sorted(NLLB_LANGUAGE_MAP))}")
    return NLLB_LANGUAGE_MAP[key]

def restore_custom_lang_ids(tokenizer, codes):
    """
    Keep compatibility across Transformers versions.
    Always resolve id via convert_tokens_to_ids and sanity-check it's not <unk>.
    """
    for c in codes:
        tid = tokenizer.convert_tokens_to_ids(c)
        if tid == tokenizer.unk_token_id:
            raise ValueError(
                f"Language token {c} resolves to <unk>. "
                f"Make sure it's in tokenizer.additional_special_tokens."
            )
    if hasattr(tokenizer, "lang_code_to_id") and isinstance(tokenizer.lang_code_to_id, dict):
        for c in codes:
            tokenizer.lang_code_to_id[c] = tokenizer.convert_tokens_to_ids(c)

def get_lang_id(tokenizer, code: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(code)
    if tid == tokenizer.unk_token_id:
        raise ValueError(f"{code} -> <unk>; not in vocab/special tokens.")
    return tid

def _normalize_series(s: pd.Series, lang_hint: str) -> pd.Series:
    s = s.fillna("").astype(str)
    try:
        import unicodedata
        s = s.map(lambda t: unicodedata.normalize("NFKC", t))
    except Exception:
        pass
    if HAVE_SACREMOSES:
        mpn = MosesPunctNormalizer(lang="en")
        s = s.map(lambda t: mpn.normalize(t))
    return s

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _make_run_dir(base_out: Optional[str], tag: str) -> Path:
    ts = time.strftime("%Y%m%d-%H%M%S")
    root = Path(base_out) if base_out else Path("runs") / tag / ts
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    return root

# ------------------------------ encoding & step (UNCHANGED CORE) -------------------
def _encode_pair_batch(
    tokenizer: NllbTokenizer,
    src_texts: List[str],
    tgt_texts: List[str],
    src_code: str,
    max_length: int,
    device: torch.device,
):
    # Coerce to string + set correct source LID
    src_texts = ["" if x is None else str(x) for x in src_texts]
    tgt_texts = ["" if x is None else str(x) for x in tgt_texts]

    tokenizer.src_lang = src_code

    enc = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

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

    labels = lab["input_ids"]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id; EOS is required.")

    with torch.no_grad():
        lengths = (labels != pad_id).sum(dim=1)
        L = labels.size(1)
        pos = torch.clamp(lengths, max=L - 1)
        rows = torch.arange(labels.size(0), dtype=torch.long)
        labels[rows, pos] = eos_id

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
    pad = tokenizer.pad_token_id
    dec_in = labels.clone().masked_fill(labels == -100, pad)
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
    import math

    prev_training = model.training
    model.eval()

    def gen_dir(src_texts, from_code, to_code):
        tokenizer.src_lang = from_code
        enc = tokenizer(
            src_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        forced_id = get_lang_id(tokenizer, to_code)
        gen_out = model.generate(
            **enc,
            num_beams=1,
            max_new_tokens=48,
            min_new_tokens=2,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=1.05,
            early_stopping=True,
            forced_bos_token_id=forced_id,
            decoder_start_token_id=forced_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
        return tokenizer.batch_decode(seqs, skip_special_tokens=True)

    def avg_loss_dir(src_texts, tgt_texts, from_code):
        total_loss = 0.0
        total_tokens = 0
        for i in range(0, len(src_texts), batch_size):
            batch_src = ["" if s is None else str(s) for s in src_texts[i:i+batch_size]]
            batch_tgt = ["" if t is None else str(t) for t in tgt_texts[i:i+batch_size]]

            tokenizer.src_lang = from_code
            enc = tokenizer(
                batch_src, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, return_attention_mask=True, return_token_type_ids=False,
            )
            lab = tokenizer(
                batch_tgt, return_tensors="pt", padding=True, truncation=True,
                max_length=max_length, return_attention_mask=False, return_token_type_ids=False,
                add_special_tokens=False,
            )

            labels = lab["input_ids"]
            pad_id = tokenizer.pad_token_id
            eos_id = tokenizer.eos_token_id
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
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            loss = outputs.loss
            total_loss += loss.item() * max(non_pad, 1)
            total_tokens += max(non_pad, 1)
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(avg_loss) if avg_loss < 50 else float("inf")
        return avg_loss, ppl

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

    all_src = [p[0] for p in val_pairs]
    all_tgt = [p[1] for p in val_pairs]
    fwd_loss, fwd_ppl = avg_loss_dir(all_src, all_tgt, src_code)
    bwd_loss, bwd_ppl = avg_loss_dir(all_tgt, all_src, tgt_code)
    print(f"[Eval] Avg token loss  (src->tgt): {fwd_loss:.4f} | ppl: {fwd_ppl:.2f}")
    print(f"[Eval] Avg token loss  (tgt->src): {bwd_loss:.4f} | ppl: {bwd_ppl:.2f}")

    if prev_training:
        model.train()

# ------------------------- data loading (single & multilingual) ----------------------
def smart_find_columns(
    df: pd.DataFrame,
    src_lang: str,
    tgt_lang: str,
    override_src: Optional[str],
    override_tgt: Optional[str],
) -> Tuple[str, str]:
    if override_src and override_tgt:
        return override_src, override_tgt

    cols = set(df.columns)
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

def load_splits(csv_path: Path,
                make_val: float = 0.05,
                make_test: float = 0.05,
                seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        tr = df[df["split"].astype(str).str.lower().eq("train")]
        va = df[df["split"].astype(str).str.lower().isin(["valid", "val", "validate"])]
        te = df[df["split"].astype(str).str.lower().eq("test")]
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

def load_splits_multilingual(
    csv_path: Path,
    langs: List[str],
    tgt_lang: str,
    seed: int = 42,
    make_val: float = 0.05,
    make_test: float = 0.05,
    normalize: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], str]:
    """
    Returns dicts lang -> DataFrame with columns [formosan_sentence, <target_col>].

    <target_col> is:
      - 'chinese_sentence' when tgt_lang == 'chinese'
      - 'english_sentence' when tgt_lang == 'english'
    """
    df = pd.read_csv(csv_path)

    tgt_lang_l = tgt_lang.lower()
    if tgt_lang_l == "chinese":
        tgt_col = "chinese_sentence"
    elif tgt_lang_l == "english":
        tgt_col = "english_sentence"
    else:
        sys.exit(f"Multilingual loader only supports tgt_lang in {{'chinese','english'}}, got '{tgt_lang}'.")

    want_cols = ["lang_code", "formosan_sentence", tgt_col, "split"]
    missing = [c for c in ["lang_code", "formosan_sentence", tgt_col] if c not in df.columns]

    if missing:
        sys.exit(f"CSV missing required columns: {missing}")

    # Normalize lang codes
    def canon(x: str) -> Optional[str]:
        if not isinstance(x, str):
            return None
        key = x.strip().lower()
        return CSV_LANG_ALIAS.get(key)

    df["canon_lang"] = df["lang_code"].map(canon)
    df = df[df["canon_lang"].isin(langs)].copy()

    # Filter to rows that have targets for this stage
    df = df[~df["formosan_sentence"].isna() & ~df[tgt_col].isna()].copy()

    # Optional normalization
    if normalize:
        df["formosan_sentence"] = _normalize_series(df["formosan_sentence"], "formosan")
        df[tgt_col] = _normalize_series(df[tgt_col], tgt_lang_l)

    # Split per language
    train_by, val_by, test_by = {}, {}, {}
    for L in sorted(df["canon_lang"].unique()):
        sub = df[df["canon_lang"] == L].copy()
        if "split" in sub.columns and sub["split"].notna().any():
            tr = sub[sub["split"].astype(str).str.lower().eq("train")]
            va = sub[sub["split"].astype(str).str.lower().isin(["valid","val","validate"])]
            te = sub[sub["split"].astype(str).str.lower().eq("test")]
        else:
            sub = sub.sample(frac=1.0, random_state=seed).reset_index(drop=True)
            n = len(sub); n_val = int(make_val*n); n_test = int(make_test*n)
            te = sub.iloc[:n_test]; va = sub.iloc[n_test:n_test+n_val]; tr = sub.iloc[n_test+n_val:]

        # Keep only necessary columns
        cols = ["formosan_sentence", tgt_col]
        tr = tr[cols].dropna().reset_index(drop=True)
        va = va[cols].dropna().reset_index(drop=True)
        te = te[cols].dropna().reset_index(drop=True)
        if len(tr):
            train_by[L] = tr
        if len(va):
            val_by[L] = va
        if len(te):
            test_by[L] = te
    return train_by, val_by, test_by, tgt_col

# --------------------------- multilingual eval wrapper ------------------------------
@torch.no_grad()
def multilingual_eval(
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    val_by_lang: Dict[str, pd.DataFrame],
    tgt_code: str,
    lang2src_code: Dict[str, str],
    tgt_col: str,
    tgt_label: str,
    max_length: int,
    device: torch.device,
    eval_samples_per_lang: int = 12,
    eval_beams: int = 4,
):

    """
    Runs tiny_eval for each language (src<->zh). Prints per-language losses + global means.
    """
    all_fwd, all_bwd = [], []
    print("\n================ Multilingual Eval (per language) ================")
    for L, df in val_by_lang.items():
        pairs = list(zip(
            df["formosan_sentence"].astype(str).tolist(),
            df[tgt_col].astype(str).tolist(),
        ))

        if not pairs:
            continue
        pairs = pairs[:eval_samples_per_lang]
        src_code = lang2src_code[L]
        print(f"\n[Lang={L}] {len(pairs)} samples")
        tiny_eval(
            tokenizer, model, pairs, src_code, tgt_code,
            max_length, device, n_show=min(3, len(pairs)), num_beams=eval_beams
        )

        # Collect numeric losses again quietly for macro averages
        # (Reuse avg_loss_dir from tiny_eval by re-calling without prints)
        # Simpler: recompute quickly here (no text printing)
        # Forward src->tgt
        fwd_src = [p[0] for p in pairs]
        fwd_tgt = [p[1] for p in pairs]
        # Backward tgt->src
        bwd_src = fwd_tgt
        bwd_tgt = fwd_src

        def avg_loss_dir(src_texts, tgt_texts, from_code):
            total_loss = 0.0
            total_tokens = 0
            batch_size = 64
            for i in range(0, len(src_texts), batch_size):
                bs = src_texts[i:i+batch_size]
                bt = tgt_texts[i:i+batch_size]
                tokenizer.src_lang = from_code
                enc = tokenizer(bs, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length, return_attention_mask=True,
                                return_token_type_ids=False)
                lab = tokenizer(bt, return_tensors="pt", padding=True, truncation=True,
                                max_length=max_length, return_attention_mask=False,
                                return_token_type_ids=False, add_special_tokens=False)
                labels = lab["input_ids"]
                pad_id = tokenizer.pad_token_id
                eos_id = tokenizer.eos_token_id
                with torch.no_grad():
                    lengths = (labels != pad_id).sum(dim=1)
                    Ls = labels.size(1)
                    pos = torch.clamp(lengths, max=Ls - 1)
                    rows = torch.arange(labels.size(0), dtype=torch.long)
                    labels[rows, pos] = eos_id
                labels = labels.to(device)
                labels[labels == pad_id] = -100
                enc = {k: v.to(device) for k, v in enc.items()}
                loss = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels).loss
                non_pad = (labels != -100).sum().item()
                total_loss += loss.item() * max(non_pad, 1)
                total_tokens += max(non_pad, 1)
            return total_loss / max(total_tokens, 1)

        fwd_loss = avg_loss_dir(fwd_src, fwd_tgt, from_code=src_code)
        bwd_loss = avg_loss_dir(bwd_src, bwd_tgt, from_code=tgt_code)
        all_fwd.append(fwd_loss); all_bwd.append(bwd_loss)

        print(f"[Lang={L}] token-loss src->{tgt_label}: {fwd_loss:.4f} | {tgt_label}->src: {bwd_loss:.4f}")

    if all_fwd and all_bwd:
        print(
            f"\n[Eval Global] mean token-loss src->{tgt_label}: "
            f"{float(np.mean(all_fwd)):.4f} | {tgt_label}->src: {float(np.mean(all_bwd)):.4f}"
        )


    print("==================================================================\n")

# --------------------------------------- main ---------------------------------------
def main():
    ap = argparse.ArgumentParser()
    # Mode control
    ap.add_argument("--multilingual", action="store_true",
                    help="If set, train across many Formosan->Chinese pairs using temperature sampling.")

    # Single-pair args (unchanged)
    ap.add_argument("--src-lang", help="e.g. amis, paiwan, tsou, ...")
    ap.add_argument("--tgt-lang", required=True, help="must be 'chinese' for multilingual; single-pair can be english or chinese")
    ap.add_argument("--src-lid", default=None, help="Override NLLB LID, e.g. ami_Latn")
    ap.add_argument("--tgt-lid", default=None, help="Override NLLB LID, e.g. zho_Hans")

    ap.add_argument("--tokenizer", required=True, help="path to tokenizer dir from setup")
    ap.add_argument("--model",     required=True, help="path to model dir from setup")
    ap.add_argument("--input", type=Path, required=True, help="CSV with parallel data")

    # Optional explicit column names (single-pair)
    ap.add_argument("--src-col", default=None)
    ap.add_argument("--tgt-col", default=None)

    # Multilingual controls
    ap.add_argument("--langs", default=",".join(sorted(FORMOSAN_SET)),
                    help="Comma-separated canonical names for languages to include (amis,atai...).")
    ap.add_argument("--min-per-lang", type=int, default=50,
                    help="Exclude languages with fewer than this many training pairs.")
    ap.add_argument("--temperature", type=float, default=5.0,
                    help="Sampling temperature T (p ∝ n^(1/T)); higher T upweights low-resource.")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Alternative smoothing exponent α (p ∝ n^α). If provided, overrides --temperature.")

    # Training hyperparams
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--clip-threshold", type=float, default=1.0)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)

    # Direction mix (bidirectional training)
    ap.add_argument("--p-src2tgt", type=float, default=0.5,
                    help="Probability of src->tgt each step (else tgt->src).")

    # Logging / saving / eval
    ap.add_argument("--output-dir", default=None,
                    help="Root run directory; default: runs/<tag>/<timestamp>")
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=1000)
    ap.add_argument("--eval-interval", type=int, default=5000)
    ap.add_argument("--eval-samples", type=int, default=12)
    ap.add_argument("--eval-beams", type=int, default=4)

    # Device / precision
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--fp16", action="store_true")

    # Preprocessing
    ap.add_argument("--normalize", action="store_true", help="Apply NFKC + Moses punctuation normalization")

    # Repro
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # Device
    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available())
              else (args.device if args.device != "auto" else "cpu"))
    device = torch.device(device)

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # Model & tokenizer
    tok = NllbTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    # Resize embeddings if tokenizer has grown
    emb_rows = model.get_input_embeddings().num_embeddings
    tok_rows = len(tok)
    if tok_rows > emb_rows:
        model.resize_token_embeddings(tok_rows)
    elif tok_rows < emb_rows:
        raise SystemExit(
            f"❌ Tokenizer vocab ({tok_rows}) < model embeddings ({emb_rows}). "
            f"Load the same tokenizer used for the model."
        )

    model.to(device).train()
    # Resolve tgt code (Chinese for multilingual)
    tgt_code = args.tgt_lid or get_nllb_code(args.tgt_lang)
    restore_custom_lang_ids(tok, [tgt_code])

    # Optim + sched
    optimizer = Adafactor(
        (p for p in model.parameters() if p.requires_grad),
        scale_parameter=False, relative_step=False, lr=args.learning_rate,
        clip_threshold=args.clip_threshold, weight_decay=args.weight_decay,
    )
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps)
    scaler = GradScaler(device.type if (args.fp16 and device.type == "cuda") else "cpu",
                        enabled=(args.fp16 and device.type == "cuda"))
    optimizer.zero_grad(set_to_none=True)

    # ------------------------ SINGLE-PAIR MODE (unchanged) ------------------------
    if not args.multilingual:
        if not args.src_lang:
            sys.exit("--src-lang is required in single-pair mode.")
        src_code = args.src_lid or get_nllb_code(args.src_lang)
        restore_custom_lang_ids(tok, [src_code])

        df_train_all, df_val_all, df_test_all = load_splits(args.input)
        s_col, t_col = smart_find_columns(df_train_all, args.src_lang, args.tgt_lang, args.src_col, args.tgt_col)

        if args.normalize:
            for df_ in (df_train_all, df_val_all, df_test_all):
                df_[s_col] = _normalize_series(df_[s_col], args.src_lang)
                df_[t_col] = _normalize_series(df_[t_col], args.tgt_lang)

        df_train = df_train_all[[s_col, t_col]].dropna().reset_index(drop=True)
        df_val   = df_val_all[[s_col, t_col]].dropna().reset_index(drop=True)

        run_dir = _make_run_dir(args.output_dir, f"{args.src_lang}_{args.tgt_lang}")
        (run_dir / "final").mkdir(parents=True, exist_ok=True)
        # Save initial checkpoint
        init_ckpt = run_dir / "checkpoints" / "step-000000"
        init_ckpt.mkdir(parents=True, exist_ok=True)
        tok.save_pretrained(str(init_ckpt)); model.save_pretrained(str(init_ckpt))

        print(f"\nRun dir: {run_dir}")
        print(f"Starting single-pair training on {device} for {args.steps} steps...")
        print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum_steps} | Max len: {args.max_length} | FP16: {bool(scaler.is_enabled())}")
        print(f"Direction mix p(src->tgt)={args.p_src2tgt:.2f}")
        print(f"Using columns: {s_col} -> {t_col}")

        losses: List[float] = []
        pbar = trange(args.steps, desc="Training", dynamic_ncols=True)
        src_lang_id = get_lang_id(tok, src_code)
        tgt_lang_id = get_lang_id(tok, tgt_code)

        for step in pbar:
            try:
                idx = np.random.randint(0, len(df_train), size=args.batch_size)
                if random.random() < args.p_src2tgt:
                    src_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                    tgt_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                    s_code = src_code; forced_id = tgt_lang_id; dir_tag = "src->tgt"
                else:
                    src_texts = df_train.iloc[idx][t_col].astype(str).tolist()
                    tgt_texts = df_train.iloc[idx][s_col].astype(str).tolist()
                    s_code = tgt_code; forced_id = src_lang_id; dir_tag = "tgt->src"

                    # NOTE: core logic preserved

                loss = training_step(tok, model, src_texts, tgt_texts, s_code,
                                     args.max_length, device, scaler.is_enabled(), scaler, forced_id)
                losses.append(loss.item())

                if (step + 1) % args.grad_accum_steps == 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        scaler.step(optimizer); scaler.update()
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
                    tiny_eval(tok, model, pairs[:args.eval_samples], src_code, tgt_code,
                              args.max_length, device, n_show=min(3, args.eval_samples),
                              num_beams=args.eval_beams)
                    model.train()

                if step > 0 and step % args.save_interval == 0:
                    ckpt_dir = run_dir / "checkpoints" / f"step-{step:06d}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    print(f"\n[Save] Step {step} -> {ckpt_dir}")
                    model.save_pretrained(str(ckpt_dir)); tok.save_pretrained(str(ckpt_dir))
                    with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                        f.write(json.dumps({"step": step, "loss": float(np.mean(losses[-args.log_interval:]))}) + "\n")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                    optimizer.zero_grad(set_to_none=True); cleanup_cuda(); continue
                raise

        final_dir = run_dir / "final"
        print(f"\n[Final Save] -> {final_dir}")
        model.save_pretrained(str(final_dir)); tok.save_pretrained(str(final_dir))
        print(f"✅ Training complete! Artifacts in: {run_dir}")
        return

    # ------------------------ MULTILINGUAL MODE (new) ------------------------
    tgt_lang = args.tgt_lang.lower()
    if tgt_lang not in {"chinese", "english"}:
        sys.exit("Multilingual mode currently supports only <Formosan> <-> {Chinese, English} for now.")


    # Prepare per-language datasets
    canon_langs = [x.strip().lower() for x in args.langs.split(",") if x.strip()]
    for L in canon_langs:       
        if L not in FORMOSAN_SET:
            sys.exit(f"--langs includes unsupported language '{L}'. Valid: {sorted(FORMOSAN_SET)}")

    train_by, val_by, test_by, tgt_col = load_splits_multilingual(
        args.input, langs=canon_langs, tgt_lang=tgt_lang, seed=args.seed, normalize=args.normalize
    )

    # Drop languages with too little training data
    train_by = {L: df for L, df in train_by.items() if len(df) >= args.min_per_lang}
    kept_langs = sorted(train_by.keys())
    if not kept_langs:
        sys.exit("No languages meet --min-per-lang threshold for training.")
    
        # Keep validation data only for languages we actually train on
    val_by_kept: Dict[str, pd.DataFrame] = {
        L: df for L, df in val_by.items() if L in kept_langs and len(df)
    }

    # Build sampling distribution p(L) ∝ n_L^(1/T)  (or α if provided)
    counts = np.array([len(train_by[L]) for L in kept_langs], dtype=float)
    if args.alpha is not None:
        weights = counts ** float(args.alpha)
        mix_note = f"alpha={args.alpha}"
    else:
        invT = 1.0 / float(args.temperature)
        weights = counts ** invT
        mix_note = f"T={args.temperature}"
    probs = weights / weights.sum()

    # Precompute language codes and LID ids
    lang2src_code: Dict[str, str] = {L: get_nllb_code(L) for L in kept_langs}
    restore_custom_lang_ids(tok, list(lang2src_code.values()) + [tgt_code])
    lid_src: Dict[str, int] = {L: get_lang_id(tok, lang2src_code[L]) for L in kept_langs}
    lid_tgt = get_lang_id(tok, tgt_code)

    # Numpy arrays for fast sampling per language
    arr_by_lang = {
        L: (
            train_by[L]["formosan_sentence"].astype(str).to_numpy(),
            train_by[L][tgt_col].astype(str).to_numpy(),
        )
        for L in kept_langs
    }

    # Run directory (shared multilingual tag)
    tag = f"formosan_multilingual_to_{args.tgt_lang}"
    run_dir = _make_run_dir(args.output_dir, tag)
    (run_dir / "final").mkdir(parents=True, exist_ok=True)
    init_ckpt = run_dir / "checkpoints" / "step-000000"
    init_ckpt.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(str(init_ckpt)); model.save_pretrained(str(init_ckpt))

    # Print sampling table
    print("\n[Multilingual] Languages kept and sampling probabilities:")
    for L, n, p in zip(kept_langs, counts.tolist(), probs.tolist()):
        print(f"  {L:11s}  n={int(n):6d}  p={p:.4f}")
    print(f"[Multilingual] Sampling scheme: p(L) ∝ n^(1/T) w/ {mix_note}.")
    print(f"\nRun dir: {run_dir}")
    print(f"Starting multilingual training on {device} for {args.steps} steps...")
    print(f"Batch size: {args.batch_size} | Grad accum: {args.grad_accum_steps} | Max len: {args.max_length} | FP16: {bool(scaler.is_enabled())}")
    tgt_short = "zh" if tgt_lang == "chinese" else "en"
    print(f"Direction mix p(src->{tgt_short})={args.p_src2tgt:.2f}")

    losses: List[float] = []
    pbar = trange(args.steps, desc="Training", dynamic_ncols=True)

    for step in pbar:
        try:
            # 1) Sample a language by temperature-smoothed distribution
            L = np.random.choice(kept_langs, p=probs)
            f_src_arr, zh_tgt_arr = arr_by_lang[L]
            # 2) Sample indices from that language
            idx = np.random.randint(0, len(f_src_arr), size=args.batch_size)

            if random.random() < args.p_src2tgt:
                # (Formosan L) -> Chinese
                src_texts = f_src_arr[idx].tolist()
                tgt_texts = zh_tgt_arr[idx].tolist()
                s_code = lang2src_code[L]
                forced_id = lid_tgt
                dir_tag = f"{L}->{tgt_short}"
            else:
                # Chinese -> (Formosan L)
                src_texts = zh_tgt_arr[idx].tolist()
                tgt_texts = f_src_arr[idx].tolist()
                s_code = tgt_code
                forced_id = lid_src[L]
                dir_tag = f"{tgt_short}->{L}"

            # 3) Core unchanged step
            loss = training_step(tok, model, src_texts, tgt_texts, s_code,
                                 args.max_length, device, scaler.is_enabled(), scaler, forced_id)
            losses.append(loss.item())

            # 4) Optim step
            if (step + 1) % args.grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            # Logs
            if step % args.log_interval == 0 and losses:
                recent = losses[-min(len(losses), args.log_interval):]
                pbar.set_postfix(loss=f"{np.mean(recent):.4f}", dir=dir_tag)
                print(f"Step {step} | Loss: {np.mean(recent):.4f} | Direction: {dir_tag}")

            # Eval across ALL languages
            if (
                args.eval_interval
                and step > 0
                and step % args.eval_interval == 0
                and len(val_by_kept)
            ):
                model.eval()
                multilingual_eval(
                    tok,
                    model,
                    val_by_lang=val_by_kept,
                    tgt_code=tgt_code,
                    lang2src_code=lang2src_code,
                    tgt_col=tgt_col,
                    tgt_label=("zh" if tgt_lang == "chinese" else "en"),
                    max_length=args.max_length,
                    device=device,
                    eval_samples_per_lang=args.eval_samples,
                    eval_beams=args.eval_beams,
                )

                model.train()
                
            # Save
            if step > 0 and step % args.save_interval == 0:
                ckpt_dir = run_dir / "checkpoints" / f"step-{step:06d}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[Save] Step {step} -> {ckpt_dir}")
                model.save_pretrained(str(ckpt_dir)); tok.save_pretrained(str(ckpt_dir))
                with open(run_dir / "train_log.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "loss": float(np.mean(losses[-args.log_interval:]))}) + "\n")

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n[OOM] step {step}: {e}. Clearing cache and continuing.")
                optimizer.zero_grad(set_to_none=True); cleanup_cuda(); continue
            raise

    final_dir = run_dir / "final"
    print(f"\n[Final Save] -> {final_dir}")
    model.save_pretrained(str(final_dir)); tok.save_pretrained(str(final_dir))
    print(f"✅ Multilingual training complete! Artifacts in: {run_dir}")

if __name__ == "__main__":
    main()
