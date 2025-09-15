#!/usr/bin/env python3
"""
Evaluate a fine-tuned mBART-50 model on a Formosan <-> (English|Chinese) test set.

- Loads your saved tokenizer/model (from the setup/training scripts).
- Uses correct mBART-50 formatting: encode with source lang prefix and generate
  with forced_bos_token_id set to the target lang code.
- Evaluates in BOTH directions (src→tgt and tgt→src) on the CSV test split.
- Reports SacreBLEU, chrF, and TER.

Example
-------
python eval_formosan_mbart.py \
  --src-lang amis --tgt-lang chinese \
  --tokenizer formosan_multilingual_mbart_tokenizer \
  --model     formosan_multilingual_mbart_model \
  --input     ami_zh.csv \
  --batch-size 16 --max-length 128 --beam 5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

# sacrebleu >= 2.x API
try:
    import sacrebleu
    from sacrebleu.metrics import CHRF, TER
except Exception as e:
    raise SystemExit(
        "❌ sacrebleu is required. Install with:\n  pip install sacrebleu\n"
        f"Import error: {e}"
    )

# ----------------------- language maps (align with setup/training) -----------------------
MBART_LANGUAGE_MAP: Dict[str, str] = {
    # Formosan (custom tokens you added in setup)
    "ami": "ami_XX", "bnn": "bnn_XX", "ckv": "ckv_XX", "dru": "dru_XX",
    "pwn": "pwn_XX", "pyu": "pyu_XX", "ssf": "ssf_XX", "sxr": "sxr_XX",
    "szy": "szy_XX", "tao": "tao_XX", "tay": "tay_XX", "trv": "trv_XX",
    "tsu": "tsu_XX", "xnb": "xnb_XX", "xsy": "xsy_XX",

    # Names some of your files/scripts use (amis, paiwan, tsou, ...)
    "amis": "ami_XX", "paiwan": "pwn_XX", "tsou": "tsu_XX",
    "bunun": "bnn_XX", "rukai": "dru_XX", "puyuma": "pyu_XX",
    "atayal": "tay_XX", "seediq": "trv_XX",

    # Built-ins
    "english": "en_XX",
    "chinese": "zh_CN",  # mBART-50 uses zh_CN; works fine for Traditional too if you added chars
}

FORMOSAN_ALIASES = {
    "amis": {"amis", "ami"},
    "paiwan": {"paiwan", "pwn"},
    "tsou": {"tsou", "tsu"},
    "bunun": {"bunun", "bnn"},
    "rukai": {"rukai", "dru"},
    "puyuma": {"puyuma", "pyu"},
    "atayal": {"atayal", "tay"},
    "seediq": {"seediq", "trv"},
    "ami": {"amis", "ami"},
    "bnn": {"bunun", "bnn"},
    "ckv": {"ckv"},
    "dru": {"rukai", "dru"},
    "pwn": {"paiwan", "pwn"},
    "pyu": {"puyuma", "pyu"},
    "ssf": {"ssf"},
    "sxr": {"sxr"},
    "szy": {"szy"},
    "tao": {"tao"},
    "tay": {"atayal", "tay"},
    "trv": {"seediq", "trv"},
    "tsu": {"tsou", "tsu"},
    "xnb": {"xnb"},
    "xsy": {"xsy"},
}
ALIASES = {
    "english": {"english", "en", "en_sentence", "english_sentence"},
    "chinese": {"chinese", "zh", "zh_cn", "chinese_sentence", "traditional_chinese", "zh_trad"},
    **FORMOSAN_ALIASES,
}

# ------------------------------ helpers -----------------------------------
def get_mbart_code(name: str) -> str:
    key = name.lower()
    if key not in MBART_LANGUAGE_MAP:
        raise SystemExit(f"❌ Unsupported language '{name}'. "
                         f"Supported: {', '.join(sorted(MBART_LANGUAGE_MAP))}")
    return MBART_LANGUAGE_MAP[key]

def load_tokenizer_model(tok_dir: str, model_dir: str, device: torch.device):
    tok = MBart50Tokenizer.from_pretrained(tok_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tok))  # HF-recommended resizing
    model.to(device)
    return tok, model

def restore_custom_lang_ids(tokenizer: MBart50Tokenizer, codes: List[str]) -> None:
    """Re-wire custom language codes after reload (if missing)."""
    for c in codes:
        if c not in tokenizer.lang_code_to_id:
            tid = tokenizer.convert_tokens_to_ids(c)
            tokenizer.lang_code_to_id[c] = tid
            tokenizer.id_to_lang_code[tid] = c

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

    # common names seen in your corpora
    if "formosan_sentence" in cols and "chinese_sentence" in cols and tgt_lang.lower() == "chinese":
        return "formosan_sentence", "chinese_sentence"
    if "formosan_sentence" in cols and "english_sentence" in cols and tgt_lang.lower() == "english":
        return "formosan_sentence", "english_sentence"

    # language-ish names
    def pick(lang: str) -> Optional[str]:
        cand = [c for c in cols if c.lower() in ALIASES.get(lang.lower(), set())]
        return cand[0] if cand else None
    s = pick(src_lang)
    t = pick(tgt_lang)
    if s and t:
        return s, t

    # fallback: first two text columns
    text_cols = [c for c in df.columns if df[c].dtype == "object"]
    if len(text_cols) >= 2:
        return text_cols[0], text_cols[1]

    raise SystemExit("❌ Could not infer language columns. Use --src-col and --tgt-col.")

def load_test_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        test = df[df["split"].astype(str).str.lower().eq("test")].copy()
        if len(test) == 0:
            print("⚠️ No 'test' rows found in 'split' column; evaluating on ALL rows.")
            test = df.copy()
        return test.reset_index(drop=True)
    print("ℹ️ No 'split' column; evaluating on ALL rows.")
    return df.reset_index(drop=True)

@torch.no_grad()
def batched_generate(
    tokenizer: MBart50Tokenizer,
    model: MBartForConditionalGeneration,
    src_texts: List[str],
    from_code: str,
    to_code: str,
    device: torch.device,
    max_length: int = 128,
    num_beams: int = 4,
    batch_size: int = 16,
) -> List[str]:
    outs: List[str] = []

    # Ensure tokenizer knows the src special tokens (mBART-50)
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(from_code)
    else:
        tokenizer.src_lang = from_code

    forced_id = tokenizer.lang_code_to_id[to_code]
    n = len(src_texts)
    for i in range(0, n, batch_size):
        chunk = src_texts[i:i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        gen = model.generate(
            **enc,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_id,  # REQUIRED for mBART-50
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs

def score_all(
    sys_out: List[str],
    ref: List[str],
) -> Dict[str, float]:
    bleu = sacrebleu.corpus_bleu(sys_out, [ref])
    chrf = CHRF().corpus_score(sys_out, [ref])
    ter  = TER().corpus_score(sys_out, [ref])
    return {
        "BLEU": float(bleu.score),
        "chrF2": float(chrf.score),   # default is chrF2
        "TER": float(ter.score),
    }

def pretty_print(title: str, metrics: Dict[str, float], n: int, samples: List[Tuple[str, str, str]]):
    print(f"\n===== {title} =====")
    print(f"Samples: {n}")
    print(f"BLEU:  {metrics['BLEU']:.2f}")
    print(f"chrF2: {metrics['chrF2']:.2f}")
    print(f"TER:   {metrics['TER']:.2f}")
    print("\n--- Examples ---")
    for s, r, h in samples[:3]:
        print(f"SRC: {s}\nREF: {r}\nHYP: {h}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-lang", required=True, help="e.g., amis, paiwan, tsou, english, chinese")
    ap.add_argument("--tgt-lang", required=True, help="e.g., chinese or english")
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer dir")
    ap.add_argument("--model", required=True, help="Path to model dir")
    ap.add_argument("--input", type=Path, required=True, help="CSV with parallel data (must contain test split or will eval on all)")

    # Optional explicit columns
    ap.add_argument("--src-col", default=None)
    ap.add_argument("--tgt-col", default=None)

    # Generation / batching
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate on first N examples (for quick runs)")

    # Device
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Output
    ap.add_argument("--save-json", default=None, help="Path to save metrics+predictions JSON")

    args = ap.parse_args()

    # Resolve device
    dev = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        (args.device if args.device != "auto" else "cpu")
    )
    device = torch.device(dev)
    print(f"Device: {device}")

    # Language codes
    src_code = get_mbart_code(args.src_lang)
    tgt_code = get_mbart_code(args.tgt_lang)
    print(f"Lang codes: {args.src_lang} -> {src_code} | {args.tgt_lang} -> {tgt_code}")

    # Load model
    tokenizer, model = load_tokenizer_model(args.tokenizer, args.model, device)
    # Rewire any custom lang codes that might not be in mapping after reload
    restore_custom_lang_ids(tokenizer, [src_code, tgt_code])

    # Data
    df_test_all = load_test_split(args.input)
    s_col, t_col = smart_find_columns(df_test_all, args.src_lang, args.tgt_lang, args.src_col, args.tgt_col)
    df_test = df_test_all[[s_col, t_col]].dropna().reset_index(drop=True)

    if args.limit is not None:
        df_test = df_test.iloc[:args.limit].reset_index(drop=True)

    src_texts = df_test[s_col].astype(str).tolist()
    tgt_texts = df_test[t_col].astype(str).tolist()
    n = len(df_test)
    assert n > 0, "No test rows to evaluate."

    print(f"Evaluating on {n} examples | columns: '{s_col}' -> '{t_col}'")

    # ---- Direction 1: src -> tgt ----
    sys_out_1 = batched_generate(
        tokenizer, model, src_texts, src_code, tgt_code, device,
        max_length=args.max_length, num_beams=args.beam, batch_size=args.batch_size
    )
    metrics_1 = score_all(sys_out_1, tgt_texts)

    # ---- Direction 2: tgt -> src ----
    sys_out_2 = batched_generate(
        tokenizer, model, tgt_texts, tgt_code, src_code, device,
        max_length=args.max_length, num_beams=args.beam, batch_size=args.batch_size
    )
    metrics_2 = score_all(sys_out_2, src_texts)

    # Pretty print + (optional) save JSON
    ex_1 = list(zip(src_texts, tgt_texts, sys_out_1))
    ex_2 = list(zip(tgt_texts, src_texts, sys_out_2))
    pretty_print(f"{args.src_lang} → {args.tgt_lang}", metrics_1, n, ex_1)
    pretty_print(f"{args.tgt_lang} → {args.src_lang}", metrics_2, n, ex_2)

    if args.save_json:
        out = {
            "pair": {"src": args.src_lang, "tgt": args.tgt_lang,
                     "src_code": src_code, "tgt_code": tgt_code},
            "n_examples": n,
            "settings": {
                "batch_size": args.batch_size, "max_length": args.max_length, "beam": args.beam,
            },
            "metrics": {
                f"{args.src_lang}->{args.tgt_lang}": metrics_1,
                f"{args.tgt_lang}->{args.src_lang}": metrics_2,
            },
            "examples": {
                f"{args.src_lang}->{args.tgt_lang}": ex_1[:10],
                f"{args.tgt_lang}->{args.src_lang}": ex_2[:10],
            },
        }
        Path(os.path.dirname(args.save_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved results JSON to: {args.save_json}")

if __name__ == "__main__":
    main()
