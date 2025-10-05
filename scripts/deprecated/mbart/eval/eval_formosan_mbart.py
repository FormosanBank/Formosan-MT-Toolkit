#!/usr/bin/env python3
"""
Evaluate a fine-tuned mBART-50 model on a Formosan↔(EN/ZH) test split.

CSV shape (required):
  - Column 1: formosan code (e.g. "ami")
  - Column 2: target code   (e.g. "en" or "zh")
  - Column "split": must contain "test" for evaluation rows

Example
-------
python eval_formosan_mbart.py \
  --tokenizer formosan_multilingual_mbart_tokenizer \
  --model     formosan_multilingual_mbart_model \
  --input     ami_en.csv \
  --batch-size 16 --max-length 128 --beam 5 \
  --csv-out runs/ami_en_eval.csv --save-json runs/ami_en_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration


# progress bar (nice in terminals & notebooks)
try:
    from tqdm.auto import tqdm
    _TQDM_AVAILABLE = True
except Exception:
    _TQDM_AVAILABLE = False
    class _DummyPB:
        def __init__(self, *a, **k): pass
        def update(self, *a, **k): pass
        def close(self): pass
    def tqdm(*a, **k):  # type: ignore
        return _DummyPB()
    
# sacrebleu >= 2.x
try:
    import sacrebleu
    from sacrebleu.metrics import CHRF, TER
except Exception as e:
    raise SystemExit(
        "❌ sacrebleu is required. Install with:\n  pip install sacrebleu\n"
        f"Import error: {e}"
    )

# ----------------------- code → mBART LID map -----------------------
CODE_TO_M50_LID: Dict[str, str] = {
    # Formosan (your custom tokens)
    "ami": "ami_XX", "bnn": "bnn_XX", "ckv": "ckv_XX", "dru": "dru_XX",
    "pwn": "pwn_XX", "pyu": "pyu_XX", "ssf": "ssf_XX", "sxr": "sxr_XX",
    "szy": "szy_XX", "tao": "tao_XX", "tay": "tay_XX", "trv": "trv_XX",
    "tsu": "tsu_XX", "xnb": "xnb_XX", "xsy": "xsy_XX",
    # Common aliases you might see
    "amis": "ami_XX", "bunun": "bnn_XX", "kavalan": "ckv_XX", "rukai": "dru_XX",
    "paiwan": "pwn_XX", "puyuma": "pyu_XX", "thao": "ssf_XX", "saaroa": "sxr_XX",
    "sakizaya": "szy_XX", "atayal": "tay_XX", "seediq": "trv_XX", "tsou": "tsu_XX",
    # Targets
    "en": "en_XX", "eng": "en_XX", "english": "en_XX",
    "zh": "zh_CN", "zho": "zh_CN", "zh_cn": "zh_CN", "chinese": "zh_CN",
}

def lid_from_code(code: str) -> str:
    """Accepts direct mBART LIDs (e.g., 'en_XX') or maps short codes like 'ami', 'en', 'zh'."""
    c = str(code).strip()
    if c.endswith("_XX"):  # already an mBART LID
        return c
    key = c.lower()
    if key not in CODE_TO_M50_LID:
        raise SystemExit(f"❌ Unrecognized language code '{code}'. Add it to CODE_TO_M50_LID or pass --src-lid/--tgt-lid.")
    return CODE_TO_M50_LID[key]

# ------------------------------ helpers -----------------------------------
def load_tokenizer_model(tok_dir: str, model_dir: str, device: torch.device):
    tok = MBart50Tokenizer.from_pretrained(tok_dir)
    model = MBartForConditionalGeneration.from_pretrained(model_dir)
    model.resize_token_embeddings(len(tok))
    model.to(device)
    model.eval()
    return tok, model

def restore_custom_lang_ids(tokenizer: MBart50Tokenizer, codes: List[str]) -> None:
    """Re-wire custom language codes after reload (if missing)."""
    for c in codes:
        if c not in tokenizer.lang_code_to_id:
            tid = tokenizer.convert_tokens_to_ids(c)
            tokenizer.lang_code_to_id[c] = tid
            tokenizer.id_to_lang_code[tid] = c

def ensure_lang_token(tokenizer: MBart50Tokenizer, code: str) -> int:
    """Get a valid BOS id for code; error if it resolves to unk."""
    # prefer the mapping if present
    if code in tokenizer.lang_code_to_id:
        tid = tokenizer.lang_code_to_id[code]
    else:
        tid = tokenizer.convert_tokens_to_ids(code)
    if tid == tokenizer.unk_token_id:
        raise SystemExit(
            f"❌ Language token {code} resolves to <unk>. "
            "Ensure your tokenizer includes this code as a special token."
        )
    return tid

def pick_columns(df: pd.DataFrame, src_col: Optional[str], tgt_col: Optional[str]) -> Tuple[str, str]:
    """If names provided use them; else first two non-'split' columns in order."""
    cols = [c for c in df.columns if c.lower() != "split"]
    if src_col and tgt_col:
        if src_col not in df.columns or tgt_col not in df.columns:
            raise SystemExit(f"❌ Columns not found. Available: {list(df.columns)}")
        return src_col, tgt_col
    if len(cols) < 2:
        raise SystemExit("❌ Need at least two text columns (plus 'split').")
    return cols[0], cols[1]

def load_test_rows(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" not in df.columns:
        raise SystemExit("❌ CSV must contain a 'split' column.")
    test = df[df["split"].astype(str).str.lower().eq("test")].copy()
    if len(test) == 0:
        raise SystemExit("❌ No rows with split == 'test'.")
    return test.reset_index(drop=True)

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
    progress: bool = True,
    desc: Optional[str] = None,
) -> List[str]:
    # set source language special tokens
    if hasattr(tokenizer, "set_src_lang_special_tokens"):
        tokenizer.set_src_lang_special_tokens(from_code)
    else:
        tokenizer.src_lang = from_code

    forced_id = ensure_lang_token(tokenizer, to_code)

    # simple length bucket
    order = np.argsort([-len(s) for s in src_texts])
    restore = np.argsort(order)
    texts_sorted = [src_texts[i] for i in order]

    outs_sorted: List[str] = []

    total = len(texts_sorted)
    use_pb = progress and (_TQDM_AVAILABLE)
    if progress and not _TQDM_AVAILABLE:
        print("ℹ️ tqdm not found; install with `pip install tqdm` for progress bars.")

    pbar = tqdm(
        total=total,
        unit="ex",
        dynamic_ncols=True,
        smoothing=0.1,
        desc=desc or "generate",
        disable=not use_pb,
    )

    for i in range(0, total, batch_size):
        chunk = texts_sorted[i:i + batch_size]
        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)

        gen = model.generate(
            **enc,
            max_length=max_length,
            num_beams=num_beams,
            forced_bos_token_id=forced_id,  # REQUIRED for mBART-50
        )
        outs_sorted.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        pbar.update(len(chunk))

    pbar.close()

    # restore original order
    outs = [outs_sorted[i] for i in restore]
    return outs


def score_all(sys_out: List[str], ref: List[str]) -> Dict[str, float]:
    tok = "zh"  # for Chinese references
    bleu = sacrebleu.corpus_bleu(sys_out, [ref], tokenize=tok)     
    chrf = CHRF().corpus_score(sys_out, [ref])
    ter  = TER().corpus_score(sys_out, [ref])
    return {"BLEU": float(bleu.score), "chrF2": float(chrf.score), "TER": float(ter.score)}

def pretty_print(title: str, metrics: Dict[str, float], n: int, examples: List[Tuple[str, str, str]]):
    print(f"\n===== {title} =====")
    print(f"Samples: {n}")
    print(f"BLEU:  {metrics['BLEU']:.2f}")
    print(f"chrF2: {metrics['chrF2']:.2f}")
    print(f"TER:   {metrics['TER']:.2f}")
    print("\n--- Examples ---")
    for s, r, h in examples[:3]:
        print(f"SRC: {s}\nREF: {r}\nHYP: {h}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer dir")
    ap.add_argument("--model", required=True, help="Path to model dir")
    ap.add_argument("--input", type=Path, required=True, help="CSV with columns: <src_code>, <tgt_code>, split")

    # Optional explicit column names (else: first two non-'split' columns)
    ap.add_argument("--src-col", default=None, help="Name of source text column (e.g. ami)")
    ap.add_argument("--tgt-col", default=None, help="Name of target text column (e.g. en)")

    # Optional explicit LIDs override (else: inferred from column names)
    ap.add_argument("--src-lid", default=None, help="Override mBART LID, e.g. ami_XX")
    ap.add_argument("--tgt-lid", default=None, help="Override mBART LID, e.g. zh_CN")

    # Generation / batching
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None, help="Evaluate on first N test examples")

    # Device
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Outputs
    ap.add_argument("--save-json", default=None, help="Path to save metrics+samples JSON")
    ap.add_argument("--csv-out", default=None, help="Optional CSV with src/ref/hyp in both directions")

    args = ap.parse_args()

    # Device
    dev = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
           (args.device if args.device != "auto" else "cpu"))
    device = torch.device(dev)
    print(f"Device: {device}")

    # Load
    tokenizer, model = load_tokenizer_model(args.tokenizer, args.model, device)

    # Data: only test split
    df_test = load_test_rows(args.input)
    src_col, tgt_col = pick_columns(df_test, args.src_col, args.tgt_col)

    # Infer mBART LIDs from column names (unless overridden)
    src_lid = args.src_lid or lid_from_code(src_col)
    tgt_lid = args.tgt_lid or lid_from_code(tgt_col)

    # Ensure custom tokens are wired
    restore_custom_lang_ids(tokenizer, [src_lid, tgt_lid])
    ensure_lang_token(tokenizer, src_lid)
    ensure_lang_token(tokenizer, tgt_lid)

    # Slice & (optionally) limit
    df = df_test[[src_col, tgt_col]].dropna().reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)

    src_texts = df[src_col].astype(str).tolist()
    tgt_texts = df[tgt_col].astype(str).tolist()
    n = len(df)
    if n == 0:
        raise SystemExit("❌ No test rows to evaluate after filtering.")
    print(f"Columns: '{src_col}' (→ {src_lid})  |  '{tgt_col}' (→ {tgt_lid})")
    print(f"Evaluating on {n} test examples")

    # ---- Direction 1: src -> tgt ----
    sys_out_1 = batched_generate(
        tokenizer, model, src_texts, src_lid, tgt_lid, device,
        max_length=args.max_length, num_beams=args.beam, batch_size=args.batch_size
    )
    metrics_1 = score_all(sys_out_1, tgt_texts)

    # ---- Direction 2: tgt -> src ----
    sys_out_2 = batched_generate(
        tokenizer, model, tgt_texts, tgt_lid, src_lid, device,
        max_length=args.max_length, num_beams=args.beam, batch_size=args.batch_size
    )
    metrics_2 = score_all(sys_out_2, src_texts)

    # Pretty print
    ex_1 = list(zip(src_texts, tgt_texts, sys_out_1))
    ex_2 = list(zip(tgt_texts, src_texts, sys_out_2))
    pretty_print(f"{src_col} → {tgt_col}", metrics_1, n, ex_1)
    pretty_print(f"{tgt_col} → {src_col}", metrics_2, n, ex_2)

    # Save JSON
    if args.save_json:
        out = {
            "pair": {"src_col": src_col, "tgt_col": tgt_col, "src_lid": src_lid, "tgt_lid": tgt_lid},
            "n_examples": n,
            "settings": {"batch_size": args.batch_size, "max_length": args.max_length, "beam": args.beam},
            "metrics": {f"{src_col}->{tgt_col}": metrics_1, f"{tgt_col}->{src_col}": metrics_2},
            "examples": {f"{src_col}->{tgt_col}": ex_1[:10], f"{tgt_col}->{src_col}": ex_2[:10]},
        }
        Path(os.path.dirname(args.save_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved results JSON to: {args.save_json}")

    # Save CSV
    if args.csv_out:
        df_out = pd.DataFrame({
            f"src_{src_col}": src_texts,
            f"ref_{tgt_col}": tgt_texts,
            f"hyp_{tgt_col}": sys_out_1,
            f"hyp_{src_col}": sys_out_2,
        })
        Path(os.path.dirname(args.csv_out) or ".").mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.csv_out, index=False)
        print(f"💾 Saved predictions CSV to: {args.csv_out}")

if __name__ == "__main__":
    main()
