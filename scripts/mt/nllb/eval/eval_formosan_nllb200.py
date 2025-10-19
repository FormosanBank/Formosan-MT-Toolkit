#!/usr/bin/env python3
"""
Evaluate an NLLB-200 fine-tuned model on a Formosan↔(EN/ZH) test split.

CSV shape (required):
  - Column 1: formosan code (e.g. "ami")
  - Column 2: target code   (e.g. "en" or "zh")
  - Column "split": with value "test" for evaluation rows

Example
-------
python eval_formosan_nllb.py \
  --tokenizer formosan_multilingual_nllb_spm_tokenizer \
  --model     formosan_multilingual_nllb_spm_model \
  --input     ami_en.csv \
  --batch-size 16 --max-length 192 --beam 5 \
  --max-new-tokens 64 --min-new-tokens 2 \
  --no-repeat-ngram-size 3 --repetition-penalty 1.1 --length-penalty 1.05 \
  --detok-latin --bleu-lowercase \
  --csv-out runs/ami_en_eval.csv --save-json runs/ami_en_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from transformers import NllbTokenizer, AutoModelForSeq2SeqLM

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
    from sacrebleu.metrics import BLEU, CHRF, TER
except Exception as e:
    raise SystemExit(
        "❌ sacrebleu is required. Install with:\n  pip install sacrebleu\n"
        f"Import error: {e}"
    )

# Optional detokenizer for Latin scripts
try:
    from sacremoses import MosesDetokenizer
    _HAVE_MOSES = True
except Exception:
    _HAVE_MOSES = False

# Optional normalization (like NLLB preprocessing)
try:
    from sacremoses import MosesPunctNormalizer
    _HAVE_MPN = True
except Exception:
    _HAVE_MPN = False

# ----------------------- code → NLLB LID map -----------------------
CODE_TO_LID: Dict[str, str] = {
    # Formosan (custom)
    "ami": "ami_Latn", "bnn": "bnn_Latn", "ckv": "ckv_Latn", "dru": "dru_Latn",
    "pwn": "pwn_Latn", "pyu": "pyu_Latn", "ssf": "ssf_Latn", "sxr": "sxr_Latn",
    "szy": "szy_Latn", "tao": "tao_Latn", "tay": "tay_Latn", "trv": "trv_Latn",
    "tsu": "tsu_Latn", "xnb": "xnb_Latn", "xsy": "xsy_Latn",

    # chinese 
    "chinese": "zho_Hant",
    # Targets
    "en":  "eng_Latn", "eng": "eng_Latn", "english": "eng_Latn",
    # Default Chinese to Traditional; override with --src-lid/--tgt-lid for Simplified
    "zh":  "zho_Hant", "zho": "zho_Hant", "zh_hant": "zho_Hant", "zh_trad": "zho_Hant",
    "zh_hans": "zho_Hans", "zh_cn": "zho_Hans", "zh_simp": "zho_Hans",
}

# ------------------------------ helpers -----------------------------------
def _normalize_list_like_train(texts: List[str]) -> List[str]:
    """Match training-time normalization: NFKC + MosesPunctNormalizer(lang='en')."""
    out: List[str] = []
    try:
        import unicodedata
        has_unicodedata = True
    except Exception:
        has_unicodedata = False
    mpn = MosesPunctNormalizer(lang="en") if _HAVE_MPN else None

    for t in texts:
        s = "" if t is None else str(t)
        if has_unicodedata:
            try:
                s = unicodedata.normalize("NFKC", s)
            except Exception:
                pass
        if mpn is not None:
            s = mpn.normalize(s)
        out.append(s)
    return out

def _bleu_tokenizer_for_lid(lid: str) -> str:
    lid = (lid or "").lower()
    return "zh" if lid.startswith("zho_") or lid in {"zh", "zho", "zh_hans", "zh_hant"} else "13a"

def lid_from_code(code: str) -> str:
    """
    Turn a column/language code into an NLLB LID.
    Accepts direct LIDs (e.g., 'eng_Latn', 'zho_Hant'); otherwise uses CODE_TO_LID.
    """
    c = str(code).strip()
    if "_" in c and len(c) >= 7:
        return c  # assume already an NLLB LID like ami_Latn or zho_Hans
    key = c.lower()
    if key not in CODE_TO_LID:
        raise SystemExit(f"❌ Unrecognized language code '{code}'. Add it to CODE_TO_LID or pass --src-lid/--tgt-lid.")
    return CODE_TO_LID[key]

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


def ensure_lang_token(tokenizer: NllbTokenizer, code: str) -> int:
    """Return the id for a language code, but fail if mapping is fishy."""
    tid = tokenizer.convert_tokens_to_ids(code)
    problems = []
    if tid == tokenizer.unk_token_id:
        problems.append("maps to <unk>")
    # must be an added *special* token in NLLB (language codes live here)
    add_specs = getattr(tokenizer, "additional_special_tokens", []) or []
    if code not in add_specs:
        problems.append("not present in tokenizer.additional_special_tokens")
    # round-trip: the id must decode back to the very same string token
    decoded = tokenizer.decode([tid], skip_special_tokens=False)
    if decoded != code:
        problems.append(f"decodes back to {decoded!r} instead of {code!r}")
    if problems:
        raise SystemExit(
            "❌ Language token mapping is broken for "
            f"{code!r} (id={tid}): " + "; ".join(problems) + "\n"
            "Fix: rebuild the tokenizer so all language codes live in "
            "`additional_special_tokens` with stable ids, and reload the "
            "model with exactly that tokenizer directory."
        )
    return tid


def pick_columns(df: pd.DataFrame, src_col: Optional[str], tgt_col: Optional[str]) -> Tuple[str, str]:
    """
    If explicit names provided, use them.
    Else: take the first two non-'split' columns in order.
    """
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

def _looks_tokenized_latin(s: str) -> bool:
    # Common markers that SacreBLEU warns about (spaces before punctuation, ``/'' quotes, etc.)
    return bool(re.search(r"\s[.,!?;:]", s) or "``" in s or "''" in s)

def maybe_detok_latin(texts: List[str]) -> List[str]:
    """
    If text *looks* tokenized (Moses-style), detokenize for fair scoring.
    Uses sacremoses when available; otherwise a minimal regex fallback.
    """
    out = []
    if _HAVE_MOSES:
        md = MosesDetokenizer(lang="en")  # neutral-en works fine for Latin punctuation joining
        for t in texts:
            if _looks_tokenized_latin(t):
                out.append(md.detokenize(t.split()))
            else:
                out.append(t)
        return out
    # Fallback: collapse spaces before punctuation + a couple of common joins
    for t in texts:
        if _looks_tokenized_latin(t):
            t2 = re.sub(r"\s+([.,!?;:])", r"\1", t)
            t2 = t2.replace(" `` ", " “").replace(" '' ", " ”")
            t2 = t2.replace(" n't", "n't").replace(" 's", "'s")
            out.append(t2)
        else:
            out.append(t)
    return out

@torch.no_grad()
def batched_generate(
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    src_texts: List[str],
    from_code: str,
    to_code: str,
    device: torch.device,
    enc_max_length: int = 128,
    num_beams: int = 4,
    batch_size: int = 16,
    progress: bool = True,
    desc: Optional[str] = None,
    gen_kwargs: Optional[Dict] = None,
) -> List[str]:
    """
    Encode with correct source LID and generate with BOTH forced_bos_token_id and
    decoder_start_token_id set to the target LID for HF version robustness.
    Prefer max_new_tokens/min_new_tokens over legacy max_length for decoding.
    """
    gen_kwargs = dict(gen_kwargs or {})
    # Resolve/validate the target language id once
    forced_id = ensure_lang_token(tokenizer, to_code)
    gen_kwargs.setdefault("forced_bos_token_id", forced_id)
    gen_kwargs.setdefault("decoder_start_token_id", forced_id)
    gen_kwargs.setdefault("eos_token_id", tokenizer.eos_token_id)
    gen_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)

    # length-bucket for throughput
    order = np.argsort([-len(s) for s in src_texts])
    restore = np.argsort(order)
    texts_sorted = [src_texts[i] for i in order]

    outs_sorted: List[str] = []

    total = len(texts_sorted)
    use_pb = progress and (_TQDM_AVAILABLE)
    if progress and not _TQDM_AVAILABLE:
        print("ℹ️ tqdm not found; install with `pip install tqdm` for progress bars.")
    pbar = tqdm(total=total, unit="ex", dynamic_ncols=True, smoothing=0.1,
                desc=desc or "generate", disable=not use_pb)

    for i in range(0, len(texts_sorted), batch_size):
        chunk = texts_sorted[i:i + batch_size]

        # Encode with correct source language
        tokenizer.src_lang = from_code
        enc = tokenizer(
            chunk, return_tensors="pt", padding=True, truncation=True, max_length=enc_max_length
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        # Generate
        gen_out = model.generate(
            **enc,
            num_beams=num_beams,
            **gen_kwargs,
            return_dict_in_generate=True,
            output_scores=False,
        )

        # Optional sanity check: first token should be the target LID
        try:
            first_tokens = gen_out.sequences[:, 0].tolist()
            if any(t != forced_id for t in first_tokens):
                print(f"[warn] First decoder token != {to_code} for some samples.")
        except Exception:
            pass

        outs_sorted.extend(tokenizer.batch_decode(gen_out.sequences, skip_special_tokens=True))
        pbar.update(len(chunk))

    pbar.close()

    # restore original order
    outs = [outs_sorted[i] for i in restore]
    return outs


def score_all(sys_out: List[str], ref: List[str], bleu_tok: str = "13a", lowercase: bool = False) -> Dict[str, float]:
    bleu_metric = BLEU(tokenize=bleu_tok, effective_order=True, lowercase=lowercase)
    chrf_metric = CHRF()
    ter_metric  = TER()
    return {
        "BLEU":  float(bleu_metric.corpus_score(sys_out, [ref]).score),
        "chrF2": float(chrf_metric.corpus_score(sys_out, [ref]).score),
        "TER":   float(ter_metric.corpus_score(sys_out, [ref]).score),
    }


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

    # Optional explicit column names (else: use first two non-'split' columns)
    ap.add_argument("--src-col", default=None, help="Name of source text column (e.g. ami)")
    ap.add_argument("--tgt-col", default=None, help="Name of target text column (e.g. en)")

    # Optional explicit LIDs override (else: inferred from column names)
    ap.add_argument("--src-lid", default=None, help="Override NLLB LID, e.g. ami_Latn")
    ap.add_argument("--tgt-lid", default=None, help="Override NLLB LID, e.g. zho_Hans")

    # after other add_argument calls, before parse_args()
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate only the first N test examples")
    # Generation / batching
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256, help="Encoder truncation length")
    ap.add_argument("--beam", type=int, default=4)                           # was 4

    # Modern generation knobs
    ap.add_argument("--max-new-tokens", type=int, default=32)                # was 64
    ap.add_argument("--min-new-tokens", type=int, default=2)                 # was 2
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)           # was 3
    ap.add_argument("--repetition-penalty", type=float, default=1.1)         # was 1.1
    ap.add_argument("--length-penalty", type=float, default=0.9)             # was 1.05


    # Scoring options
    ap.add_argument("--bleu-lowercase", action="store_true",
                    help="Compute BLEU in lowercase (for noisy casing)")
    ap.add_argument("--detok-latin", action="store_true",
                    help="Detokenize Latin-script refs/hyps before scoring (when CSV looks Moses-tokenized)")

    # Device
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    
    # --- add a CLI switch in your argparse setup ---
    # args
    ap.add_argument("--normalize", dest="normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)

    # Outputs
    ap.add_argument("--save-json", default=None, help="Path to save metrics+samples JSON")
    ap.add_argument("--csv-out", default=None, help="Optional CSV with src/ref/hyp in both directions")

    args = ap.parse_args()

    # Device
    dev = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
           (args.device if args.device != "auto" else "cpu"))
    device = torch.device(dev)
    print(f"Device: {device}")

    # Load model/tokenizer
    tokenizer, model = load_tok_model(args.tokenizer, args.model, device)

    # Data (only test split)
    df_test = load_test_rows(args.input)
    src_col, tgt_col = pick_columns(df_test, args.src_col, args.tgt_col)

    # Infer codes from column names, then map to NLLB LIDs (unless overridden)
    src_code_str = src_col
    tgt_code_str = tgt_col
    src_lid = args.src_lid or lid_from_code(src_code_str)
    tgt_lid = args.tgt_lid or lid_from_code(tgt_code_str)

    # Sanity: LID tokens must exist
    ensure_lang_token(tokenizer, src_lid)
    ensure_lang_token(tokenizer, tgt_lid)

    # Slice & (optionally) limit
    df = df_test[[src_col, tgt_col]].dropna().reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[:args.limit].reset_index(drop=True)

    # --- just after you build src_texts/tgt_texts (before generation) ---
    src_texts = df[src_col].astype(str).tolist()
    tgt_texts = df[tgt_col].astype(str).tolist()

    if args.normalize:
        # normalize inputs used for generation and references used for scoring
        src_texts = _normalize_list_like_train(src_texts)
        tgt_texts = _normalize_list_like_train(tgt_texts)

    n = len(df)
    if n == 0:
        raise SystemExit("❌ No test rows to evaluate after filtering.")
    print(f"Columns: '{src_col}' (→ {src_lid})  |  '{tgt_col}' (→ {tgt_lid})")
    print(f"Evaluating on {n} test examples")

    # ---- Generation settings (modern) ----
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        # early_stopping is deprecated in HF >= 4.47 and ignored; omit to be version-safe
    }

    # --- replace your current scoring-prep block so hyps/refs are normalized consistently ---
    sys_out_1 = batched_generate(
        tokenizer, model, src_texts, src_lid, tgt_lid, device,
        enc_max_length=args.max_length, num_beams=args.beam,
        batch_size=args.batch_size, gen_kwargs=gen_kwargs, desc=f"{src_col}->{tgt_col}"
    )
    sys_out_2 = batched_generate(
        tokenizer, model, tgt_texts, tgt_lid, src_lid, device,
        enc_max_length=args.max_length, num_beams=args.beam,
        batch_size=args.batch_size, gen_kwargs=gen_kwargs, desc=f"{tgt_col}->{src_col}"
    )

    # Prepare refs/hyps for metrics
    ref1, ref2 = tgt_texts, src_texts
    hyp1, hyp2 = sys_out_1, sys_out_2

    # (Optional) apply the same normalization to hypotheses before scoring
    if args.normalize:
        hyp1 = _normalize_list_like_train(hyp1)
        hyp2 = _normalize_list_like_train(hyp2)

    # (Optional) detokenize Latin scripts after normalization, before scoring
    if args.detok_latin:
        if not tgt_lid.startswith("zho_"):
            ref1 = maybe_detok_latin(ref1)
            hyp1 = maybe_detok_latin(hyp1)
        if not src_lid.startswith("zho_"):
            ref2 = maybe_detok_latin(ref2)
            hyp2 = maybe_detok_latin(hyp2)

    # Scoring (unchanged, but now uses ref*/hyp*)
    bleu_tok_1 = _bleu_tokenizer_for_lid(tgt_lid)
    bleu_tok_2 = _bleu_tokenizer_for_lid(src_lid)
    metrics_1 = score_all(hyp1, ref1, bleu_tok=bleu_tok_1, lowercase=args.bleu_lowercase)
    metrics_2 = score_all(hyp2, ref2, bleu_tok=bleu_tok_2, lowercase=args.bleu_lowercase)

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
            "settings": {
                "batch_size": args.batch_size,
                "enc_max_length": args.max_length,
                "beam": args.beam,
                "gen": {
                    "max_new_tokens": args.max_new_tokens,
                    "min_new_tokens": args.min_new_tokens,
                    "no_repeat_ngram_size": args.no_repeat_ngram_size,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                },
                "bleu_lowercase": args.bleu_lowercase,
                "detok_latin": args.detok_latin,
            },
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
