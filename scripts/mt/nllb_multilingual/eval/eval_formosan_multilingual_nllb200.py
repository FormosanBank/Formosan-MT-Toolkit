#!/usr/bin/env python3
"""
Evaluate an NLLB-200 fine-tuned model on Formosan↔Chinese/English (EN/ZH) test splits.

Modes
-----
1) Single-pair mode (default, backwards compatible with old script):
   CSV shape:
     - Two text columns (e.g. 'ami' and 'zh' or 'english')
     - Column 'split' with value 'test' for evaluation rows

   Example:
   python eval_formosan_nllb.py \
     --tokenizer formosan_multilingual_nllb_spm_tokenizer \
     --model     formosan_multilingual_nllb_spm_model \
     --input     ami_zh_processed.csv \
     --src-col   ami --tgt-col zh \
     --batch-size 16 --max-length 192 --beam 5 \
     --max-new-tokens 64 --min-new-tokens 2 \
     --no-repeat-ngram-size 3 --repetition-penalty 1.1 --length-penalty 1.05 \
     --detok-latin --bleu-lowercase \
     --csv-out runs/ami_zh_eval.csv --save-json runs/ami_zh_eval.json

2) Multilingual mode (for new corpus):
   CSV shape (required):
     lang_code, formosan_sentence, chinese_sentence OR english_sentence, source, dialect, split

   - 'lang_code' is 3-letter like 'ami', 'bnn', ...
   - 'split' values: train / valid|val / test (we only use 'test')
   - Target column is chosen via --tgt-lang {chinese,english}:
        chinese -> uses 'chinese_sentence'
        english -> uses 'english_sentence'
   - We compute BLEU/chrF++ per language and global.

   Example (Formosan↔Chinese):
   python eval_formosan_nllb.py \
     --multilingual \
     --tgt-lang chinese \
     --tokenizer formosan_multilingual_nllb_tokenizer \
     --model     formosan_multilingual_nllb_model \
     --input     big_corpus_zh.csv \
     --batch-size 16 --max-length 192 --beam 5 \
     --max-new-tokens 64 --min-new-tokens 2 \
     --no-repeat-ngram-size 3 --repetition-penalty 1.1 --length-penalty 1.05 \
     --detok-latin --bleu-lowercase \
     --csv-out runs/multilingual_zh_eval.csv --save-json runs/multilingual_zh_eval.json

   Example (Formosan↔English):
   python eval_formosan_nllb.py \
     --multilingual \
     --tgt-lang english \
     --tokenizer formosan_multilingual_nllb_tokenizer \
     --model     formosan_multilingual_nllb_model \
     --input     big_corpus_en.csv \
     --batch-size 16 --max-length 192 --beam 5 \
     --max-new-tokens 64 --min-new-tokens 2 \
     --no-repeat-ngram-size 3 --repetition-penalty 1.1 --length-penalty 1.05 \
     --detok-latin --bleu-lowercase \
     --csv-out runs/multilingual_en_eval.csv --save-json runs/multilingual_en_eval.json
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

    # chinese/english (for multilingual corpus)
    "chinese": "zho_Hant",
    "english": "eng_Latn",

    # Targets / general
    "en":  "eng_Latn", "eng": "eng_Latn",
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
    Turn a language code into an NLLB LID.
    Accepts direct LIDs (e.g., 'eng_Latn', 'zho_Hant'); otherwise uses CODE_TO_LID.
    """
    c = str(code).strip()
    if "_" in c and len(c) >= 7:
        return c  # assume already an NLLB LID like ami_Latn or zho_Hans
    key = c.lower()
    if key not in CODE_TO_LID:
        raise SystemExit(
            f"❌ Unrecognized language code '{code}'. Add it to CODE_TO_LID or pass "
            f"--src-lid/--tgt-lid in single-pair mode."
        )
    return CODE_TO_LID[key]


def load_tok_model(tok_dir: str, model_dir: str, device: torch.device):
    tok = NllbTokenizer.from_pretrained(tok_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    model.config.decoder_start_token_id = tok.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tok.eos_token_id

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
    Single-pair mode:
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
    Encode with correct source LID and generate with forced_bos_token_id set to
    the target LID. Keep decoder_start_token_id as EOS (`</s>`), which is the
    M2M100/NLLB decoder start used by Transformers 4.56.
    Prefer max_new_tokens/min_new_tokens over legacy max_length for decoding.
    """
    gen_kwargs = dict(gen_kwargs or {})
    # Resolve/validate the target language id once
    forced_id = ensure_lang_token(tokenizer, to_code)
    gen_kwargs.setdefault("forced_bos_token_id", forced_id)
    gen_kwargs.setdefault("decoder_start_token_id", tokenizer.eos_token_id)
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

        # Optional sanity check: generation should start </s>, target LID.
        try:
            start_tokens = gen_out.sequences[:, 0].tolist()
            target_tokens = gen_out.sequences[:, 1].tolist()
            if any(t != tokenizer.eos_token_id for t in start_tokens):
                print("[warn] First decoder token != </s> for some samples.")
            if any(t != forced_id for t in target_tokens):
                print(f"[warn] First generated token != {to_code} for some samples.")
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
    ter_metric = TER()
    return {
        "BLEU": float(bleu_metric.corpus_score(sys_out, [ref]).score),
        "chrF2": float(chrf_metric.corpus_score(sys_out, [ref]).score),
        "TER": float(ter_metric.corpus_score(sys_out, [ref]).score),
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


# ------------------------- Multilingual eval (new) -------------------------
def eval_multilingual(
    args,
    tokenizer: NllbTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: torch.device,
):
    """
    New mode for big multilingual corpus with columns:
      lang_code, formosan_sentence, chinese_sentence OR english_sentence, source, dialect, split

    - Evaluates all languages in 'lang_code' that have split == 'test'.
    - Target column chosen via args.tgt_lang in {'chinese','english'}.
    - Computes metrics per language and global (micro-averaged over sentences).
    """
    df = pd.read_csv(args.input)

    tgt_lang = getattr(args, "tgt_lang", "chinese").lower()
    if tgt_lang == "chinese":
        tgt_col = "chinese_sentence"
        tgt_label = "zh"
        default_tgt_code = "zh"
    elif tgt_lang == "english":
        tgt_col = "english_sentence"
        tgt_label = "en"
        default_tgt_code = "en"
    else:
        raise SystemExit(
            f"❌ Multilingual mode: --tgt-lang must be 'chinese' or 'english', got {tgt_lang!r}"
        )

    required = ["lang_code", "formosan_sentence", tgt_col, "split"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"❌ Multilingual mode: CSV is missing required columns: {missing}")

    # Filter to test split and rows with both sides
    df = df[df["split"].astype(str).str.lower().eq("test")].copy()
    df = df[~df["formosan_sentence"].isna() & ~df[tgt_col].isna()]
    if df.empty:
        raise SystemExit(
            f"❌ Multilingual mode: no test rows with both formosan_sentence and {tgt_col}."
        )

    # Normalize lang_code strings
    df["lang_code_norm"] = df["lang_code"].astype(str).str.strip().str.lower()
    all_langs = sorted(df["lang_code_norm"].unique().tolist())

    # Filter to codes we know how to map
    valid_langs = []
    for lc in all_langs:
        try:
            _ = lid_from_code(lc)
            valid_langs.append(lc)
        except SystemExit as e:
            print(f"[warn] Skipping unsupported lang_code={lc!r}: {e}")

    if not valid_langs:
        raise SystemExit("❌ Multilingual mode: no supported languages found in lang_code.")

    # Global aggregators (after normalization/detok) for micro-averaged metrics
    global_ref_src2tgt: List[str] = []
    global_hyp_src2tgt: List[str] = []
    global_ref_tgt2src: List[str] = []
    global_hyp_tgt2src: List[str] = []

    # For optional CSV export
    csv_records: List[Dict[str, str]] = []

    # Target LID (zh or en) same for all langs
    tgt_lid_global = args.tgt_lid or lid_from_code(default_tgt_code)
    ensure_lang_token(tokenizer, tgt_lid_global)

    metrics_by_lang: Dict[str, Dict[str, Dict[str, float]]] = {}
    counts_by_lang: Dict[str, int] = {}

    print("\n================ Multilingual BLEU/chrF++ Eval (per language) ================")

    for lc in valid_langs:
        sub = df[df["lang_code_norm"] == lc].copy().reset_index(drop=True)
        if args.limit is not None:
            sub = sub.iloc[:args.limit].reset_index(drop=True)
        if sub.empty:
            continue

        src_lid = lid_from_code(lc)
        ensure_lang_token(tokenizer, src_lid)

        # Formosan (src) + target (tgt: Chinese or English)
        src_texts = sub["formosan_sentence"].astype(str).tolist()
        tgt_texts = sub[tgt_col].astype(str).tolist()

        # Normalize like training if requested
        if args.normalize:
            src_texts = _normalize_list_like_train(src_texts)
            tgt_texts = _normalize_list_like_train(tgt_texts)

        n = len(sub)
        print(f"\n[Lang={lc}] Evaluating on {n} test examples")

        # Generation settings (same as single-pair)
        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
        }

        # Formosan -> target
        sys_out_1 = batched_generate(
            tokenizer,
            model,
            src_texts,
            src_lid,
            tgt_lid_global,
            device,
            enc_max_length=args.max_length,
            num_beams=args.beam,
            batch_size=args.batch_size,
            gen_kwargs=gen_kwargs,
            desc=f"{lc}->{tgt_label}",
        )
        # Target -> Formosan
        sys_out_2 = batched_generate(
            tokenizer,
            model,
            tgt_texts,
            tgt_lid_global,
            src_lid,
            device,
            enc_max_length=args.max_length,
            num_beams=args.beam,
            batch_size=args.batch_size,
            gen_kwargs=gen_kwargs,
            desc=f"{tgt_label}->{lc}",
        )

        # Prepare refs/hyps for metrics
        ref1, hyp1 = tgt_texts, sys_out_1  # Formosan -> target
        ref2, hyp2 = src_texts, sys_out_2  # Target -> Formosan

        # Optional: normalize hyps too
        if args.normalize:
            hyp1 = _normalize_list_like_train(hyp1)
            hyp2 = _normalize_list_like_train(hyp2)

        # Optional detok for Latin scripts
        if args.detok_latin:
            # direction 1 target is zh/en; detok only if non-Chinese (i.e., English)
            if not tgt_lid_global.startswith("zho_"):
                ref1 = maybe_detok_latin(ref1)
                hyp1 = maybe_detok_latin(hyp1)
            # direction 2 target is Formosan (Latin) -> detok ok
            if not src_lid.startswith("zho_"):
                ref2 = maybe_detok_latin(ref2)
                hyp2 = maybe_detok_latin(hyp2)

        # Per-language metrics
        bleu_tok_1 = _bleu_tokenizer_for_lid(tgt_lid_global)
        bleu_tok_2 = _bleu_tokenizer_for_lid(src_lid)
        metrics_1 = score_all(hyp1, ref1, bleu_tok=bleu_tok_1, lowercase=args.bleu_lowercase)
        metrics_2 = score_all(hyp2, ref2, bleu_tok=bleu_tok_2, lowercase=args.bleu_lowercase)

        # Examples (use *pre*-normalized src/tgt + raw sys_out for readability)
        ex_1 = list(zip(src_texts, tgt_texts, sys_out_1))
        ex_2 = list(zip(tgt_texts, src_texts, sys_out_2))
        pretty_print(f"{lc} → {tgt_label}", metrics_1, n, ex_1)
        pretty_print(f"{tgt_label} → {lc}", metrics_2, n, ex_2)

        metrics_by_lang[lc] = {
            f"src2{tgt_label}": metrics_1,
            f"{tgt_label}2src": metrics_2,
        }
        counts_by_lang[lc] = n

        # Extend global pools (after scoring-prep normalization/detok)
        global_ref_src2tgt.extend(ref1)
        global_hyp_src2tgt.extend(hyp1)
        global_ref_tgt2src.extend(ref2)
        global_hyp_tgt2src.extend(hyp2)

        # CSV rows (one row per example)
        if args.csv_out:
            for s_f, s_tgt, h_tgt, h_f in zip(src_texts, tgt_texts, sys_out_1, sys_out_2):
                csv_records.append({
                    "lang_code": lc,
                    "src_formosan": s_f,
                    f"ref_{tgt_label}": s_tgt,
                    f"hyp_{tgt_label}": h_tgt,
                    "hyp_formosan": h_f,
                })

    # Global micro-averaged metrics over all sentences
    if global_ref_src2tgt:
        bleu_tok_global_src2tgt = _bleu_tokenizer_for_lid(tgt_lid_global)
        bleu_tok_global_tgt2src = "13a"  # all Formosan targets are Latin
        global_metrics_src2tgt = score_all(
            global_hyp_src2tgt,
            global_ref_src2tgt,
            bleu_tok=bleu_tok_global_src2tgt,
            lowercase=args.bleu_lowercase,
        )
        global_metrics_tgt2src = score_all(
            global_hyp_tgt2src,
            global_ref_tgt2src,
            bleu_tok=bleu_tok_global_tgt2src,
            lowercase=args.bleu_lowercase,
        )

        print("\n================ Global (all languages combined) ================")
        print(f"Total sentences (Formosan→{tgt_label}): {len(global_ref_src2tgt)}")
        print(f"BLEU:  {global_metrics_src2tgt['BLEU']:.2f}")
        print(f"chrF2: {global_metrics_src2tgt['chrF2']:.2f}")
        print(f"TER:   {global_metrics_src2tgt['TER']:.2f}")

        print(f"\nTotal sentences ({tgt_label}→Formosan): {len(global_ref_tgt2src)}")
        print(f"BLEU:  {global_metrics_tgt2src['BLEU']:.2f}")
        print(f"chrF2: {global_metrics_tgt2src['chrF2']:.2f}")
        print(f"TER:   {global_metrics_tgt2src['TER']:.2f}")
        print("=================================================================\n")
    else:
        global_metrics_src2tgt = {}
        global_metrics_tgt2src = {}

    # Save JSON (per-language + global)
    if args.save_json:
        out = {
            "mode": "multilingual",
            "target_language": tgt_lang,
            "target_label": tgt_label,
            "n_examples_total": int(len(global_ref_src2tgt)),
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
                "normalize": args.normalize,
            },
            "global_metrics": {
                f"Formosan->{tgt_label}": global_metrics_src2tgt,
                f"{tgt_label}->Formosan": global_metrics_tgt2src,
            },
            "per_language": {
                lc: {
                    "n_examples": int(counts_by_lang.get(lc, 0)),
                    "src_lid": lid_from_code(lc),
                    "tgt_lid": tgt_lid_global,
                    "metrics": metrics_by_lang[lc],
                }
                for lc in metrics_by_lang
            },
        }
        Path(os.path.dirname(args.save_json) or ".").mkdir(parents=True, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Saved multilingual results JSON to: {args.save_json}")

    # Save CSV (all langs)
    if args.csv_out and csv_records:
        df_out = pd.DataFrame(csv_records)
        Path(os.path.dirname(args.csv_out) or ".").mkdir(parents=True, exist_ok=True)
        df_out.to_csv(args.csv_out, index=False)
        print(f"💾 Saved multilingual predictions CSV to: {args.csv_out}")


# ------------------------------- main --------------------------------------
def main():
    ap = argparse.ArgumentParser()

    # Mode switch
    ap.add_argument(
        "--multilingual",
        action="store_true",
        help=(
            "If set, interpret CSV as multilingual corpus with "
            "lang_code, formosan_sentence, <target>_sentence, split; "
            "<target> is chosen via --tgt-lang {chinese,english}."
        ),
    )

    # Multilingual target language (for corpus column choice + LID)
    ap.add_argument(
        "--tgt-lang",
        default="chinese",
        help="(Multilingual) target language: 'chinese' (uses chinese_sentence) or 'english' (uses english_sentence).",
    )

    # Model paths
    ap.add_argument("--tokenizer", required=True, help="Path to tokenizer dir")
    ap.add_argument("--model", required=True, help="Path to model dir")
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="CSV with either single-pair (old) or multilingual (new) format",
    )

    # Single-pair-only options (ignored in multilingual mode)
    ap.add_argument(
        "--src-col",
        default=None,
        help="(Single-pair mode) Name of source text column (e.g. ami)",
    )
    ap.add_argument(
        "--tgt-col",
        default=None,
        help="(Single-pair mode) Name of target text column (e.g. en)",
    )

    # Optional explicit LIDs override for single-pair (multilingual uses CODE_TO_LID)
    ap.add_argument(
        "--src-lid",
        default=None,
        help="(Single-pair) Override NLLB LID, e.g. ami_Latn",
    )
    ap.add_argument(
        "--tgt-lid",
        default=None,
        help="Override NLLB LID, e.g. zho_Hans (used globally in multilingual mode)",
    )

    # Limit
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Single-pair: limit total test examples. "
            "Multilingual: limit per language."
        ),
    )

    # Generation / batching
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256, help="Encoder truncation length")
    ap.add_argument("--beam", type=int, default=4)

    # Modern generation knobs
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--min-new-tokens", type=int, default=1)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=0)
    ap.add_argument("--repetition-penalty", type=float, default=1.0)
    ap.add_argument("--length-penalty", type=float, default=1.0)

    # Scoring options
    ap.add_argument(
        "--bleu-lowercase",
        action="store_true",
        help="Compute BLEU in lowercase (for noisy casing)",
    )
    ap.add_argument(
        "--detok-latin",
        action="store_true",
        help="Detokenize Latin-script refs/hyps before scoring (when CSV looks Moses-tokenized)",
    )

    # Device
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    # Normalization toggle
    ap.add_argument("--normalize", dest="normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)

    # Outputs
    ap.add_argument("--save-json", default=None, help="Path to save metrics+samples JSON")
    ap.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV with src/ref/hyp. Single-pair: one pair. Multilingual: all languages.",
    )

    args = ap.parse_args()

    # Device
    dev = (
        "cuda"
        if (args.device == "auto" and torch.cuda.is_available())
        else (args.device if args.device != "auto" else "cpu")
    )
    device = torch.device(dev)
    print(f"Device: {device}")

    # Load model/tokenizer
    tokenizer, model = load_tok_model(args.tokenizer, args.model, device)

    # ----------------- Multilingual mode (new) -----------------
    if args.multilingual:
        eval_multilingual(args, tokenizer, model, device)
        return

    # ----------------- Single-pair mode (original behavior) ----
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

    # ---- Generation settings ----
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
    }

    # Single-pair generation both directions
    sys_out_1 = batched_generate(
        tokenizer,
        model,
        src_texts,
        src_lid,
        tgt_lid,
        device,
        enc_max_length=args.max_length,
        num_beams=args.beam,
        batch_size=args.batch_size,
        gen_kwargs=gen_kwargs,
        desc=f"{src_col}->{tgt_col}",
    )
    sys_out_2 = batched_generate(
        tokenizer,
        model,
        tgt_texts,
        tgt_lid,
        src_lid,
        device,
        enc_max_length=args.max_length,
        num_beams=args.beam,
        batch_size=args.batch_size,
        gen_kwargs=gen_kwargs,
        desc=f"{tgt_col}->{src_col}",
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

    # Scoring
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
            "mode": "single_pair",
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
                "normalize": args.normalize,
            },
            "metrics": {
                f"{src_col}->{tgt_col}": metrics_1,
                f"{tgt_col}->{src_col}": metrics_2,
            },
            "examples": {
                f"{src_col}->{tgt_col}": ex_1[:10],
                f"{tgt_col}->{src_col}": ex_2[:10],
            },
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
