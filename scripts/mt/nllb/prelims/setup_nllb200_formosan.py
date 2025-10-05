#!/usr/bin/env python3
"""
setup_nllb200_formosan.py

Robust NLLB-200-distilled-600M tokenizer/model setup for Formosan languages + zho_Hant (+ optional en)
- Rebuilds the tokenizer so it contains:
    • ALL stock NLLB language codes, plus your Formosan codes (e.g., ami_Latn)
    • '<mask>' kept as the last special token (NLLB quirk)
- Two paths for <unk> mitigation:
    (A) --add-mode chars  : add frequent unknown characters as added tokens
    (B) --add-mode spm    : train a small SPM on your corpus and surgically merge into NLLB SPM  ← default
- Resizes model embeddings exactly once (after *all* tokenizer changes)
- Warm-starts new embedding rows by averaging old-piece embeddings (fallback to <unk>)
  and seeds new *_Latn language-code tokens from eng_Latn
- Saves tokenizer/model + a JSON report
- Smoke-test generation for each Formosan LID:
    * Formosan -> Traditional Chinese (zho_Hant)
    * Traditional Chinese -> Formosan
    * (optional) + English if --also-eng is given

Usage (examples)
---------------
# Fast path: add frequent unknown characters only (no SPM surgery)
python setup_nllb200_formosan.py \
  --input big_corpus_combined.csv \
  --output-prefix formosan_multilingual_nllb \
  --add-mode chars --min-char-frequency 3 \
  --run-eval --samples-per-lang 1

# SPM path (recommended for zh_Hant): retrain/merge sentencepiece for cleaner segmentation
python setup_nllb200_formosan.py \
  --input big_corpus_combined.csv \
  --output-prefix formosan_multilingual_nllb \
  --add-mode spm --spm-vocab 16384 --min-char-frequency 3 \
  --run-eval --samples-per-lang 1

CSV columns expected:
- REQUIRED: lang_code, formosan_sentence, chinese_sentence
- OPTIONAL: english_sentence (included in SPM/char scans if present)

Notes:
- Use tokenizer.src_lang / forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang) for generation.  # HF NLLB docs
"""

from __future__ import annotations
import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Set, Iterable, Optional, List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm, trange

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    NllbTokenizer,
)

# Optional (only if --add-mode spm)
try:
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
    HAVE_SPM = True
except Exception:
    HAVE_SPM = False

# Base language codes from Transformers (list of stock FAIRSEQ/NLLB codes).
try:
    from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
except Exception:
    FAIRSEQ_LANGUAGE_CODES = None  # fallback handled below


# ───────────────────────────── config: language maps ──────────────────────────
# Map your short tags to NLLB-style language codes. Most Formosan orthographies are Latin.
FORMOSAN_LANGUAGE_MAP: Dict[str, str] = {
    "ami": "ami_Latn", "bnn": "bnn_Latn", "ckv": "ckv_Latn", "dru": "dru_Latn",
    "pwn": "pwn_Latn", "pyu": "pyu_Latn", "ssf": "ssf_Latn", "sxr": "sxr_Latn",
    "szy": "szy_Latn", "tao": "tao_Latn", "tay": "tay_Latn", "trv": "trv_Latn",
    "tsu": "tsu_Latn", "xnb": "xnb_Latn", "xsy": "xsy_Latn",
}
FORMOSAN_LANGS = set(FORMOSAN_LANGUAGE_MAP.keys())

TARGET_LANGUAGE_MAP: Dict[str, str] = {
    "chinese_traditional": "zho_Hant",
    "english": "eng_Latn",
}

BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"


# ────────────────────────────── utilities ─────────────────────────────────────
def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    # Moses-like core: NFKC; (punct/control handled downstream if you later add MosesPunctNormalizer)
    return unicodedata.normalize("NFKC", s)


def load_corpus(input_path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    df = pd.read_csv(input_path)
    required = {"lang_code", "formosan_sentence", "chinese_sentence"}
    if not required <= set(df.columns):
        sys.exit("❌ CSV must have columns: lang_code, formosan_sentence, chinese_sentence (optional: english_sentence)")
    uniq = set(df["lang_code"].dropna().unique().tolist())
    supported = uniq & FORMOSAN_LANGS
    missing = uniq - FORMOSAN_LANGS
    if missing:
        print(f"⚠️  Skipping unsupported lang_code(s): {sorted(missing)}")
    print(f"✅ Loaded {len(df):,} rows; processing {len(supported)} Formosan langs")
    return df, supported


def get_device(arg_choice: str) -> str:
    if arg_choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg_choice


def iter_unique_chars(texts: Iterable[str]) -> Set[str]:
    charset = set()
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        t = clean_text(t)
        charset.update(list(t))
    # drop pure whitespace & C0 controls
    return {c for c in charset if not c.isspace() and not (0 <= ord(c) < 32)}


def find_unknown_chars(tokenizer, chars: Set[str]) -> Set[str]:
    """
    Test each character as a standalone tokenization unit (no specials).
    If any subpiece maps to <unk>, we consider that char unknown.
    """
    unk = tokenizer.unk_token_id
    unknown = set()
    for c in chars:
        ids = tokenizer(c, add_special_tokens=False).input_ids
        if len(ids) == 0 or any(i == unk for i in ids):
            unknown.add(c)
    return unknown


def count_char_frequency(texts: Iterable[str], target_chars: Set[str]) -> Dict[str, int]:
    cnt = Counter()
    tgt = target_chars
    for t in texts:
        if not isinstance(t, str):
            continue
        t = clean_text(t)
        for ch in t:
            if ch in tgt:
                cnt[ch] += 1
    return dict(cnt)


# ─────────────────────────── tokenizer surgery (≥4.38) ────────────────────────
def _safe_reload_with_langs_and_optional_spm(base_tokenizer, new_spm_path, extra_lang_codes):
    import os, shutil, tempfile
    tmp = tempfile.mkdtemp(prefix="nllb_tok_")
    base_tokenizer.save_pretrained(tmp)

    if new_spm_path is not None:
        shutil.copy(new_spm_path, os.path.join(tmp, "sentencepiece.bpe.model"))

    tok = NllbTokenizer.from_pretrained(tmp, use_fast=False)

    # APPEND truly new codes; DO NOT replace the existing special list
    existing = set(tok.additional_special_tokens or [])
    missing = [c for c in extra_lang_codes if c not in existing]
    if missing:
        tok.add_special_tokens(
            {"additional_special_tokens": missing},
            replace_additional_special_tokens=False,   # ← THIS is the key
        )

    # (optional) keep "<mask>" last
    m = tok.mask_token
    if m and tok.additional_special_tokens and tok.additional_special_tokens[-1] != m:
        lst = [t for t in tok.additional_special_tokens if t != m] + [m]
        tok.add_special_tokens({"additional_special_tokens": lst}, replace_additional_special_tokens=True)

    shutil.rmtree(tmp, ignore_errors=True)
    return tok




def _merge_spm_models(nllb_tokenizer, merged_spm_out: str, corpus_txt_path: str,
                      spm_vocab: int, required_chars: str):
    """
    Train a small SPM on your corpus, then append any missing normal pieces to NLLB SPM.
    Saves combined SPM to merged_spm_out.
    """
    if not HAVE_SPM:
        sys.exit("❌ sentencepiece not installed; pip install sentencepiece")

    # Train small SPM
    spm_prefix = Path(merged_spm_out).with_suffix("").as_posix() + "_tmp"
    spm.SentencePieceTrainer.train(
        input=corpus_txt_path,
        model_prefix=spm_prefix,
        vocab_size=int(spm_vocab),
        character_coverage=1.0,
        num_threads=max(1, os.cpu_count() or 1),
        train_extremely_large_corpus=False,
        add_dummy_prefix=False,
        max_sentencepiece_length=128,
        max_sentence_length=4192 * 4,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        required_chars=required_chars,
    )

    # Parse both models
    sp_trained = spm.SentencePieceProcessor(model_file=f"{spm_prefix}.model")
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())

    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(nllb_tokenizer.sp_model.serialized_model_proto())

    nllb_tokens_set = {p.piece for p in base_spm.pieces}
    prev_min_score = base_spm.pieces[-1].score

    # Only copy NORMAL (type == 1) pieces
    added = 0
    for p in added_spm.pieces:
        if getattr(p, "type", 1) != 1:
            continue
        piece = p.piece
        if piece not in nllb_tokens_set:
            new_p = sp_pb2_model.ModelProto().SentencePiece()
            new_p.piece = piece
            new_p.score = p.score + prev_min_score
            base_spm.pieces.append(new_p)
            added += 1

    with open(merged_spm_out, "wb") as f:
        f.write(base_spm.SerializeToString())

    # Cleanup temp SPM artifacts
    for ext in (".model", ".vocab"):
        try:
            os.remove(f"{spm_prefix}{ext}")
        except OSError:
            pass

    print(f"🧩 SPM merged; added {added} pieces -> {merged_spm_out}")
    return added


def _seed_new_langcode_rows(model, tokenizer, formosan_langcodes: List[str]):
    """
    For newly added language-code tokens (e.g., ami_Latn), initialize their embeddings
    from a 'similar' built-in language code. For Latin-script Formosan, use eng_Latn.
    """
    if not formosan_langcodes:
        return 0
    emb = model.model.shared.weight.data
    try:
        src_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    except Exception:
        return 0
    if src_id is None or src_id < 0:
        return 0

    seeded = 0
    for lid in formosan_langcodes:
        # Only affect *_Latn codes (safe default)
        if not lid.endswith("_Latn"):
            continue
        tid = tokenizer.convert_tokens_to_ids(lid)
        if tid is None or tid < 0:
            continue
        # If row already has non-zero norm (e.g., copied/avg), leave it alone
        if torch.norm(emb[tid]).item() == 0.0:
            emb[tid] = emb[src_id]
            seeded += 1
    return seeded


def _warm_start_new_rows(model, tokenizer_old, tokenizer_new):
    """
    Warm-start any *new* embedding rows:
      1) If the token existed in the old tokenizer vocab, copy that exact row.
      2) Else try decomposing with the old tokenizer and average piece embeddings.
      3) Else fall back to <unk>.
    """
    vocab_old = tokenizer_old.get_vocab()        # str -> id
    vocab_new = tokenizer_new.get_vocab()

    new_tokens = sorted(set(vocab_new.keys()) - set(vocab_old.keys()))
    if not new_tokens:
        return 0

    emb = model.model.shared.weight.data
    unk_old = tokenizer_old.unk_token_id

    for tok in new_tokens:
        new_id = tokenizer_new.convert_tokens_to_ids(tok)
        if tok in vocab_old:
            emb[new_id] = emb[vocab_old[tok]]
            continue

        ids_old = tokenizer_old(tok, add_special_tokens=False).input_ids
        if not ids_old:
            ids_old = [unk_old]
        emb[new_id] = emb[ids_old].mean(0)

    # lm_head is usually tied; if not, expand it
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and lm_head.weight.data.shape[0] < emb.shape[0]:
        old_rows, dim = lm_head.weight.data.shape
        extra = emb[old_rows:]
        lm_head.weight.data = torch.cat([lm_head.weight.data, extra.clone()], dim=0)

    return len(new_tokens)


# ───────────────────────────── core setup flow ────────────────────────────────
def setup_tokenizer_and_model(add_mode: str,
                              supported_langs: Set[str],
                              df: pd.DataFrame,
                              output_prefix: str,
                              spm_vocab: int,
                              min_char_freq: int,
                              device: str,
                              also_eng: bool) -> Tuple:
    print("🔧 Loading base NLLB tokenizer/model...")
    tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

    extra_lang_codes = [FORMOSAN_LANGUAGE_MAP[k] for k in sorted(supported_langs)]

    # 1) Optional SPM surgery path
    new_spm_path = None
    added_spm_pieces = 0
    if add_mode == "spm":
        # Dump cleaned corpus text (Formosan + zh + optional eng)
        corpus_txt = Path(f"{output_prefix}_alltext_cleaned.txt")
        with open(corpus_txt, "w", encoding="utf-8") as f:
            for col in ["formosan_sentence", "chinese_sentence", "english_sentence"]:
                if col in df.columns:
                    for t in df[col].dropna().astype(str).tolist():
                        t = clean_text(t)
                        if t.strip():
                            f.write(t + "\n")

        # Required chars: char freq ≥ min_char_freq (skip spaces)
        chars_cnt = Counter()
        with open(corpus_txt, "r", encoding="utf-8") as f:
            for line in f:
                for ch in line.rstrip("\n"):
                    if ch != " ":
                        chars_cnt[ch] += 1
        required_chars = "".join([c for c, n in chars_cnt.items() if n >= min_char_freq])

        new_spm_path = f"{output_prefix}_merged_spm.model"
        added_spm_pieces = _merge_spm_models(tokenizer_base, new_spm_path, str(corpus_txt),
                                             spm_vocab, required_chars)

    # 2) Rebuild tokenizer cleanly with new langs (+ optional new SPM)
    tokenizer = _safe_reload_with_langs_and_optional_spm(tokenizer_base, new_spm_path, extra_lang_codes)

    # Create convenience maps for all codes (built-ins + Formosan)
    if not hasattr(tokenizer, "lang_code_to_id") or not tokenizer.lang_code_to_id:
        tokenizer.lang_code_to_id = {}
        tokenizer.id_to_lang_code = {}

    all_langs = set(FAIRSEQ_LANGUAGE_CODES or []) | set(extra_lang_codes)
    for lid in sorted(all_langs):
        tid = tokenizer.convert_tokens_to_ids(lid)
        if tid == tokenizer.unk_token_id:
            continue
        tokenizer.lang_code_to_id[lid] = tid
        tokenizer.id_to_lang_code[tid] = lid

    # 3) Unknown char path (chars mode): add frequent unknown characters as added tokens
    added_char_tokens = 0
    if add_mode == "chars":
        print("🔎 Scanning corpus for unknown characters...")
        texts = []
        for col in ["formosan_sentence", "chinese_sentence", "english_sentence"]:
            if col in df.columns:
                texts.extend(df[col].dropna().astype(str).tolist())
        uniq = iter_unique_chars(texts)
        unknown = find_unknown_chars(tokenizer, uniq)
        print(f"📊 Unique <unk> chars (pre-filter): {len(unknown):,}")
        if unknown:
            freq = count_char_frequency(texts, unknown)
            candidates = sorted([c for c, n in freq.items() if n >= min_char_freq],
                                key=lambda c: freq[c], reverse=True)
            if candidates:
                print(f"➕ Adding {len(candidates)} frequent unknown chars (min_freq={min_char_freq}) as added tokens")
                added_char_tokens = tokenizer.add_tokens(candidates)
            else:
                print(f"ℹ️  No unknown chars meet min_freq ≥ {min_char_freq}. Skipping char additions.")
        else:
            print("✅ No unknown characters found.")

    # 4) Single embedding resize after *all* tokenizer changes
    total_new = added_char_tokens > 0 or added_spm_pieces > 0 or len(extra_lang_codes) > 0
    if total_new:
        print(f"📏 Resizing embeddings to len(tokenizer) = {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # Warm-start any *new* rows
        warmed = _warm_start_new_rows(model, tokenizer_base, tokenizer)
        if warmed:
            print(f"🔥 Warm-started {warmed} new embedding rows")

        # Seed *_Latn language codes from eng_Latn for stability
        seeded = _seed_new_langcode_rows(model, tokenizer, extra_lang_codes)
        if seeded:
            print(f"🌱 Seeded {seeded} Formosan *_Latn language-code embeddings from eng_Latn")

    # Move to device
    model = model.to(torch.device(device))
    print(f"📱 Model on: {device}")

    return tokenizer, model, extra_lang_codes, added_char_tokens, added_spm_pieces


# ─────────────────────────── generation & eval harness ────────────────────────
@torch.inference_mode()
def _gen_once(tokenizer, model, device, text: str, src_lid: str, tgt_lid: str,
              num_beams: int, max_new_tokens: int) -> str:
    tokenizer.src_lang = src_lid  # HF docs: set src_lang on the tokenizer
    enc = tokenizer(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lid),
        decoder_start_token_id=tokenizer.convert_tokens_to_ids(tgt_lid),  # add this
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def _pick_example(df: pd.DataFrame, lang: str, col: str) -> str:
    rows = df[df["lang_code"] == lang][col].astype(str)
    for x in rows.tolist():
        x = clean_text(x or "")
        if x.strip():
            return x
    return "kiso test" if col == "formosan_sentence" else "你好"


def run_smoke_tests(df: pd.DataFrame,
                    tokenizer,
                    model,
                    supported_langs: Set[str],
                    device: str,
                    samples_per_lang: int,
                    num_beams: int,
                    max_new_tokens: int,
                    also_eng: bool,
                    save_jsonl: Optional[str] = None) -> None:
    model.eval()
    results: List[dict] = []
    zh = TARGET_LANGUAGE_MAP["chinese_traditional"]
    en = TARGET_LANGUAGE_MAP["english"]

    print("\n🧪 Smoke tests: Formosan <-> zho_Hant" + (" (+ eng_Latn)" if also_eng else ""))
    for lang in sorted(supported_langs):
        lid = FORMOSAN_LANGUAGE_MAP[lang]

        # A) Formosan -> Chinese (and optional English)
        for _ in range(samples_per_lang):
            src_f = _pick_example(df, lang, "formosan_sentence")
            try:
                hyp = _gen_once(tokenizer, model, device, src_f, lid, zh, num_beams, max_new_tokens)
                print(f"✅ {lang}→zho_Hant | src[:30]={src_f[:30]!r} | hyp[:60]={hyp[:60]!r}")
                results.append({"dir": f"{lang}->zho_Hant", "src": src_f, "hyp": hyp, "ok": True})
            except Exception as e:
                print(f"❌ {lang}→zho_Hant | {e}")
                results.append({"dir": f"{lang}->zho_Hant", "src": src_f, "hyp": "", "ok": False, "err": str(e)})

            if also_eng:
                try:
                    hyp = _gen_once(tokenizer, model, device, src_f, lid, en, num_beams, max_new_tokens)
                    print(f"✅ {lang}→eng_Latn | src[:30]={src_f[:30]!r} | hyp[:60]={hyp[:60]!r}")
                    results.append({"dir": f"{lang}->eng_Latn", "src": src_f, "hyp": hyp, "ok": True})
                except Exception as e:
                    print(f"❌ {lang}→eng_Latn | {e}")
                    results.append({"dir": f"{lang}->eng_Latn", "src": src_f, "hyp": "", "ok": False, "err": str(e)})

        # B) Chinese -> Formosan
        for _ in range(samples_per_lang):
            src_c = _pick_example(df, lang, "chinese_sentence")
            try:
                hyp = _gen_once(tokenizer, model, device, src_c, zh, lid, num_beams, max_new_tokens)
                print(f"✅ zho_Hant→{lang} | src[:30]={src_c[:30]!r} | hyp[:60]={hyp[:60]!r}")
                results.append({"dir": f"zho_Hant->{lang}", "src": src_c, "hyp": hyp, "ok": True})
            except Exception as e:
                print(f"❌ zho_Hant→{lang} | {e}")
                results.append({"dir": f"zho_Hant->{lang}", "src": src_c, "hyp": "", "ok": False, "err": str(e)})

    if save_jsonl:
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"🧾 Saved eval report -> {save_jsonl}")


# ────────────────────────────── save report ───────────────────────────────────
def save_everything(tokenizer,
                    model,
                    supported_langs: Set[str],
                    output_prefix: str,
                    added_char_tokens: int,
                    added_spm_pieces: int):
    tok_dir = f"{output_prefix}_tokenizer"
    mdl_dir = f"{output_prefix}_model"
    print("💾 Saving tokenizer/model...")
    tokenizer.save_pretrained(tok_dir)
    model.save_pretrained(mdl_dir)

    lang_info = {
        "supported_formosan_languages": sorted(list(supported_langs)),
        "language_tokens": {k: FORMOSAN_LANGUAGE_MAP[k] for k in supported_langs},
        "lang_code_to_id": getattr(tokenizer, "lang_code_to_id", {}),
        "vocab_size_reported": getattr(tokenizer, "vocab_size", None),
        "len_tokenizer": len(tokenizer),
        "added_char_tokens": int(added_char_tokens),
        "added_spm_pieces": int(added_spm_pieces),
        "base_model": BASE_MODEL_NAME,
    }
    with open(f"{output_prefix}_language_info.json", "w", encoding="utf-8") as f:
        json.dump(lang_info, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Tokenizer -> {tok_dir}")
    print(f"   ✅ Model     -> {mdl_dir}")
    print(f"   📝 Lang info -> {output_prefix}_language_info.json")


# ────────────────────────────── main ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with columns: lang_code, formosan_sentence, chinese_sentence (optional: english_sentence)")
    ap.add_argument("--output-prefix", default="formosan_multilingual_nllb")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--add-mode", default="spm", choices=["none", "chars", "spm"],
                    help="Fix unknowns by adding frequent chars, or merging a new SPM (default: spm for zh_Hant)")
    ap.add_argument("--min-char-frequency", type=int, default=3,
                    help="Min frequency to add an unknown char (chars mode) or to require in SPM")
    ap.add_argument("--spm-vocab", type=int, default=16384,
                    help="Vocab size for the auxiliary SPM (spm mode)")
    # eval harness flags
    ap.add_argument("--run-eval", action="store_true", help="Run the generation smoke tests.")
    ap.add_argument("--samples-per-lang", type=int, default=1)
    ap.add_argument("--num-beams", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--save-eval-json", type=str, default=None)
    ap.add_argument("--also-eng", action="store_true", help="Also test → eng_Latn from Formosan")
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"❌ Input file not found: {args.input}")

    device = get_device(args.device)
    print(f"🚀 NLLB-200 setup\n📁 Input: {args.input}\n📱 Device: {device}\n📂 Out: {args.output_prefix}\n🧩 Mode: {args.add_mode}")

    df, supported_langs = load_corpus(args.input)

    tokenizer, model, extra_lang_codes, added_char_tokens, added_spm_pieces = setup_tokenizer_and_model(
        add_mode=args.add_mode,
        supported_langs=supported_langs,
        df=df,
        output_prefix=args.output_prefix,
        spm_vocab=args.spm_vocab,
        min_char_freq=args.min_char_frequency,
        device=device,
        also_eng=args.also_eng,
    )

    save_everything(tokenizer, model, supported_langs, args.output_prefix, added_char_tokens, added_spm_pieces)

    if args.run_eval:
        run_smoke_tests(
            df=df,
            tokenizer=tokenizer,
            model=model,
            supported_langs=supported_langs,
            device=device,
            samples_per_lang=args.samples_per_lang,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            also_eng=args.also_eng,
            save_jsonl=args.save_eval_json,
        )

    print("\n🎉 Done. Ready for fine-tuning.\nℹ️  For training, set tokenizer.src_lang per batch and use "
          "`forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang)` in generate().")


if __name__ == "__main__":
    main()
