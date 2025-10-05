#!/usr/bin/env python3
"""
Robust mBART-50 tokenizer/model setup for Formosan languages + Traditional Chinese
- Adds custom language codes (e.g., ami_XX) as special tokens and wires them into lang_code_to_id
- Detects unknown characters (after cleaning) efficiently and adds them as tokens
- Resizes model embeddings exactly once via len(tokenizer) (HF-recommended)
- Saves tokenizer/model + a JSON with language-code -> id mapping
- **New**: Smoke-test harness that tries generation for every Formosan LID:
    * Formosan -> Chinese (zh_CN)
    * Chinese (zh_CN) -> Formosan

Usage:
  python setup_mbart50_formosan.py --input big_corpus_combined.csv \
    --min-char-frequency 3 --output-prefix formosan_multilingual_mbart \
    --run-eval --samples-per-lang 1 --save-eval-json eval_results.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple, Set, Iterable, Optional, List

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import MBart50Tokenizer, MBartForConditionalGeneration

# ───────────────────────────── config: language maps ──────────────────────────
FORMOSAN_LANGUAGE_MAP: Dict[str, str] = {
    "ami": "ami_XX", "bnn": "bnn_XX", "ckv": "ckv_XX", "dru": "dru_XX",
    "pwn": "pwn_XX", "pyu": "pyu_XX", "ssf": "ssf_XX", "sxr": "sxr_XX",
    "szy": "szy_XX", "tao": "tao_XX", "tay": "tay_XX", "trv": "trv_XX",
    "tsu": "tsu_XX", "xnb": "xnb_XX", "xsy": "xsy_XX",
}
FORMOSAN_LANGS = set(FORMOSAN_LANGUAGE_MAP.keys())

# mBART-50 languages we’ll translate with (already present)
TARGET_LANGUAGE_MAP: Dict[str, str] = {
    "english": "en_XX",
    "chinese_simplified": "zh_CN",  # mBART-50 ships zh_CN only
}

# ────────────────────────────── utilities ─────────────────────────────────────
def load_corpus(input_path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    df = pd.read_csv(input_path)
    required = {"lang_code", "formosan_sentence", "chinese_sentence"}
    if not required <= set(df.columns):
        sys.exit("❌ CSV must have columns: lang_code, formosan_sentence, chinese_sentence")
    uniq = set(df["lang_code"].dropna().unique().tolist())
    supported = uniq & FORMOSAN_LANGS
    if missing := (uniq - FORMOSAN_LANGS):
        print(f"⚠️  Skipping unsupported language codes: {sorted(missing)}")
    print(f"✅  Loaded {len(df):,} rows; processing {len(supported)} Formosan langs")
    return df, supported


def setup_tokenizer_model(supported_langs: Set[str], device: str):
    print("🔧 Loading mBART-50 tokenizer/model...")
    tok = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

    # Add all custom language codes as special tokens (one shot)
    new_lang_tokens = []
    full_vocab = tok.get_vocab()
    for lc in sorted(supported_langs):
        t = FORMOSAN_LANGUAGE_MAP[lc]
        if t not in full_vocab:
            new_lang_tokens.append(t)

    if new_lang_tokens:
        print(f"➕ Adding {len(new_lang_tokens)} language codes: {', '.join(new_lang_tokens)}")
        tok.add_special_tokens({"additional_special_tokens": new_lang_tokens})
        # Note: add_special_tokens() automatically handles never_split behavior
        # resize with the correct size
        model.resize_token_embeddings(len(tok))

        # wire new language codes into tokenizer mappings so set_src works & ids resolve
        if not hasattr(tok, "lang_code_to_id"):
            tok.lang_code_to_id = {}
        if not hasattr(tok, "id_to_lang_code"):
            tok.id_to_lang_code = {}

        for t in new_lang_tokens:
            tid = tok.convert_tokens_to_ids(t)
            tok.lang_code_to_id[t] = tid
            tok.id_to_lang_code[tid] = t
            print(f"   ✅ {t} -> id {tid}")
    else:
        print("✅ No new language codes needed (already present)")

    # Load to device
    model = model.to(torch.device(device))
    print(f"📱 Model on: {device}")
    return tok, model


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s)


def iter_unique_chars(texts: Iterable[str]) -> Set[str]:
    charset = set()
    for t in texts:
        if not isinstance(t, str) or not t:
            continue
        t = clean_text(t)
        charset.update(list(t))
    # filter trivial ASCII control/space
    return {c for c in charset if not c.isspace() and not (0 <= ord(c) < 32)}


def find_unknown_chars(tok: MBart50Tokenizer, chars: Set[str]) -> Dict[str, int]:
    """
    Find chars that become <unk> when encoded (test each char once, no special tokens).
    """
    unk = tok.unk_token_id
    oov_counts: Dict[str, int] = {}
    for c in chars:
        ids = tok(c, add_special_tokens=False).input_ids
        if len(ids) == 0 or any(i == unk for i in ids):
            oov_counts[c] = 0
    return oov_counts


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


def add_unknown_char_tokens(tok: MBart50Tokenizer,
                            model: MBartForConditionalGeneration,
                            unk_chars: Dict[str, int],
                            min_freq: int) -> int:
    if not unk_chars:
        print("✅ No <unk> characters found.")
        return 0
    filtered = [c for c, n in unk_chars.items() if n >= min_freq]
    if not filtered:
        print(f"ℹ️  <unk> chars exist but none meet min frequency ≥ {min_freq}. Skipping.")
        return 0

    filtered.sort(key=lambda c: unk_chars[c], reverse=True)
    print(f"➕ Adding {len(filtered)} frequent unknown chars (min_freq={min_freq})")
    added = tok.add_tokens(filtered)
    print(f"   ✅ tokenizer.add_tokens added: {added}")
    model.resize_token_embeddings(len(tok))
    print(f"📏 Embedding matrix resized to len(tokenizer) = {len(tok)}")
    return added


def save_everything(tok: MBart50Tokenizer,
                    model: MBartForConditionalGeneration,
                    supported_langs: Set[str],
                    output_prefix: str):
    tok_dir = f"{output_prefix}_tokenizer"
    mdl_dir = f"{output_prefix}_model"
    print("💾 Saving tokenizer/model...")
    tok.save_pretrained(tok_dir)
    model.save_pretrained(mdl_dir)

    lang_info = {
        "supported_formosan_languages": sorted(list(supported_langs)),
        "language_tokens": {k: FORMOSAN_LANGUAGE_MAP[k] for k in supported_langs},
        "lang_code_to_id": {k: tok.lang_code_to_id[k] for k in tok.lang_code_to_id},
        "vocab_size_reported": tok.vocab_size,   # base SPM size
        "len_tokenizer": len(tok),               # base + added tokens (what the model sees)
    }
    with open(f"{output_prefix}_language_info.json", "w", encoding="utf-8") as f:
        json.dump(lang_info, f, indent=2, ensure_ascii=False)

    print(f"   ✅ Tokenizer -> {tok_dir}")
    print(f"   ✅ Model     -> {mdl_dir}")
    print(f"   📝 Lang info -> {output_prefix}_language_info.json")


# ───────────────────────────── eval harness ───────────────────────────────────
def _pick_example(df: pd.DataFrame, lang: str, col: str) -> str:
    # pick the first non-empty example for this lang/column
    rows = df[df["lang_code"] == lang]
    for x in rows[col].astype(str).tolist():
        x = clean_text(x)
        if x and x.strip():
            return x
    return "kiso test" if col == "formosan_sentence" else "你好"


@torch.inference_mode()
def _gen_once(tok: MBart50Tokenizer,
              model: MBartForConditionalGeneration,
              device: str,
              text: str,
              src_lid: str,
              tgt_lid: str,
              num_beams: int,
              max_new_tokens: int) -> str:
    # mBART-50: prefix with src lang; force bos = tgt lang
    tok.src_lang = src_lid  # tokenizer will format as [src_lang_code] X [eos]
    enc = tok(text, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        forced_bos_token_id=tok.lang_code_to_id[tgt_lid],
        decoder_start_token_id=tok.eos_token_id,  # safe for MBART-50
    )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def run_smoke_tests(df: pd.DataFrame,
                    tok: MBart50Tokenizer,
                    model: MBartForConditionalGeneration,
                    supported_langs: Set[str],
                    device: str,
                    samples_per_lang: int = 1,
                    num_beams: int = 2,
                    max_new_tokens: int = 24,
                    save_jsonl: Optional[str] = None) -> None:
    model.eval()
    results: List[dict] = []

    zh = TARGET_LANGUAGE_MAP["chinese_simplified"]

    print("\n🧪 Smoke tests: Formosan <-> zh_CN")
    for lang in sorted(supported_langs):
        lid = FORMOSAN_LANGUAGE_MAP[lang]

        # A) Formosan -> Chinese
        for _ in range(samples_per_lang):
            src_f = _pick_example(df, lang, "formosan_sentence")
            try:
                hyp = _gen_once(tok, model, device, src_f, lid, zh, num_beams, max_new_tokens)
                print(f"✅ {lang}→zh_CN | src[:30]={src_f[:30]!r} | hyp[:60]={hyp[:60]!r}")
                results.append({
                    "direction": f"{lang}->zh_CN",
                    "src_lang_code": lang, "src_lid": lid, "tgt_lid": zh,
                    "src": src_f, "hyp": hyp, "ok": True
                })
            except Exception as e:
                print(f"❌ {lang}→zh_CN | {e}")
                results.append({
                    "direction": f"{lang}->zh_CN",
                    "src_lang_code": lang, "src_lid": lid, "tgt_lid": zh,
                    "src": src_f, "hyp": "", "ok": False, "error": str(e)
                })

        # B) Chinese -> Formosan
        for _ in range(samples_per_lang):
            src_c = _pick_example(df, lang, "chinese_sentence")
            try:
                hyp = _gen_once(tok, model, device, src_c, zh, lid, num_beams, max_new_tokens)
                print(f"✅ zh_CN→{lang} | src[:30]={src_c[:30]!r} | hyp[:60]={hyp[:60]!r}")
                results.append({
                    "direction": f"zh_CN->{lang}",
                    "src_lang_code": "zh", "src_lid": zh, "tgt_lid": lid,
                    "src": src_c, "hyp": hyp, "ok": True
                })
            except Exception as e:
                print(f"❌ zh_CN→{lang} | {e}")
                results.append({
                    "direction": f"zh_CN->{lang}",
                    "src_lang_code": "zh", "src_lid": zh, "tgt_lid": lid,
                    "src": src_c, "hyp": "", "ok": False, "error": str(e)
                })

    if save_jsonl:
        with open(save_jsonl, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"🧾 Saved eval report -> {save_jsonl}")


# ────────────────────────────── main ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with columns: lang_code, formosan_sentence, chinese_sentence")
    ap.add_argument("--output-prefix", default="formosan_multilingual_mbart")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--skip-add-unknown-chars", action="store_true",
                    help="Skip adding <unk> chars (not recommended if you need Traditional Chinese).")
    ap.add_argument("--min-char-frequency", type=int, default=3,
                    help="Min frequency to add an unknown char to the vocab.")
    # eval harness flags
    ap.add_argument("--run-eval", action="store_true", help="Run the generation smoke tests.")
    ap.add_argument("--samples-per-lang", type=int, default=1)
    ap.add_argument("--num-beams", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--save-eval-json", type=str, default=None)
    args = ap.parse_args()

    if not args.input.exists():
        sys.exit(f"❌ Input file not found: {args.input}")

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else (
        args.device if args.device != "auto" else "cpu"
    )
    print(f"🚀 mBART-50 setup\n📁 Input: {args.input}\n📱 Device: {device}\n📂 Out: {args.output_prefix}")

    df, supported_langs = load_corpus(args.input)
    tok, model = setup_tokenizer_model(supported_langs, device)

    # 1) Collect unique chars across Formosan + Chinese texts
    formosan_texts = df["formosan_sentence"].dropna().astype(str).tolist()
    chinese_texts   = df["chinese_sentence"].dropna().astype(str).tolist()

    all_unique_chars = iter_unique_chars(formosan_texts + chinese_texts)
    print(f"🔎 Unique chars to test: {len(all_unique_chars):,}")

    # 2) Which of those encode to <unk>?
    unk_candidate = find_unknown_chars(tok, all_unique_chars)
    print(f"📊 Unique <unk> chars (pre-filter): {len(unk_candidate):,}")

    # 3) Count their frequency so we can filter by --min-char-frequency
    if unk_candidate:
        freq_map = count_char_frequency(chinese_texts + formosan_texts, set(unk_candidate.keys()))
        for k in list(unk_candidate.keys()):
            unk_candidate[k] = freq_map.get(k, 0)

        total_instances = sum(unk_candidate.values())
        print(f"🔢 Total <unk> instances in corpus: {total_instances:,}")
        # 4) Add them
        if not args.skip_add_unknown_chars:
            add_unknown_char_tokens(tok, model, unk_candidate, args.min_char_frequency)

    save_everything(tok, model, supported_langs, args.output_prefix)

    # 5) Eval harness
    if args.run_eval:
        run_smoke_tests(
            df=df,
            tok=tok,
            model=model,
            supported_langs=supported_langs,
            device=device,
            samples_per_lang=args.samples_per_lang,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            save_jsonl=args.save_eval_json,
        )

    print("\n🎉 Done. Ready for fine-tuning.")
    print("ℹ️  Use zh_CN for Chinese (mBART-50 ships zh_CN only); adding frequent Trad. chars reduces <unk> churn.")

if __name__ == "__main__":
    main()
