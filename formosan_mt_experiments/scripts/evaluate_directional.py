#!/usr/bin/env python3
"""Directional generation/evaluation with metadata-rich prediction outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

from mt_common import FORMOSAN_CODES, get_lid, read_parallel_csv, source_bucket, with_tagged_columns, write_json
from train_directional_nllb import ensure_control_tags, ensure_lang_token

try:
    from sacrebleu.metrics import BLEU, CHRF, TER
    _SACREBLEU_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
    BLEU = CHRF = TER = None  # type: ignore
    _SACREBLEU_ERROR = exc


def score(sys_out: list[str], refs: list[str], lowercase: bool = False) -> dict:
    if _SACREBLEU_ERROR is not None:
        raise SystemExit(f"sacrebleu is required for evaluation: {_SACREBLEU_ERROR}")
    return {
        "BLEU": float(BLEU(tokenize="13a", effective_order=True, lowercase=lowercase).corpus_score(sys_out, [refs]).score),
        "chrF2": float(CHRF().corpus_score(sys_out, [refs]).score),
        "TER": float(TER().corpus_score(sys_out, [refs]).score),
    }


def length_bin(tokens: int) -> str:
    if tokens <= 3:
        return "001_003"
    if tokens <= 8:
        return "004_008"
    if tokens <= 16:
        return "009_016"
    if tokens <= 32:
        return "017_032"
    return "033_plus"


def word_oov_rates(full_df: pd.DataFrame, eval_df: pd.DataFrame, direction: str) -> pd.Series:
    train = full_df[full_df["split"].astype(str).str.lower().eq("train")]
    col = "formosan_sentence" if direction == "f2en" else "english_sentence"
    vocab_by_lang = {}
    for lang, sub in train.groupby("lang_code"):
        vocab = set()
        for text in sub[col].fillna("").astype(str):
            vocab.update(text.lower().split())
        vocab_by_lang[lang] = vocab

    rates = []
    for _, row in eval_df.iterrows():
        words = str(row[col]).lower().split()
        vocab = vocab_by_lang.get(row["lang_code"], set())
        if not words:
            rates.append(0.0)
        else:
            rates.append(sum(1 for w in words if w not in vocab) / len(words))
    return pd.Series(rates, index=eval_df.index)


def formosan_fragmentation(tokenizer: NllbTokenizer, eval_df: pd.DataFrame, direction: str) -> pd.Series:
    col = "formosan_sentence"
    values = []
    for text in eval_df[col].fillna("").astype(str):
        pieces = 0
        words = 0
        for word in text.split():
            if not word:
                continue
            words += 1
            pieces += len(tokenizer.tokenize(word))
        values.append(float(pieces / max(words, 1)))
    return pd.Series(values, index=eval_df.index)


@torch.no_grad()
def generate(
    tokenizer: NllbTokenizer,
    model,
    texts: list[str],
    src_lid: str,
    tgt_lid: str,
    device: torch.device,
    args,
    desc: str,
) -> list[str]:
    forced_id = ensure_lang_token(tokenizer, tgt_lid)
    order = np.argsort([-len(t) for t in texts])
    restore = np.argsort(order)
    sorted_texts = [texts[i] for i in order]
    outs = []
    pbar = tqdm(total=len(texts), desc=desc, unit="ex", dynamic_ncols=True)
    for start in range(0, len(sorted_texts), args.batch_size):
        batch = sorted_texts[start : start + args.batch_size]
        tokenizer.src_lang = src_lid
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_token_type_ids=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            num_beams=args.beam,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty,
            length_penalty=args.length_penalty,
            forced_bos_token_id=forced_id,
            decoder_start_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        pbar.update(len(batch))
    pbar.close()
    return [outs[i] for i in restore]


def group_scores(preds: pd.DataFrame, group_col: str, lowercase: bool) -> dict:
    if group_col not in preds.columns:
        return {}
    out = {}
    for name, sub in preds.groupby(group_col, dropna=False):
        if len(sub) == 0:
            continue
        out[str(name)] = {"samples": int(len(sub))} | score(sub["hyp"].tolist(), sub["ref"].tolist(), lowercase=lowercase)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--direction", choices=["f2en", "en2f"], required=True)
    parser.add_argument("--split", default="test", choices=["test", "validate"])
    parser.add_argument("--use-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--validate-tags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--beam", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--min-new-tokens", type=int, default=1)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--limit-per-lang", type=int, default=0)
    parser.add_argument("--lowercase-bleu", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    raw = read_parallel_csv(args.input)
    if "split" not in raw.columns:
        raise SystemExit("Input CSV must have split column.")
    raw["split"] = raw["split"].astype(str).str.lower()
    raw["source_bucket"] = raw["source"].map(source_bucket)
    eval_raw = raw[raw["split"].eq(args.split)].copy()
    if eval_raw.empty:
        raise SystemExit(f"No rows with split={args.split}.")
    if args.limit_per_lang > 0:
        sampled_groups = []
        for _, group in eval_raw.groupby("lang_code", sort=False):
            sampled_groups.append(group.sample(min(len(group), args.limit_per_lang), random_state=17))
        eval_raw = pd.concat(sampled_groups, ignore_index=True)
    tokenizer = NllbTokenizer.from_pretrained(args.tokenizer)
    eval_raw["_src_oov_rate"] = word_oov_rates(raw, eval_raw, args.direction)
    eval_raw["_formosan_pieces_per_word"] = formosan_fragmentation(tokenizer, eval_raw, args.direction)
    if args.use_tags and args.validate_tags:
        ensure_control_tags(tokenizer, eval_raw, args.direction)
    eval_tagged = with_tagged_columns(eval_raw, args.direction, use_tags=args.use_tags)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    model.to(device).eval()

    rows = []
    for lang, sub in eval_tagged.groupby("lang_code", sort=True):
        if lang not in FORMOSAN_CODES:
            continue
        if args.direction == "f2en":
            src_lid, tgt_lid = get_lid(lang), get_lid("english")
            src = sub["formosan_sentence"].astype(str).tolist()
            ref = sub["english_sentence"].astype(str).tolist()
        else:
            src_lid, tgt_lid = get_lid("english"), get_lid(lang)
            src = sub["english_sentence"].astype(str).tolist()
            ref = sub["formosan_sentence"].astype(str).tolist()
        hyp = generate(tokenizer, model, src, src_lid, tgt_lid, device, args, desc=f"{lang} {args.direction}")
        for idx, (_, original_row) in enumerate(eval_raw.loc[sub.index].iterrows()):
            rows.append(
                {
                    "row_id": original_row.get("row_id", ""),
                    "lang_code": lang,
                    "direction": args.direction,
                    "eval_tier": original_row.get("eval_tier", ""),
                    "source_bucket": original_row.get("source_bucket", ""),
                    "source": original_row.get("source", ""),
                    "dialect": original_row.get("dialect", ""),
                    "src": src[idx],
                    "ref": ref[idx],
                    "hyp": hyp[idx],
                    "src_tokens": len(str(src[idx]).split()),
                    "ref_tokens": len(str(ref[idx]).split()),
                    "src_oov_rate": float(original_row.get("_src_oov_rate", 0.0)),
                    "formosan_pieces_per_word": float(original_row.get("_formosan_pieces_per_word", 0.0)),
                }
            )

    preds = pd.DataFrame(rows)
    if preds.empty:
        raise SystemExit("No predictions generated.")
    preds["length_bin"] = preds["src_tokens"].map(length_bin)

    metrics = {
        "input": str(args.input),
        "model": str(args.model),
        "tokenizer": str(args.tokenizer),
        "direction": args.direction,
        "split": args.split,
        "samples": int(len(preds)),
        "global": {"samples": int(len(preds))} | score(preds["hyp"].tolist(), preds["ref"].tolist(), lowercase=args.lowercase_bleu),
        "by_language": group_scores(preds, "lang_code", lowercase=args.lowercase_bleu),
        "by_source_bucket": group_scores(preds, "source_bucket", lowercase=args.lowercase_bleu),
        "by_length_bin": group_scores(preds, "length_bin", lowercase=args.lowercase_bleu),
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(args.output_csv, index=False)
    write_json(args.output_json, metrics)
    print(json.dumps(metrics["global"], indent=2))
    print(f"predictions: {args.output_csv}")
    print(f"metrics: {args.output_json}")


if __name__ == "__main__":
    main()
