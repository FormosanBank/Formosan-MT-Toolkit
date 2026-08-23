"""Data preparation and validation sampling for directional MT training."""

from __future__ import annotations

import math

import nllb_runtime as nllb
import numpy as np
import pandas as pd
import torch
from mt_common import (
    FORMOSAN_CODES,
    bool_series,
    is_formosan_to_target,
    read_parallel_csv,
    row_type_sampling_probs,
)
from mt_metrics import score_translations


def prepare_data(
    args,
    tokenizer,
    runtime=nllb,
) -> tuple[dict, dict, dict]:
    df = read_parallel_csv(
        args.input,
        target_col=args.target_col,
        columns=["split", "row_id", "row_type", "pivot_origin"],
    )
    if "split" not in df.columns:
        raise SystemExit("Training CSV must have split values train/validate/test.")
    df["split"] = df["split"].astype(str).str.lower()
    if "kindOf" not in df or not df["kindOf"].astype(str).str.lower().eq("standard").all():
        raise SystemExit("Training CSV must contain only kindOf=standard rows")
    if "row_id" not in df or df["row_id"].astype(str).duplicated().any():
        raise SystemExit("Training CSV must contain unique stable row_id values")
    unknown_splits = sorted(set(df["split"]) - {"train", "validate", "test"})
    if unknown_splits:
        raise SystemExit(f"Training CSV has unknown splits: {unknown_splits}")

    train = df[df["split"].eq("train")].copy()
    val = df[df["split"].isin(["validate", "valid", "val"])].copy()
    if train.empty:
        raise SystemExit("No train rows found.")
    if val.empty:
        raise SystemExit("No validation rows found")
    val_mt_eligible = bool_series(
        val["mt_eval_eligible"],
        context="training validation rows:mt_eval_eligible",
    )
    if not val_mt_eligible.all() or val["mt_normalization_confidence"].astype(str).eq("ambiguous").any():
        raise SystemExit("Validation contains MT-ineligible or ambiguous-normalization rows")

    validation_metadata_fallback = {"dialect_fallback_rows": 0}
    training_metadata_fallback = {"dialect_fallback_rows": 0}
    if args.use_tags:
        train, training_metadata_fallback = runtime.normalize_control_metadata(
            train,
            tokenizer,
            mode="oracle",
        )
        val, validation_metadata_fallback = runtime.normalize_control_metadata(
            val,
            tokenizer,
            mode=args.validation_metadata_mode,
        )
    if args.use_tags and args.validate_tags:
        runtime.ensure_source_prefix_tokens(
            tokenizer,
            train,
            args.direction,
            target_lang=args.target_lang,
            use_tags=args.use_tags,
        )
        runtime.ensure_source_prefix_tokens(
            tokenizer,
            val,
            args.direction,
            target_lang=args.target_lang,
            use_tags=args.use_tags,
        )
    pivot_origin = val.get(
        "pivot_origin",
        pd.Series("original", index=val.index),
    ).astype(str)
    if not val["row_type"].astype(str).eq("sentence").all():
        raise SystemExit("Validation must contain sentence rows only")

    train_by_lang = {}
    val_by_lang = {}
    for lang in sorted(train["lang_code"].unique()):
        if lang not in FORMOSAN_CODES:
            continue
        train_sub = train[train["lang_code"].eq(lang)].copy()
        val_sub = val[val["lang_code"].eq(lang)].copy()
        if is_formosan_to_target(args.direction):
            train_sub["src_text"] = train_sub["formosan_sentence"].astype(str)
            train_sub["tgt_text"] = train_sub[args.target_col].astype(str)
            val_sub["src_text"] = val_sub["formosan_sentence"].astype(str)
            val_sub["tgt_text"] = val_sub[args.target_col].astype(str)
        else:
            train_sub["src_text"] = train_sub[args.target_col].astype(str)
            train_sub["tgt_text"] = train_sub["formosan_sentence"].astype(str)
            val_sub["src_text"] = val_sub[args.target_col].astype(str)
            val_sub["tgt_text"] = val_sub["formosan_sentence"].astype(str)
        source_column = "formosan_sentence" if is_formosan_to_target(args.direction) else args.target_col
        val_sub["src_text"] = [
            runtime.format_source(
                row,
                str(row[source_column]),
                args.direction,
                target_lang=args.target_lang,
                use_tags=args.use_tags,
            )
            for row in val_sub.to_dict(orient="records")
        ]
        task = runtime.task_spec(
            lang,
            args.direction,
            target_lang=args.target_lang,
        )
        runtime.validate_task(tokenizer, task)
        normalized_train = train_sub.reset_index(drop=True)
        try:
            sampling_probs = row_type_sampling_probs(
                normalized_train["row_type"],
                args.lexical_row_sampling_weight,
            )
        except ValueError as exc:
            raise SystemExit(f"Cannot construct row sampling for {lang}: {exc}") from exc
        train_by_lang[lang] = {
            "df": normalized_train,
            "row_sampling_probs": np.asarray(sampling_probs, dtype=np.float64),
            "task": task,
        }
        if not val_sub.empty:
            val_by_lang[lang] = {
                "df": val_sub.reset_index(drop=True),
                "task": task,
            }

    if not train_by_lang:
        raise SystemExit("No supported Formosan languages found in train rows.")
    return (
        train_by_lang,
        val_by_lang,
        {
            "input_rows": int(len(df)),
            "train_rows": int(len(train)),
            "validate_rows": int(len(val)),
            "direction": args.direction,
            "model_family": runtime.MODEL_FAMILY,
            "target_lang": args.target_lang,
            "target_col": args.target_col,
            "use_tags": bool(args.use_tags),
            "standard_rows": int(df["kindOf"].astype(str).str.lower().eq("standard").sum()),
            "validation_metadata_fallback": validation_metadata_fallback,
            "validation_metadata_mode": args.validation_metadata_mode,
            "training_metadata_fallback": training_metadata_fallback,
            "dialect_tag_dropout": float(args.dialect_tag_dropout),
            "row_type_sampling": {
                "sentence_weight": 1.0,
                "lexical_weight": float(args.lexical_row_sampling_weight),
                "basis": "explicit XML row_type within each selected language",
                "train_by_language": {
                    lang: {
                        str(row_type): int(count)
                        for row_type, count in info["df"]["row_type"].value_counts().sort_index().items()
                    }
                    for lang, info in train_by_lang.items()
                },
            },
            "synthetic_train_rows": int(
                train.get(
                    "pivot_origin",
                    pd.Series("original", index=train.index),
                )
                .astype(str)
                .eq("synthetic")
                .sum()
            ),
            "synthetic_validate_rows": int(pivot_origin.eq("synthetic").sum()),
            "human_validate_rows": int((~pivot_origin.eq("synthetic")).sum()),
            "train_by_language": {key: int(len(value["df"])) for key, value in train_by_lang.items()},
            "validate_by_language": {key: int(len(value["df"])) for key, value in val_by_lang.items()},
        },
    )


def encode_batch(
    tokenizer,
    src_texts,
    tgt_texts,
    task,
    max_length,
    device,
    runtime=nllb,
):
    return runtime.encode_batch(
        tokenizer,
        list(src_texts),
        list(tgt_texts),
        task,
        max_length=max_length,
        device=device,
    )


def training_source_texts(
    batch: pd.DataFrame,
    *,
    args,
    runtime=nllb,
) -> tuple[list[str], dict[str, int]]:
    """Build training inputs with reproducible dialect-tag dropout."""
    if args.use_tags:
        dialect_mask = np.random.random(len(batch)) < args.dialect_tag_dropout
    else:
        dialect_mask = np.zeros(len(batch), dtype=bool)

    source_column = "formosan_sentence" if is_formosan_to_target(args.direction) else args.target_col
    texts: list[str] = []
    for index, row in enumerate(batch.to_dict(orient="records")):
        if dialect_mask[index]:
            row["dialect"] = "default"
        texts.append(
            runtime.format_source(
                row,
                str(row[source_column]),
                args.direction,
                target_lang=args.target_lang,
                use_tags=args.use_tags,
            )
        )
    return texts, {"dialect": int(dialect_mask.sum())}


def validation_subset(df: pd.DataFrame, lang: str, args) -> pd.DataFrame:
    """Select one stable validation sample reused by loss and generation metrics."""
    if args.eval_samples > 0 and len(df) > args.eval_samples:
        return df.sample(args.eval_samples, random_state=args.seed + sum(map(ord, lang))).sort_index()
    return df


def validation_sample_manifest(val_by_lang: dict, args) -> dict:
    """Record exactly which validation rows drive checkpoint selection."""
    rows = {}
    for lang, info in sorted(val_by_lang.items()):
        subset = validation_subset(info["df"], lang, args)
        rows[lang] = [
            {
                "row_id": str(row.get("row_id", index)),
                "source": str(row.get("source", "")),
            }
            for index, row in subset.iterrows()
        ]
    return {
        "seed": args.seed,
        "maximum_rows_per_language": args.eval_samples,
        "selection": "pandas.DataFrame.sample with stable per-language seed, then original index order",
        "rows": rows,
    }


@torch.no_grad()
def evaluate_loss(
    model,
    tokenizer,
    val_by_lang,
    args,
    device,
    runtime=nllb,
) -> dict:
    model.eval()
    lang_losses = {}
    total_loss = 0.0
    total_tokens = 0
    for lang, info in sorted(val_by_lang.items()):
        df = validation_subset(info["df"], lang, args)
        lang_loss = 0.0
        lang_tokens = 0
        for start in range(0, len(df), args.eval_batch_size):
            batch = df.iloc[start : start + args.eval_batch_size]
            enc, labels = encode_batch(
                tokenizer,
                batch["src_text"].tolist(),
                batch["tgt_text"].tolist(),
                info["task"],
                args.max_length,
                device,
                runtime,
            )
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            tokens = int((labels != -100).sum().item())
            lang_loss += float(outputs.loss.item()) * max(tokens, 1)
            lang_tokens += max(tokens, 1)
        if lang_tokens:
            lang_losses[lang] = lang_loss / lang_tokens
            total_loss += lang_loss
            total_tokens += lang_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    model.train()
    return {
        "mean_token_loss": float(mean_loss),
        "ppl": float(math.exp(mean_loss) if mean_loss < 50 else float("inf")),
        "by_language": {key: float(value) for key, value in lang_losses.items()},
    }


@torch.no_grad()
def evaluate_generation(
    model,
    tokenizer,
    val_by_lang,
    args,
    device,
    runtime=nllb,
) -> dict:
    training_use_cache = getattr(model.config, "use_cache", None)
    if training_use_cache is not None:
        model.config.use_cache = True
    model.eval()
    hypotheses: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    by_language: dict[str, dict] = {}
    bleu_tokenize = "zh" if args.direction == "f2zh" else "13a"
    for lang, info in sorted(val_by_lang.items()):
        df = validation_subset(info["df"], lang, args)
        lang_hypotheses: list[str] = []
        lang_references = df["tgt_text"].astype(str).tolist()
        lang_sources = df["src_text"].astype(str).tolist()
        task = info["task"]
        for start in range(0, len(df), args.generation_batch_size):
            batch = df.iloc[start : start + args.generation_batch_size]
            lang_hypotheses.extend(
                runtime.generate_batch(
                    model,
                    tokenizer,
                    batch["src_text"].astype(str).tolist(),
                    task,
                    max_length=args.max_length,
                    max_new_tokens=args.validation_max_new_tokens,
                    min_new_tokens=1,
                    num_beams=args.validation_beam,
                    no_repeat_ngram_size=0,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    device=device,
                )
            )
        by_language[lang] = {"samples": len(df)} | score_translations(
            lang_hypotheses,
            lang_references,
            sources=lang_sources,
            bleu_tokenize=bleu_tokenize,
        )
        hypotheses.extend(lang_hypotheses)
        references.extend(lang_references)
        sources.extend(lang_sources)
    if training_use_cache is not None:
        model.config.use_cache = training_use_cache
    model.train()
    return {
        "samples": len(hypotheses),
        "bleu_tokenize": bleu_tokenize,
        "global": score_translations(
            hypotheses,
            references,
            sources=sources,
            bleu_tokenize=bleu_tokenize,
        ),
        "by_language": by_language,
    }
