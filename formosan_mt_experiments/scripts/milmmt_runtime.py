#!/usr/bin/env python3
"""MiLMMT causal-LM controls for directional Formosan translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
import torch
from mt_common import is_formosan_to_target, safe_tag_value
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_FAMILY = "milmmt"

FORMOSAN_LANGUAGE_NAMES = {
    "ami": "Amis",
    "bnn": "Bunun",
    "ckv": "Kavalan",
    "dru": "Rukai",
    "pwn": "Paiwan",
    "pyu": "Puyuma",
    "ssf": "Thao",
    "sxr": "Saaroa",
    "szy": "Sakizaya",
    "tao": "Tao",
    "tay": "Atayal",
    "trv": "Seediq",
    "tsu": "Tsou",
    "xnb": "Kanakanavu",
    "xsy": "Saisiyat",
}
MAJOR_LANGUAGE_NAMES = {
    "english": "English",
    "chinese": "Chinese (Traditional)",
}


@dataclass(frozen=True)
class TaskSpec:
    language: str
    source_name: str
    target_name: str


def load_tokenizer(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    tokenizer.padding_side = "right"
    return tokenizer


def load_model(path: Path, *, dtype: torch.dtype):
    return AutoModelForCausalLM.from_pretrained(
        path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )


def configure_model(model, tokenizer) -> None:
    if tokenizer.pad_token_id is None or tokenizer.eos_token_id is None:
        raise SystemExit("MiLMMT tokenizer must define pad and EOS tokens")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.do_sample = False
        model.generation_config.top_k = None
        model.generation_config.top_p = None
        model.generation_config.cache_implementation = None


def validate_model_tokenizer(model, tokenizer) -> None:
    embedding_size = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) == embedding_size:
        return
    extra_tokens = {
        token: token_id for token, token_id in tokenizer.get_added_vocab().items() if int(token_id) >= embedding_size
    }
    if len(tokenizer) != embedding_size + 1 or extra_tokens != {"<image_soft_token>": embedding_size}:
        raise SystemExit("MiLMMT tokenizer and text embedding table have an unexpected mismatch")


def normalize_control_metadata(
    frame: pd.DataFrame,
    tokenizer,
    *,
    mode: str = "oracle",
) -> tuple[pd.DataFrame, dict[str, int]]:
    del tokenizer
    if mode not in {"default", "oracle"}:
        raise ValueError(f"Unsupported metadata mode: {mode}")
    output = frame.copy()
    if mode == "default":
        output["dialect"] = "default"
        return output, {"dialect_fallback_rows": len(output)}

    dialects = output.get(
        "dialect",
        pd.Series("default", index=output.index),
    ).map(lambda value: safe_tag_value(value, "default"))
    invalid_dialects = dialects.eq("")
    output["dialect"] = dialects.mask(invalid_dialects, "default")
    return output, {"dialect_fallback_rows": int(invalid_dialects.sum())}


def ensure_source_prefix_tokens(*args, **kwargs) -> None:
    del args, kwargs


def task_spec(
    language: str,
    direction: str,
    *,
    target_lang: str,
) -> TaskSpec:
    try:
        formosan_name = FORMOSAN_LANGUAGE_NAMES[language]
        major_name = MAJOR_LANGUAGE_NAMES[target_lang]
    except KeyError as exc:
        raise ValueError(f"Unsupported MiLMMT language: {exc.args[0]}") from exc
    if is_formosan_to_target(direction):
        return TaskSpec(language, formosan_name, major_name)
    return TaskSpec(language, major_name, formosan_name)


def format_source(
    row: Mapping,
    source_text: str,
    direction: str,
    *,
    target_lang: str,
    use_tags: bool,
) -> str:
    task = task_spec(
        str(row.get("lang_code", "")).strip().lower(),
        direction,
        target_lang=target_lang,
    )
    lines = [f"Translate this from {task.source_name} to {task.target_name}:"]
    if use_tags:
        dialect = safe_tag_value(row.get("dialect"), "default")
        lines.append(f"Dialect: {dialect}")
    lines.extend(
        [
            f"{task.source_name}: {source_text}",
            f"{task.target_name}:",
        ]
    )
    return "\n".join(lines)


def validate_task(tokenizer, task: TaskSpec) -> None:
    del tokenizer
    if not task.source_name or not task.target_name:
        raise SystemExit(f"Invalid MiLMMT task: {task}")


def _batch_token_ids(tokenizer, texts: list[str]) -> list[list[int]]:
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]
    return [list(row) for row in encoded]


def encode_batch(
    tokenizer,
    source_texts: list[str],
    target_texts: list[str],
    task: TaskSpec,
    *,
    max_length: int,
    device: torch.device | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    del task
    if max_length < 8:
        raise ValueError("MiLMMT max_length must be at least 8")
    eos = int(tokenizer.eos_token_id)
    pad = int(tokenizer.pad_token_id)
    sequences: list[list[int]] = []
    label_rows: list[list[int]] = []
    minimum_prompt_tokens = min(64, max_length // 2)
    target_limit = max(1, max_length - minimum_prompt_tokens - 1)
    prompt_rows = _batch_token_ids(tokenizer, source_texts)
    target_rows = _batch_token_ids(tokenizer, target_texts)
    for prompt_ids, target_ids in zip(prompt_rows, target_rows, strict=True):
        target_ids = target_ids[:target_limit]
        if not target_ids:
            raise ValueError("MiLMMT target tokenized to an empty sequence")
        response = target_ids + [eos]
        prompt_budget = max_length - len(response)
        if len(prompt_ids) > prompt_budget:
            prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + response
        sequences.append(input_ids)
        label_rows.append([-100] * len(prompt_ids) + response)

    width = min(
        max_length,
        ((max(len(row) for row in sequences) + 7) // 8) * 8,
    )
    input_ids = torch.full((len(sequences), width), pad, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), width), dtype=torch.long)
    labels = torch.full((len(sequences), width), -100, dtype=torch.long)
    for index, (sequence, label_row) in enumerate(zip(sequences, label_rows, strict=True)):
        length = len(sequence)
        input_ids[index, :length] = torch.tensor(sequence, dtype=torch.long)
        attention_mask[index, :length] = 1
        labels[index, :length] = torch.tensor(label_row, dtype=torch.long)
    encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if device is None:
        return encoded, labels
    return ({key: value.to(device) for key, value in encoded.items()}, labels.to(device))


@torch.no_grad()
def generate_batch(
    model,
    tokenizer,
    source_texts: list[str],
    task: TaskSpec,
    *,
    max_length: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    length_penalty: float,
    device: torch.device,
) -> list[str]:
    del task
    old_padding_side = tokenizer.padding_side
    old_truncation_side = tokenizer.truncation_side
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    try:
        encoded = tokenizer(
            source_texts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            pad_to_multiple_of=8,
            return_token_type_ids=False,
        )
    finally:
        tokenizer.padding_side = old_padding_side
        tokenizer.truncation_side = old_truncation_side
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_width = int(encoded["input_ids"].shape[1])
    generated = model.generate(
        **encoded,
        do_sample=False,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        no_repeat_ngram_size=no_repeat_ngram_size,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        eos_token_id=model.generation_config.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    completions = generated[:, prompt_width:]
    return [
        text.strip()
        for text in tokenizer.batch_decode(
            completions,
            skip_special_tokens=True,
        )
    ]
