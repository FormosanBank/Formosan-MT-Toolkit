#!/usr/bin/env python3
"""NLLB runtime controls for directional Formosan machine translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from mt_common import (
    DOMAIN_BUCKETS,
    build_prefix,
    get_lid,
    is_formosan_to_target,
    safe_tag_value,
    source_bucket,
    target_lid_for,
)
from transformers import NllbTokenizer

MODEL_FAMILY = "nllb"


@dataclass(frozen=True)
class TaskSpec:
    """NLLB language IDs for one Formosan language and direction."""

    language: str
    source_lid: str
    target_lid: str


def token_id(tokenizer, token: str) -> int:
    value = int(tokenizer.convert_tokens_to_ids(token))
    if value == tokenizer.unk_token_id:
        raise SystemExit(
            f"Required token {token!r} maps to <unk>; load the matching setup artifacts."
        )
    if tokenizer.convert_ids_to_tokens(value) != token:
        raise SystemExit(
            f"Required token {token!r} does not round-trip through the tokenizer."
        )
    return value


def token_exists(tokenizer, token: str) -> bool:
    value = int(tokenizer.convert_tokens_to_ids(token))
    return (
        value != tokenizer.unk_token_id
        and tokenizer.convert_ids_to_tokens(value) == token
    )


def normalize_control_metadata(
    frame: pd.DataFrame,
    tokenizer,
    *,
    mode: str = "oracle",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Replace held-out metadata that has no setup token with stable defaults."""
    if mode not in {"default", "oracle"}:
        raise ValueError(f"Unsupported metadata mode: {mode}")
    output = frame.copy()
    if mode == "default":
        output["source_bucket"] = "unknown"
        output["dialect"] = "default"
        return output, {
            "domain_fallback_rows": len(output),
            "dialect_fallback_rows": len(output),
        }

    def values(column: str, default: str) -> pd.Series:
        if column not in output:
            return pd.Series(default, index=output.index, dtype="object")
        return output[column].map(
            lambda value: (
                default
                if pd.isna(value) or not str(value).strip()
                else str(value)
            )
        )

    raw_buckets = values("source_bucket", "unknown")
    canonical_buckets = raw_buckets.map(
        lambda value: safe_tag_value(value, "unknown")
    )
    domain_invalid = ~canonical_buckets.isin(DOMAIN_BUCKETS)
    buckets = canonical_buckets.mask(domain_invalid, "unknown")
    dialects = values("dialect", "default")
    domain_available = {
        value: token_exists(tokenizer, f"<dom_{safe_tag_value(value, 'unknown')}>")
        for value in buckets.unique()
    }
    dialect_available = {
        value: token_exists(tokenizer, f"<dialect_{safe_tag_value(value)}>")
        for value in dialects.unique()
    }
    domain_missing = domain_invalid | ~buckets.map(domain_available)
    dialect_missing = ~dialects.map(dialect_available)
    output["source_bucket"] = buckets.mask(domain_missing, "unknown")
    output["dialect"] = dialects.mask(dialect_missing, "default")
    return output, {
        "domain_fallback_rows": int(domain_missing.sum()),
        "dialect_fallback_rows": int(dialect_missing.sum()),
    }


def load_tokenizer(path: Path):
    return NllbTokenizer.from_pretrained(path, use_fast=False)


def task_spec(
    language: str,
    direction: str,
    *,
    target_lang: str,
) -> TaskSpec:
    major_lid = target_lid_for(target_lang)
    if is_formosan_to_target(direction):
        return TaskSpec(
            language=language,
            source_lid=get_lid(language),
            target_lid=major_lid,
        )
    return TaskSpec(
        language=language,
        source_lid=major_lid,
        target_lid=get_lid(language),
    )


def source_prefix(
    row: Mapping,
    direction: str,
    *,
    target_lang: str,
    use_tags: bool,
) -> str:
    if not use_tags:
        return ""
    return build_prefix(row, direction, target_lang=target_lang)


def ensure_source_prefix_tokens(
    tokenizer,
    frame: pd.DataFrame,
    direction: str,
    *,
    target_lang: str,
    use_tags: bool,
) -> None:
    metadata = pd.DataFrame(index=frame.index)
    metadata["lang_code"] = frame["lang_code"].astype(str)
    metadata["source_bucket"] = (
        frame["source_bucket"].astype(str)
        if "source_bucket" in frame
        else frame["source"].map(source_bucket)
    )
    metadata["dialect"] = (
        frame["dialect"].astype(str)
        if "dialect" in frame
        else "default"
    )
    needed = {
        token
        for row in metadata.drop_duplicates().to_dict(orient="records")
        for token in source_prefix(
            row,
            direction,
            target_lang=target_lang,
            use_tags=use_tags,
        ).split()
    }
    bad: list[str] = []
    for token in sorted(needed):
        try:
            token_id(tokenizer, token)
        except SystemExit:
            bad.append(token)
    if bad:
        raise SystemExit(
            "NLLB source-control tokens are not single tokenizer tokens. "
            f"First missing/broken tags: {bad[:30]}"
        )


def prepare_source(tokenizer, task: TaskSpec) -> None:
    tokenizer.src_lang = task.source_lid


def prepare_target(tokenizer, task: TaskSpec) -> None:
    tokenizer.tgt_lang = task.target_lid


def generation_kwargs(tokenizer, task: TaskSpec) -> dict[str, int]:
    return {
        "forced_bos_token_id": token_id(tokenizer, task.target_lid),
        "decoder_start_token_id": int(tokenizer.eos_token_id),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
    }


def configure_model(model, tokenizer) -> None:
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.decoder_start_token_id = tokenizer.eos_token_id


def validate_task(tokenizer, task: TaskSpec) -> None:
    token_id(tokenizer, task.source_lid)
    token_id(tokenizer, task.target_lid)
