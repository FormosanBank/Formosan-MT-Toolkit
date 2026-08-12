#!/usr/bin/env python3
"""Model-family contracts for directional Formosan machine translation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from mt_common import (
    DOMAIN_BUCKETS,
    FORMOSAN_CODES,
    build_prefix,
    get_lid,
    is_formosan_to_target,
    safe_tag_value,
    target_language_from_direction,
    target_lid_for,
)
from transformers import AutoTokenizer, NllbTokenizer

MADLAD_TARGET_TOKENS = {
    "english": "<2en>",
    "chinese": "<2zh_Hant>",
}


@dataclass(frozen=True)
class TaskSpec:
    """Language-control values for one Formosan language and direction."""

    language: str
    source_lid: str | None = None
    target_lid: str | None = None
    target_selector: str | None = None


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
    """Fallback held-out metadata that was intentionally absent from setup."""
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
        value: token_exists(
            tokenizer,
            f"<dom_{safe_tag_value(value, 'unknown')}>",
        )
        for value in buckets.unique()
    }
    dialect_available = {
        value: token_exists(
            tokenizer,
            f"<dialect_{safe_tag_value(value)}>",
        )
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


def ensure_source_prefix_tokens(
    backend: "ModelBackend",
    tokenizer,
    frame: pd.DataFrame,
    direction: str,
    *,
    target_lang: str,
    use_tags: bool,
) -> None:
    needed: set[str] = set()
    for _, row in frame.iterrows():
        needed.update(
            backend.source_prefix(
                row,
                direction,
                target_lang=target_lang,
                use_tags=use_tags,
            ).split()
        )
    bad: list[str] = []
    for token in sorted(needed):
        try:
            token_id(tokenizer, token)
        except SystemExit:
            bad.append(token)
    if bad:
        raise SystemExit(
            f"{backend.family} source-control tokens are not single tokenizer tokens. "
            f"First missing/broken tags: {bad[:30]}"
        )


class ModelBackend:
    family = "base"

    def load_tokenizer(self, path: Path):
        return AutoTokenizer.from_pretrained(path, use_fast=False)

    def task_spec(
        self,
        language: str,
        direction: str,
        *,
        target_lang: str,
    ) -> TaskSpec:
        raise NotImplementedError

    def source_prefix(
        self,
        row: Mapping,
        direction: str,
        *,
        target_lang: str,
        use_tags: bool,
    ) -> str:
        raise NotImplementedError

    def prepare_source(self, tokenizer, task: TaskSpec) -> None:
        return None

    def prepare_target(self, tokenizer, task: TaskSpec) -> None:
        return None

    def generation_kwargs(self, tokenizer, model, task: TaskSpec) -> dict[str, int]:
        raise NotImplementedError

    def configure_model(self, model, tokenizer) -> None:
        raise NotImplementedError

    def validate_task(self, tokenizer, task: TaskSpec) -> None:
        return None


class NllbBackend(ModelBackend):
    family = "nllb"

    def load_tokenizer(self, path: Path):
        return NllbTokenizer.from_pretrained(path, use_fast=False)

    def task_spec(
        self,
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
        self,
        row: Mapping,
        direction: str,
        *,
        target_lang: str,
        use_tags: bool,
    ) -> str:
        if not use_tags:
            return ""
        return build_prefix(row, direction, target_lang=target_lang)

    def prepare_source(self, tokenizer, task: TaskSpec) -> None:
        tokenizer.src_lang = task.source_lid

    def prepare_target(self, tokenizer, task: TaskSpec) -> None:
        tokenizer.tgt_lang = task.target_lid

    def generation_kwargs(self, tokenizer, model, task: TaskSpec) -> dict[str, int]:
        return {
            "forced_bos_token_id": token_id(tokenizer, str(task.target_lid)),
            "decoder_start_token_id": int(tokenizer.eos_token_id),
            "eos_token_id": int(tokenizer.eos_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        }

    def configure_model(self, model, tokenizer) -> None:
        model.config.decoder_start_token_id = tokenizer.eos_token_id
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.decoder_start_token_id = tokenizer.eos_token_id

    def validate_task(self, tokenizer, task: TaskSpec) -> None:
        token_id(tokenizer, str(task.source_lid))
        token_id(tokenizer, str(task.target_lid))


class MadladBackend(ModelBackend):
    family = "madlad400"

    @staticmethod
    def target_selector(
        language: str,
        direction: str,
        *,
        target_lang: str,
    ) -> str:
        if is_formosan_to_target(direction):
            return MADLAD_TARGET_TOKENS[
                target_language_from_direction(direction, target_lang)
            ]
        return f"<2{language}>"

    def task_spec(
        self,
        language: str,
        direction: str,
        *,
        target_lang: str,
    ) -> TaskSpec:
        return TaskSpec(
            language=language,
            target_selector=self.target_selector(
                language,
                direction,
                target_lang=target_lang,
            ),
        )

    def source_prefix(
        self,
        row: Mapping,
        direction: str,
        *,
        target_lang: str,
        use_tags: bool,
    ) -> str:
        language = str(row.get("lang_code", "")).strip().lower()
        selector = self.target_selector(
            language,
            direction,
            target_lang=target_lang,
        )
        if not use_tags:
            return selector
        return (
            f"{selector} "
            f"{build_prefix(row, direction, target_lang=target_lang)}"
        )

    def generation_kwargs(self, tokenizer, model, task: TaskSpec) -> dict[str, int]:
        return {
            "decoder_start_token_id": int(model.config.decoder_start_token_id),
            "eos_token_id": int(model.config.eos_token_id),
            "pad_token_id": int(model.config.pad_token_id),
        }

    def configure_model(self, model, tokenizer) -> None:
        expected = {
            "decoder_start_token_id": 0,
            "pad_token_id": 1,
            "eos_token_id": 2,
        }
        for field, value in expected.items():
            if int(getattr(model.config, field)) != value:
                raise SystemExit(
                    f"MADLAD {field} must remain {value}; "
                    f"found {getattr(model.config, field)!r}"
                )
        if getattr(model, "generation_config", None) is not None:
            for field, value in expected.items():
                setattr(model.generation_config, field, value)

    def validate_task(self, tokenizer, task: TaskSpec) -> None:
        token_id(tokenizer, str(task.target_selector))


BACKENDS: dict[str, ModelBackend] = {
    "nllb": NllbBackend(),
    "madlad400": MadladBackend(),
}


def get_backend(profile_or_family: Mapping | str) -> ModelBackend:
    family = (
        str(profile_or_family.get("model_family", "nllb"))
        if isinstance(profile_or_family, Mapping)
        else str(profile_or_family)
    ).strip().lower()
    try:
        return BACKENDS[family]
    except KeyError as exc:
        raise SystemExit(f"Unsupported model family: {family}") from exc


def madlad_formosan_target_tokens() -> list[str]:
    return [f"<2{code}>" for code in FORMOSAN_CODES]
