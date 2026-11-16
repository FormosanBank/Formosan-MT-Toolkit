"""Fixtures shared by several test modules."""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from mt_standardization import (
    DEFAULT_PROFILE_PATH,
    StandardizationContext,
    assert_idempotent,
    load_profile,
    profile_sha256,
    standardize_text,
)

MT_PROFILE = load_profile(DEFAULT_PROFILE_PATH)
MT_PROFILE_HASH = profile_sha256(DEFAULT_PROFILE_PATH)


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def mt_contract_fields(
    text: str,
    *,
    row_type: str = "sentence",
    pivot_origin: str = "original",
    confidence: str = "unchanged",
) -> dict[str, object]:
    return {
        "kindOf": "standard",
        "standard_namespace": "formosan-mt",
        "formosan_mt_standard": text,
        "mt_standard_sha256": text_sha256(text),
        "mt_normalization_status": "accepted",
        "mt_normalization_confidence": confidence,
        "mt_eval_eligible": (
            pivot_origin == "original"
            and row_type == "sentence"
            and confidence in {"unchanged", "safe"}
        ),
        "mt_normalization_reason": "",
        "mt_standard_profile": MT_PROFILE["profile_id"],
        "mt_standard_profile_sha256": MT_PROFILE_HASH,
    }


def mt_records_for_xml(
    path: Path,
    directory: Path,
    *,
    language: str = "ami",
) -> dict[tuple[str, str, int, str], dict[str, object]]:
    relative = str(path.relative_to(directory))
    repository = Path(relative).parts[0] if Path(relative).parts else "FixtureRepo"
    records: dict[tuple[str, str, int, str], dict[str, object]] = {}
    unit_index = 0
    for element in ET.parse(path).getroot().iter():
        if element.tag not in {"S", "W", "M"}:
            continue
        element_index = unit_index
        unit_index += 1
        xml_id = (element.get("id") or "").strip()
        standard = element.find("FORM[@kindOf='standard']")
        original = element.find("FORM[@kindOf='original']")
        selected = standard if standard is not None else original
        source_standard = (
            "" if selected is None else "".join(selected.itertext()).strip()
        )
        original_text = (
            "" if original is None else "".join(original.itertext()).strip()
        )
        contains_unclear = (
            selected is not None
            and selected.find(".//UNCLEAR") is not None
        )
        row_type = {"S": "sentence", "W": "lexeme", "M": "morpheme"}[
            element.tag
        ]
        context = StandardizationContext(
            language=language,
            row_type=row_type,
            repository=repository,
            xml_path=relative,
        )
        result = standardize_text(
            source_standard,
            context=context,
            profile=MT_PROFILE,
            contains_unclear=contains_unclear,
        )
        assert_idempotent(result, context=context, profile=MT_PROFILE)
        records[(relative, element.tag, element_index, xml_id)] = {
            "standard_origin": (
                "provided" if standard is not None else "derived_from_original"
            ),
            "formosan_original_raw": original_text,
            "formosan_source_standard": source_standard,
            "formosan_mt_standard": result.text,
            "source_standard_sha256": text_sha256(source_standard),
            "mt_standard_sha256": text_sha256(result.text),
            "contains_unclear_source": contains_unclear,
            "mt_normalization_status": result.status,
            "mt_normalization_confidence": result.confidence,
            "mt_eval_eligible": result.eval_eligible,
            "mt_normalization_reason": result.reason,
            "mt_transformations": json.dumps(result.transformations),
            "mt_unresolved_markers": "|".join(result.unresolved_markers),
            "speaker_label": result.speaker_label,
            "mt_standard_profile": MT_PROFILE["profile_id"],
            "mt_standard_profile_sha256": MT_PROFILE_HASH,
        }
    return records


def add_mt_contract(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    row_type = output.get(
        "row_type",
        pd.Series("sentence", index=output.index),
    ).astype(str)
    if "xml_unit_context" not in output:
        output["xml_unit_context"] = row_type.map(
            {
                "sentence": "sentence",
                "lexeme": "standalone_word",
                "morpheme": "standalone_morpheme",
            }
        ).fillna("unknown")
    output["kindOf"] = "standard"
    output["standard_namespace"] = "formosan-mt"
    output["formosan_mt_standard"] = output["formosan_sentence"].astype(str)
    output["formosan_source_standard"] = output["formosan_sentence"].astype(str)
    output["formosan_original_raw"] = output["formosan_sentence"].astype(str)
    output["mt_standard_sha256"] = output["formosan_sentence"].map(
        lambda value: hashlib.sha256(str(value).encode()).hexdigest()
    )
    output["source_standard_sha256"] = output["mt_standard_sha256"]
    output["mt_normalization_status"] = "accepted"
    output["mt_normalization_confidence"] = "unchanged"
    output["mt_eval_eligible"] = output.get("pivot_origin", pd.Series("original", index=output.index)).astype(str).eq(
        "original"
    ) & output.get("row_type", pd.Series("sentence", index=output.index)).astype(str).eq("sentence")
    output["mt_normalization_reason"] = ""
    output["mt_standard_profile"] = MT_PROFILE["profile_id"]
    output["mt_standard_profile_sha256"] = MT_PROFILE_HASH
    return output
