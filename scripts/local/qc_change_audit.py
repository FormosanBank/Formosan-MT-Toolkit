"""Classify field-level changes made by the pinned XML cleaner."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from xml_repairs import PROVENANCE_ATTR

XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"
CARET_VARIANTS = {
    "⌃": "^",
    "‸": "^",
    "ˆ": "^",
    "＾": "^",
}
ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
FULLWIDTH_TO_ASCII = {
    "（": "(",
    "）": ")",
    "：": ":",
    "，": ",",
    "？": "?",
    "！": "!",
    "。": ".",
    "》": '"',
    "《": '"',
    "」": '"',
    "「": '"',
    "、": ",",
    "】": ")",
    "【": "(",
    "]": ")",
    "[": "(",
    "〔": "(",
    "〕": ")",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "ˈ": "'",
    "`": "'",
    "ʼ": "'",
    "ʻ": "'",
    "『": '"',
    "』": '"',
}
CHINESE_DOUBLE_QUOTES = {
    "“": "＂",
    "”": "＂",
    "「": "＂",
    "」": "＂",
    "『": "＂",
    "』": "＂",
}
CHINESE_LANGS = frozenset(
    {"zho", "zh", "cmn", "yue", "wuu", "hak", "nan"}
)
HYPHEN_LETTER_LANGS = frozenset({"bnn", "ssf"})


def snapshot_cleaner_fields(
    corpus_dir: Path,
) -> dict[str, dict[str, str]]:
    fields: dict[str, dict[str, str]] = {}
    for xml_file in sorted(corpus_dir.rglob("*.xml")):
        root = ET.parse(xml_file).getroot()
        relative = str(xml_file.relative_to(corpus_dir))
        root_language = (root.get(XML_LANG) or "").strip().lower()
        for unit in root.iter():
            if unit.tag not in {"S", "W", "M"}:
                continue
            token = unit.get(PROVENANCE_ATTR, "")
            if not token:
                raise SystemExit(
                    "Cannot snapshot cleaner fields without transform "
                    f"provenance at {relative}:{unit.tag}:"
                    f"{unit.get('id', '')}"
                )
            occurrences: Counter[str] = Counter()
            for field in unit:
                if field.tag not in {"FORM", "TRANSL"}:
                    continue
                occurrence = occurrences[field.tag]
                occurrences[field.tag] += 1
                key = f"{token}:{field.tag}:{occurrence}"
                fields[key] = {
                    "xml_path": relative,
                    "xml_id": unit.get("id", ""),
                    "unit_tag": unit.tag,
                    "field_tag": field.tag,
                    "field_kind": (
                        field.get("kindOf") or ""
                    ).strip().lower(),
                    "language": (
                        field.get(XML_LANG)
                        or unit.get(XML_LANG)
                        or root_language
                    ).strip().lower(),
                    "explicit_language": (
                        field.get(XML_LANG) or ""
                    ).strip().lower(),
                    "text": field.text or "",
                }
    return fields


def _replace_characters(
    text: str,
    replacements: dict[str, str],
) -> str:
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def _normalize_cleaner_whitespace(text: str) -> str:
    text = re.sub(r" {2,}", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_cleaner_field_changes(
    before: dict[str, dict[str, str]],
    after: dict[str, dict[str, str]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    modified_keys: set[str] = set()
    metadata_fields_modified = 0
    unclassified: list[dict[str, str]] = []

    for key, before_row in before.items():
        after_row = after.get(key)
        if after_row is None:
            continue
        before_language = before_row.get(
            "explicit_language",
            before_row["language"],
        )
        after_language = after_row.get(
            "explicit_language",
            after_row["language"],
        )
        if before_language != after_language:
            modified_keys.add(key)
            metadata_fields_modified += 1
            language_rule = {
                ("en", "eng"): (
                    "normalize_translation_language_en_to_eng"
                ),
                ("zh", "zho"): (
                    "normalize_translation_language_zh_to_zho"
                ),
                ("", "eng"): (
                    "infer_hundred_paiwan_gloss_language_eng"
                ),
            }.get((before_language, after_language))
            if language_rule:
                counts[language_rule] += 1
            else:
                counts[
                    "unclassified_cleaner_metadata_change"
                ] += 1
        original = before_row["text"]
        expected = after_row["text"]
        if original == expected:
            continue
        modified_keys.add(key)
        working = original

        def apply(rule: str, value: str) -> None:
            nonlocal working
            if value != working:
                counts[rule] += 1
                working = value

        apply(
            "replace_non_breaking_space",
            working.replace("\u00a0", " "),
        )
        if before_row["field_tag"] == "FORM":
            apply(
                "unicode_nfc",
                unicodedata.normalize("NFC", working),
            )
        apply("html_unescape", html.unescape(working))
        apply(
            "remove_zero_width_characters",
            ZERO_WIDTH_RE.sub("", working),
        )
        apply(
            "normalize_caret_variants",
            _replace_characters(working, CARET_VARIANTS),
        )

        language = before_row["language"]
        is_chinese = (
            language in CHINESE_LANGS
            or language.startswith("zh")
        )
        if (
            before_row["field_tag"] == "TRANSL"
            and is_chinese
        ):
            apply(
                "normalize_chinese_double_quotes",
                _replace_characters(
                    working,
                    CHINESE_DOUBLE_QUOTES,
                ),
            )
        else:
            apply(
                "normalize_punctuation",
                _replace_characters(
                    working,
                    FULLWIDTH_TO_ASCII,
                ),
            )
        apply(
            "normalize_whitespace",
            _normalize_cleaner_whitespace(working),
        )
        repeated = re.sub(r"([?!])\1+", r"\1", working)
        repeated = re.sub(r"--+", "-", repeated)
        apply("trim_repeated_punctuation", repeated)

        if (
            before_row["unit_tag"] == "S"
            and before_row["field_tag"] == "FORM"
            and before_row["field_kind"] == "standard"
        ):
            segmentation = re.sub(r"Ø-|-Ø|Ø", "", working)
            segmentation = segmentation.replace("=", "")
            if language not in HYPHEN_LETTER_LANGS:
                segmentation = segmentation.replace("-", "")
            apply(
                "remove_standard_segmentation_markers",
                segmentation,
            )

        if working != expected:
            counts["unclassified_cleaner_change"] += 1
            if len(unclassified) < 20:
                unclassified.append(
                    {
                        "xml_path": before_row["xml_path"],
                        "xml_id": before_row["xml_id"],
                        "unit_tag": before_row["unit_tag"],
                        "field_tag": before_row["field_tag"],
                        "field_kind": before_row["field_kind"],
                        "before_sha256": hashlib.sha256(
                            original.encode("utf-8")
                        ).hexdigest(),
                        "after_sha256": hashlib.sha256(
                            expected.encode("utf-8")
                        ).hexdigest(),
                    }
                )

    added_fields = len(set(after) - set(before))
    if added_fields:
        counts["unexpected_cleaner_fields_added"] += added_fields
    return {
        "fields_scanned": len(before),
        "fields_modified": len(modified_keys),
        "metadata_fields_modified": metadata_fields_modified,
        "rule_counts": dict(sorted(counts.items())),
        "unclassified_examples": unclassified,
    }
