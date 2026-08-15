"""Conservative, language-aware quality rules for Formosan MT pairs."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
CJK_RE = re.compile(r"[\u3400-\u9FFF\U00020000-\U0002B81F\uF900-\uFAFF]")
KANA_HANGUL_RE = re.compile(r"[\u3040-\u30FF\uAC00-\uD7AF]")
LATIN_RE = re.compile(r"[A-Za-z\u00C0-\u024F]")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[A-Za-z/!][^>]*>")
REDACTION_RE = re.compile(r"(?<![A-Za-z])X{3,}(?![A-Za-z])", re.IGNORECASE)
REPEATED_CHAR_RE = re.compile(r"([^\s])\1{7,}")
MISSING_TRANSLATION_RE = re.compile(
    r"^\s*(?:\[?\s*translation\s+missing\s*\]?|\(?no\s+record\)?|"
    r"無翻譯|缺翻譯|n/?a|null|nan)\s*$",
    re.IGNORECASE,
)
TARGET_META_RE = re.compile(
    r"^\s*[\[\【(（]\s*(?:介|虛|虚|名|動|动|形|副|代|助|連|连|量|嘆|叹|"
    r"語助|语助|語氣|语气|疑問|疑问|感嘆|感叹|pos|particle|prep(?:osition)?|"
    r"noun|verb|adj(?:ective)?|adv(?:erb)?)\s*[\]\】)）]\s*$",
    re.IGNORECASE,
)
STAGE_ONLY_RE = re.compile(
    r"^\s*(?:換下一個(?:說)?|我(?:分享|說明)到這裡|全文紀錄|中文紀錄|女子全名|"
    r"人物生平\s*[-—–]\s*[-—–])\s*[.!。！]?\s*$",
    re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(r"^[A-Z][：:]\s*")
LEXICAL_PATH_HINTS = (
    "dictionary",
    "dict/",
    "dict-",
    "dicts",
    "lexicon",
    "wordlist",
    "vocab",
    "詞表",
    "學習詞表",
    "learning_vocabulary",
)
NON_TRANSLATION_KINDS = frozenset({"gloss", "interlinear-gloss"})
GLOSS_TAGS = frozenset(
    """
    ABS ACC ACT AF APPL AUX AV CAU CAUS CN COMP CONJ CONV COP COS
    COSHAB CV DAT DEM DEON DIST ERG EVI EVID EXCL FILL FIN FOC FUT
    GEN HAB IMP INCL INST INTR IPFV IRR LNK LOC LOCNMLZ LV NAV NCM
    NEG NFUT NFIN NIA NMLZ NOM NPST NTOP OBL PART PASS PERF PF PFV
    PI PIV PL PLN PN POSS PPN PRF PROG PROS PROX PRS PRT PST PV REAL
    RED REL RL RV SG STA STAT SUB TOP TR UV VBLZ VCL
    """.split()
)
LEXICAL_GLOSS_TAGS = GLOSS_TAGS | frozenset(
    {"DEIC", "EN", "EN1", "EN2", "NEUT", "NM", "REFL", "SA", "STIM", "THM"}
)
GLOSS_PART_RE = re.compile(r"[.=_-]+")
PERSON_NUMBER_GLOSS_RE = re.compile(r"^[123](?:SG|PL)(?:INCL|EXCL)?$")
GENERIC_GLOSS_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
LEXICAL_STRONG_BOUNDARY_RE = re.compile(r"\S[=_]\S")
LEXICAL_AMBIGUOUS_BOUNDARY_RE = re.compile(r"\S[-/]\S")
GLOSS_TOKEN_EDGE_PUNCTUATION = "()[]{}<>,;:!?\"“”‘’'"


@dataclass(frozen=True)
class TextNormalization:
    text: str
    transformations: tuple[str, ...]


@dataclass(frozen=True)
class QualityDecision:
    disposition: str
    reason: str
    flags: tuple[str, ...] = ()


def normalize_text(value: object) -> TextNormalization:
    raw = "" if value is None else str(value)
    transformations: list[str] = []
    text = raw
    cleaned_controls = CONTROL_RE.sub(" ", text)
    cleaned_controls = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf", "Cs"} else character for character in cleaned_controls
    )
    if cleaned_controls != text:
        transformations.append("remove_controls")
        text = cleaned_controls
    unescaped = html.unescape(text)
    if unescaped != text:
        transformations.append("html_unescape")
        text = unescaped
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        transformations.append("unicode_nfc")
        text = normalized
    squeezed = WHITESPACE_RE.sub(" ", text).strip()
    if squeezed != text:
        transformations.append("whitespace")
        text = squeezed
    return TextNormalization(text=text, transformations=tuple(transformations))


def letters_and_marks(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFC", value).casefold()
        if unicodedata.category(character)[0] in {"L", "N", "M"}
    )


def exact_key(value: str) -> str:
    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value).casefold()).strip()


def is_only_punctuation_or_symbols(value: str) -> bool:
    if not value.strip():
        return True
    return not any(unicodedata.category(character)[0] in {"L", "N", "M"} for character in value)


def token_count(value: str) -> int:
    return len(value.split()) if value.strip() else 0


def target_units(value: str, target_language: str) -> int:
    if target_language == "chinese":
        cjk = len(CJK_RE.findall(value))
        non_cjk = CJK_RE.sub(" ", value)
        return cjk + token_count(non_cjk)
    return token_count(value)


def normalized_row_type(value: object, source_path: object) -> str:
    existing = str(value or "").strip().lower()
    if existing in {"sentence", "lexeme", "morpheme"}:
        return existing
    path = str(source_path or "").lower()
    if any(hint in path for hint in LEXICAL_PATH_HINTS):
        return "lexeme"
    return "unknown"


def script_counts(value: str) -> dict[str, int]:
    return {
        "cjk": len(CJK_RE.findall(value)),
        "kana_hangul": len(KANA_HANGUL_RE.findall(value)),
        "latin": len(LATIN_RE.findall(value)),
    }


def normalized_translation_kind(value: object) -> str:
    return re.sub(r"[\s_]+", "-", str(value or "").strip().casefold())


def has_annotation_gloss_structure(value: str) -> bool:
    """Detect unlabelled interlinear notation without matching normal prose."""
    gloss_hits = 0
    separated_tokens = 0
    tokens = value.split()
    for raw_token in tokens:
        token = raw_token.strip(GLOSS_TOKEN_EDGE_PUNCTUATION)
        parts = [part for part in GLOSS_PART_RE.split(token) if part]
        token_hits = sum(
            part in GLOSS_TAGS or PERSON_NUMBER_GLOSS_RE.fullmatch(part) is not None
            for part in parts
        )
        if not token_hits:
            continue
        gloss_hits += token_hits
        if GLOSS_PART_RE.search(token):
            separated_tokens += 1

    return (
        separated_tokens >= 2
        or (separated_tokens >= 1 and gloss_hits >= 2)
        or (separated_tokens >= 1 and len(tokens) <= 3)
    )


def target_gloss_reason(
    value: object,
    *,
    translation_kind: object,
    target_language: str,
) -> str:
    """Return a stable rejection reason for explicit or unlabelled glosses."""
    if normalized_translation_kind(translation_kind) in NON_TRANSLATION_KINDS:
        return "target_gloss_translation"
    if target_language == "english" and has_annotation_gloss_structure(str(value)):
        return "target_annotation_gloss"
    return ""


def is_gloss_code(value: str) -> bool:
    return (
        value in LEXICAL_GLOSS_TAGS
        or PERSON_NUMBER_GLOSS_RE.fullmatch(value) is not None
        or GENERIC_GLOSS_CODE_RE.fullmatch(value) is not None
    )


def has_lexical_morphological_gloss(
    value: object,
    *,
    target_language: str,
) -> bool:
    """Detect high-confidence gloss notation in lexical target strings."""
    text = str(value).strip()
    if not text:
        return False
    if has_annotation_gloss_structure(text) or LEXICAL_STRONG_BOUNDARY_RE.search(text):
        return True

    tokens = text.split()
    for raw_token in tokens:
        token = raw_token.strip(GLOSS_TOKEN_EDGE_PUNCTUATION)
        if len(tokens) == 1 and is_gloss_code(token):
            return True
        if not GLOSS_PART_RE.search(token):
            continue
        parts = [part for part in GLOSS_PART_RE.split(token) if part]
        if any(is_gloss_code(part) for part in parts):
            return True

    if target_language == "chinese":
        return bool(LEXICAL_AMBIGUOUS_BOUNDARY_RE.search(text) and re.search(r"[A-Z]", text))
    return False


def lexical_quality_reason(
    value: object,
    *,
    row_type: object,
    xml_unit_context: object,
    target_language: str,
) -> str:
    """Return a reason that keeps questionable word glosses out of MT."""
    normalized_type = str(row_type or "").strip().casefold()
    if normalized_type not in {"lexeme", "morpheme"}:
        return ""
    if normalized_type == "lexeme" and str(xml_unit_context or "").strip() != "standalone_word":
        return "ambiguous_lexical_structure"
    if has_lexical_morphological_gloss(
        value,
        target_language=target_language,
    ):
        return "target_morphological_gloss"

    text = str(value).strip()
    if LEXICAL_AMBIGUOUS_BOUNDARY_RE.search(text):
        if target_language == "chinese" or len(text.split()) == 1:
            return "ambiguous_lexical_translation"
    return ""


def fertility_reason(
    source: str,
    target: str,
    target_language: str,
    row_type: str,
) -> str:
    source_units = token_count(source)
    target_count = target_units(target, target_language)
    if not source_units or not target_count:
        return "empty_units"
    ratio = target_count / source_units
    low, high = (0.03, 30.0) if row_type in {"lexeme", "morpheme"} else (0.15, 12.0)
    return "extreme_fertility" if ratio < low or ratio > high else ""


def quality_decision(
    row: Mapping[str, object],
    *,
    source_column: str,
    target_column: str,
    target_language: str,
    keep_redactions: bool,
) -> QualityDecision:
    source = str(row[source_column])
    target = str(row[target_column])
    row_type = str(row["row_type"])
    flags: list[str] = []

    if str(row.get("kindOf", "standard")).strip().lower() != "standard":
        return QualityDecision("rejected", "non_standard_form")
    if str(row.get("standard_namespace", "")).strip() != "formosan-mt":
        return QualityDecision("rejected", "non_mt_standard_namespace")
    mt_status = str(row.get("mt_normalization_status", "")).strip().lower()
    if mt_status != "accepted":
        reason = str(row.get("mt_normalization_reason", "")).strip() or mt_status or "missing"
        return QualityDecision("quarantine", f"mt_standard:{reason}")
    if str(row.get("formosan_mt_standard", "")) != source:
        return QualityDecision("rejected", "mt_standard_alias_mismatch")
    if "456otca" in source.casefold():
        return QualityDecision("rejected", "source_artifact_marker")
    if "*" in source:
        return QualityDecision("rejected", "source_annotation_marker")
    gloss_reason = target_gloss_reason(
        target,
        translation_kind=row.get("translation_kind", ""),
        target_language=target_language,
    )
    if gloss_reason:
        disposition = "rejected" if gloss_reason == "target_gloss_translation" else "quarantine"
        return QualityDecision(disposition, gloss_reason)
    lexical_reason = lexical_quality_reason(
        target,
        row_type=row_type,
        xml_unit_context=row.get("xml_unit_context", ""),
        target_language=target_language,
    )
    if lexical_reason:
        return QualityDecision("quarantine", lexical_reason)
    if MISSING_TRANSLATION_RE.match(source) or MISSING_TRANSLATION_RE.match(target):
        return QualityDecision("rejected", "missing_translation_marker")
    if is_only_punctuation_or_symbols(source) or is_only_punctuation_or_symbols(target):
        return QualityDecision("rejected", "empty_or_punctuation_only")
    if not keep_redactions and (REDACTION_RE.search(source) or REDACTION_RE.search(target)):
        return QualityDecision("rejected", "redaction_placeholder")
    if CJK_RE.search(source) or KANA_HANGUL_RE.search(source):
        return QualityDecision("quarantine", "non_formosan_script_in_source")
    if row_type == "sentence" and ("=" in source or "Ø" in source):
        return QualityDecision("quarantine", "segmentation_marker_in_standard_sentence")
    if URL_RE.search(source) or URL_RE.search(target):
        return QualityDecision("quarantine", "url")
    if HTML_TAG_RE.search(source) or HTML_TAG_RE.search(target):
        return QualityDecision("quarantine", "markup")
    if REPEATED_CHAR_RE.search(source) or REPEATED_CHAR_RE.search(target):
        return QualityDecision("quarantine", "repeated_character_noise")
    if STAGE_ONLY_RE.match(source) or STAGE_ONLY_RE.match(target):
        return QualityDecision("rejected", "presentation_scaffolding")
    if TARGET_META_RE.match(target):
        return QualityDecision("rejected", "target_meta_label_only")
    target_scripts = script_counts(target)
    if target_language == "english":
        if target_scripts["kana_hangul"]:
            return QualityDecision("quarantine", "english_target_script_mismatch")
        if target_scripts["cjk"] >= 2 and target_scripts["cjk"] > max(2, target_scripts["latin"] // 2):
            return QualityDecision("quarantine", "english_target_script_mismatch")
    elif target_language == "chinese":
        if target_scripts["kana_hangul"]:
            return QualityDecision("quarantine", "chinese_target_script_mismatch")
        if target_scripts["cjk"] == 0 and target_scripts["latin"] > 0:
            return QualityDecision("quarantine", "chinese_target_without_han")

    source_key = letters_and_marks(source)
    target_key = letters_and_marks(target)
    if len(source_key) >= 4 and source_key == target_key:
        return QualityDecision("quarantine", "source_target_identity")

    fertility = fertility_reason(source, target, target_language, row_type)
    if fertility:
        return QualityDecision("quarantine", fertility)
    if str(row.get("contains_unclear", "")).strip().lower() in {"1", "true", "yes"}:
        flags.append("contains_unclear")
    if SPEAKER_PREFIX_RE.match(source) or SPEAKER_PREFIX_RE.match(target):
        flags.append("speaker_prefix_preserved")
    if row_type == "unknown":
        flags.append("unknown_row_type")
    return QualityDecision("accepted", "", tuple(flags))


def normalize_dataframe(
    frame: pd.DataFrame,
    source_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, Counter[str]]:
    output = frame.copy()
    transformations: Counter[str] = Counter()
    for column, raw_column, ledger_column in (
        (source_column, "formosan_raw", "formosan_transformations"),
        (target_column, "target_raw", "target_transformations"),
    ):
        output[raw_column] = output[column].astype(str)
        normalized: list[str] = []
        ledgers: list[str] = []
        for value in output[column].tolist():
            result = normalize_text(value)
            normalized.append(result.text)
            ledgers.append("|".join(result.transformations))
            transformations.update(result.transformations)
        output[column] = normalized
        output[ledger_column] = ledgers
    output["row_type"] = [
        normalized_row_type(row_type, source)
        for row_type, source in zip(
            output.get("row_type", pd.Series([""] * len(output))),
            output.get("source", pd.Series([""] * len(output))),
            strict=True,
        )
    ]
    return output, transformations


def apply_quality_rules(
    frame: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    target_language: str,
    keep_redactions: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, Counter[str]]:
    dispositions: list[str] = []
    reasons: list[str] = []
    flags: list[str] = []
    counts: Counter[str] = Counter()
    decision_columns = list(
        dict.fromkeys(
            [
                source_column,
                target_column,
                "row_type",
                "kindOf",
                "standard_namespace",
                "mt_normalization_status",
                "mt_normalization_reason",
                "formosan_mt_standard",
                "contains_unclear",
                "translation_kind",
                "xml_unit_context",
            ]
        )
    )
    decision_frame = frame.reindex(columns=decision_columns, fill_value="")
    for values in decision_frame.itertuples(index=False, name=None):
        decision = quality_decision(
            dict(zip(decision_columns, values, strict=True)),
            source_column=source_column,
            target_column=target_column,
            target_language=target_language,
            keep_redactions=keep_redactions,
        )
        dispositions.append(decision.disposition)
        reasons.append(decision.reason)
        flags.append("|".join(decision.flags))
        counts[f"{decision.disposition}:{decision.reason or 'ok'}"] += 1

    work = frame.copy()
    work["quality_flags"] = flags
    work["disposition"] = dispositions
    work["disposition_reason"] = reasons
    accepted = work[work["disposition"].eq("accepted")].copy()
    rejected = work[~work["disposition"].eq("accepted")].copy()
    return accepted, rejected, counts


def deduplicate_pairs(
    frame: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame.copy(), frame.iloc[0:0].copy()
    work = frame.copy()
    work["_pair_key"] = [
        f"{exact_key(source)}\u241f{exact_key(target)}"
        for source, target in zip(
            work[source_column],
            work[target_column],
            strict=True,
        )
    ]
    priority = {"sentence": 0, "lexeme": 1, "morpheme": 2, "unknown": 3}
    work["_priority"] = work["row_type"].map(priority).fillna(4)
    work["_order"] = range(len(work))
    work = work.sort_values(["_pair_key", "_priority", "_order"], kind="stable")
    canonical_ids = work.groupby("_pair_key", sort=False)["row_id"].first().to_dict()
    group_sizes = work.groupby("_pair_key", sort=False).size().to_dict()
    duplicate_mask = work.duplicated("_pair_key", keep="first")

    duplicates = work[duplicate_mask].copy()
    duplicates["disposition"] = "deduplicated"
    duplicates["disposition_reason"] = "duplicate_pair"
    duplicates["canonical_row_id"] = duplicates["_pair_key"].map(canonical_ids)

    accepted = work[~duplicate_mask].copy()
    accepted["duplicate_group_size"] = accepted["_pair_key"].map(group_sizes).astype(int)
    accepted["pair_fingerprint"] = accepted["_pair_key"].map(hashlib_sha256)
    drop_columns = ["_pair_key", "_priority", "_order"]
    return (
        accepted.sort_values("_order", kind="stable").drop(columns=drop_columns).reset_index(drop=True),
        duplicates.drop(columns=drop_columns).sort_index(kind="stable"),
    )


def hashlib_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def reason_counts(rows: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in rows).items()))


def compact_provenance(row: pd.Series) -> str:
    fields = {
        key: row.get(key, "")
        for key in (
            "row_id",
            "source_record_id",
            "repository",
            "repository_commit",
            "xml_path",
            "xml_id",
            "qc_final_xml_id",
            "target_lang",
            "translation_index",
        )
    }
    return json.dumps(fields, ensure_ascii=False, sort_keys=True)
