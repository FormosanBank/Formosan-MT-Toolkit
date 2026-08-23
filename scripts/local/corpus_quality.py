"""Conservative, language-aware quality rules for Formosan MT pairs."""

from __future__ import annotations

import hashlib
import html
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
EMBEDDED_MISSING_TRANSLATION_RE = re.compile(
    r"(?:[\[\u3010(\uff08]\s*)?(?:translation\s+(?:missing|unavailable|not\s+available)|"
    r"missing\s+translation|no\s+translation|\u7121\u7ffb\u8b6f|\u7f3a\u7ffb\u8b6f)"
    r"(?:\s*[\]\u3011)\uff09])?",
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
LEXICAL_GLOSS_TAGS = GLOSS_TAGS | frozenset({"DEIC", "EN", "EN1", "EN2", "NEUT", "NM", "REFL", "SA", "STIM", "THM"})
GLOSS_PART_RE = re.compile(r"[.=_-]+")
GLOSS_CHAIN_RE = re.compile(r"[=_-]+")
PERSON_NUMBER_GLOSS_RE = re.compile(r"^[123](?:SG|PL)(?:INCL|EXCL)?$")
GENERIC_GLOSS_CODE_RE = re.compile(r"^[A-Z][A-Z0-9]{1,9}$")
MIXED_CASE_GLOSS_CODE_RE = re.compile(r"^(?=.{2,8}$)(?=.*[a-z])(?=(?:[^A-Z]*[A-Z]){2})[A-Za-z][A-Za-z0-9]*$")
LEXICAL_STRONG_BOUNDARY_RE = re.compile(r"\S[=_]\S")
LEXICAL_AMBIGUOUS_BOUNDARY_RE = re.compile(r"\S[-/]\S")
GLOSS_TOKEN_EDGE_PUNCTUATION = "()[]{}<>,;:!?\"“”‘’'"
TERMINAL_PUNCTUATION = ".!?。！？"
TRAILING_CLOSERS = "\"'”’)]}》」』"
ENGLISH_TITLE_WORD_RE = re.compile(r"^[A-Z][A-Za-z'’-]*$")
CHINESE_LATIN_GLOSS_BOUNDARY_RE = re.compile(r"(?:[\u3400-\u9FFF][=_-][A-Za-z]|[A-Za-z][=_-][\u3400-\u9FFF])")
CHINESE_STRONG_GLOSS_BOUNDARY_RE = re.compile(r"[\u3400-\u9FFF]=[\u3400-\u9FFF]")
CHINESE_SHORT_GLOSS_BOUNDARY_RE = re.compile(r"[\u3400-\u9FFF]-[\u3400-\u9FFF]")
ENGLISH_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?")
ENGLISH_ANCHOR_WORDS = frozenset(
    """
    a an and are as at be because been being but by can could did do does for
    from had has have he her here hers him his how i if in into is it its may
    me might must my no nor not of on or our ours she should so than that the
    their theirs them then there these they this those through to under up us
    was we were what when where which who whom whose why will with would yes
    you your yours
    """.split()
)
FORMOSAN_SPECIFIC_TARGET_RE = re.compile(
    r"[ɐɑɒɓɔɕɖɗəɛɜɣɨɪɬɮɯɲŋɳɴɾʀʂʃʈʔʉʋʐʑʒθχʰ]",
    re.IGNORECASE,
)
MALFORMED_ESCAPE_RE = re.compile(r"\\+(?=[A-Za-z\"“”])")
REPEATED_CLOSING_QUOTE_RE = re.compile(r"[\"”]{2,}\s*$")
ENGLISH_GRAMMAR_PHRASE_RE = re.compile(
    r"\b(?:case\s+marker|"
    r"(?:nominative|oblique|genitive|accusative|ergative)\s+(?:case|marker)|"
    r"(?:actor|agent|patient|object|subject|locative)\s+focus)\b",
    re.IGNORECASE,
)
ENGLISH_ROOT_ANALYSIS_RE = re.compile(r"\bthe\s+root\s+is\b", re.IGNORECASE)
ENGLISH_GLOSS_CODE_RE = re.compile(
    r"\b(?:" + "|".join(sorted(map(re.escape, GLOSS_TAGS), key=len, reverse=True)) + r")\b"
)
MORPHEME_ANALYSIS_RE = re.compile(r"(?<!\w)[A-Za-z'’()]+(?:-[A-Za-z'’()]+)+\s*:")
CHINESE_GRAMMAR_NOTE_RE = re.compile(
    r"[（(][^）)\n]{0,300}(?:"
    r"主事焦點|受事焦點|處所焦點|參考焦點|工具焦點|焦點句|"
    r"主格標記|斜格標記|屬格標記|關係子句|"
    r"主事者|受事者|受動者|使動者|受役者|"
    r"動詞\s*[-－—]|語法分析|詞根(?:是|為)"
    r")"
)
CHINESE_DIRECT_GRAMMAR_NOTE_RE = re.compile(
    r"(?:\u8a9e\u5f59\u4e2d\u7684[^\n]{0,100}\u8a5e\u6839(?:\u662f|\u70ba)|"
    r"\u8a5e\u6839(?:\u662f|\u70ba)\s*[A-Za-z'\u2019]|"
    r"(?:\u4e3b\u4e8b|\u53d7\u4e8b|\u8655\u6240|\u53c3\u8003|\u5de5\u5177)\u7126\u9ede|"
    r"(?:\u4e3b\u683c|\u659c\u683c|\u5c6c\u683c)\u6a19\u8a18|\u8a9e\u6cd5\u5206\u6790)"
)
TARGET_PROMPT_PREFIX_RE = re.compile(
    r"^\s*(?:source|target|translation|reference|english|chinese|formosan)\s*:\s*",
    re.IGNORECASE,
)
TARGET_PROVENANCE_NOTE_RE = re.compile(
    r"(?:^|[\s(\uff08\[])(?:source|sources|citation|reference|references|note|"
    r"author|book\s+title)\s*:\s*\S|"
    r"(?:^|[\[(\uff08])\s*(?:(?:this|the)\s+(?:passage|article|text|story)\s+"
    r"(?:is|was)\s+)?(?:excerpted\s+and\s+)?(?:adapted|retold)\s+from\s+\S|"
    r"(?:\u8cc7\u6599\u4f86\u6e90|\u4f86\u6e90|\u51fa\u8655)\s*[:\uff1a]\s*\S",
    re.IGNORECASE,
)
TARGET_TRANSLATION_COMMENTARY_RE = re.compile(
    r"\b(?:(?:literal|word[- ]for[- ]word|free)\s+translation|"
    r"literal\s+meaning|translation\s+note)\s*:\s*\S|"
    r"(?:\u76f4\u8b6f|\u9010\u5b57\u7ffb\u8b6f|\u610f\u8b6f|\u7ffb\u8b6f\u8aaa\u660e)\s*[:\uff1a]\s*\S",
    re.IGNORECASE,
)
NUMBERED_REFERENCE_RE = re.compile(r"(?:^|\s)\(?([1-9]\d?)\)?[.)\u3001:\uff1a]\s*")
VOCABULARY_MAPPING_RE = re.compile(
    r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f'\u2019-]{0,30}\s*"
    r"[\uff08(][^\uff09)\n]{0,20}[\u3400-\u9fff][^\uff09)\n]{0,20}[\uff09)]"
)
DELIMITER_PAIRS = (
    ("(", ")"),
    ("[", "]"),
    ("{", "}"),
    ("（", "）"),
    ("【", "】"),
)
FORMOSAN_INITIAL_CLUSTERS = (
    "cm",
    "cn",
    "kn",
    "mn",
    "mq",
    "nq",
    "pd",
    "pq",
    "spg",
    "tk",
    "tq",
)
COMMON_ENGLISH_INITIAL_CLUSTERS = frozenset({"chr", "phr", "sch", "scr", "shr", "spl", "spr", "squ", "str", "thr"})


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
    return sum(
        any(unicodedata.category(character)[0] in {"L", "N", "M"} for character in token) for token in value.split()
    )


def target_units(value: str, target_language: str) -> int:
    if target_language == "chinese":
        cjk = len(CJK_RE.findall(value))
        non_cjk = CJK_RE.sub(" ", value)
        return cjk + token_count(non_cjk)
    return token_count(value)


def has_terminal_punctuation(value: str) -> bool:
    text = value.rstrip().rstrip(TRAILING_CLOSERS).rstrip()
    return bool(text) and text[-1] in TERMINAL_PUNCTUATION


def normalized_row_type(value: object) -> str:
    existing = str(value or "").strip().lower()
    if existing in {"sentence", "lexeme", "morpheme"}:
        return existing
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
    long_gloss_chain = False
    tokens = value.split()
    for raw_token in tokens:
        token = raw_token.strip(GLOSS_TOKEN_EDGE_PUNCTUATION)
        has_boundary = GLOSS_PART_RE.search(token) is not None
        has_morphological_boundary = GLOSS_CHAIN_RE.search(token) is not None
        parts = [part for part in GLOSS_PART_RE.split(token) if part]
        token_hits = sum(
            is_annotation_gloss_code(
                part,
                allow_mixed_case=has_morphological_boundary,
            )
            for part in parts
        )
        long_gloss_chain = long_gloss_chain or len(GLOSS_CHAIN_RE.findall(token)) >= 4
        if not token_hits:
            continue
        gloss_hits += token_hits
        if has_boundary:
            separated_tokens += 1

    return (
        long_gloss_chain
        or separated_tokens >= 2
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
    text = str(value)
    if TARGET_PROMPT_PREFIX_RE.match(text):
        return "target_prompt_scaffolding"
    if has_annotation_gloss_structure(text):
        return "target_annotation_gloss"
    if has_appended_linguistic_analysis(text, target_language=target_language):
        return "target_linguistic_analysis"
    if target_language == "chinese":
        if (
            CHINESE_LATIN_GLOSS_BOUNDARY_RE.search(text)
            or CHINESE_STRONG_GLOSS_BOUNDARY_RE.search(text)
            or (
                target_units(text, "chinese") <= 6
                and not has_terminal_punctuation(text)
                and CHINESE_SHORT_GLOSS_BOUNDARY_RE.search(text)
            )
        ):
            return "target_annotation_gloss"
    return ""


def has_numbered_sequence(value: object) -> bool:
    numbered = [int(item) for item in NUMBERED_REFERENCE_RE.findall(str(value))]
    return any(second == first + 1 for first, second in zip(numbered, numbered[1:], strict=False))


def target_metadata_reason(value: object, *, source: object = "") -> str:
    """Detect reference metadata and multi-sense entries embedded in targets."""
    text = str(value).strip()
    if EMBEDDED_MISSING_TRANSLATION_RE.search(text):
        return "embedded_missing_translation_marker"
    if TARGET_PROVENANCE_NOTE_RE.search(text):
        return "target_provenance_note"
    if TARGET_TRANSLATION_COMMENTARY_RE.search(text):
        return "target_translation_commentary"
    if has_numbered_sequence(text) and not has_numbered_sequence(source) and token_count(str(source)) <= 12:
        return "target_numbered_multi_reference"
    return ""


def has_appended_linguistic_analysis(value: str, *, target_language: str) -> bool:
    """Detect grammatical analysis appended to an otherwise free translation."""
    if target_language == "chinese":
        return bool(
            CHINESE_GRAMMAR_NOTE_RE.search(value) or CHINESE_DIRECT_GRAMMAR_NOTE_RE.search(value) or "=" in value
        )
    if target_language != "english":
        return False

    gloss_codes = ENGLISH_GLOSS_CODE_RE.findall(value)
    return bool(
        ENGLISH_GRAMMAR_PHRASE_RE.search(value)
        or len(gloss_codes) >= 2
        or (gloss_codes and MORPHEME_ANALYSIS_RE.search(value))
        or (ENGLISH_ROOT_ANALYSIS_RE.search(value) and (gloss_codes or MORPHEME_ANALYSIS_RE.search(value)))
        or len(MORPHEME_ANALYSIS_RE.findall(value)) >= 2
    )


def has_malformed_escaping(value: str) -> bool:
    """Detect literal escape debris and duplicated closing quotation marks."""
    return bool(MALFORMED_ESCAPE_RE.search(value) or REPEATED_CLOSING_QUOTE_RE.search(value))


def has_unbalanced_target_delimiters(value: str) -> bool:
    """Detect delimiter imbalance without treating it as automatic data loss."""
    if any(value.count(opening) != value.count(closing) for opening, closing in DELIMITER_PAIRS):
        return True
    return value.count('"') % 2 == 1 or value.count("“") != value.count("”")


def english_language_quality(value: str) -> tuple[str, tuple[str, ...]]:
    """Return a quarantine reason or train-only flag for doubtful English."""
    if CJK_RE.search(value) or KANA_HANGUL_RE.search(value):
        return "english_target_script_mismatch", ()
    words = [match.group(0).casefold() for match in ENGLISH_WORD_RE.finditer(value)]
    anchor_candidates = {re.split(r"['’]", word, maxsplit=1)[0] for word in words}
    if len(words) < 4 or ENGLISH_ANCHOR_WORDS.intersection(anchor_candidates):
        return "", ()
    orthography_score = 0
    for word in dict.fromkeys(words):
        if re.search(r"q(?!u)|q$", word):
            orthography_score += 1
        if word.startswith(FORMOSAN_INITIAL_CLUSTERS):
            orthography_score += 1
        initial_cluster = re.match(r"^[bcdfghjklmnpqrstvwxz]{3,}", word)
        if initial_cluster and initial_cluster.group(0)[:3] not in COMMON_ENGLISH_INITIAL_CLUSTERS:
            orthography_score += 1
        if "'" in word or "’" in word:
            if not re.search(r"(?:'s|'t|'re|'ve|'ll|'d|'m|’s|’t|’re|’ve|’ll|’d|’m)$", word):
                orthography_score += 1
    if orthography_score >= 2 or (
        FORMOSAN_SPECIFIC_TARGET_RE.search(value) and has_unbalanced_target_delimiters(value)
    ):
        return "english_target_language_mismatch", ()
    return "", ("english_language_uncertain",)


def target_alignment_artifact_reason(
    source: str,
    target: str,
    *,
    target_language: str,
) -> str:
    """Detect source text copied in front of a free target translation."""
    if target_language != "chinese":
        return ""
    first_han = CJK_RE.search(target)
    if first_han is None:
        return ""
    prefix = target[: first_han.start()].strip()
    if not prefix or not LATIN_RE.search(prefix):
        return ""
    normalized_prefix = letters_and_marks(prefix)
    if len(normalized_prefix) < 4 or normalized_prefix not in letters_and_marks(source):
        return ""
    if re.search(r"[.!?\u3002\uff01\uff1f]\s*$", prefix) or "=" in prefix:
        return "target_copied_source_clause"
    return ""


def has_vocabulary_mapping_list(value: str) -> bool:
    """Return true for long word-to-gloss lists serialized as sentences."""
    return len(VOCABULARY_MAPPING_RE.findall(value)) >= 6


def is_explicit_gloss_code(value: str) -> bool:
    return (
        value in LEXICAL_GLOSS_TAGS
        or PERSON_NUMBER_GLOSS_RE.fullmatch(value) is not None
        or GENERIC_GLOSS_CODE_RE.fullmatch(value) is not None
    )


def is_annotation_gloss_code(
    value: str,
    *,
    allow_mixed_case: bool,
) -> bool:
    return (
        value in LEXICAL_GLOSS_TAGS
        or PERSON_NUMBER_GLOSS_RE.fullmatch(value) is not None
        or (allow_mixed_case and MIXED_CASE_GLOSS_CODE_RE.fullmatch(value) is not None)
    )


def is_gloss_code(value: str) -> bool:
    return is_explicit_gloss_code(value) or MIXED_CASE_GLOSS_CODE_RE.fullmatch(value) is not None


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


def model_length_reason(
    source: str,
    target: str,
    *,
    target_language: str,
    max_units_per_side: int,
) -> str:
    """Reject pairs outside the conservative pretokenization length bound."""
    if token_count(source) > max_units_per_side:
        return "formosan_model_length_overflow"
    if target_units(target, target_language) > max_units_per_side:
        return "target_model_length_overflow"
    return ""


def is_english_heading(value: str) -> bool:
    """Return true for short title-case labels, not ordinary punctuated text."""
    text = value.strip()
    if not text or has_terminal_punctuation(text):
        return False
    words = [word.strip("()[]{}<>,;:\"“”‘’'") for word in text.split()]
    words = [word for word in words if word]
    return 2 <= len(words) <= 4 and all(ENGLISH_TITLE_WORD_RE.fullmatch(word) is not None for word in words)


def alignment_quality(
    source: str,
    target: str,
    *,
    target_language: str,
    row_type: str,
) -> tuple[str, tuple[str, ...]]:
    """Find high-confidence alignment failures and risky explanatory rows."""
    if row_type != "sentence":
        return "", ()

    source_units = token_count(source)
    target_count = target_units(target, target_language)
    if (
        (source_units >= 10 and target_count <= 2)
        or (source_units >= 12 and target_count <= 3)
        or (source_units >= 18 and target_count <= 5)
    ):
        return "obvious_alignment_mismatch", ()
    heading_like_target = target_language == "english" and target_count <= 3 and is_english_heading(target)
    if source_units >= 7 and heading_like_target:
        return "target_heading_alignment_mismatch", ()

    flags: list[str] = []
    if target_language == "chinese" and has_vocabulary_mapping_list(target):
        flags.append("lexical_content_sentence")
    if source_units >= 4 and heading_like_target:
        flags.append("heading_like_target")
    target_is_punctuated = has_terminal_punctuation(target)
    if target_language == "english" and source_units >= 4 and target_count <= 3 and not target_is_punctuated:
        flags.append("target_fragment")
    first_target_letter = next(
        (character for character in target if unicodedata.category(character)[0] == "L"),
        "",
    )
    english_lexical_evidence = target_language == "english" and (
        ";" in target or target.lstrip().startswith("(") or (first_target_letter and first_target_letter.islower())
    )
    if source_units <= 4 and english_lexical_evidence:
        flags.append("lexical_content_sentence")
    if source_units <= 3 and target_count >= 12:
        flags.append("length_asymmetry")
    explanatory_markers = (
        (";" in target or "cf." in target.casefold() or target.count("(") >= 1)
        if target_language == "english"
        else ("；" in target or ";" in target or target.count("（") + target.count("(") >= 1)
    )
    if source_units <= 3 and target_count >= 8 and explanatory_markers:
        flags.append("definition_like_sentence")
    return "", tuple(flags)


def quality_decision(
    row: Mapping[str, object],
    *,
    source_column: str,
    target_column: str,
    target_language: str,
    keep_redactions: bool,
    max_units_per_side: int | None = None,
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
    if normalized_translation_kind(row.get("translation_kind", "")) in NON_TRANSLATION_KINDS:
        return QualityDecision("rejected", "target_gloss_translation")
    lexical_reason = lexical_quality_reason(
        target,
        row_type=row_type,
        xml_unit_context=row.get("xml_unit_context", ""),
        target_language=target_language,
    )
    if lexical_reason:
        return QualityDecision("quarantine", lexical_reason)
    gloss_reason = target_gloss_reason(
        target,
        translation_kind="",
        target_language=target_language,
    )
    if gloss_reason:
        return QualityDecision("quarantine", gloss_reason)
    metadata_reason = target_metadata_reason(target, source=source)
    if metadata_reason:
        disposition = "rejected" if metadata_reason == "embedded_missing_translation_marker" else "quarantine"
        return QualityDecision(disposition, metadata_reason)
    if has_malformed_escaping(source):
        return QualityDecision("quarantine", "malformed_source_escaping")
    if has_malformed_escaping(target):
        return QualityDecision("quarantine", "malformed_target_escaping")
    if MISSING_TRANSLATION_RE.match(source) or MISSING_TRANSLATION_RE.match(target):
        return QualityDecision("rejected", "missing_translation_marker")
    if is_only_punctuation_or_symbols(source) or is_only_punctuation_or_symbols(target):
        return QualityDecision("rejected", "empty_or_punctuation_only")
    if max_units_per_side is not None:
        length_reason = model_length_reason(
            source,
            target,
            target_language=target_language,
            max_units_per_side=max_units_per_side,
        )
        if length_reason:
            return QualityDecision("quarantine", length_reason)
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
        language_reason, language_flags = english_language_quality(target)
        if language_reason:
            return QualityDecision("quarantine", language_reason)
        flags.extend(language_flags)
    elif target_language == "chinese":
        if target_scripts["kana_hangul"]:
            return QualityDecision("quarantine", "chinese_target_script_mismatch")
        if target_scripts["cjk"] == 0 and target_scripts["latin"] > 0:
            return QualityDecision("quarantine", "chinese_target_without_han")

    source_key = letters_and_marks(source)
    target_key = letters_and_marks(target)
    if len(source_key) >= 4 and source_key == target_key:
        return QualityDecision("quarantine", "source_target_identity")

    alignment_artifact = target_alignment_artifact_reason(
        source,
        target,
        target_language=target_language,
    )
    if alignment_artifact:
        return QualityDecision("quarantine", alignment_artifact)

    alignment_reason, alignment_flags = alignment_quality(
        source,
        target,
        target_language=target_language,
        row_type=row_type,
    )
    if alignment_reason:
        return QualityDecision("quarantine", alignment_reason)
    flags.extend(alignment_flags)
    if has_unbalanced_target_delimiters(target):
        flags.append("unbalanced_target_delimiters")
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
        normalized_row_type(row_type) for row_type in output.get("row_type", pd.Series([""] * len(output)))
    ]
    return output, transformations


def apply_quality_rules(
    frame: pd.DataFrame,
    *,
    source_column: str,
    target_column: str,
    target_language: str,
    keep_redactions: bool,
    max_units_per_side: int | None = None,
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
            max_units_per_side=max_units_per_side,
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
