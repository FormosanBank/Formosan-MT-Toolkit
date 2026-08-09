"""Deterministic, source-aware Formosan standardization for MT corpora."""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_PATH = PROJECT_ROOT / "config" / "mt_standardization.json"

CONTROL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
FORMAT_CHARACTERS = frozenset("\u200b\u200c\u200d\ufeff")
WHITESPACE_RE = re.compile(r"\s+")
WIKI_HEADING_RE = re.compile(r"(^|\s)==+\s*([^=\n]+?)\s*==+(?=\s|$)")
ANGLE_CONTENT_RE = re.compile(r"<([^<>\s]{1,24})>")
BRACE_CONTENT_RE = re.compile(r"\{([^{}\s]{1,24})}")
OPTIONAL_CONTENT_RE = re.compile(r"\(([^()\s]{1,16})\)")
SIMPLE_ALTERNATIVE_RE = re.compile(r"(?<!\S)([^\s/]+)\s*/\s*([^\s/]+)(?!\S)")
SPACED_TILDE_RE = re.compile(r"(?<!\S)([^\s~]+)\s+~\s+([^\s~]+)(?!\S)")
SPEAKER_PREFIX_RE = re.compile(r"^([A-Z][A-Za-z0-9_. -]{0,31}):\s+")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
SUPERSCRIPT_FOOTNOTE_RE = re.compile(r"(?<=\D)[⁰¹²³⁴⁵⁶⁷⁸⁹]+(?=\s|$)")
EXERCISE_BLANK_RE = re.compile(r"_{2,}")


@dataclass(frozen=True)
class StandardizationContext:
    language: str
    row_type: str
    repository: str
    xml_path: str


@dataclass(frozen=True)
class StandardizationResult:
    text: str
    status: str
    confidence: str
    eval_eligible: bool
    transformations: tuple[dict[str, Any], ...]
    unresolved_markers: tuple[str, ...]
    reason: str = ""
    speaker_label: str = ""


def load_profile(path: Path = DEFAULT_PROFILE_PATH) -> dict[str, Any]:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot load MT standardization profile {path}: {exc}") from exc
    if profile.get("schema_version") != 1:
        raise SystemExit(f"Unsupported MT standardization profile schema: {profile.get('schema_version')}")
    if not str(profile.get("profile_id") or "").strip():
        raise SystemExit(f"MT standardization profile has no profile_id: {path}")
    if profile.get("unicode_normalization") != "NFC":
        raise SystemExit("MT standardization currently requires NFC")
    expected_implementation = str(
        profile.get("implementation_sha256") or ""
    )
    actual_implementation = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    if expected_implementation != actual_implementation:
        raise SystemExit(
            "MT standardization implementation hash mismatch: "
            f"profile={expected_implementation or '<missing>'}, "
            f"loaded={actual_implementation}"
        )
    if not isinstance(profile.get("policy"), dict):
        raise SystemExit(f"MT standardization profile has no policy mapping: {path}")
    if not isinstance(profile.get("source_overrides", []), list):
        raise SystemExit(f"MT standardization source_overrides must be a list: {path}")
    return profile


def profile_sha256(path: Path = DEFAULT_PROFILE_PATH) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_orthographic_character(character: str) -> bool:
    if not character:
        return False
    return unicodedata.category(character)[0] in {"L", "M", "N"} or character in {"'", "’", "ʼ"}


def insertion_content_is_safe(value: str) -> bool:
    return bool(value) and all(is_orthographic_character(character) for character in value)


def source_matches(selector: dict[str, Any], context: StandardizationContext) -> bool:
    for key, actual in (
        ("language", context.language),
        ("row_type", context.row_type),
        ("repository", context.repository),
    ):
        expected = str(selector.get(key) or "*")
        if expected not in {"*", actual}:
            return False
    path_pattern = str(selector.get("path_regex") or "")
    return not path_pattern or re.search(path_pattern, context.xml_path) is not None


def effective_policy(profile: dict[str, Any], context: StandardizationContext) -> tuple[dict[str, Any], bool]:
    policy = dict(profile["policy"])
    reviewed_ambiguous = False
    for override in profile.get("source_overrides", []):
        if not isinstance(override, dict) or not source_matches(override, context):
            continue
        values = override.get("policy", {})
        if not isinstance(values, dict):
            raise ValueError(f"Source override policy must be a mapping: {override}")
        policy.update(values)
        reviewed_ambiguous = reviewed_ambiguous or bool(override.get("reviewed_ambiguous"))
    return policy, reviewed_ambiguous


class _StandardizationRun:
    def __init__(self, text: str) -> None:
        self.text = text
        self.transformations: list[dict[str, Any]] = []
        self.ambiguous = False
        self.reason = ""
        self.speaker_label = ""

    def replace(self, rule: str, new_text: str, *, count: int = 1, ambiguous: bool = False) -> None:
        if new_text == self.text:
            return
        self.text = new_text
        self.transformations.append({"rule": rule, "count": int(max(count, 1))})
        self.ambiguous = self.ambiguous or ambiguous

    def regex_sub(
        self,
        rule: str,
        pattern: re.Pattern[str],
        replacement: str | Any,
        *,
        ambiguous: bool = False,
    ) -> None:
        new_text, count = pattern.subn(replacement, self.text)
        if count:
            self.replace(rule, new_text, count=count, ambiguous=ambiguous)


def unwrap_infixes(run: _StandardizationRun, pattern: re.Pattern[str], rule: str) -> None:
    count = 0

    def replacement(match: re.Match[str]) -> str:
        nonlocal count
        content = match.group(1)
        if not insertion_content_is_safe(content):
            return match.group(0)
        count += 1
        return content

    new_text = pattern.sub(replacement, run.text)
    if count:
        run.replace(rule, new_text, count=count)


def include_optional_segments(run: _StandardizationRun) -> None:
    original = run.text
    working = original
    total = 0
    while True:
        count = 0

        def replacement(match: re.Match[str]) -> str:
            nonlocal count
            start, end = match.span()
            previous = working[start - 1] if start else ""
            following = working[end] if end < len(working) else ""
            content = match.group(1)
            previous_is_boundary = (
                is_orthographic_character(previous)
                or previous in {"-", "=", "+", "~"}
            )
            following_is_boundary = (
                is_orthographic_character(following)
                or following in {"-", "=", "+", "~"}
            )
            if not (
                insertion_content_is_safe(content)
                and previous_is_boundary
                and following_is_boundary
            ):
                return match.group(0)
            count += 1
            return content

        updated = OPTIONAL_CONTENT_RE.sub(replacement, working)
        if not count:
            break
        total += count
        working = updated
    if total:
        run.replace(
            "include_optional_intraword_segment",
            working,
            count=total,
            ambiguous=True,
        )


def remove_boundary_character(run: _StandardizationRun, marker: str, rule: str) -> None:
    output: list[str] = []
    removed = 0
    index = 0
    while index < len(run.text):
        character = run.text[index]
        if character != marker:
            output.append(character)
            index += 1
            continue
        end = index + 1
        while end < len(run.text) and run.text[end] == marker:
            end += 1
        previous = run.text[index - 1] if index else ""
        following = run.text[end] if end < len(run.text) else ""
        if (is_orthographic_character(previous) or is_orthographic_character(following)) and not (
            previous.isdigit() and following.isdigit()
        ):
            removed += end - index
            index = end
            continue
        output.append(run.text[index:end])
        index = end
    if removed:
        run.replace(rule, "".join(output), count=removed)


def normalize_morphological_boundaries(
    run: _StandardizationRun,
    policy: dict[str, Any],
) -> None:
    while True:
        before = run.text
        if policy.get("remove_clitic_boundaries") and "=" in run.text:
            count = run.text.count("=")
            run.replace(
                "remove_clitic_boundary",
                run.text.replace("=", ""),
                count=count,
            )
        if policy.get("remove_intraword_hyphens"):
            remove_boundary_character(
                run,
                "-",
                "remove_hyphen_boundary",
            )
        if policy.get("remove_intraword_tildes"):
            remove_boundary_character(
                run,
                "~",
                "remove_tilde_boundary",
            )
        if policy.get("remove_intraword_pluses"):
            remove_boundary_character(
                run,
                "+",
                "remove_plus_boundary",
            )
        if policy.get("include_optional_intraword_segments"):
            include_optional_segments(run)
        if run.text == before:
            return
        if len(run.text) >= len(before):
            raise ValueError(
                "Morphological normalization did not strictly reduce text"
            )


def select_simple_alternatives(run: _StandardizationRun) -> None:
    if URL_RE.search(run.text):
        run.reason = "url_in_formosan_standard"
        return
    slash_count = run.text.count("/")
    if not slash_count:
        return
    if slash_count > 4:
        run.reason = "complex_slash_notation"
        return
    working = run.text
    total = 0
    while True:
        working, count = SIMPLE_ALTERNATIVE_RE.subn(lambda match: match.group(1), working)
        total += count
        if not count:
            break
    if total:
        run.replace("select_first_slash_alternative", working, count=total, ambiguous=True)
    if "/" in run.text:
        run.reason = "unresolved_slash_notation"


def select_tilde_alternatives(run: _StandardizationRun) -> None:
    working = run.text
    total = 0
    while True:
        working, count = SPACED_TILDE_RE.subn(lambda match: match.group(1), working)
        total += count
        if not count:
            break
    if total:
        run.replace("select_first_tilde_alternative", working, count=total, ambiguous=True)


def unresolved_markers(text: str, configured: Iterable[str]) -> tuple[str, ...]:
    markers = {marker for marker in configured if marker and marker in text}
    if EXERCISE_BLANK_RE.search(text):
        markers.add("exercise_blank")
    return tuple(sorted(markers))


def standardize_text(
    value: object,
    *,
    context: StandardizationContext,
    profile: dict[str, Any],
    contains_unclear: bool = False,
) -> StandardizationResult:
    raw = "" if value is None else str(value)
    if contains_unclear:
        return StandardizationResult(
            text="",
            status="ineligible",
            confidence="none",
            eval_eligible=False,
            transformations=(),
            unresolved_markers=(),
            reason="contains_unclear",
        )
    if not raw.strip():
        return StandardizationResult(
            text="",
            status="ineligible",
            confidence="none",
            eval_eligible=False,
            transformations=(),
            unresolved_markers=(),
            reason="empty_source_standard",
        )

    policy, reviewed_ambiguous = effective_policy(profile, context)
    run = _StandardizationRun(raw)

    controls = CONTROL_RE.sub(" ", run.text)
    controls = "".join("" if character in FORMAT_CHARACTERS else character for character in controls)
    run.replace("remove_control_and_format_characters", controls)
    run.replace("html_unescape", html.unescape(run.text))
    run.replace("unicode_nfc", unicodedata.normalize("NFC", run.text))
    run.replace(
        "normalize_whitespace",
        WHITESPACE_RE.sub(" ", run.text).strip(),
    )

    if policy.get("strip_wiki_heading_markup"):
        run.regex_sub("strip_wiki_heading_markup", WIKI_HEADING_RE, lambda match: f"{match.group(1)}{match.group(2)}")

    if policy.get("remove_null_morphemes"):
        for marker in ("Ø-", "∅-", "-Ø", "-∅", "Ø", "∅"):
            count = run.text.count(marker)
            if count:
                run.replace("remove_null_morpheme", run.text.replace(marker, ""), count=count)

    if policy.get("unwrap_letter_infixes"):
        unwrap_infixes(run, ANGLE_CONTENT_RE, "unwrap_angle_infix")
        unwrap_infixes(run, BRACE_CONTENT_RE, "unwrap_braced_morpheme")

    # Remove annotation characters before boundary and alternative handling so
    # cleanup cannot expose notation that is processed only on a later pass.
    if "_" in run.text and not EXERCISE_BLANK_RE.search(run.text):
        count = run.text.count("_")
        run.replace("remove_annotation_underscore", run.text.replace("_", ""), count=count, ambiguous=True)
    if "\\" in run.text:
        count = run.text.count("\\")
        run.replace("remove_escape_backslash", run.text.replace("\\", ""), count=count)

    normalize_morphological_boundaries(run, policy)

    if policy.get("select_first_simple_alternative"):
        select_simple_alternatives(run)
        select_tilde_alternatives(run)

    if policy.get("strip_trailing_superscript_footnotes"):
        run.regex_sub("strip_superscript_footnote", SUPERSCRIPT_FOOTNOTE_RE, "")

    squeezed = WHITESPACE_RE.sub(" ", run.text).strip()
    run.replace("normalize_whitespace", squeezed)

    speaker = SPEAKER_PREFIX_RE.match(run.text)
    if speaker:
        run.speaker_label = speaker.group(1)

    markers = unresolved_markers(run.text, profile.get("unresolved_annotation_characters", []))
    reason = run.reason
    if markers and not reason:
        reason = "unresolved_annotation_markers"
    if not run.text or not any(unicodedata.category(character)[0] in {"L", "M", "N"} for character in run.text):
        return StandardizationResult(
            text=run.text,
            status="ineligible",
            confidence="none",
            eval_eligible=False,
            transformations=tuple(run.transformations),
            unresolved_markers=markers,
            reason="empty_or_nonlinguistic_after_standardization",
            speaker_label=run.speaker_label,
        )
    if reason or markers:
        return StandardizationResult(
            text=run.text,
            status="quarantine",
            confidence="ambiguous",
            eval_eligible=False,
            transformations=tuple(run.transformations),
            unresolved_markers=markers,
            reason=reason,
            speaker_label=run.speaker_label,
        )

    confidence = "ambiguous" if run.ambiguous else ("safe" if run.transformations else "unchanged")
    return StandardizationResult(
        text=run.text,
        status="accepted",
        confidence=confidence,
        eval_eligible=(context.row_type == "sentence" and (not run.ambiguous or reviewed_ambiguous)),
        transformations=tuple(run.transformations),
        unresolved_markers=(),
        speaker_label=run.speaker_label,
    )


def assert_idempotent(
    result: StandardizationResult,
    *,
    context: StandardizationContext,
    profile: dict[str, Any],
) -> None:
    if result.status != "accepted":
        return
    repeated = standardize_text(result.text, context=context, profile=profile)
    if (
        repeated.text != result.text
        or repeated.status != "accepted"
        or repeated.transformations
        or repeated.speaker_label != result.speaker_label
    ):
        raise ValueError(
            f"MT standardization is not idempotent for {context.repository}/{context.xml_path}: "
            f"{result.text!r} -> {repeated.text!r}"
        )
