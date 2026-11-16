"""One definition of text length for filtering, pivoting, splitting, and reports.

A unit is the smallest piece that length policy counts: a whitespace-delimited
token that contains a letter, number, or mark, or one Han character in Chinese
text. Text is counted as written. Unicode compatibility folding would split a
word at a spacing accent (``lami´``), so it is never applied here.
"""

from __future__ import annotations

import re
import unicodedata

# Unified ideographs, compatibility ideographs, and the radical blocks that
# PDF extraction substitutes for ordinary characters.
HAN_RE = re.compile(r"[⺀-⿟㐀-䶿一-鿿豈-﫿\U00020000-\U000323AF]")
TERMINAL_PUNCTUATION = ".!?。！？"
TRAILING_CLOSERS = "\"'”’)]}》」』"


def token_count(value: object) -> int:
    """Count whitespace-delimited tokens that contain a letter, number, or mark."""
    return sum(
        any(unicodedata.category(character)[0] in "LNM" for character in token)
        for token in ("" if value is None else str(value)).split()
    )


def target_units(value: object, target_language: str) -> int:
    """Count Han characters individually in Chinese; count tokens otherwise."""
    text = "" if value is None else str(value)
    if target_language != "chinese":
        return token_count(text)
    remainder, han_characters = HAN_RE.subn(" ", text)
    return han_characters + token_count(remainder)


def has_terminal_punctuation(value: object) -> bool:
    """Return true when text ends a sentence, ignoring closing quotes and brackets."""
    text = ("" if value is None else str(value)).rstrip().rstrip(TRAILING_CLOSERS).rstrip()
    return bool(text) and text[-1] in TERMINAL_PUNCTUATION
