#!/usr/bin/env python3
"""Apply the corpus MT standardization profile to Formosan inference input."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
LOCAL_SCRIPTS = PROJECT_ROOT / "scripts" / "local"
if LOCAL_SCRIPTS.is_dir():
    sys.path.insert(0, str(LOCAL_SCRIPTS))

from mt_standardization import (  # noqa: E402
    StandardizationContext,
    assert_idempotent,
    load_profile,
    standardize_text,
)

PACKAGED_PROFILE = SCRIPT_DIR / "mt_standardization_profile.json"
PROJECT_PROFILE = PROJECT_ROOT / "config" / "mt_standardization.json"


def default_profile_path() -> Path:
    if PACKAGED_PROFILE.is_file():
        return PACKAGED_PROFILE
    return PROJECT_PROFILE


def normalize_formosan(
    text: str,
    lang_code: str,
    *,
    row_type: str = "sentence",
    source_repository: str = "inference",
    source_path: str = "interactive",
    profile_path: Path | None = None,
) -> str:
    """Return model-ready Formosan text or reject unresolved notation."""
    path = (profile_path or default_profile_path()).expanduser().resolve()
    profile = load_profile(path)
    context = StandardizationContext(
        language=lang_code.strip().lower(),
        row_type=row_type,
        repository=source_repository,
        xml_path=source_path,
    )
    result = standardize_text(text, context=context, profile=profile)
    if result.status != "accepted":
        raise ValueError(
            "Formosan input is not safe to normalize automatically: "
            f"{result.reason or result.status}"
        )
    assert_idempotent(result, context=context, profile=profile)
    return result.text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    parser.add_argument("--lang-code", required=True)
    parser.add_argument("--row-type", default="sentence")
    parser.add_argument("--profile", type=Path)
    args = parser.parse_args()
    print(
        normalize_formosan(
            args.text,
            args.lang_code,
            row_type=args.row_type,
            profile_path=args.profile,
        )
    )


if __name__ == "__main__":
    main()
