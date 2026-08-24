"""Canonical target-specific corpus split ratios."""

from __future__ import annotations

from typing import Any

TARGET_LANGUAGES = ("english", "chinese")
SPLIT_NAMES = ("train", "validate", "test")


def target_split_ratios(splits: dict[str, Any], target_language: str) -> dict[str, float]:
    """Return the human-corpus split ratios for one parallel corpus."""
    target = str(target_language).strip().lower()
    if target not in TARGET_LANGUAGES:
        raise ValueError(f"Unsupported split target language: {target_language!r}")
    ratios = splits.get("ratios_by_target", {}).get(target)
    if not isinstance(ratios, dict):
        raise ValueError(f"Missing split ratios for {target}")
    return {name: float(ratios[name]) for name in SPLIT_NAMES}


def validate_target_split_ratios(splits: dict[str, Any]) -> None:
    """Validate the complete English and Chinese human-split contract."""
    ratios_by_target = splits.get("ratios_by_target")
    if not isinstance(ratios_by_target, dict) or set(ratios_by_target) != set(
        TARGET_LANGUAGES
    ):
        raise ValueError("Split policy must define exactly English and Chinese ratios")
    for target in TARGET_LANGUAGES:
        ratios = target_split_ratios(splits, target)
        if (
            set(ratios_by_target[target]) != set(SPLIT_NAMES)
            or not all(0 <= ratio <= 1 for ratio in ratios.values())
            or abs(sum(ratios.values()) - 1.0) > 1e-9
        ):
            raise ValueError(f"Invalid human-corpus split ratios for {target}")
