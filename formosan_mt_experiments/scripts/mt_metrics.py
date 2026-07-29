#!/usr/bin/env python3
"""Shared corpus-level generation metrics for training and evaluation."""

from __future__ import annotations

import random
from collections.abc import Sequence

try:
    from sacrebleu.metrics import BLEU, CHRF, TER

    _SACREBLEU_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
    BLEU = CHRF = TER = None  # type: ignore
    _SACREBLEU_ERROR = exc


def score_translations(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    lowercase: bool = False,
    bleu_tokenize: str = "13a",
) -> dict[str, object]:
    """Return comparable corpus MT metrics plus basic degeneration diagnostics."""
    if _SACREBLEU_ERROR is not None:
        raise RuntimeError(f"sacrebleu is required for MT metrics: {_SACREBLEU_ERROR}")
    if len(hypotheses) != len(references):
        raise ValueError("Hypothesis/reference counts do not match.")
    if not hypotheses:
        raise ValueError("Cannot score an empty corpus.")

    hyps = [str(text).strip() for text in hypotheses]
    refs = [str(text).strip() for text in references]
    hyp_chars = sum(len(text) for text in hyps)
    ref_chars = sum(len(text) for text in refs)
    bleu = BLEU(
        tokenize=bleu_tokenize,
        effective_order=True,
        lowercase=lowercase,
    )
    chrf = CHRF()
    ter = TER(
        normalized=True,
        asian_support=bleu_tokenize == "zh",
        case_sensitive=not lowercase,
    )
    bleu_score = bleu.corpus_score(hyps, [refs])
    chrf_score = chrf.corpus_score(hyps, [refs])
    ter_score = ter.corpus_score(hyps, [refs])
    return {
        "BLEU": float(bleu_score.score),
        "chrF2": float(chrf_score.score),
        "TER": float(ter_score.score),
        "exact_match_rate": float(sum(hyp == ref for hyp, ref in zip(hyps, refs, strict=True)) / len(hyps)),
        "empty_output_rate": float(sum(not hyp for hyp in hyps) / len(hyps)),
        "character_length_ratio": float(hyp_chars / max(ref_chars, 1)),
        "signatures": {
            "BLEU": str(bleu.get_signature()),
            "chrF2": str(chrf.get_signature()),
            "TER": str(ter.get_signature()),
        },
    }


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def bootstrap_confidence_intervals(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    strata: Sequence[str] | None = None,
    samples: int = 200,
    seed: int = 42,
    lowercase: bool = False,
    bleu_tokenize: str = "13a",
) -> dict[str, object]:
    """Return deterministic stratified 95% bootstrap intervals."""
    if samples <= 0:
        return {"samples": 0, "seed": seed, "metrics": {}}
    if len(hypotheses) != len(references):
        raise ValueError("Hypothesis/reference counts do not match")
    if strata is None:
        strata = ["all"] * len(hypotheses)
    if len(strata) != len(hypotheses):
        raise ValueError("Bootstrap strata count does not match outputs")
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(strata):
        groups.setdefault(str(name), []).append(index)
    rng = random.Random(seed)
    values = {"BLEU": [], "chrF2": [], "TER": []}
    for _ in range(samples):
        indexes = [
            rng.choice(group)
            for group in groups.values()
            for _ in range(len(group))
        ]
        metrics = score_translations(
            [str(hypotheses[index]) for index in indexes],
            [str(references[index]) for index in indexes],
            lowercase=lowercase,
            bleu_tokenize=bleu_tokenize,
        )
        for metric in values:
            values[metric].append(float(metrics[metric]))
    return {
        "samples": samples,
        "seed": seed,
        "stratification": "language",
        "confidence": 0.95,
        "metrics": {
            metric: {
                "lower": percentile(scores, 0.025),
                "median": percentile(scores, 0.5),
                "upper": percentile(scores, 0.975),
            }
            for metric, scores in values.items()
        },
    }
