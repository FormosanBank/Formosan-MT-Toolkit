#!/usr/bin/env python3
"""Shared corpus-level generation metrics for training and evaluation."""

from __future__ import annotations

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
) -> dict[str, float]:
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
    return {
        "BLEU": float(
            BLEU(tokenize=bleu_tokenize, effective_order=True, lowercase=lowercase)
            .corpus_score(hyps, [refs])
            .score
        ),
        "chrF2": float(CHRF().corpus_score(hyps, [refs]).score),
        "TER": float(TER().corpus_score(hyps, [refs]).score),
        "exact_match_rate": float(sum(hyp == ref for hyp, ref in zip(hyps, refs, strict=True)) / len(hyps)),
        "empty_output_rate": float(sum(not hyp for hyp in hyps) / len(hyps)),
        "character_length_ratio": float(hyp_chars / max(ref_chars, 1)),
    }
