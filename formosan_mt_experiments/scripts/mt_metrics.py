#!/usr/bin/env python3
"""Shared corpus-level generation metrics for training and evaluation."""

from __future__ import annotations

import multiprocessing
import random
from collections.abc import Sequence

try:
    from sacrebleu.metrics import BLEU, CHRF, TER

    _SACREBLEU_ERROR = None
except Exception as exc:  # pragma: no cover - exercised only in incomplete envs
    BLEU = CHRF = TER = None  # type: ignore
    _SACREBLEU_ERROR = exc


_BOOTSTRAP_HYPOTHESES: tuple[str, ...] = ()
_BOOTSTRAP_REFERENCES: tuple[str, ...] = ()
_BOOTSTRAP_LOWERCASE = False
_BOOTSTRAP_BLEU_TOKENIZE = "13a"


def _initialize_bootstrap_worker(
    hypotheses: tuple[str, ...],
    references: tuple[str, ...],
    lowercase: bool,
    bleu_tokenize: str,
) -> None:
    global _BOOTSTRAP_HYPOTHESES
    global _BOOTSTRAP_REFERENCES
    global _BOOTSTRAP_LOWERCASE
    global _BOOTSTRAP_BLEU_TOKENIZE

    _BOOTSTRAP_HYPOTHESES = hypotheses
    _BOOTSTRAP_REFERENCES = references
    _BOOTSTRAP_LOWERCASE = lowercase
    _BOOTSTRAP_BLEU_TOKENIZE = bleu_tokenize


def _score_bootstrap_indexes(indexes: list[int]) -> tuple[float, float, float]:
    metrics = score_translations(
        [_BOOTSTRAP_HYPOTHESES[index] for index in indexes],
        [_BOOTSTRAP_REFERENCES[index] for index in indexes],
        lowercase=_BOOTSTRAP_LOWERCASE,
        bleu_tokenize=_BOOTSTRAP_BLEU_TOKENIZE,
    )
    return (
        float(metrics["BLEU"]),
        float(metrics["chrF2"]),
        float(metrics["TER"]),
    )


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
    workers: int = 1,
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
    if workers <= 0:
        raise ValueError("Bootstrap workers must be positive")
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(strata):
        groups.setdefault(str(name), []).append(index)
    rng = random.Random(seed)
    values = {"BLEU": [], "chrF2": [], "TER": []}

    def sampled_indexes() -> list[int]:
        return [
            rng.choice(group)
            for group in groups.values()
            for _ in range(len(group))
        ]

    hypotheses_tuple = tuple(str(value) for value in hypotheses)
    references_tuple = tuple(str(value) for value in references)
    worker_count = min(workers, samples)
    initializer_args = (
        hypotheses_tuple,
        references_tuple,
        lowercase,
        bleu_tokenize,
    )
    indexes = (sampled_indexes() for _ in range(samples))
    if worker_count == 1:
        _initialize_bootstrap_worker(*initializer_args)
        scores = map(_score_bootstrap_indexes, indexes)
        for bleu, chrf, ter in scores:
            values["BLEU"].append(bleu)
            values["chrF2"].append(chrf)
            values["TER"].append(ter)
    else:
        start_method = (
            "fork"
            if "fork" in multiprocessing.get_all_start_methods()
            else "spawn"
        )
        context = multiprocessing.get_context(start_method)
        with context.Pool(
            processes=worker_count,
            initializer=_initialize_bootstrap_worker,
            initargs=initializer_args,
        ) as pool:
            for bleu, chrf, ter in pool.imap(
                _score_bootstrap_indexes,
                indexes,
                chunksize=1,
            ):
                values["BLEU"].append(bleu)
                values["chrF2"].append(chrf)
                values["TER"].append(ter)
    return {
        "samples": samples,
        "seed": seed,
        "workers": worker_count,
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
