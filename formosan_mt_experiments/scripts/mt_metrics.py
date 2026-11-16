#!/usr/bin/env python3
"""Corpus MT metrics built on per-sentence sufficient statistics.

BLEU, chrF2, and TER are corpus-level functions of per-sentence counts.
Extracting those counts once lets every slice, bootstrap resample, and paired
system comparison sum rows instead of scoring the text again.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sacrebleu.metrics import BLEU, CHRF, TER

METRIC_NAMES = ("BLEU", "chrF2", "TER")
CONFIDENCE = 0.95
# Bounds one block of bootstrap resample counts to about 32 MB.
RESAMPLE_BLOCK_CELLS = 4_000_000


def bleu_tokenizer_for(direction: str) -> str:
    """Chinese output has no word boundaries, so it needs sacreBLEU's zh tokenizer."""
    return "zh" if str(direction).strip().lower() == "f2zh" else "13a"


@dataclass(frozen=True)
class SentenceStatistics:
    """Sufficient statistics and diagnostics for one hypothesis set.

    ``counts`` holds one row per sentence with the BLEU, chrF2, and TER count
    columns side by side. Summing rows gives the counts of any subcorpus.
    """

    counts: np.ndarray
    exact: np.ndarray
    empty: np.ndarray
    hypothesis_characters: np.ndarray
    reference_characters: np.ndarray
    source_copy: np.ndarray | None
    _metrics: dict[str, object]
    _columns: dict[str, slice]

    def __len__(self) -> int:
        return len(self.counts)

    @property
    def signatures(self) -> dict[str, str]:
        return {name: str(metric.get_signature()) for name, metric in self._metrics.items()}

    def scores_from_totals(self, totals: np.ndarray) -> dict[str, float]:
        """Score one row of summed counts."""
        return {
            name: float(metric._compute_score_from_stats(totals[self._columns[name]]).score)
            for name, metric in self._metrics.items()
        }

    def score(self, rows: np.ndarray | None = None) -> dict[str, object]:
        """Return metrics and degeneration diagnostics for all rows or a subset."""

        def select(values: np.ndarray) -> np.ndarray:
            return values if rows is None else values[rows]

        counts = select(self.counts)
        if not len(counts):
            raise ValueError("Cannot score an empty corpus.")
        metrics: dict[str, object] = dict(self.scores_from_totals(counts.sum(axis=0)))
        metrics["exact_match_rate"] = float(select(self.exact).mean())
        metrics["empty_output_rate"] = float(select(self.empty).mean())
        metrics["character_length_ratio"] = float(
            select(self.hypothesis_characters).sum() / max(int(select(self.reference_characters).sum()), 1)
        )
        metrics["signatures"] = self.signatures
        if self.source_copy is not None:
            metrics["source_copy_rate"] = float(select(self.source_copy).mean())
        return metrics


def sentence_statistics(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    sources: Sequence[str] | None = None,
    lowercase: bool = False,
    bleu_tokenize: str = "13a",
) -> SentenceStatistics:
    """Tokenize and count every sentence pair exactly once."""
    if len(hypotheses) != len(references):
        raise ValueError("Hypothesis/reference counts do not match.")
    if sources is not None and len(sources) != len(hypotheses):
        raise ValueError("Source/hypothesis counts do not match.")
    if not len(hypotheses):
        raise ValueError("Cannot score an empty corpus.")

    hyps = [str(text).strip() for text in hypotheses]
    refs = [str(text).strip() for text in references]
    metrics = {
        "BLEU": BLEU(tokenize=bleu_tokenize, effective_order=True, lowercase=lowercase),
        "chrF2": CHRF(),
        "TER": TER(
            normalized=True,
            asian_support=bleu_tokenize == "zh",
            case_sensitive=not lowercase,
        ),
    }
    blocks = [
        np.asarray(metric._extract_corpus_statistics(hyps, [refs]), dtype=np.float64) for metric in metrics.values()
    ]
    columns: dict[str, slice] = {}
    offset = 0
    for name, block in zip(metrics, blocks, strict=True):
        columns[name] = slice(offset, offset + block.shape[1])
        offset += block.shape[1]
    source_copy = None
    if sources is not None:
        source_copy = np.array(
            [
                " ".join(hyp.casefold().split()) == " ".join(str(source).casefold().split())
                for hyp, source in zip(hyps, sources, strict=True)
            ],
            dtype=bool,
        )
    return SentenceStatistics(
        counts=np.hstack(blocks),
        exact=np.array([hyp == ref for hyp, ref in zip(hyps, refs, strict=True)], dtype=bool),
        empty=np.array([not hyp for hyp in hyps], dtype=bool),
        hypothesis_characters=np.array([len(text) for text in hyps], dtype=np.int64),
        reference_characters=np.array([len(text) for text in refs], dtype=np.int64),
        source_copy=source_copy,
        _metrics=metrics,
        _columns=columns,
    )


def score_translations(
    hypotheses: Sequence[str],
    references: Sequence[str],
    *,
    sources: Sequence[str] | None = None,
    lowercase: bool = False,
    bleu_tokenize: str = "13a",
) -> dict[str, object]:
    """Return comparable corpus MT metrics plus basic degeneration diagnostics."""
    return sentence_statistics(
        hypotheses,
        references,
        sources=sources,
        lowercase=lowercase,
        bleu_tokenize=bleu_tokenize,
    ).score()


def stratum_rows(strata: Sequence[str] | None, size: int) -> dict[str, np.ndarray]:
    """Map each stratum name to its row positions, in sorted name order."""
    if strata is None:
        return {"all": np.arange(size)}
    if len(strata) != size:
        raise ValueError("Strata count does not match the scored sentences")
    names = np.asarray([str(name) for name in strata], dtype=object)
    return {name: np.flatnonzero(names == name) for name in sorted(set(names.tolist()))}


# Scores are kept per view: the pooled corpus, the unweighted stratum average,
# and each stratum. Internally a view is keyed by ``(report key, stratum)``.
View = tuple[str, str]
POOLED: View = ("metrics", "")
MACRO: View = ("macro", "")


def _with_macro(by_stratum: dict[str, dict], pooled: dict) -> dict[View, dict]:
    return {
        POOLED: pooled,
        MACRO: {name: np.mean([scores[name] for scores in by_stratum.values()], axis=0) for name in METRIC_NAMES},
        **{("by_stratum", stratum): scores for stratum, scores in by_stratum.items()},
    }


def _report(views: dict[View, object]) -> dict[str, object]:
    return {
        "metrics": views[POOLED],
        "macro": views[MACRO],
        "by_stratum": {stratum: value for (key, stratum), value in views.items() if key == "by_stratum"},
    }


def _observed_scores(statistics: SentenceStatistics, groups: dict[str, np.ndarray]) -> dict[View, dict]:
    return _with_macro(
        {name: statistics.scores_from_totals(statistics.counts[rows].sum(axis=0)) for name, rows in groups.items()},
        statistics.scores_from_totals(statistics.counts.sum(axis=0)),
    )


def macro_average(statistics: SentenceStatistics, strata: Sequence[str]) -> dict[str, object]:
    """Average each metric over strata so large languages do not dominate."""
    groups = stratum_rows(strata, len(statistics))
    macro = _observed_scores(statistics, groups)[MACRO]
    return {"groups": len(groups), **{name: float(macro[name]) for name in METRIC_NAMES}}


def _resampled_totals(
    counts: np.ndarray,
    groups: dict[str, np.ndarray],
    samples: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Return ``(samples, width)`` summed counts per stratum.

    Drawing a stratum of ``m`` rows with replacement is a multinomial count
    over its rows, so each block of resamples is one matrix product.
    """
    rng = np.random.default_rng(seed)
    totals: dict[str, np.ndarray] = {}
    for name, rows in groups.items():
        block = counts[rows]
        size = len(rows)
        probabilities = np.full(size, 1.0 / size)
        output = np.empty((samples, counts.shape[1]), dtype=np.float64)
        step = max(1, RESAMPLE_BLOCK_CELLS // size)
        for start in range(0, samples, step):
            draws = rng.multinomial(size, probabilities, size=min(step, samples - start))
            output[start : start + len(draws)] = draws.astype(np.float64) @ block
        totals[name] = output
    return totals


def _resampled_scores(statistics: SentenceStatistics, totals: dict[str, np.ndarray]) -> dict[View, dict]:
    def scored(values: np.ndarray) -> dict[str, np.ndarray]:
        rows = [statistics.scores_from_totals(row) for row in values]
        return {name: np.array([row[name] for row in rows]) for name in METRIC_NAMES}

    return _with_macro(
        {name: scored(values) for name, values in totals.items()},
        scored(sum(totals.values())),
    )


def _interval(values: np.ndarray) -> dict[str, float]:
    tail = (1.0 - CONFIDENCE) / 2 * 100
    lower, median, upper = np.percentile(values, [tail, 50.0, 100.0 - tail])
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}


def bootstrap_confidence_intervals(
    statistics: SentenceStatistics,
    *,
    strata: Sequence[str] | None = None,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Return deterministic stratified bootstrap intervals.

    ``metrics`` covers the pooled corpus, ``macro`` the unweighted stratum
    average, and ``by_stratum`` each stratum on its own.
    """
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    groups = stratum_rows(strata, len(statistics))
    resampled = _resampled_scores(statistics, _resampled_totals(statistics.counts, groups, samples, seed))
    return {
        "samples": samples,
        "seed": seed,
        "confidence": CONFIDENCE,
        **_report(
            {view: {name: _interval(scores[name]) for name in METRIC_NAMES} for view, scores in resampled.items()}
        ),
    }


def paired_bootstrap_comparison(
    baseline: SentenceStatistics,
    system: SentenceStatistics,
    *,
    strata: Sequence[str] | None = None,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, object]:
    """Test whether two systems differ on the same test sentences.

    Both systems are scored on identical stratified resamples. ``delta`` is
    system minus baseline; lower TER is better. The two-sided p-value is the
    shifted paired bootstrap test: the share of resampled differences, centered
    on their mean, that are at least as large as the observed difference.
    """
    if samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    if len(baseline) != len(system):
        raise ValueError("Paired comparison needs the same sentences for both systems")
    if baseline.signatures != system.signatures:
        raise ValueError("Paired comparison needs identical metric configurations")
    groups = stratum_rows(strata, len(baseline))
    width = baseline.counts.shape[1]
    totals = _resampled_totals(np.hstack([baseline.counts, system.counts]), groups, samples, seed)
    baseline_resampled = _resampled_scores(baseline, {name: values[:, :width] for name, values in totals.items()})
    system_resampled = _resampled_scores(system, {name: values[:, width:] for name, values in totals.items()})
    baseline_observed = _observed_scores(baseline, groups)
    system_observed = _observed_scores(system, groups)

    def compared(view: View) -> dict[str, dict[str, float]]:
        output = {}
        for name in METRIC_NAMES:
            delta = float(system_observed[view][name] - baseline_observed[view][name])
            differences = system_resampled[view][name] - baseline_resampled[view][name]
            interval = _interval(differences)
            centered = np.abs(differences - differences.mean())
            output[name] = {
                "baseline": float(baseline_observed[view][name]),
                "system": float(system_observed[view][name]),
                "delta": delta,
                "delta_lower": interval["lower"],
                "delta_upper": interval["upper"],
                "p_value": float((np.sum(centered >= abs(delta)) + 1) / (samples + 1)) if delta else 1.0,
            }
        return output

    return {
        "samples": samples,
        "seed": seed,
        "confidence": CONFIDENCE,
        "sentences": len(baseline),
        **_report({view: compared(view) for view in system_observed}),
    }
