# Formosan MT Experiments

The supported training stack for the 2026 Formosan MT experiments. It consumes
the final `in_domain_hard` corpus artifacts, validates them independently,
extends NLLB-200 with an 8k Formosan-aware SentencePiece model, trains one model
per direction, and evaluates both final and best checkpoints.

The May 2026 E0-E4 sweeps remain in this directory for provenance. Their scores
and older model cards are historical comparisons, not the active data or
launcher defaults.

## Production Experiment

Two corpus scopes are trained independently:

- `public_no_bible`: XML available from the public FormosanBank corpus tree;
- `private_no_bible`: all repositories visible to the configured GitHub token.

Both exclude exactly `Formosan-Taiwan-Bible-Society-Bibles` and train these
unidirectional models:

| Direction | Input | Output |
|---|---|---|
| `f2en` | tagged Formosan | English |
| `en2f` | tagged English | Formosan |
| `f2zh` | tagged Formosan | Traditional Chinese |
| `zh2f` | tagged Traditional Chinese | Formosan |

The active flight snapshot is
[`manifests/no_bible_v1_20260712.json`](manifests/no_bible_v1_20260712.json).
It records corpus checksums, split totals, code commit, hyperparameters, and all
validation/setup/training/evaluation job IDs.

New submission manifests also contain a SHA-256 inventory of every active
launcher, Slurm wrapper, Python module, configuration file, dependency list,
and tokenizer-setup implementation. Manifest generation fails if any required
code artifact is absent. This makes the code actually deployed to Andromeda
auditable independently of the Git commit label.

Validate its structure and, when the local ignored builds are present, every
file checksum and split count with `scripts/verify_experiment_manifest.py`.

## Data Gate

`scripts/validate_experiment.py` is a mandatory pre-training gate. For every
language it verifies at least 7.5% test and 2.5% validation against the complete
final corpus denominator. It also requires:

- no lexemes in validation or test;
- zero normalized source, target, or pair overlap across train/evaluation;
- zero punctuation/spacing skeleton overlap across train/evaluation;
- zero one-edit train/evaluation source or target conflicts.

The splitter prefers human references for evaluation. If all eligible human
sentence groups are exhausted before a language reaches its ratio floor, it
may use synthetic sentence references for the residual deficit. This is
reported explicitly; synthetic references are never silently admitted and XML
lexemes remain training-only.

Run the gate locally with:

```bash
python scripts/validate_experiment.py \
  --input ../corpus_builds/public_no_bible/pivot_corpora_final/big_corpus_en_in_domain_hard.csv \
  --target-lang english \
  --min-test-ratio 0.075 \
  --min-validate-ratio 0.025
```

On Andromeda, `slurm/validate_corpus.sl` runs the same code in separate CPU
jobs. Tokenizer setup uses `afterok` on those validators, so invalid remote data
cannot start GPU training.

The tokenizer setup wrapper invokes the versioned implementation at
`../scripts/mt/nllb/prelims/setup_formosan_nllb200.py`. The Andromeda mirror at
`/home/$USER/nllb-scripts/` is checksum-pinned by `setup_spm_sweep.sl`, so an
untracked or stale helper cannot silently alter a flight.

## Input Tags

Training and inference prefix each source with control tags:

```text
<to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra to demak nira.
<to_ami> <src_zh> <dom_unknown> <dialect_default> 他回家了。
```

The tokenizer setup adds direction, source-language, source-domain, and dialect
tags as special tokens. Metadata normalization and weighted sampling live in
`scripts/mt_common.py`.

## Current Recipe

The canonical configuration is `configs/default_experiment.json`:

| Setting | Value |
|---|---:|
| Base model | `facebook/nllb-200-distilled-600M` |
| Formosan SPM extension | 8,192 |
| Updates | 300,000 maximum |
| Microbatch / accumulation | 16 / 4 |
| Effective batch | 64 |
| Maximum length | 384 |
| Learning rate | `2e-5` |
| Precision | bf16 |
| Generation validation | every 10,000 updates |
| Validation sample | 128 rows per language |
| Selection metric | chrF2 |
| Early stopping | 5 non-improving validations after step 30,000 |

Validation logs corpus and per-language BLEU, chrF2, TER, exact match, empty
output rate, output/reference length ratio, token loss, and perplexity. Training
logs loss, learning rate, gradient norm, throughput, and peak CUDA memory.

Every generation validation writes an atomic `resume/` checkpoint containing
model, optimizer, scheduler, scaler, and random state. A Slurm rerun resumes the
latest complete checkpoint. Successful completion retains deployable `best/`
and `final/` directories and removes transient resume state.

## Andromeda Layout

The tracked jobs use the canonical cluster layout:

```text
/home/$USER/workspace/projects/mt/formosan_mt_experiments   code
/projects/prudlab/formosan_parallel_corpora/<corpus>        corpora + provenance
/scratch/$USER/projects/mt/formosan_mt_experiments/data     tokenizers/base models
/scratch/$USER/projects/mt/formosan_mt_experiments/runs     training outputs
/scratch/$USER/projects/mt/formosan_mt_experiments/reports  final metrics/predictions
```

Transfer data with provenance:

```bash
rsync -avP ../corpus_builds/public_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/public_no_bible/
rsync -avP ../corpus_builds/private_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/private_no_bible/
```

The corpus directory should also contain `provenance/mt_build_manifest.json`,
the pivot manifest, and local validation reports.

Submit each complete flight:

```bash
CORPUS_NAME=public_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_v1_spm8k_directional.sh
CORPUS_NAME=private_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_v1_spm8k_directional.sh
```

The launcher intentionally refuses an unnamed corpus. For each corpus it
queues two validators, two tokenizer setups, four trainers, and eight
evaluations. A fixed `RUN_STAMP` is idempotent: pending, running, or completed
jobs are reused; terminal failed jobs are resubmitted. Evaluations depend on
successful training and target both `final/` and `best/`.

Operational defaults match the current flight: `medium`/48h trainers on one
40GB-or-larger GPU, `short` CPU setup/validation, and `medium`/24h evaluation.
All resources remain overridable through the launcher's environment variables.

## Script Ownership

| Component | Responsibility |
|---|---|
| `build_experiment_splits.py` | Connected hard groups, ratio floors, fallbacks, leakage pruning, reports. |
| `validate_experiment.py` | Independent data contract verification. |
| `setup_tokenizer_sweep.py` | SPM extension, NLLB resize, token audit, smoke generation. |
| `train_directional_nllb.py` | Sampling, optimization, validation metrics, checkpointing, resume. |
| `evaluate_directional.py` | Full test generation and global/per-language/source-bin metrics. |
| `mt_metrics.py` | Shared BLEU/chrF2/TER and diagnostic metric implementation. |
| `training_code_inventory.py` | Required production-code inventory and SHA-256 provenance. |
| `slurm/submit_v1_spm8k_directional.sh` | Idempotent production DAG submission and manifest emission. |

## Historical Experiments

The following are retained to reproduce the May 2026 architecture search:

- `legacy_rebuild_robust_mt_splits.py`;
- `pretrain_dae_nllb.py` and `slurm/pretrain_dae.sl`;
- `slurm/submit_e2_e3_e4_andromeda.sh`;
- `slurm/submit_e3_zh_andromeda.sh`;
- `slurm/train_legacy_multilingual.sl`;
- `hf_cards/` and `reports/README.md`.

That comparison established SPM8k directional NLLB as the strongest practical
600M architecture. Its metrics used older data and must not be compared as if
they were results from the current no-Bible corpora.

## NLLB Invariants

- `decoder_start_token_id` is the tokenizer EOS token.
- target language selection uses `forced_bos_token_id`.
- every custom Formosan code and control tag must be a tokenizer special token.
- final and best models each include the tokenizer needed for standalone use.

## Verification

```bash
python -m unittest discover -s ../tests -v
ruff check scripts ../scripts/local/*.py ../scripts/local/scripts/pivot ../tests
python -m compileall -q scripts ../scripts/local ../tests
bash -n slurm/*.sh slurm/*.sl
```
