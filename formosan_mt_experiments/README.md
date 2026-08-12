# Formosan MT Experiments

This package trains and evaluates directional NLLB-200 and MADLAD-400 models
from a completed corpus pipeline v3 bundle. It is model-family aware but shares
one data, sampling, metrics, provenance, and checkpoint contract.

## Directions

| ID | Source | Target |
|---|---|---|
| `f2en` | Formosan | English |
| `en2f` | English | Formosan |
| `f2zh` | Formosan | Traditional Chinese |
| `zh2f` | Traditional Chinese | Formosan |

Inputs are prefixed with target, source-language, domain, and dialect controls:

```text
# NLLB
<to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra.

# MADLAD
<2en> <to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra.
```

MADLAD uses its first input token for target selection. The native English and
Traditional Chinese selectors are retained, while the 15 Formosan selectors
and metadata controls are added during setup.

## Required Corpus Contract

Use only `big_corpus_<en|zh>_in_domain_hard.csv` from a completed v3 provenance
bundle. Before model setup, `scripts/validate_experiment.py` verifies:

- the corpus, profile, standardization namespace, and artifact hashes;
- roughly 90/2.5/7.5 train/validation/test proportions within every language
  and source corpus;
- sentence references only in evaluation, preferring human rows before valid
  synthetic pivot fallbacks;
- no lexical, morpheme, lexical-source, short, or ambiguous-normalization
  evaluation rows;
- zero exact, skeleton, one-edit, and configured character n-gram conflicts
  across split boundaries.

Document overlap is diagnostic rather than a gate because a source corpus may
store thousands of independent records in one XML file.

The local release pipeline also runs per-language TAME-MT exposure checks in
both translation directions. Tokenizer/model setup consumes training rows only.

## Profiles

`configs/default_experiment.json` is the production NLLB-200 recipe:

| Setting | Value |
|---|---:|
| Base model | NLLB-200 distilled 600M, pinned revision |
| Tokenizer | Formosan-aware SentencePiece extension |
| Added pieces | 8,192 |
| Max updates | 300,000 |
| Microbatch / accumulation | 16 / 4 |
| Maximum length | 384 |
| Learning rate | `2e-5` |
| Precision | bf16 |
| Selection metric | chrF2 |

`configs/madlad400_3b_native.json` is the MADLAD-400 3B recipe:

| Setting | Value |
|---|---:|
| Base model | MADLAD-400 3B MT, pinned revision |
| Tokenizer | Native 256k SentencePiece plus controls |
| Max updates | 50,000 |
| Microbatch / accumulation | 1 / 32 |
| Maximum length | 384 |
| Learning rate | `1e-5` |
| Precision / load dtype | bf16 / bf16 |
| Gradient checkpointing | enabled |
| Selection metric | chrF2 |

NLLB generation starts the decoder with EOS and selects the target with
`forced_bos_token_id`. MADLAD preserves decoder start ID 0 and never uses NLLB
forced-BOS behavior. `scripts/model_backends.py` enforces these differences.

## Slurm Submission

The Slurm launchers are portable examples. Configure paths for the target
cluster rather than editing tracked files:

```bash
cd formosan_mt_experiments
export EXP_DIR="$PWD"
export PROJECT_DATA=/shared/formosan_parallel_corpora
export SCRATCH=/scratch/$USER/formosan_mt
export JOBS_DIR="$SCRATCH/jobs"
export MANIFEST_DIR="$SCRATCH/manifests"
```

The data layout is:

```text
$PROJECT_DATA/<corpus-name>/
  big_corpus_en_in_domain_hard.csv
  big_corpus_zh_in_domain_hard.csv
  provenance/mt_build_manifest.json
```

Submit NLLB:

```bash
CORPUS_NAME=public_no_bible \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

Submit MADLAD:

```bash
CORPUS_NAME=public_no_bible \
PROFILE="$EXP_DIR/configs/madlad400_3b_native.json" \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

The launcher queues two CPU validators, family-specific setup, four trainers,
and a `best/` checkpoint evaluation for each trainer. A fixed `RUN_STAMP` is idempotent:
active or completed jobs are reused, while terminal failures are resubmitted.
Training resumes only when corpus, code, profile, and setup hashes match.

Resource defaults can be overridden with `VALIDATE_*`, `SETUP_*`, `TRAIN_*`,
and `EVAL_*` environment variables. NLLB expects a 40GB-or-larger GPU. MADLAD
expects an 80GB-or-larger GPU with the supplied full-finetuning profile. The
Slurm files assume a `miniconda` module and `formosan_mt` environment; adapt
those two setup lines to the cluster environment when needed.

## Metrics And Checkpoints

Training logs loss, learning rate, gradient norm, throughput, peak CUDA memory,
validation loss, perplexity, BLEU, chrF2, TER, exact match, empty-output rate,
output/reference length ratio, and applied metadata-dropout counts. Direction
and language tags are always retained. Domain and dialect tags independently
fall back to `unknown` and `default` for 25% of training presentations.
Checkpoint selection uses `unknown`/`default` metadata to match headline
inference. Generation metrics include per-language breakdowns.

Each generation validation atomically updates `resume/` with model, optimizer,
scheduler, scaler, random state, and run-contract hash. Successful training
retains deployable `best/` and `final/` directories. The default Slurm flight
evaluates only the validation-selected `best/` checkpoint with realistic
default metadata. Set `EVAL_CHECKPOINTS="best final"` or
`METADATA_MODES="default,oracle"` only for a specific ablation.

Headline and grouped metrics are written immediately after generation.
Bootstrap confidence intervals are optional because they are not needed for
routine model selection. Set `SUBMIT_BOOTSTRAP=1` to add a parallel CPU-only
confidence-interval job; the GPU is not held while resampling.

## Component Ownership

| Component | Responsibility |
|---|---|
| `validate_experiment.py` | Independent corpus and leakage gate. |
| `audit_corpus_exposure.py` | Exact TAME-MT exposure reports. |
| `setup_formosan_nllb200.py` | NLLB SentencePiece and embedding setup. |
| `setup_formosan_madlad400.py` | MADLAD controls and embedding/head resize. |
| `model_backends.py` | Model-family prompts and generation behavior. |
| `train_directional.py` | Sampling, optimization, validation, and resume. |
| `evaluate_directional.py` | Selected-checkpoint test evaluation. |
| `bootstrap_predictions.py` | Optional CPU confidence intervals. |
| `mt_metrics.py` | BLEU, chrF2, TER, signatures, and confidence intervals. |
| `publish_huggingface_models.py` | Audited standalone Hub packages. |
| `slurm/submit_directional_experiment.sh` | Idempotent Slurm DAG. |

Generated data, run manifests, checkpoints, predictions, and reports are
ignored. Store them with the corresponding corpus release or in
access-controlled experiment storage, never in this source repository.

## Verification

From the repository root:

```bash
python scripts/check_public_release.py
pytest -q
ruff check scripts formosan_mt_experiments/scripts tests
python -m compileall -q scripts formosan_mt_experiments/scripts tests
bash -n build_corpora.sh formosan_mt_experiments/slurm/*.sh \
  formosan_mt_experiments/slurm/*.sl
```
