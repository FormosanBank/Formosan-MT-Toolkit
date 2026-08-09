# Formosan MT Experiments

The supported NLLB-200 and MADLAD-400 3B training stack for the 2026 Formosan
MT experiments. It consumes final `in_domain_hard` artifacts, validates them
independently, prepares a family-specific tokenizer/model, trains one model per
direction, and evaluates both final and best checkpoints.

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

The historical July 2026 flight snapshot is
[`manifests/no_bible_v1_20260712.json`](manifests/no_bible_v1_20260712.json).
It records corpus checksums, split totals, code commit, hyperparameters, and all
validation/setup/training/evaluation job IDs. It is not a corpus pipeline v3
release manifest and must not be used to authorize new training.

New submission manifests also contain a SHA-256 inventory of every active
launcher, Slurm wrapper, Python module, configuration file, and tokenizer-setup
implementation. Manifest generation fails if any required
code artifact is absent. This makes the code actually deployed to Andromeda
auditable independently of the Git commit label.

Validate its structure and, when the local ignored builds are present, every
file checksum and split count with `scripts/verify_experiment_manifest.py`.

## Data Gate

`scripts/validate_experiment.py` is a mandatory pre-training gate. For every
language it verifies at least 7.5% test and 2.5% validation against the complete
final corpus denominator. It also requires:

- no lexemes in validation or test;
- no synthetic rows in validation or test;
- exact `formosan-mt` namespace/profile provenance on every row;
- no ambiguous or MT-ineligible normalization in validation or test;
- source-document holdout or a declared small-language fallback;
- zero normalized source, target, or pair overlap across train/evaluation;
- zero punctuation/spacing skeleton overlap across train/evaluation;
- zero one-edit and high character n-gram conflicts across every split
  boundary.

The splitter uses only human sentence references for evaluation. Synthetic,
lexical, and morpheme rows remain training-only. TAME-MT exact exposure audits
run per `lang_code` task, matching the model's language-control tag, as part of
the local corpus release before transfer to Andromeda.

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

NLLB setup invokes `scripts/setup_formosan_nllb200.py`; the generic submitter
hashes that implementation and `setup_spm_sweep.sl` verifies it. MADLAD setup
runs `scripts/setup_formosan_madlad400.py` on a CPU compute node, uses both
training corpora to establish one shared vocabulary contract, and records
every input and output hash.

## Input Tags

Training and inference prefix each source with control tags:

```text
# NLLB
<to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra to demak nira.
<to_ami> <src_zh> <dom_unknown> <dialect_default> 他回家了。

# MADLAD
<2en> <to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra to demak nira.
<2ami> <to_ami> <src_zh> <dom_unknown> <dialect_default> 他回家了。
```

MADLAD’s first token selects the target language. `<2en>` and `<2zh_Hant>` are
native; the 15 `<2iso>` Formosan selectors and metadata controls are added and
their input/output rows are initialized deterministically. Metadata
normalization and weighted sampling live in `scripts/mt_common.py`.

## Supported Recipes

The canonical configuration is `configs/default_experiment.json`:

| Setting | Value |
|---|---:|
| Base model | `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d` |
| Auxiliary SPM | 8,192 pieces, Formosan train text only |
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

Tokenizer setup realigns shared NLLB embeddings by token identity after
SentencePiece changes and hashes every artifact. Training verifies the corpus,
independent validation, setup, profile, and file hashes before loading data.
Validation logs corpus and per-language BLEU, chrF2, TER, exact match, empty
output rate, output/reference length ratio, token loss, and perplexity.
Training logs loss, learning rate, gradient norm, throughput, and peak CUDA
memory.

Every generation validation writes an atomic `resume/` checkpoint containing
model, optimizer, scheduler, scaler, random state, and the immutable run
contract hash. A Slurm rerun refuses mismatched data/code/setup and otherwise
resumes the latest complete checkpoint. Successful completion retains
deployable `best/` and `final/` directories.

`configs/madlad400_3b_native.json` is the matched 3B recipe:

| Setting | Value |
|---|---:|
| Base model | `google/madlad400-3b-mt@fa184c675da0b5c9e1c8694fccd4e12e2d422094` |
| Tokenizer | Native 256k SentencePiece plus target/control tokens |
| Updates | 50,000 maximum |
| Microbatch / accumulation | 1 / 32 |
| Maximum length | 384 |
| Learning rate | `1e-5` |
| Precision / load dtype | bf16 / bf16 |
| Gradient checkpointing | enabled |
| Selection metric | chrF2 |

MADLAD does not use the NLLB SPM8k merge, `src_lang`, forced BOS, or EOS
decoder-start override. The shared backend module enforces those differences.

## Andromeda Layout

The tracked jobs use the canonical cluster layout:

```text
/home/$USER/workspace/projects/mt/formosan_mt_experiments   code
/projects/prudlab/formosan_parallel_corpora/<corpus>        corpora + provenance
/scratch/$USER/projects/mt/formosan_mt_experiments/data     tokenizers/base models
/scratch/$USER/projects/mt/formosan_mt_experiments/runs     training outputs
/scratch/$USER/projects/mt/formosan_mt_experiments/reports  final metrics/predictions
```

Transfer the final directory. The corpus builder creates `provenance/`
automatically:

```bash
rsync -avP ../corpus_builds/public_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/public_no_bible/
rsync -avP ../corpus_builds/private_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/private_no_bible/
```

The submitter requires `provenance/mt_build_manifest.json`; validators produce
fresh runtime reports before tokenizer setup can start.

Submit each complete flight:

```bash
# NLLB
CORPUS_NAME=public_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
CORPUS_NAME=private_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh

# MADLAD
CORPUS_NAME=private_no_bible \
PROFILE="$PWD/configs/madlad400_3b_native.json" \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

The launcher intentionally refuses an unnamed corpus. For each corpus it
queues two validators, family-specific setup, four trainers, and eight
evaluations. NLLB has two target-specific SPM setups; MADLAD has one shared
setup. A fixed `RUN_STAMP` is idempotent: pending, running, or completed jobs
are reused; terminal failed jobs are resubmitted. Evaluations depend on
successful training and target both `final/` and `best/`.

NLLB defaults to one 40GB-or-larger GPU. MADLAD defaults to one
80GB-or-larger GPU, bf16 loading, microbatch 1, and gradient checkpointing.
Setup and validation run on CPU compute nodes. All resources remain
overridable through launcher environment variables.

## Hugging Face Publication

`scripts/publish_huggingface_models.py` builds four standalone Hub packages
from a matched profile, flight manifest, best/final checkpoints, and hard-test
reports. It accepts single-file or sharded safetensors and emits NLLB- or
MADLAD-correct inference examples. Formosan-source packages include
`formosan_mt_inference.py`, the standardizer, and the exact profile so
interactive source text receives the same idempotent transformation as
training data. `--publish` replaces stale Hub files and
records every published commit and file SHA-256.

The production publication is recorded in
[`manifests/huggingface_private_no_bible_20260716.json`](manifests/huggingface_private_no_bible_20260716.json).
The cards disclose checkpoint selection, split constraints, full global and
per-language metrics, BLEU tokenization, and reference provenance.

## Script Ownership

| Component | Responsibility |
|---|---|
| `build_experiment_splits.py` | Connected hard groups, ratio floors, fallbacks, leakage pruning, reports. |
| `validate_experiment.py` | Independent data contract verification. |
| `audit_corpus_exposure.py` | Exact TAME-MT exposure reports and release gates. |
| `setup_formosan_nllb200.py` | Formosan SentencePiece extension and NLLB embedding initialization. |
| `setup_tokenizer_sweep.py` | SPM extension, NLLB resize, token audit, smoke generation. |
| `setup_formosan_madlad400.py` | Native MADLAD controls, untied embedding/head resize, train-only audits. |
| `model_backends.py` | Family-specific prompting, tokenizer state, and generation contract. |
| `formosan_mt_inference.py` | Applies the pinned V3 Formosan standardizer at inference. |
| `train_directional.py` | Shared sampling, optimization, validation metrics, checkpointing, resume. |
| `evaluate_directional.py` | Default-tag headline and oracle-metadata diagnostic evaluation. |
| `mt_metrics.py` | BLEU/chrF2/TER, signatures, diagnostics, and bootstrap intervals. |
| `training_code_inventory.py` | Required production-code inventory and SHA-256 provenance. |
| `publish_huggingface_models.py` | Builds audited Hub packages and atomically replaces the four production model repos. |
| `slurm/submit_directional_experiment.sh` | Profile-driven idempotent DAG submission and manifest emission. |

## NLLB Invariants

- `decoder_start_token_id` is the tokenizer EOS token.
- target language selection uses `forced_bos_token_id`.
- every custom Formosan code and control tag must be a tokenizer special token.
- final and best models each include the tokenizer needed for standalone use.

## MADLAD Invariants

- source text starts with `<2en>`, `<2zh_Hant>`, or a learned Formosan
  `<2iso>` target selector;
- decoder start, pad, and EOS remain model IDs 0, 1, and 2;
- generation never sets `forced_bos_token_id`;
- input embeddings and the untied output head are both resized by token
  identity, with existing rows unchanged;
- native SentencePiece is retained; SPM8k merging is not part of this recipe.

## Verification

```bash
python -m unittest discover -s ../tests -v
ruff check scripts ../scripts/local/*.py ../tests
python -m compileall -q scripts ../scripts/local ../tests
bash -n slurm/*.sh slurm/*.sl
```
