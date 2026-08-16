# Formosan MT Experiments

This package trains and evaluates directional NLLB-200 and MiLMMT models from
a completed corpus pipeline v3 bundle.

## Directions

| ID | Source | Target |
|---|---|---|
| `f2en` | Formosan | English |
| `en2f` | English | Formosan |
| `f2zh` | Formosan | Traditional Chinese |
| `zh2f` | Traditional Chinese | Formosan |

NLLB inputs use target, source-language, domain, and dialect controls:

```text
<to_eng> <src_ami> <dom_ntu> <dialect_coastal> Pa'araw cingra.
```

## Required Corpus Contract

Use only `big_corpus_<en|zh>_in_domain_hard.csv` from a completed v3 provenance
bundle. Before model setup, `scripts/validate_experiment.py` verifies:

- the corpus, profile, standardization namespace, and artifact hashes;
- 5/10 validation/test proportions from all deduplicated pairs in
  every language, with capacity-aware source-corpus representation;
- human sentence references only in evaluation; synthetic pivots are train-only;
- no lexical, morpheme, short, or ambiguous-normalization
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
| Microbatch / accumulation | 8 / 8 |
| Maximum length | 384 |
| Learning rate | `2e-5` |
| Precision | bf16 |
| Selection metric | chrF2 |

NLLB generation starts the decoder with EOS and selects the target with
`forced_bos_token_id`. `scripts/nllb_runtime.py` owns this runtime contract.

`configs/milmmt_1b_experiment.json` is the experimental MiLMMT recipe:

| Setting | Value |
|---|---:|
| Base model | MiLMMT-46-1B-v1.0, pinned revision |
| Tokenizer | Native Gemma 3 tokenizer |
| Objective | Full-parameter response-only causal SFT |
| Max updates | 20,000 |
| Microbatch / accumulation | 2 / 16 |
| Maximum length | 512 |
| Learning rate | `2e-5` |
| Optimizer / schedule | AdamW / inverse square root |
| Precision | bf16 |
| Generation | Greedy |
| Checkpoint selection | Macro validation chrF2 across languages |

MiLMMT follows its official language-name prompt. Formosan names are learned
during fine-tuning because the released model does not claim native Formosan
support. The initial recipe intentionally omits metadata context and does not
merge the NLLB SPM8k vocabulary.

The recipe is based on the
[MiLMMT model card](https://huggingface.co/xiaomi-research/MiLMMT-46-1B-v1.0),
[GemmaX training code](https://github.com/xiaomi-research/gemmax), and the
[MiLMMT post-training paper](https://arxiv.org/abs/2608.10812). It adapts the
official supervised fine-tuning stage, not the multi-model GRPO stage.

Audit native-tokenizer efficiency on training data before choosing a longer
sequence limit:

```bash
python scripts/tokenizer_audit.py \
  --tokenizer "$MODEL" \
  --input "$PROJECT_DATA/private_no_bible/big_corpus_en_in_domain_hard.csv" \
  --target-lang english \
  --model-family milmmt \
  --direction f2en \
  --max-length 512 \
  --split train \
  --output-json tokenizer_audit_f2en.json \
  --output-csv tokenizer_audit_f2en.csv
```

The audit reports sentence and word fragmentation, unknown tokens,
Formosan-to-target token ratios, sequence-length percentiles, and the number
of examples that would be truncated. For initial learning-rate pilots, submit
separate run stamps with exported `LEARNING_RATE=5e-6` and `LEARNING_RATE=1e-5`.
Keep `2e-5` as the paper-faithful comparison point.
Use distinct `RUN_STAMP` values and export `SEED=42`, `SEED=43`, or `SEED=44`
when running replicated final experiments.

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

```bash
CORPUS_NAME=public_no_bible \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

For MiLMMT, add:

```bash
PROFILE="$EXP_DIR/configs/milmmt_1b_experiment.json"
```

The launcher queues two CPU validators, the profile's model setup, four
trainers, and a `best/` checkpoint evaluation for each trainer. NLLB has one
train-only SPM setup per target corpus; MiLMMT has one shared pinned base-model
snapshot. A fixed `RUN_STAMP` is idempotent: active or completed jobs are
reused, while terminal failures are resubmitted. Training resumes only when
corpus, code, profile, and setup hashes match.

Resource defaults can be overridden with `VALIDATE_*`, `SETUP_*`, `TRAIN_*`,
and `EVAL_*` environment variables. Both profiles expect a 40GB-or-larger GPU.
The Slurm files assume a `miniconda` module and `formosan_mt` environment;
adapt those two setup lines to the cluster environment when needed.

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
| `nllb_runtime.py` | NLLB language controls and generation behavior. |
| `setup_milmmt.py` | Pinned MiLMMT snapshot verification. |
| `milmmt_runtime.py` | MiLMMT prompts, causal loss, and generation. |
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
