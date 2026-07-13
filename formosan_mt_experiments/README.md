# Formosan MT Experiment Stack

This directory is a self-contained experiment layer for Formosan MT against English and Traditional Chinese. It does not replace the older NLLB scripts; it wraps or reuses them where useful and keeps new split, tokenizer, training, DAE, and eval logic here.

## Directory Map

- `data/`: generated split CSVs, tokenizer sweeps, tokenizer audits.
- `scripts/`: corpus builders, tokenizer setup wrappers, training, DAE pretraining, evaluation, and diagnostics.
- `configs/`: reproducible defaults.
- `slurm/`: Andromeda job templates.
- `reports/`: generated prediction/metric reports.

Generated files under `data/` and `reports/` are intentionally ignored by git except for README placeholders. The canonical expensive pivot corpora stay in `../pivot_corpora_final/`, and protected checksum copies live in `../protected_corpora/deepl_pivots/`.

Expected generated subdirectories:

- `data/splits_en_v1/`: tiered English MT splits and validation reports.
- `data/splits_zh_v1/`: tiered Traditional Chinese MT splits and validation reports.
- `data/tokenizer_sweep*/`: tokenizer/model setup outputs and tokenizer fragmentation audits.
- `data/runs/` or `/scratch/.../formosan_mt_experiments/runs/`: local or cluster training outputs.
- `reports/E*/`: prediction CSVs and metric JSONs.

The old one-off robust splitter is preserved as `scripts/legacy_rebuild_robust_mt_splits.py` for reference. Use `scripts/build_experiment_splits.py` for new runs.

## Build Tiered Splits

Build all split tiers from the raw English corpus and ignore its current `split` column:

```bash
python formosan_mt_experiments/scripts/build_experiment_splits.py \
  --input pivot_corpora_final/big_corpus_en.csv \
  --output-dir formosan_mt_experiments/data/splits_en_v1 \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075
```

Build the matching Traditional Chinese splits:

```bash
python formosan_mt_experiments/scripts/build_experiment_splits.py \
  --input pivot_corpora_final/big_corpus_zh.csv \
  --target-lang chinese \
  --target-col chinese_sentence \
  --output-prefix big_corpus_zh \
  --output-dir formosan_mt_experiments/data/splits_zh_v1 \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075
```

Outputs:

- `big_corpus_en_lexical.csv`: honest lexeme/template eval.
- `big_corpus_en_in_domain_hard.csv`: headline benchmark.
- `big_corpus_en_hard_global.csv`: hardest domain-transfer stress test.
- `big_corpus_zh_*.csv`: same tiers for Traditional Chinese.
- `report_all_tiers.json`: leakage and count diagnostics.

The splitter keeps lexemes and DeepL-synthetic references out of validation and
test. It assigns connected exact and punctuation/spacing skeleton clusters to
one split globally, then removes any remaining training row within one
character insertion, deletion, or substitution of Formosan or target
evaluation text. Ratios are calculated against human-reference rows so a large
synthetic training expansion cannot starve evaluation. The headline tier uses
minimum desired floors of 500 test rows and 150 validation rows per language;
when strict hard-domain candidates are scarce, it fills from non-lexical human
sentences while retaining the same leakage checks.

Validate any tier:

```bash
python formosan_mt_experiments/scripts/validate_experiment.py \
  --input formosan_mt_experiments/data/splits_zh_v1/big_corpus_zh_in_domain_hard.csv \
  --target-lang chinese \
  --direction f2zh
```

## E1: Build SPM Tokenizers

Run the NLLB setup in SPM mode, add experiment control tags, and audit fragmentation:

```bash
python formosan_mt_experiments/scripts/setup_tokenizer_sweep.py \
  --input pivot_corpora_final/big_corpus_en.csv \
  --output-dir formosan_mt_experiments/data/tokenizer_sweep \
  --spm-vocabs 8192,16384,32768 \
  --setup-splits train,validate,valid,val \
  --run-smoke
```

Use the tokenizer with the lowest fragmentation that does not bloat outputs. The current deployable v1 recipe is `spm8192`.

For Traditional Chinese, the selected deployable recipe is E3 SPM8k directional:

```bash
python formosan_mt_experiments/scripts/setup_tokenizer_sweep.py \
  --input formosan_mt_experiments/data/splits_zh_v1/big_corpus_zh_in_domain_hard.csv \
  --target-lang chinese \
  --output-dir formosan_mt_experiments/data/tokenizer_sweep_zh_spm8192 \
  --spm-vocabs 8192 \
  --setup-splits train,validate \
  --run-smoke
```

## V1: SPM8k Directional MT

The deployable v1 architecture is the four-checkpoint E3 SPM8k recipe used for
the published FormosanBank NLLB models:

- base model: `facebook/nllb-200-distilled-600M`
- tokenizer: fresh 8k Formosan-aware SentencePiece extension trained on the
  training/validation side of the current corpus
- directions: `f2en`, `en2f`, `f2zh`, `zh2f`
- metadata tags: target direction, source language, source bucket, dialect
- steps: `300000`
- batch size: `16`, gradient accumulation: `4`, effective batch size: `64`
- max sequence length: `384`
- learning rate: `2e-5`
- warmup steps: `4000`
- label smoothing: `0.1`
- language sampling alpha: `0.5`
- easy-source weight: `0.05` for Formosan→target, `0.15` for target→Formosan

For a self-contained public/private corpus copied under
`/projects/prudlab/formosan_parallel_corpora/<CORPUS_NAME>/`, queue the full
four-direction v1 stack with:

```bash
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  CORPUS_NAME=public_no_bible \
  /home/scheppat/workspace/projects/mt/formosan_mt_experiments/slurm/submit_v1_spm8k_directional.sh

RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  CORPUS_NAME=private_no_bible \
  /home/scheppat/workspace/projects/mt/formosan_mt_experiments/slurm/submit_v1_spm8k_directional.sh
```

The submitter records accepted job IDs under
`/home/scheppat/jobs/mt/submission_state_v1_spm8k_<CORPUS_NAME>_<RUN_STAMP>/`
and is safe to rerun with the same `RUN_STAMP`. Recorded pending, running, or
completed jobs are reused. Failed, canceled, timed-out, preempted, node-failed,
or out-of-memory jobs are resubmitted, and training automatically resumes from
its last complete validation checkpoint. Setup reuse requires actual tokenizer
and model files, not just stale directories.
Before tokenizer setup, separate CPU jobs independently validate the remote
English and Chinese corpora for per-language split floors, exact and skeleton
leakage, one-edit conflicts, and lexeme routing. Setup and all downstream GPU
jobs require those validators to succeed.

## E1: Directional MT Training

Formosan to English:

```bash
python formosan_mt_experiments/scripts/train_directional_nllb.py \
  --input formosan_mt_experiments/data/splits_en_v1/big_corpus_en_in_domain_hard.csv \
  --tokenizer formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_tokenizer \
  --model formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_model \
  --output-dir formosan_mt_experiments/data/runs/E1_in_domain_hard_f2en \
  --direction f2en \
  --steps 300000 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --precision bf16
```

English to Formosan:

```bash
python formosan_mt_experiments/scripts/train_directional_nllb.py \
  --input formosan_mt_experiments/data/splits_en_v1/big_corpus_en_in_domain_hard.csv \
  --tokenizer formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_tokenizer \
  --model formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_model \
  --output-dir formosan_mt_experiments/data/runs/E1_in_domain_hard_en2f \
  --direction en2f \
  --steps 300000 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --precision bf16
```

Defaults include effective batch size 64, max length 384, learning rate `2e-5`,
language sampling `alpha=0.5`, label smoothing `0.1`, and easy-source
downweighting.

Production runs also perform fixed, per-language generation validation every
10,000 updates. They log corpus and per-language BLEU, chrF2, TER, exact-match,
empty-output, and output/reference length-ratio diagnostics alongside
teacher-forced token loss and perplexity. The default best checkpoint is the
highest validation chrF2 checkpoint; training stops after five evaluations
without an improvement of at least 0.05 chrF2 after step 30,000. This avoids
selecting a fluent-looking but worse translator solely from token loss.

Each validation writes an atomic `resume/` checkpoint containing model,
optimizer, scheduler, scaler, and random-number-generator state. Slurm reruns
resume it automatically and successful completion removes it, retaining only
`best/` and `final/`. `train_log.jsonl` includes loss, learning rate, gradient
norm, throughput, and peak CUDA memory. `eval_log.jsonl` is the authoritative
validation history. Non-interactive jobs disable the high-frequency progress
bar to keep Slurm logs compact.

Traditional Chinese directions use the same training code with `--target-lang chinese` and directions `f2zh` / `zh2f`:

```bash
python formosan_mt_experiments/scripts/train_directional_nllb.py \
  --input formosan_mt_experiments/data/splits_zh_v1/big_corpus_zh_in_domain_hard.csv \
  --target-lang chinese \
  --tokenizer formosan_mt_experiments/data/tokenizer_sweep_zh_spm8192/formosan_multilingual_nllb_spm8192_tokenizer \
  --model formosan_mt_experiments/data/tokenizer_sweep_zh_spm8192/formosan_multilingual_nllb_spm8192_model \
  --output-dir formosan_mt_experiments/data/runs/E3_zh_spm8192_f2zh \
  --direction f2zh \
  --steps 300000 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --learning-rate 2e-5 \
  --precision bf16
```

## E2: DAE Pre-Adaptation

Train corrupted Formosan to clean Formosan first:

```bash
python formosan_mt_experiments/scripts/pretrain_dae_nllb.py \
  --input formosan_mt_experiments/data/splits_en_v1/big_corpus_en_in_domain_hard.csv \
  --tokenizer formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_tokenizer \
  --model formosan_mt_experiments/data/tokenizer_sweep/formosan_multilingual_nllb_spm16384_model \
  --output-dir formosan_mt_experiments/data/runs/E2_dae \
  --steps 100000 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --precision bf16
```

Then use `E2_dae/final` as `--model` for `train_directional_nllb.py`.

## E3/E4/E5

- `E3`: repeat E1/E2 with `spm8192`, `spm16384`, and `spm32768`; select by validation chrF/BLEU and tokenizer fragmentation.
- `E4`: build the custom tokenizer/model with `setup_tokenizer_sweep.py --base-model facebook/nllb-200-1.3B`, then use LoRA with `train_directional_nllb.py --lora-r 16`. Install `peft` in the cluster env first with `pip install -r formosan_mt_experiments/requirements-extras.txt`.
- `E5`: run per-language second-stage fine-tuning from the best multilingual checkpoint by filtering the CSV to one `lang_code` or by adding a filtered input CSV.

## Evaluation

Evaluate a trained directional model:

```bash
python formosan_mt_experiments/scripts/evaluate_directional.py \
  --input formosan_mt_experiments/data/splits_en_v1/big_corpus_en_in_domain_hard.csv \
  --tokenizer formosan_mt_experiments/data/runs/E1_in_domain_hard_f2en/final \
  --model formosan_mt_experiments/data/runs/E1_in_domain_hard_f2en/final \
  --direction f2en \
  --output-csv formosan_mt_experiments/reports/E1_f2en_predictions.csv \
  --output-json formosan_mt_experiments/reports/E1_f2en_metrics.json
```

The evaluator writes global, per-language, per-source-bucket, and per-length-bin metrics. Use `in_domain_hard` as the headline benchmark and `hard_global` as the stress test.

For Traditional Chinese:

```bash
python formosan_mt_experiments/scripts/evaluate_directional.py \
  --input formosan_mt_experiments/data/splits_zh_v1/big_corpus_zh_in_domain_hard.csv \
  --target-lang chinese \
  --tokenizer formosan_mt_experiments/data/runs/E3_zh_spm8192_f2zh/final \
  --model formosan_mt_experiments/data/runs/E3_zh_spm8192_f2zh/final \
  --direction f2zh \
  --output-csv formosan_mt_experiments/reports/E3_zh_f2zh_predictions.csv \
  --output-json formosan_mt_experiments/reports/E3_zh_f2zh_metrics.json
```

## Completed E0-E4 Comparison

The May 2026 Andromeda comparison used the `in_domain_hard` test split with `36,559` examples per direction. Final checkpoints outperformed the small-validation "best" checkpoints in almost every run.

| Experiment | F→EN BLEU | F→EN chrF2 | EN→F BLEU | EN→F chrF2 | Notes |
|---|---:|---:|---:|---:|---|
| E0 legacy bidirectional | 8.14 | 27.30 | 3.80 | 25.39 | Old multilingual recipe on the new hard split. |
| E1 SPM16k directional | 8.11 | 27.28 | 5.64 | 30.04 | Directional 600M models, tags, weighted sampling. |
| E2 DAE + SPM16k | 7.58 | 27.02 | 4.52 | 26.90 | DAE pre-adaptation did not help. |
| E3 SPM8k directional | 8.23 | 27.35 | 5.77 | 30.24 | Best small/deployable default. |
| E3 SPM32k directional | 7.83 | 26.90 | 5.93 | 30.28 | Similar EN→F to SPM8k, weaker F→EN. |
| E4 1.3B LoRA | 9.02 | 27.76 | 6.46 | 30.27 | Best overall, slower/heavier. |

## Andromeda

Copy this directory to the cluster, then use the templates in `slurm/`. The templates assume:

- experiment directory: `/home/scheppat/workspace/projects/mt/formosan_mt_experiments`
- corpus: `/projects/prudlab/formosan_parallel_corpora/big_corpus_en.csv`
- Chinese corpus: `/projects/prudlab/formosan_parallel_corpora/big_corpus_zh.csv`
- outputs: `/scratch/scheppat/projects/mt/formosan_mt_experiments`

Override any path with environment variables, for example:

```bash
sbatch --export=ALL,DIRECTION=en2f,TIER=in_domain_hard formosan_mt_experiments/slurm/train_directional.sl
```

For public/private corpus comparisons built by `build_corpora.sh
--build-public-private`, copy each self-contained final corpus directory to the
project corpus root:

```bash
rsync -avP corpus_builds/public_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/public_no_bible/
rsync -avP corpus_builds/private_no_bible/pivot_corpora_final/ \
  andromeda:/projects/prudlab/formosan_parallel_corpora/private_no_bible/
```

The Slurm templates accept `CORPUS_NAME`, so the default input becomes
`/projects/prudlab/formosan_parallel_corpora/<CORPUS_NAME>/big_corpus_<en|zh>_<tier>.csv`:

```bash
sbatch --export=ALL,CORPUS_NAME=public_no_bible,TARGET_LANG=english,DIRECTION=f2en \
  formosan_mt_experiments/slurm/train_directional.sl
sbatch --export=ALL,CORPUS_NAME=private_no_bible,TARGET_LANG=chinese,DIRECTION=f2zh \
  formosan_mt_experiments/slurm/train_directional.sl
```

To queue the full follow-up comparison stack on Andromeda, use the idempotent submitter:

```bash
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  /home/scheppat/jobs/mt/submit_e2_e3_e4_andromeda.sh
```

That submitter queues:

- `E0`: legacy multilingual training/eval scripts with the old char-added NLLB tokenizer/model on the new `in_domain_hard` split. This is the old-method control for separating split difficulty from E1-E4 method changes. It keeps the late-April sampling knobs (`alpha=0.5`, easy-source weight `0.1`) and trains bidirectionally with `p-src2tgt=0.5`.
- `E2`: 16k SPM DAE pre-adaptation, then F→EN and EN→F MT from the DAE best checkpoint.
- `E3`: 8k and 32k SPM setup plus directional MT. The already-running E1 16k SPM jobs provide the 16k comparison point.
- `E4`: NLLB-1.3B 16k SPM setup plus LoRA directional MT.

To queue the Chinese E3 SPM8k directional run:

```bash
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  /home/scheppat/jobs/mt/submit_e3_zh_andromeda.sh
```

The Chinese submitter builds or reuses `splits_zh_v1/big_corpus_zh_in_domain_hard.csv`, builds an SPM8k tokenizer/model with `<to_zh>` and `<src_zh>` control tags, then trains and evaluates `f2zh` and `zh2f` final/best checkpoints.

If Slurm is temporarily down, rerunning the submitter with the same `RUN_STAMP` is safe; it records accepted job IDs in `/home/scheppat/jobs/mt/submission_state_<RUN_STAMP>/`.

Before copying data to the cluster, verify the protected local corpora:

```bash
cd protected_corpora/deepl_pivots
shasum -a 256 -c SHA256SUMS
```

## NLLB Invariants

All scripts preserve the Transformers 4.56 NLLB behavior:

- `decoder_start_token_id` is `tokenizer.eos_token_id`.
- generation uses `forced_bos_token_id` for the target language.
- custom Formosan language codes must exist as tokenizer special tokens.
