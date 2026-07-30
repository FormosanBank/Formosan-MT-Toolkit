# Experiment Data

This directory is for generated experiment artifacts. It is ignored by git except for this README and `.gitkeep`.

Expected generated layout:

- `splits_en_v1/` and `splits_zh_v1/`: the sole supported
  `in_domain_hard` split plus independent validation and exposure reports.
- `tokenizer_sweep_en_spm8192/` and `tokenizer_sweep_zh_spm8192/`:
  generated NLLB SPM8k tokenizer/model outputs and train-only audits.
- `madlad400_3b_native_directional_v1/setup/`: generated shared MADLAD
  tokenizer/model with Formosan target/control tokens and train-only audits.
- `runs/`: optional local training outputs. On Andromeda, put runs under `/scratch/$USER/projects/mt/formosan_mt_experiments/runs/`.

Current public/private no-Bible corpora live in ignored named build directories
under `../../corpus_builds/`. Paid cache backups live under
`../../protected_corpora/deepl_pivots/`.

Regenerate a named build's primary tiered splits with:

```bash
python formosan_mt_experiments/scripts/build_experiment_splits.py \
  --input corpus_builds/public_no_bible/pivot_corpora_final/big_corpus_en.csv \
  --output-dir corpus_builds/public_no_bible/formosan_mt_experiments/data/splits_en_v1 \
  --target-lang english \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075
```

The production orchestrator's `--resplit-only` mode is preferred because it
also refreshes final copies and checksummed build manifests.
