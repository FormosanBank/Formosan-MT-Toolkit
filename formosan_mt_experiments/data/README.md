# Experiment Data

This directory is for generated experiment artifacts. It is ignored by git except for this README and `.gitkeep`.

Expected generated layout:

- `splits_en_v1/`: regenerated tiered English MT splits (`lexical`, `in_domain_hard`, `hard_global`) plus split validation reports.
- `tokenizer_sweep_spm8192/`, `tokenizer_sweep_en_spm8192/`, `tokenizer_sweep_zh_spm8192/`: generated v1 SPM8k tokenizer/model setup outputs and tokenizer audits.
- `runs/`: optional local training outputs. On Andromeda, put runs under `/scratch/$USER/projects/mt/formosan_mt_experiments/runs/`.

Current public/private no-Bible corpora live in ignored named build directories
under `../../corpus_builds/`. `../../pivot_corpora_final/` and
`../../protected_corpora/deepl_pivots/` retain an older unscoped pivot snapshot.

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
