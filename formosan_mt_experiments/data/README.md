# Experiment Data

This directory is for generated experiment artifacts. It is ignored by git except for this README and `.gitkeep`.

Expected generated layout:

- `splits_en_v1/`: regenerated tiered English MT splits (`lexical`, `in_domain_hard`, `hard_global`) plus split validation reports.
- `tokenizer_sweep/`, `tokenizer_sweep_spm8192/`, `tokenizer_sweep_spm32768/`: generated tokenizer/model setup outputs and tokenizer audits.
- `runs/`: optional local training outputs. On Andromeda, put runs under `/scratch/$USER/formosan_mt_experiments/runs/`.

The canonical expensive English/Chinese pivot corpora remain in `../../pivot_corpora_final/`. Protected checksum copies live in `../../protected_corpora/deepl_pivots/`.

Regenerate the primary tiered splits with:

```bash
python formosan_mt_experiments/scripts/build_experiment_splits.py \
  --input pivot_corpora_final/big_corpus_en.csv \
  --output-dir formosan_mt_experiments/data/splits_en_v1 \
  --train-ratio 0.90 \
  --val-ratio 0.025 \
  --test-ratio 0.075
```
