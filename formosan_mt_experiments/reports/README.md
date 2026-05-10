# Experiment Reports

This directory is for generated metrics, prediction CSVs, and analysis reports. It is ignored by git except for this README and `.gitkeep`.

Expected generated layout:

- `E*_*.json` / `E*_*.csv`: expected output pattern for new experiment metrics and predictions.

Keep large prediction exports here instead of the repository root.

## E0-E4 Final Summary

Metrics below are global on the `in_domain_hard` test set with `36,559` examples per direction.

| Experiment | Checkpoint | F→EN BLEU | F→EN chrF2 | EN→F BLEU | EN→F chrF2 |
|---|---|---:|---:|---:|---:|
| E0 legacy bidirectional | final | 8.14 | 27.30 | 3.80 | 25.39 |
| E1 SPM16k directional | final | 8.11 | 27.28 | 5.64 | 30.04 |
| E2 DAE + SPM16k | final | 7.58 | 27.02 | 4.52 | 26.90 |
| E3 SPM8k directional | final | 8.23 | 27.35 | 5.77 | 30.24 |
| E3 SPM32k directional | final | 7.83 | 26.90 | 5.93 | 30.28 |
| E4 1.3B LoRA | final | 9.02 | 27.76 | 6.46 | 30.27 |
