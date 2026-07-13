# Pipeline Architecture

## Supported Workflow

`scripts/local/build_mt_corpus.py` is the single orchestration boundary. The
top-level `build_corpora.sh` only selects the repository virtualenv and forwards
arguments. Individual stage scripts remain callable for debugging, but a
production corpus should come from the orchestrator so paths, cleanup, pivot
cache reuse, hard splitting, validation, and checksummed manifests stay
consistent.

### 1. Acquisition

`fetch_xml.py` enumerates repositories visible through GitHub, resolves each
default branch, and downloads XML with bounded workers, retries, exponential
backoff, and a shared metadata/blob cache. Public mode restricts input to the
public FormosanBank corpus tree. Private mode means every eligible repository
visible to the token.

`--exclude-bible` excludes one exact repository/corpus root:
`Formosan-Taiwan-Bible-Society-Bibles`. It is not a fuzzy path-name filter.

### 2. FormosanBank QC

`clean_xml.py` runs the canonical FormosanBank cleaner and standardizer with a
valid package root and Python path. It handles current XML that already has a
standard tier instead of assuming every sentence needs an original-to-standard
copy. Validation can be enabled when a publication-grade XML audit is needed.

### 3. Extraction

`make_corpus.py` extracts `FORM kindOf="standard"` and target translations. It
preserves source path, XML ID, dialect, source kind, and row type. Word-level
XML material is marked as lexical data rather than inferred later from length
alone.

### 4. MT Filtering

`filter_split_corpus.py` performs Unicode/text normalization, removes explicit
missing markers and presentation artifacts, rejects source-side CJK leakage,
filters empty/redacted/performative rows, deduplicates exact pairs, and applies
separate sentence/lexeme fertility bounds. It records drop reasons and reject
samples. Pairwise splits are useful diagnostics; the experiment splitter later
rebuilds the authoritative hard split from the aggregate corpus.

### 5. Aggregation And Pivoting

The orchestrator combines all pairwise rows while retaining provenance and
`row_type`. `scripts/local/scripts/pivot/pivot.py` fills missing English or
Chinese targets, rotates all configured DeepL keys, batches within API limits,
backs off on transient errors, and writes each successful response immediately
to a content-keyed JSONL cache. Rebuilds read shared and build-local caches
before spending quota.

Pivot rows retain provider, direction, source text, origin, and cache key. A
cached response is data provenance, not permission to treat the row as human.

### 6. Hard Splitting

`formosan_mt_experiments/scripts/build_experiment_splits.py` ignores legacy
pairwise split labels and builds the model-facing split. It:

1. classifies lexical rows and locks them to training;
2. forms connected components through normalized and punctuation/spacing
   skeleton source and target keys, covering one-to-many and many-to-one rows;
3. selects hard human sentence groups first;
4. broadens to other human sentence groups when a language floor is short;
5. uses synthetic sentence groups only for any remaining floor deficit;
6. removes training rows one character edit from evaluation on either side;
7. minimally trims low-value training rows only when indivisible groups leave a
   final denominator just below the required ratio;
8. fails if any language still has less than 7.5% test or 2.5% validation.

The final denominator includes human, synthetic, sentence, and lexical rows.
This prevents pivot growth from making evaluation statistically negligible.

### 7. Independent Validation

`validate_experiment.py` recomputes split ratios and leakage properties from the
emitted CSV. It does not trust the split-builder report. The same validator runs
on Andromeda before tokenizer setup; Slurm `afterok` dependencies turn the data
contract into a hard training gate.

### 8. NLLB Training

`setup_tokenizer_sweep.py` learns the 8k Formosan extension, adds metadata tags,
resizes NLLB, audits tokenization, and smoke-tests generation.
Its legacy NLLB surgery implementation remains versioned under
`scripts/mt/nllb/prelims/`; the cluster mirror is checksum-pinned before use.

`train_directional_nllb.py` trains one direction per checkpoint with
source-bucket weighting and language-temperature sampling. It performs fixed
per-language generation validation, selects best by chrF2, supports early
stopping, and atomically saves full resume state.

`evaluate_directional.py` evaluates the entire test split and writes predictions
plus global, language, source-bucket, dialect, and length-bin metrics.

## Data Products

| Product | Lifetime | Versioned |
|---|---|---|
| Released 2025 pairwise corpora | Historical release | Yes |
| Downloaded XML and raw/filtered rebuild CSVs | Regenerable | No |
| DeepL JSONL response caches | Expensive source artifact | No; checksum and back up |
| Named final no-Bible corpora | Current experiment input | No; manifest/checksum yes |
| Split/validation reports | Current provenance | Copied with corpus; summary manifest in git |
| Tokenizers, checkpoints, predictions | Cluster experiment output | No |
| Experiment configuration and job graph | Reproducibility record | Yes |

## Legacy Boundary

`processed_corpora/`, `scripts/mt/`, historical Slurm submitters, DAE/LoRA code,
and the May model cards remain intentionally. They reproduce earlier papers and
architecture searches. New production work should use the named corpus builder,
`in_domain_hard`, `configs/default_experiment.json`, and
`submit_v1_spm8k_directional.sh`.
