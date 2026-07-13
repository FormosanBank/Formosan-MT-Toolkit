# Formosan-MT-Toolkit

Reproducible corpus construction and NLLB-200 training for multilingual
Formosan machine translation. The current production workflow builds separate
public and private/all-repository corpora, excludes the exact Taiwan Bible
Society repository, completes English/Chinese coverage with cached DeepL
pivots, creates leakage-resistant evaluation splits, and trains four
unidirectional SPM8k models per corpus variant.

The repository also retains the corpora and baseline code from the 2025
FormosanMT paper. Those released artifacts are historical inputs, not the
default 2026 experiment path.

## Current Production Path

```text
FormosanBank Final_XML
  -> fetch exact public/private scope
  -> FormosanBank QC + standardization
  -> extract FORM kindOf="standard"
  -> MT filtering + lexeme classification
  -> multilingual EN/ZH aggregates
  -> cached DeepL pivot completion
  -> connected-group hard splits
  -> independent corpus validation
  -> SPM8k NLLB setup
  -> f2en / en2f / f2zh / zh2f training
  -> final and best-checkpoint evaluation
```

The current no-Bible corpus inventory, checksums, split counts, training
configuration, and Andromeda job graph are tracked in
[`formosan_mt_experiments/manifests/no_bible_v1_20260712.json`](formosan_mt_experiments/manifests/no_bible_v1_20260712.json).
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component ownership and
[`docs/CURRENT_EXPERIMENT.md`](docs/CURRENT_EXPERIMENT.md) for the active flight.

## Repository Layout

| Path | Role |
|---|---|
| `build_corpora.sh` | Stable wrapper for the end-to-end corpus builder. |
| `scripts/local/` | Fetch, QC, XML extraction, pair filtering, aggregation, and orchestration. |
| `scripts/local/scripts/pivot/` | DeepL key rotation, caching, and pivot provenance. |
| `formosan_mt_experiments/` | Current split, validation, SPM8k training, evaluation, and Slurm stack. |
| `corpus_builds/<name>/` | Ignored self-contained public/private generated builds. |
| `processed_corpora/` | Released 2025 pairwise corpora and historical model inputs. |
| `scripts/mt/` | Historical NLLB/OpenNMT implementations retained for reproducibility. |
| `protected_corpora/` | Ignored paid-pivot snapshots plus tracked checksums; not the current build namespace. |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For repository development and linting, use `requirements-dev.txt` instead.

For XML fetching, configure `GITHUB_TOKEN`. For new pivot translations,
configure `DEEPL_API_KEY` and any numbered `DEEPL_API_KEY_N` variables. The
scripts load an ignored root `.env`, discover numbered keys in numeric order,
and never print key values.

The cleaner prefers a sibling `../FormosanBank` checkout so it runs the current
canonical QC package. It can sync the required QC tree when that checkout is
not present.

## Build Corpora

Build both isolated public/private variants with complete pivoting while
excluding exactly `FormosanBank/Formosan-Taiwan-Bible-Society-Bibles`:

```bash
./build_corpora.sh --build-public-private --with-pivot --exclude-bible
```

Reuse downloaded XML and paid translation caches without making DeepL calls:

```bash
./build_corpora.sh --build-public-private --with-pivot --exclude-bible \
  --skip-fetch --pivot-skip-translation
```

Regenerate only the final hard splits and manifests after split-policy changes:

```bash
python scripts/local/build_mt_corpus.py --corpus-name public_no_bible \
  --public --exclude-bible --with-pivot --resplit-only --tiers in_domain_hard
python scripts/local/build_mt_corpus.py --corpus-name private_no_bible \
  --exclude-bible --with-pivot --resplit-only --tiers in_domain_hard
```

Named builds do not overwrite one another. Full builds clear stale generated
outputs but preserve paid pivot caches unless explicitly told otherwise. Do not
combine `--skip-fetch` with a first-time Bible exclusion because stale fetched
XML could remain.

Fetches use bounded concurrency, retries, exponential backoff, and a shared
GitHub snapshot cache. If GitHub rate-limits a run, resume with existing files
and lower concurrency:

```bash
./build_corpora.sh --build-public-private --with-pivot --exclude-bible \
  --fetch-workers 2 --keep-downloaded
```

Detailed rebuild and storage instructions are in
[`docs/NO_BIBLE_CORPUS_REBUILD.md`](docs/NO_BIBLE_CORPUS_REBUILD.md).

## Split Contract

The model-facing artifact is
`big_corpus_<en|zh>_in_domain_hard.csv`. Its contract is enforced locally and
again on Andromeda before any GPU dependency can start:

- extract source text from `FORM kindOf="standard"`;
- route XML lexemes and morphemes to training only;
- reserve at least 7.5% test and 2.5% validation for every language, measured
  against all final rows;
- prefer human evaluation references, admitting synthetic sentence references
  only when eligible human hard groups cannot meet a language floor;
- keep exact normalized and punctuation/spacing skeleton source, target, and
  pair overlap at zero across train/evaluation;
- remove train rows one insertion, deletion, or substitution away from an
  evaluation source or target;
- keep connected one-to-many and many-to-one equivalence groups in one split;
- report every fallback, synthetic evaluation row, removed conflict, count,
  and checksum.

Validate a built corpus directly:

```bash
python formosan_mt_experiments/scripts/validate_experiment.py \
  --input corpus_builds/public_no_bible/pivot_corpora_final/big_corpus_en_in_domain_hard.csv \
  --target-lang english --min-test-ratio 0.075 --min-validate-ratio 0.025
```

Verify the tracked experiment snapshot against all four local files and
checksums:

```bash
python formosan_mt_experiments/scripts/verify_experiment_manifest.py \
  --manifest formosan_mt_experiments/manifests/no_bible_v1_20260712.json \
  --check-files
```

## Train On Andromeda

Transfer each final build with its provenance files, sync
`formosan_mt_experiments/`, then submit one flight per corpus:

```bash
CORPUS_NAME=public_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  formosan_mt_experiments/slurm/submit_v1_spm8k_directional.sh
CORPUS_NAME=private_no_bible RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  formosan_mt_experiments/slurm/submit_v1_spm8k_directional.sh
```

The submitter requires a named corpus, validates EN and ZH on CPU, creates or
reuses the SPM8k tokenizer/model, trains all four directions, and queues final
and best evaluations with `afterok` dependencies. Submission state is
idempotent for a fixed run stamp, and the launcher writes a machine-readable
manifest after Slurm accepts the graph.

See [`formosan_mt_experiments/README.md`](formosan_mt_experiments/README.md) for
the training contract and metrics.

## Quality Checks

```bash
python -m unittest discover -s tests -v
ruff check scripts/local/*.py scripts/local/scripts/pivot \
  formosan_mt_experiments/scripts tests
python -m compileall -q scripts/local formosan_mt_experiments/scripts tests
bash -n build_corpora.sh formosan_mt_experiments/slurm/*.sh \
  formosan_mt_experiments/slurm/*.sl
```

Generated XML, corpora, caches, models, prediction files, and Slurm logs are
ignored. Commit source, documentation, small manifests, checksums, and released
historical corpora only. Paid DeepL caches need an off-machine backup; a second
directory on the same workstation is not disaster recovery.

## Citation

This repository accompanies **FormosanMT: A Multilingual Parallel Corpus of the
Formosan Language Family** ([COMPUTEL 2025](https://aclanthology.org/2025.computel-main.19/)).
Check each upstream corpus's metadata and permissions before redistribution.
