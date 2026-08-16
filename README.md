# Formosan MT Toolkit

Formosan MT Toolkit builds leakage-controlled Formosan-English and
Formosan-Traditional Chinese corpora from the public
[FormosanBank](https://github.com/FormosanBank/FormosanBank) XML collection.
It also contains reproducible directional NLLB-200 and MiLMMT training and
evaluation code.

The supported workflow is corpus pipeline v3. It keeps the supplied XML
`kindOf="standard"` tier, derives a separate model-facing standardization,
records every transformation, keeps approved standalone lexical rows in
training, and creates source-balanced hard evaluation splits. Human sentence
references are preferred; validated pivot sentences are used only when a
source lacks enough human parallel sentences.

## Quick Start

```bash
git clone https://github.com/FormosanBank/Formosan-MT-Toolkit.git
cd Formosan-MT-Toolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

A GitHub token is optional for public data but raises the API rate limit. Add
it to the ignored `.env` file:

```text
GITHUB_TOKEN=your_token
```

Build the public corpus without paid pivot translation:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --exclude-bible
```

Build the complete public English and Chinese corpora with DeepL pivoting:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible
```

Set `DEEPL_API_KEY` and optional numbered keys such as `DEEPL_API_KEY_2` in
`.env`. Keys are rotated in numeric order and their values are never printed.

To rebuild from downloaded XML and existing paid response caches without any
DeepL network calls:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible \
  --skip-fetch \
  --pivot-skip-translation
```

The last command fails if a required cache entry is missing. It never silently
publishes partial pivot output.

## Outputs

Each named build is isolated under `corpus_builds/<name>/`. The model-facing
files are:

```text
corpus_builds/public_no_bible/pivot_corpora_final/
  big_corpus_en_in_domain_hard.csv
  big_corpus_zh_in_domain_hard.csv
  provenance/
```

The provenance bundle contains source repository commits, blob hashes, the
pinned FormosanBank QC revision, cleaning and rejection ledgers, pivot status,
split diagnostics, TAME-MT exposure reports, configurations, and final artifact
checksums.

The pipeline excludes exactly
`FormosanBank/Formosan-Taiwan-Bible-Society-Bibles`. It does not use fuzzy
matching for repositories whose names happen to contain `bible`.

## Data Contract

Production builds fail unless all of these conditions hold:

- every fetched repository, XML file, and extracted row is accounted for;
- existing nonempty `kindOf="standard"` tiers are preserved;
- model text comes from the separate, versioned `formosan-mt` namespace;
- malformed, missing, rejected, quarantined, and deduplicated rows are logged;
- sentence-internal `<W>` annotations, explicit interlinear gloss targets, and
  detected morphological gloss structures are excluded from model corpora;
- appended grammatical analyses, wrong-language targets, and malformed escape
  artifacts are quarantined; uncertain English and unbalanced delimiters are
  retained for training but cannot enter evaluation;
- DeepL is limited to MT-eligible sentence rows with at least four Formosan and
  pivot-source units; completion is explicit and every response is validated;
- only structurally standalone `<W>` entries with natural target text may enter
  as train-only lexemes; ambiguous lexical structures and translations are
  quarantined; source provenance never determines row eligibility;
- each language reserves 10% of all deduplicated pairs for test and 5% for
  validation, using only eligible sentence rows; source-corpus targets
  follow the same proportions where eligible capacity permits;
- evaluation references are sentence-level, with human rows preferred before
  validated synthetic pivot rows;
- exact, punctuation-skeleton, one-edit, and high character n-gram leakage
  checks pass across split boundaries;
- exact TAME-MT source, target, and pair exposure at 0.95 is zero;
- final files and their build environment are checksummed.

See [Pipeline Architecture](docs/ARCHITECTURE.md) for stage ownership and
[Public Corpus Rebuild](docs/NO_BIBLE_CORPUS_REBUILD.md) for rebuild and cache
details.

## Training

The `formosan_mt_experiments/` package trains four unidirectional models:

| Direction | Input | Output |
|---|---|---|
| `f2en` | Formosan | English |
| `en2f` | English | Formosan |
| `f2zh` | Formosan | Traditional Chinese |
| `zh2f` | Traditional Chinese | Formosan |

The default profile uses NLLB-200 with the established 8k Formosan
SentencePiece extension and metadata controls. An experimental MiLMMT-46 1B
profile uses Xiaomi's native tokenizer and translation prompt with
response-only causal loss.

On a Slurm cluster, place a completed `pivot_corpora_final` directory under a
shared data root and submit from `formosan_mt_experiments/`:

```bash
export EXP_DIR="$PWD"
export PROJECT_DATA=/shared/formosan_parallel_corpora
export SCRATCH=/scratch/$USER/formosan_mt

CORPUS_NAME=public_no_bible \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

Select MiLMMT explicitly:

```bash
CORPUS_NAME=public_no_bible \
PROFILE="$EXP_DIR/configs/milmmt_1b_experiment.json" \
RUN_STAMP=$(date +%Y%m%d-%H%M%S) \
  slurm/submit_directional_experiment.sh
```

Cluster partitions, constraints, memory, time, and paths are environment
overrides. The tracked Slurm files contain no personal account or filesystem
defaults. See [Experiment Training](formosan_mt_experiments/README.md).

## Repository Layout

| Path | Purpose |
|---|---|
| `build_corpora.sh` | Stable public entrypoint. |
| `config/` | Versioned MT standardization policy. |
| `scripts/local/` | Acquisition, QC, extraction, filtering, pivoting, and release orchestration. |
| `formosan_mt_experiments/` | Split validation, model setup, training, evaluation, and Slurm launchers. |
| `tests/` | Corpus, training, inference, and provenance contract tests. |

Corpus rows, downloaded XML, credentials, paid caches, private repository
inventories, model files, and run outputs are intentionally not versioned.
Run `python scripts/check_public_release.py` before publishing. The full policy
is in [Public Data Policy](docs/DATA_POLICY.md).

## Development

```bash
pip install -r requirements-dev.txt
python scripts/check_public_release.py
pytest -q
ruff check scripts formosan_mt_experiments/scripts tests
python -m compileall -q scripts formosan_mt_experiments/scripts tests
bash -n build_corpora.sh formosan_mt_experiments/slurm/*.sh \
  formosan_mt_experiments/slurm/*.sl
```

## Data Rights And Citation

This toolkit does not grant rights to upstream corpus material or DeepL
outputs. Check the FormosanBank terms, each source corpus's metadata, and all
required citations before using or redistributing a generated dataset. See the
[license notice](LICENSE.md).

This repository accompanies **FormosanMT: A Multilingual Parallel Corpus of
the Formosan Language Family**
([COMPUTEL 2025](https://aclanthology.org/2025.computel-main.19/)).
