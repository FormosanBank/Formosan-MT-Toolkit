# Formosan-MT-Toolkit

MT tooling and corpus creation repo. It is part of the FormosanBank organization.

## Repository At A Glance

| Field | Value |
| --- | --- |
| Type | Machine translation corpus and tooling |
| Language(s) | Formosan languages |
| Source | https://aclanthology.org/2025.computel-main.19/ |
| Main output | `processed_corpora/` released processed corpora |
| Status | Active FormosanBank repository; verify repo-specific notes before regenerating or reusing outputs. |

## Source And Scope

The upstream source or related project page is https://aclanthology.org/2025.computel-main.19/. Review the detailed notes below for source-specific handling, permissions, and processing caveats.

## What Is In This Repository

- `requirements.txt` - repository file used by this corpus or tool.
- `processed_corpora/` - repository content used by this corpus or tool.
- `scripts/` - repository content used by this corpus or tool.

## Outputs

The main expected output is: `processed_corpora/` released processed corpora. When XML is present, it is intended to follow the [FormosanBank XML format](https://ai4commsci.gitbook.io/formosanbank/the-bank-architecture/formosanbank-xml-format).

## Reproduce Or Update

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use the repo-specific commands and notes below when rebuilding data or generated artifacts. Avoid overwriting checked-in outputs until the regenerated files have been reviewed.

## Quality Checks

Run any repo-specific tests, validation scripts, or manual checks described below. For XML corpora, use the FormosanBank QC tools where applicable before publishing regenerated XML.

## Citation And Reuse

Citation, licensing, and permission requirements can vary by source. Check the source notes, existing documentation, and upstream resource terms before reusing data from this repository.

## Contributing

Issues and pull requests are welcome for corrections, documentation improvements, reproducibility fixes, and source metadata updates. Keep generated outputs reviewable and document any manual cleanup steps.

## Detailed Notes

The following section preserves the repository-specific documentation that existed before this README was standardized.

## FormosanMT

This repository accompanies the paper:

**FormosanMT: A Multilingual Parallel Corpus of the Formosan Language Family**

It contains the released corpora, the XML-to-corpus pipeline used to build them from FormosanBank, and the model training and evaluation code used for the machine translation experiments.

A detailed tutorial on using this architecture in Google Colab can be found [Here](https://medium.com/@hunterschep/fine-tuning-nllb-200-for-a-new-language-in-2025-fcae209d9980)

### What's In This Repo

- `processed_corpora/`
  Released pairwise corpora used in the paper's training runs. The current release contains 19 training-ready corpora with 395,710 aligned pairs.
- `processed_corpora/untrained/`
  Additional released English corpora not used in the paper's training runs. The current release contains 11 such corpora with 47,412 aligned pairs.
- `processed_corpora/helpers/`
  Utilities for building multilingual aggregate files from the released pairwise corpora.
- `processed_corpora/pivot/`
  Pivot-specific helper scripts. Generated DeepL pivot CSVs/caches are intentionally ignored and should be restored from `protected_corpora/` or rebuilt only when needed.
- `pivot_corpora_final/`
  Canonical working location for the final DeepL-pivoted English and Chinese multilingual corpora. The large CSVs are generated artifacts and are ignored by git.
- `protected_corpora/`
  Local protected checksum copies of the expensive DeepL pivot corpora and caches. Git tracks manifests/checksums only; mirror the ignored payloads off-box for real disaster recovery.
- `formosan_mt_experiments/`
  Self-contained experimental stack for leakage-resistant splits, SPM tokenizer sweeps, directional NLLB training, DAE pre-adaptation, LoRA experiments, and tiered evaluation.
- `scripts/local/`
  The XML acquisition, cleaning, extraction, and filtering pipeline.
- `scripts/local/scripts/pivot/`
  DeepL-based pivot augmentation for multilingual aggregate corpora.
- `scripts/mt/nllb/`
  Bilingual NLLB-200 setup, training, and evaluation.
- `scripts/mt/nllb_multilingual/`
  Multilingual NLLB-200 training and evaluation.
- `scripts/mt/opennmt/`
  OpenNMT baseline preparation.
- `formosan_mt_experiments/slurm/`
  Tracked Andromeda Slurm templates for the newer experiment stack. Local one-off launchers under `bash/` are ignored because they are machine/user specific.

The release contains 30 processed corpora in total: 15 Formosan-Chinese corpora and 15 Formosan-English corpora.

Each processed CSV stores:

`<formosan_lang>`, `english|chinese`, `source`, `kindOf`, `dialect`, `row_type`, `split`

The `split` column is the released train/validate/test split used by the experiment code. The `row_type` column distinguishes lexical entries (`lexeme`) from sentence-level pairs (`sentence`).

### Architecture

The corpus pipeline is:

`FormosanBank XML -> fetch_xml.py -> clean_xml.py -> make_corpus.py -> filter_split_corpus.py -> processed CSV -> NLLB/OpenNMT prep`

Main stages:

1. `scripts/local/fetch_xml.py`
   Downloads XML from FormosanBank and filters by source language.
2. `scripts/local/clean_xml.py`
   Syncs the official FormosanBank QC scripts and runs them in place on the downloaded XML.
3. `scripts/local/make_corpus.py`
   Extracts sentence and word rows from `FORM kindOf="standard"` by default and preserves `source`, `kindOf`, `dialect`, `row_type`, and `xml_id` metadata.
4. `scripts/local/filter_split_corpus.py`
   Normalizes text, removes noisy rows, preserves XML lexical rows, labels obvious vocabulary/dictionary rows as `lexeme`, deduplicates, filters language-aware length-ratio outliers, and assigns train/validate/test splits. It writes an audit report with drop counts and reject samples beside each processed CSV.

Two properties of the released data are important for the experiments:

- `lexeme` rows are routed to `train` only; if a train lexeme would still leak an exact or skeleton source/target/pair key into validation/test, that train row is pruned rather than moved into eval.
- Sentence splits are assigned by punctuation/spacing-insensitive shared source or target keys to reduce train/eval leakage across one-to-many, many-to-one, and near-duplicate pairs.
- Aggregate and pivot corpora keep `row_type`, so the hard-split builder can continue excluding XML lexical rows from validation/test after the pairwise CSVs are combined.

### Repository Hygiene

This repository intentionally separates source code and released corpora from generated artifacts:

- Keep source scripts, configs, small manifests, checksums, and released processed corpora in git.
- Do not commit virtual environments, caches, generated model checkpoints, tokenizer/model artifacts, Slurm logs, prediction dumps, or DeepL cache payloads.
- Keep final costly DeepL pivot CSVs in `pivot_corpora_final/`; they are ignored by git.
- Keep protected local backups in `protected_corpora/deepl_pivots/`; git tracks only `README.md`, `MANIFEST.json`, and `SHA256SUMS`.
- Before deleting or copying protected pivot data, run `shasum -a 256 -c protected_corpora/deepl_pivots/SHA256SUMS` from the `protected_corpora/deepl_pivots/` directory.

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you only use the released processed corpora, no GitHub token is required.

If you want to rebuild the corpora from XML, set a GitHub token first:

```bash
export GITHUB_TOKEN=...
```

The local corpus scripts also load `.env` from the repository root if you prefer to store the token there.

If you want to build the optional DeepL pivot corpora, set one or more DeepL API keys:

```bash
export DEEPL_API_KEY=...
export DEEPL_API_KEY_2=...
```

The pivot builder also loads these keys from `.env` at the repository root.

### Language Naming

Processed CSV filenames use three-letter language codes such as `ami`, `bnn`, and `tay`.

The bilingual NLLB trainer uses canonical language names in `--src-lang`:

`ami->amis`, `bnn->bunun`, `ckv->kavalan`, `dru->rukai`, `pwn->paiwan`, `pyu->puyuma`, `ssf->thao`, `sxr->saaroa`, `szy->sakizaya`, `tao->tao`, `tay->atayal`, `trv->seediq`, `tsu->tsou`, `xnb->kanakanavu`, `xsy->saisiyat`

### Rebuild The Corpora

The preferred rebuild entrypoint is now `scripts/local/build_mt_corpus.py`. The
top-level shell script is only a compatibility wrapper around it:

```bash
./build_corpora.sh
```

This runs the end-to-end MT corpus pipeline:

1. remove stale `downloaded_<lang>/` directories unless `--keep-downloaded` is passed, then download `Final_XML/` files for each Formosan language;
2. clean with the real FormosanBank QC package (`clean_xml.py` plus
   `standardize.py --copy`);
3. extract sentence and word-level rows from standard forms to raw pairwise CSVs;
4. run MT-specific filtering, deduplication, lexeme routing, and train/eval overlap pruning on each pair;
5. build multilingual English/Chinese aggregate corpora while preserving `row_type`;
6. optionally rebuild pivot corpora from DeepL cache or new DeepL translations;
7. build leakage-resistant hard split tiers with exact and punctuation/spacing
   skeleton overlap checks and lexeme-in-eval validation.

By default the cleaner uses a sibling `../FormosanBank` checkout when present,
so current FormosanBank QC imports such as `QC.validation._dialect_inventory`
work correctly. If that checkout is not available, the script can sync the
minimal `QC/` and `Orthographies/` trees into `scripts/.formosan_qc_repo/`
using `GITHUB_TOKEN`.

Useful rebuild commands:

```bash
# Rebuild a small language subset from already downloaded XML.
./build_corpora.sh --languages ami,tay --skip-fetch

# Public-only rebuild from FormosanBank/Corpora XML.
./build_corpora.sh --public

# Rebuild through pivot outputs using existing DeepL cache only.
./build_corpora.sh --skip-fetch --with-pivot --pivot-skip-translation

# Full rebuild with pivot, excluding Taiwan Bible Society Bible XML at fetch time.
./build_corpora.sh --with-pivot --exclude-bible

# Same Bible-excluded rebuild, but do not spend DeepL characters.
./build_corpora.sh --with-pivot --pivot-skip-translation --exclude-bible

# Build separate public and private/all-data no-Bible corpora for model comparison.
./build_corpora.sh --build-public-private --with-pivot --exclude-bible

# Same public/private comparison build, but do not spend DeepL characters.
./build_corpora.sh --build-public-private --with-pivot --pivot-skip-translation --exclude-bible

# Rebuild everything from current GitHub XML, but do not spend DeepL characters.
./build_corpora.sh --with-pivot --pivot-skip-translation

# Build only the public no-Bible corpus into its own namespace.
./build_corpora.sh --public --with-pivot --exclude-bible --corpus-name public_no_bible

# Build only the private/all-data no-Bible corpus into its own namespace.
./build_corpora.sh --with-pivot --exclude-bible --corpus-name private_no_bible

# Plan a full command sequence without writing anything.
./build_corpora.sh --languages ami --dry-run
```

`--exclude-bible` is applied during fetch. Use it with the default fresh-download
behavior so stale Bible XML is removed from `downloaded_<lang>/`; do not combine
it with `--skip-fetch` when the goal is to remove Bible data from the corpus.

Named builds write every generated artifact under `corpus_builds/<name>/`, so
public and private/all-data runs do not overwrite each other. The comparison
outputs to train from are:

- `corpus_builds/public_no_bible/pivot_corpora_final/big_corpus_en.csv`
- `corpus_builds/public_no_bible/pivot_corpora_final/big_corpus_zh.csv`
- `corpus_builds/private_no_bible/pivot_corpora_final/big_corpus_en.csv`
- `corpus_builds/private_no_bible/pivot_corpora_final/big_corpus_zh.csv`

Hard split tiers for those corpora live under
`corpus_builds/<name>/formosan_mt_experiments/data/splits_en_v1/` and
`corpus_builds/<name>/formosan_mt_experiments/data/splits_zh_v1/`. In this
context, `private` means the default all-repos fetch available to your
`GITHUB_TOKEN`, not the public `FormosanBank/Corpora` release snapshot.

Named full rebuilds remove stale generated CSVs and split files before
rebuilding, while preserving existing pivot cache files. Use
`--keep-build-output` only for deliberate incremental debugging. Named pivot
builds also read the existing root cache at `processed_corpora/pivot/cache/`
without writing to it, so `--pivot-skip-translation` can reuse protected DeepL
work. Disable that with `--no-shared-pivot-cache` if you need a fully isolated
cache experiment.

Each build writes `mt_build_manifest.json` in its build root. For completed
non-dry runs, the manifest includes row counts, byte sizes, and SHA-256 checksums
for the final aggregate and split CSVs. Use `--skip-artifact-checksums` only when
you need to avoid hashing large files during exploratory runs.

The generated root-level pairwise and aggregate CSVs are ignored by git:

- `raw_corpora/*.csv`
- `processed_corpora/*_processed.csv`
- `processed_corpora/big_corpus*.csv`
- `processed_corpora/filter_reports/`
- `formosan_mt_experiments/data/splits_*_v1/`
- `pivot_corpora_final/*.csv` when `--with-pivot` is used
- `corpus_builds/` for named public/private comparison builds

To rebuild a single pair manually, the underlying stages remain available:

```bash
python scripts/local/fetch_xml.py --src-lang ami --public --exclude-bible
python scripts/local/clean_xml.py --src-lang ami
python scripts/local/make_corpus.py \
  --xml-dir downloaded_ami \
  --target chinese \
  --out raw_corpora/ami_zh.csv
python scripts/local/filter_split_corpus.py \
  --input raw_corpora/ami_zh.csv \
  --output processed_corpora/ami_zh_processed.csv \
  --workers 32
```

The rebuild path creates intermediate XML in `downloaded_*`, raw extracted CSVs
in `raw_corpora/`, filtered pairwise CSVs in `processed_corpora/`, aggregate
corpora in `processed_corpora/big_corpus_*.csv`, and hard split tiers in
`formosan_mt_experiments/data/splits_*_v1/`. Filtering reports are written under
`processed_corpora/filter_reports/`; inspect `summary.json` and
`reject_samples.csv` when changing filtering thresholds.

The hard split builder ignores the old pairwise `split` assignments when it
creates tiered experiment files, but it does use the preserved `row_type`.
Rows marked `lexeme` are never assigned to validation/test in `lexical`,
`in_domain_hard`, or `hard_global`; they remain train-only unless removed
because they would leak an exact or skeleton source/target/pair key into eval.

### Replicate The NLLB Experiments

The NLLB experiments consume the released `split` column directly. You do not need to re-split the data.

#### 1. Build multilingual aggregate files

```bash
python processed_corpora/helpers/big_corpus_for_tokenizer.py
```

This writes:

- `processed_corpora/big_corpus_zh.csv`
- `processed_corpora/big_corpus_en.csv`
- `processed_corpora/big_corpus_combined.csv`

These aggregate files are built from `processed_corpora/` only; they do not include `processed_corpora/untrained/`. This matches the paper training setup.

The helper also accepts explicit input and output directories, and can normalize already-aggregated corpora such as the pivot outputs:

```bash
python processed_corpora/helpers/big_corpus_for_tokenizer.py \
  --input-dir processed_corpora/pivot \
  --output-dir /tmp/formosan_pivot_big_corpora
```

This is header-driven rather than position-driven, so it works with both the released pairwise corpora and the pivot CSVs that append provenance columns. It also writes a tokenizer-compatible `big_corpus_combined.csv` with columns `lang_code, formosan_sentence, chinese_sentence, english_sentence, source, dialect, split`. Rows with an empty target sentence are skipped and reported.

#### 2. Optional: build DeepL pivot corpora

The pivot builder creates synthetic target text by translating the multilingual aggregate corpora through DeepL:

- `processed_corpora/pivot/big_corpus_zh_pivot.csv`
- `processed_corpora/pivot/big_corpus_en_pivot.csv`

Run:

```bash
python scripts/local/scripts/pivot/pivot.py \
  --api-key-env DEEPL_API_KEY,DEEPL_API_KEY_2 \
  --directions both \
  --splits all
```

What it does:

- preserves all original rows from `big_corpus_zh.csv` and `big_corpus_en.csv`
- preserves the released `split` column on synthetic rows
- skips DeepL calls by default when the same `(lang_code, formosan_sentence)` already exists in the target corpus, which avoids paying for rows that already have a real translation and reduces leakage risk
- caches successful DeepL responses immediately in `processed_corpora/pivot/cache/`
- writes `processed_corpora/pivot/pivot_manifest.json`

If a run is interrupted or a key exhausts its quota, rerun the same command. Cached translations are reused and only missing texts are sent to DeepL.

The default output schema keeps the original core columns and appends pivot provenance fields such as `pivot_origin`, `pivot_provider`, `pivot_direction`, and `pivot_cache_key`. Use `--minimal-schema` if you want output shaped like the original aggregate CSVs.

Treat DeepL pivot outputs as costly artifacts. In this working tree, the canonical final working copies are:

- `pivot_corpora_final/big_corpus_en.csv`
- `pivot_corpora_final/big_corpus_zh.csv`

Protected local copies with checksums and provenance are stored under `protected_corpora/deepl_pivots/`. Git tracks only the README, manifest, and checksums from that directory; the actual CSV/cache payloads are intentionally ignored. Verify a local protected payload with:

```bash
cd protected_corpora/deepl_pivots
shasum -a 256 -c SHA256SUMS
```

This is still a local copy on the same machine, so mirror `protected_corpora/deepl_pivots/` to Andromeda, external storage, or another durable backup location before deleting or regenerating any pivot data.

Useful flags:

- `--respect-usage-limit` reads the DeepL `/usage` endpoint and caps the run to the remaining reported characters
- `--skip-translation` rebuilds the pivot CSVs from the existing cache without making new API calls
- `--dry-run` prints the translation plan without calling DeepL or writing output

Pivot helper scripts:

```bash
python processed_corpora/pivot/summary_stats.py
python processed_corpora/pivot/standardize_dialects.py
```

`standardize_dialects.py` rewrites the pivot CSVs in place, so treat it as a normalization pass on generated data rather than a read-only report.

If you want a tokenizer-compatible combined pivot corpus after this step, run:

```bash
python processed_corpora/helpers/big_corpus_for_tokenizer.py \
  --input-dir processed_corpora/pivot \
  --output-dir processed_corpora/pivot
```

This writes:

- `processed_corpora/pivot/big_corpus_zh.csv`
- `processed_corpora/pivot/big_corpus_en.csv`
- `processed_corpora/pivot/big_corpus_combined.csv`

#### 3. Initialize the custom NLLB tokenizer and model

Run this once before bilingual or multilingual fine-tuning:

```bash
mkdir -p artifacts
python scripts/mt/nllb/prelims/setup_formosan_nllb200.py \
  --input processed_corpora/big_corpus_combined.csv \
  --output-prefix artifacts/formosan_multilingual_nllb \
  --add-mode spm \
  --spm-vocab 16384
```

This creates:

- `artifacts/formosan_multilingual_nllb_tokenizer/`
- `artifacts/formosan_multilingual_nllb_model/`

For pivot-augmented tokenizer setup, first regenerate or restore `processed_corpora/pivot/big_corpus_combined.csv`, then point `--input` at that file. If you only need the final English/Chinese pivot corpora, use `pivot_corpora_final/` or restore from `protected_corpora/deepl_pivots/generated/`.

Generated model/tokenizer artifacts should live in ignored output directories such as `artifacts/` or `/scratch/.../formosan_mt_experiments/data/`, not inside `scripts/mt/nllb/prelims/`.

For the newer leakage-resistant experimental workflow, see `formosan_mt_experiments/README.md`. That stack builds tiered eval splits from `pivot_corpora_final/big_corpus_en.csv` and keeps generated split/model/report outputs under `formosan_mt_experiments/data/` and `formosan_mt_experiments/reports/`.

#### 4. Bilingual NLLB

Example: Amis-English

```bash
python scripts/mt/nllb/training/train_formosan_nllb200.py \
  --src-lang amis \
  --tgt-lang english \
  --tokenizer artifacts/formosan_multilingual_nllb_tokenizer \
  --model artifacts/formosan_multilingual_nllb_model \
  --input processed_corpora/ami_en_processed.csv \
  --normalize \
  --steps 35000 \
  --batch-size 8 \
  --max-length 192 \
  --fp16 \
  --device cuda
```

Evaluation:

```bash
python scripts/mt/nllb/eval/eval_formosan_nllb200.py \
  --tokenizer runs/amis_english/<timestamp>/final \
  --model runs/amis_english/<timestamp>/final \
  --input processed_corpora/ami_en_processed.csv \
  --batch-size 16 \
  --max-length 192 \
  --beam 4 \
  --csv-out runs/amis_english/<timestamp>/eval.csv \
  --save-json runs/amis_english/<timestamp>/eval.json
```

To reproduce another bilingual pair, swap in the corresponding processed CSV and the correct canonical `--src-lang` value.

#### 5. Multilingual NLLB

Chinese-target multilingual training:

```bash
python scripts/mt/nllb_multilingual/training/train_formosan_multilingual_nllb200.py \
  --multilingual \
  --tgt-lang chinese \
  --tokenizer artifacts/formosan_multilingual_nllb_tokenizer \
  --model artifacts/formosan_multilingual_nllb_model \
  --input processed_corpora/big_corpus_zh.csv \
  --temperature 5 \
  --normalize \
  --steps 150000 \
  --batch-size 8 \
  --max-length 192 \
  --fp16 \
  --device cuda
```

Chinese-target multilingual evaluation:

```bash
python scripts/mt/nllb_multilingual/eval/eval_formosan_multilingual_nllb200.py \
  --multilingual \
  --tgt-lang chinese \
  --tokenizer runs/formosan_multilingual_to_chinese/<timestamp>/final \
  --model runs/formosan_multilingual_to_chinese/<timestamp>/final \
  --input processed_corpora/big_corpus_zh.csv \
  --batch-size 16 \
  --max-length 192 \
  --beam 4 \
  --normalize \
  --csv-out runs/formosan_multilingual_to_chinese/<timestamp>/eval.csv \
  --save-json runs/formosan_multilingual_to_chinese/<timestamp>/eval.json
```

English-target multilingual training:

```bash
python scripts/mt/nllb_multilingual/training/train_formosan_multilingual_nllb200.py \
  --multilingual \
  --tgt-lang english \
  --tokenizer artifacts/formosan_multilingual_nllb_tokenizer \
  --model artifacts/formosan_multilingual_nllb_model \
  --input processed_corpora/big_corpus_en.csv \
  --temperature 5 \
  --normalize \
  --steps 60000 \
  --batch-size 8 \
  --max-length 192 \
  --fp16 \
  --device cuda
```

English-target evaluation is the same as the Chinese-target evaluation command above, but with `--tgt-lang english` and `--input processed_corpora/big_corpus_en.csv`.

For augmented multilingual runs, the multilingual training and evaluation scripts can read generated pivot corpora directly because they only require `lang_code`, `formosan_sentence`, the target sentence column, and `split`. After restoring or rebuilding the pivot outputs, replace the aggregate input with:

- `processed_corpora/pivot/big_corpus_zh_pivot.csv` for Chinese-target runs
- `processed_corpora/pivot/big_corpus_en_pivot.csv` for English-target runs

The newer tracked Slurm templates live in `formosan_mt_experiments/slurm/`. Any local launchers under ignored `bash/` are machine-specific scratch notes, not portable defaults.

### Replicate The OpenNMT Baselines

The OpenNMT pipeline is pair-specific. Start from a processed pairwise CSV such as `ami_zh_processed.csv` or `ami_en_processed.csv`.

#### 1. Export a processed corpus into OpenNMT format

```bash
python scripts/mt/opennmt/setup/prep_opennmt.py \
  --csv processed_corpora/ami_zh_processed.csv \
  --outdir processed_corpora/opennmt/data
```

This creates a pair directory:

`processed_corpora/opennmt/data/ami_zh/`

The export contains:

- `raw/train/`, `raw/valid/`, `raw/test/`
- directional files such as `ami2zh.src`, `ami2zh.tgt`, `zh2ami.src`
- bidirectional files `bi.src` and `bi.tgt`
- test references such as `ami2zh.ref.zh` and `zh2ami.ref.ami`
- `user_defined_symbols.txt` with the target tags used for bidirectional training, for example `<2ami>,<2zh>`

The bidirectional files prepend a target tag on the source side:

- `<2zh> ...` for Formosan to Chinese or English
- `<2ami> ...` for Chinese or English to Formosan

#### 2. Train SentencePiece and encode the OpenNMT splits

Train a pair-specific SentencePiece model on the bidirectional training text:

```bash
python scripts/mt/opennmt/setup/prep_spm.py \
  --pair_dir processed_corpora/opennmt/data/ami_zh
```

Optional flags:

```bash
python scripts/mt/opennmt/setup/prep_spm.py \
  --pair_dir processed_corpora/opennmt/data/ami_zh \
  --vocab_size 8000 \
  --model_type unigram \
  --retrain
```

This writes:

- `spm.model`
- `spm.vocab`
- `train/bi.spm.src`, `train/bi.spm.tgt`
- `valid/bi.spm.src`, `valid/bi.spm.tgt`
- `test/bi.spm.src`, `test/bi.spm.tgt`

#### 3. Train the OpenNMT model

The checked-in OpenNMT configuration is `scripts/mt/opennmt/training/config.yaml`. The OpenNMT launcher used during development was machine-specific and is not part of the portable source tree.

If you write a new launcher, set at least:

- `DATA_ROOT`
- `RUN_ROOT`
- any helper script paths used for SentencePiece encode/decode

The training config/launcher should read a pair directory like `ami_zh` containing:

- `${DATA_ROOT}/ami_zh/train/bi.spm.src`
- `${DATA_ROOT}/ami_zh/train/bi.spm.tgt`
- `${DATA_ROOT}/ami_zh/valid/bi.spm.src`
- `${DATA_ROOT}/ami_zh/valid/bi.spm.tgt`

Example shape after you create a local launcher:

```bash
./your_train_opennmt_pair.sh ami_zh
```

#### 4. What the training script does

The training launcher:

- builds OpenNMT vocab files
- trains a bidirectional BiLSTM encoder-decoder with attention
- selects the best checkpoint by validation perplexity
- translates the held-out test set in both directions
- detokenizes the outputs with SentencePiece
- computes BLEU and chrF++
- writes per-direction TSV outputs and a combined 4-column comparison file

The main outputs are:

- `checkpoints/`
- `logs/train.onmt.log`
- `eval/scores.csv`
- `eval/<direction>.triples.tsv`
- `eval/bi.<pair>.4col.tsv`

#### 5. SentencePiece helper scripts

Two small helper scripts are included for manual debugging or custom evaluation:

- `scripts/mt/opennmt/training/spm_encode_stdin.py`
- `scripts/mt/opennmt/training/spm_decode_file.py`

These are the same utilities used by the OpenNMT training launcher when it prepares tagged source text for translation and decodes model outputs back into text.
