# Public No-Bible Corpus Rebuild

This procedure builds the public corpus pipeline v3 release. Generated XML,
CSVs, paid DeepL caches, and provenance manifests stay under the ignored
`corpus_builds/` directory.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

`GITHUB_TOKEN` is optional for public data and recommended to avoid the low
anonymous API limit. `DEEPL_API_KEY` is required only when `--with-pivot` needs
translations that are not already cached.

The builder accepts a sibling `../FormosanBank` checkout only when its clean
HEAD exactly matches the pinned QC commit. Otherwise it downloads and verifies
the pinned QC source.

## Full Build

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible
```

This refreshes repository heads, reuses valid content-addressed stages, reuses
DeepL responses by content key, and reruns every downstream stage affected by
changed XML or code.

Force corpus stages to rerun while still reusing paid DeepL responses:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible \
  --no-stage-cache
```

Use existing downloads and caches without GitHub or DeepL calls:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible \
  --skip-fetch \
  --pivot-skip-translation
```

Do not use `--skip-fetch` when changing acquisition or exclusion policy. It is
appropriate only when the existing build has the intended immutable source
snapshot.

## Split-Only Rebuild

After a split-policy change, rebuild final splits without fetching XML,
rerunning cleaning, or calling DeepL:

```bash
python scripts/local/build_mt_corpus.py \
  --corpus-name public_no_bible \
  --public \
  --exclude-bible \
  --with-pivot \
  --resplit-only \
  --tiers in_domain_hard
```

The source aggregate and pivot manifests must already be complete.

## Performance And Logs

Three language preparation pipelines run concurrently by default. Use
`--language-workers 1` for serial debugging. Normal output contains one
progress display plus stage timing and rule summaries. Full child commands and
raw output are written under `corpus_builds/public_no_bible/logs/`. Add
`--verbose` only when live subprocess output is needed.

Raw XML is cached by verified Git blob ID. A full build resolves each public
repository snapshot once and parses each XML blob once before routing it by
language. Large internal tables use checksummed Parquet companions while CSV
remains the release format.

If GitHub rate limits a run, resume with the existing partial cache and lower
download concurrency:

```bash
./build_corpora.sh \
  --corpus-name public_no_bible \
  --public \
  --with-pivot \
  --exclude-bible \
  --fetch-workers 2 \
  --keep-downloaded
```

## Release Outputs

```text
corpus_builds/public_no_bible/
  mt_build_manifest.json
  pivot_corpora_final/
    big_corpus_en_in_domain_hard.csv
    big_corpus_zh_in_domain_hard.csv
    provenance/
      bundle_manifest.json
      mt_build_manifest.json
```

The provenance directory also contains split validation, TAME-MT exposure,
pivot, source snapshot, standardization, and configuration records. Row counts
and hashes belong in these generated manifests, not in source documentation.

The build fails when acquisition is incomplete, row accounting does not
conserve inputs, unresolved pivots remain, evaluation contains lexical-like
rows, language/source split ratios are missed, leakage gates fail, or release
hashes cannot be written. Human sentence references are preferred, but a
validated synthetic sentence may be used when a source stratum lacks enough
human parallel rows.

## Authorized Nonpublic Sources

The orchestrator can ingest repositories visible to an authorized GitHub
token. Private discovery accepts top-level `Final_XML/` and `XML/` trees. Those
source snapshots, inventories, corpora, manifests, metrics, and models must
remain outside this repository. Use a separate named build and external
access-controlled storage. Never commit or attach them to a public pull request.
