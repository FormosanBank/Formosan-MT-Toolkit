# No-Bible Corpus Rebuild

This is the canonical corpus pipeline v2 release procedure. Generated CSVs and
paid DeepL caches are intentionally ignored by git. A completed build is
self-describing through its checksummed manifests and portable provenance
bundle.

## Rebuild

Prerequisites:

- a clean checkout of this repository;
- `GITHUB_TOKEN` in `.env` for private repositories;
- `DEEPL_API_KEY` plus any numbered `DEEPL_API_KEY_N` variables.

A sibling `../FormosanBank` checkout is optional. It is accepted only when its
clean HEAD equals the pinned QC commit.

Run the complete fresh build:

```bash
./build_corpora.sh --build-public-private --with-pivot --exclude-bible
```

Rebuild all data while reusing completed translations and making DeepL calls
only for new eligible rows:

```bash
./build_corpora.sh --build-public-private --with-pivot \
  --exclude-bible
```

Rebuild without any DeepL network calls, failing if the caches do not cover
every eligible row:

```bash
./build_corpora.sh --build-public-private --with-pivot \
  --pivot-skip-translation --exclude-bible
```

Regenerate only the final leakage-controlled splits and checksummed manifests,
without downloading XML, rerunning cleaning, or calling DeepL:

```bash
python scripts/local/build_mt_corpus.py --corpus-name public_no_bible \
  --public --exclude-bible --with-pivot --resplit-only --tiers in_domain_hard
python scripts/local/build_mt_corpus.py --corpus-name private_no_bible \
  --exclude-bible --with-pivot --resplit-only --tiers in_domain_hard
```

The pipeline excludes exactly
`FormosanBank/Formosan-Taiwan-Bible-Society-Bibles`, discovers all configured
DeepL keys in numeric order, reuses response caches by content key, preserves
`pivot_origin`/`pivot_direction`, and requires zero missing eligible synthetic
rows before these artifacts are considered complete. Responses that fail
synthetic target-script, identity, markup, or fertility checks are excluded and
written to checksummed `pivot_rejections_<direction>.csv` quarantine ledgers.
They do not count as missing because the provider response was received and
audited; absent cache entries and provider errors remain release-blocking.

Each public/private build records one immutable
`source_repository_snapshot.json`. Every language fetch reuses those exact
repository commits, and private tree discovery is limited to `Final_XML`.
This keeps one build internally consistent and avoids resolving and traversing
every organization repository once per language.

## Release Outputs

Each named build produces:

- `pivot_corpora_final/big_corpus_en_in_domain_hard.csv`;
- `pivot_corpora_final/big_corpus_zh_in_domain_hard.csv`;
- `mt_build_manifest.json`;
- `pivot_corpora_final/provenance/bundle_manifest.json`;
- split, independent validation, and TAME-MT exposure reports.

Rows and hashes are release outputs, not constants in source documentation.
Read them from the build and bundle manifests.

## Evaluation Policy

- XML lexemes/morphemes: training only.
- DeepL/synthetic rows: training only.
- Evaluation: human sentence references only.
- Final per-language split minimum: 7.5% test and 2.5% validation, measured
  against every emitted row including synthetic training rows.
- Small-language desired floors: 500 test and 150 validation rows.
- Exact normalized Formosan, target, and pair overlap: zero.
- Punctuation/spacing skeleton overlap: zero.
- One-character insertion/deletion/substitution train/eval overlap: removed on
  both Formosan and target sides.
- Character 4-gram Jaccard conflicts at or above 0.82: zero across all split
  boundaries.
- TAME-MT exact source/target/pair overlap and exposure at 0.95: zero in both
  translation directions within every `lang_code` task.

The builder expands the human sentence candidate pool when the strict hard pool
cannot meet those final-corpus percentages, with a declared small-language
fallback. It fails rather than silently emitting an undersized or synthetic
evaluation split.

## Historical Warning

`formosan_mt_experiments/manifests/no_bible_v1_20260712.json` and the July 2026
row counts describe the superseded v1 experiment. They predate the standard-tier
and provenance repairs and are retained only to reproduce the published model
history. Do not train new models from those CSVs.

## Storage

Do not delete paid caches under
`corpus_builds/*/processed_corpora/pivot/cache/`, final inputs under
`corpus_builds/*/pivot_corpora_final/`, or build/split manifests. Ignored
`.github_metadata_cache` and `.github_raw_xml_cache` directories are disposable
after a successful checksummed build.
