# No-Bible Corpus Rebuild

This document identifies the canonical public/private corpus matrix rebuilt on
2026-07-12/13. Generated CSVs and paid DeepL caches are intentionally ignored
by git; this tracked record makes their provenance and expected outputs
auditable.

## Rebuild

Prerequisites:

- sibling `../FormosanBank` checkout for canonical QC;
- `GITHUB_TOKEN` in `.env` for private repositories;
- `DEEPL_API_KEY` plus any numbered `DEEPL_API_KEY_N` variables.

Run the complete fresh build:

```bash
./build_corpora.sh --build-public-private --with-pivot --exclude-bible
```

Resume without spending DeepL quota after caches are complete:

```bash
./build_corpora.sh --build-public-private --with-pivot \
  --pivot-skip-translation --exclude-bible --keep-build-output
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
rows before these artifacts are considered complete.

## Headline Artifacts

| Artifact | Rows | SHA-256 |
|---|---:|---|
| public English hard | 461,875 | `579bd9bb84cf0ef91ea45888a7df2d766acc64785f79217fa3b2525051108118` |
| public Chinese hard | 485,369 | `b16eb04e4dcc43b7c2412672f17e5eb3505853f308a3ed075804d9d1a592ac04` |
| private English hard | 759,493 | `b9c0ad5147b116abcb7e9cef3878a5c2f25a282c47e5dfc85a1bde7d86b91db9` |
| private Chinese hard | 791,330 | `39a69195b92a608512f991b1e67f0132cdcb59c2092c6155ef747c463b59cddd` |

The authoritative machine-readable inventories are each build's
`mt_build_manifest.json`, `processed_corpora/pivot/pivot_manifest.json`, and
split-directory `report_all_tiers.json`.

## Evaluation Policy

- XML lexemes/morphemes: training only.
- Evaluation references prefer human translations. Synthetic sentence
  references are admitted only for a per-language residual shortfall after all
  eligible human sentence groups are exhausted; reports quantify these rows.
- Final per-language split minimum: 7.5% test and 2.5% validation, measured
  against every emitted row including synthetic training rows.
- Small-language desired floors: 500 test and 150 validation rows.
- Exact normalized Formosan, target, and pair overlap: zero.
- Punctuation/spacing skeleton overlap: zero.
- One-character insertion/deletion/substitution train/eval overlap: removed on
  both Formosan and target sides.

The builder expands the human sentence candidate pool when the strict hard pool
cannot meet those final-corpus percentages. It fails rather than silently
emitting an undersized evaluation split. Lexemes remain training-only.

## Storage

Do not delete paid caches under
`corpus_builds/*/processed_corpora/pivot/cache/`, final inputs under
`corpus_builds/*/pivot_corpora_final/`, or build/split manifests. Ignored
`.github_metadata_cache` and `.github_raw_xml_cache` directories are disposable
after a successful checksummed build.
