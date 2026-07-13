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

The pipeline excludes exactly
`FormosanBank/Formosan-Taiwan-Bible-Society-Bibles`, discovers all configured
DeepL keys in numeric order, reuses response caches by content key, preserves
`pivot_origin`/`pivot_direction`, and requires zero missing eligible synthetic
rows before these artifacts are considered complete.

## Headline Artifacts

| Artifact | Rows | SHA-256 |
|---|---:|---|
| public English hard | 451,049 | `277d8fd936334f7ef73bad74351f267816ac4ce0bed792f434555fe3e75afec6` |
| public Chinese hard | 485,406 | `5989474b6a9357a5aa2c0a9723af89f2761e6293971421cd12097870cd7bc37a` |
| private English hard | 751,474 | `1ecd445b870d0039dfcb2ab2fc7f5dacf1dc5c3af56f2896d29663efdcd36955` |
| private Chinese hard | 793,692 | `ca2af32a044ee815a966b120b175a045529efadf122d03393e2113d9dba5ddf7` |

The authoritative machine-readable inventories are each build's
`mt_build_manifest.json`, `processed_corpora/pivot/pivot_manifest.json`, and
split-directory `report_all_tiers.json`.

## Evaluation Policy

- DeepL rows and XML lexemes/morphemes: training only.
- Evaluation references: human translations only.
- Human split target: approximately 90% train, 7.5% test, 2.5% validation.
- Small-language desired floors: 500 test and 150 validation rows.
- Exact normalized Formosan, target, and pair overlap: zero.
- Punctuation/spacing skeleton overlap: zero.
- One-character insertion/deletion/substitution train/eval overlap: removed on
  both Formosan and target sides.

The four rebuilt human-reference subsets land near 88-90% train, 7.6-9.3%
test, and 2.8-3.4% validation. Public Puyuma English has 498 test rows because
indivisible leakage groups and final one-edit pruning take precedence over
meeting the nominal floor exactly.

## Storage

Do not delete paid caches under
`corpus_builds/*/processed_corpora/pivot/cache/`, final inputs under
`corpus_builds/*/pivot_corpora_final/`, or build/split manifests. Ignored
`.github_metadata_cache` and `.github_raw_xml_cache` directories are disposable
after a successful checksummed build.
