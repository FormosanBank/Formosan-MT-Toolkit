# Public Data Policy

This repository tracks source code, configuration, documentation, and tests.
It does not track corpus content or credentials, even when a corpus is already
public elsewhere.

## Never Commit

- `.env` files, API keys, tokens, passwords, or private keys;
- fetched XML and repository snapshots;
- raw, cleaned, pivoted, split, or rejected corpus rows;
- DeepL caches, provider responses, or paid-output inventories;
- private repository names, paths, commits, counts, or build manifests;
- model weights, checkpoints, predictions, metrics tied to nonpublic data, or
  cluster logs;
- personal machine, account, or storage paths.

Keep generated data under the ignored `corpus_builds/`,
`protected_corpora/`, `formosan_mt_experiments/data/`,
`formosan_mt_experiments/reports/`, or external storage roots.

## May Be Committed

- pipeline and training source code;
- small, data-free configuration files;
- documentation that uses public examples and portable paths;
- synthetic test fixtures that contain no source corpus text;
- aggregate public research results when their redistribution terms permit it.

## Release Check

Run this before every push intended for a public branch:

```bash
python scripts/check_public_release.py
```

CI runs the same check. It rejects forbidden tracked paths, likely secrets,
personal cluster paths, and files larger than 5 MiB. This guard checks the
current tree. Before changing a private repository to public, its complete Git
history must also be reviewed and purged of old credentials and data blobs.
