#!/usr/bin/env python3
"""Package and optionally publish a public or private Formosan MT corpus."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from pipeline_common import atomic_write_json, sha256_file, utc_now

CONFIGS = {
    "formosan-en": {
        "short": "en",
        "target_column": "english_sentence",
        "target_code": "en",
        "directions": ("f2en", "en2f"),
    },
    "formosan-zh": {
        "short": "zh",
        "target_column": "chinese_sentence",
        "target_code": "zh",
        "directions": ("f2zh", "zh2f"),
    },
}

PROVENANCE_COLUMNS = [
    "row_id",
    "source_record_id",
    "content_sha256",
    "repository",
    "repository_commit",
    "xml_path",
    "xml_id",
    "kindOf",
    "standard_namespace",
    "standard_origin",
    "pivot_origin",
    "pivot_provider",
    "pivot_direction",
    "source_bucket",
    "row_type",
    "eval_tier",
    "document_id",
]

LANGUAGE_NAMES = {
    "ami": "Amis",
    "bnn": "Bunun",
    "ckv": "Kavalan",
    "dru": "Rukai",
    "pwn": "Paiwan",
    "pyu": "Puyuma",
    "ssf": "Thao",
    "sxr": "Saaroa",
    "szy": "Sakizaya",
    "tao": "Tao / Yami",
    "tay": "Atayal",
    "trv": "Seediq / Truku",
    "tsu": "Tsou",
    "xnb": "Kanakanavu",
    "xsy": "Saisiyat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-root",
        type=Path,
        default=Path("corpus_builds/public_no_bible"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--repo-id",
        help=(
            "Hugging Face dataset repository. Defaults to FormosanBank/formosan-mt "
            "or FormosanBank/formosan-mt-private with --private."
        ),
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Package a private build and require a private Hugging Face repository.",
    )
    parser.add_argument("--upload", action="store_true")
    return parser.parse_args()


def read_complete_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("complete") is not True:
        raise SystemExit(f"Incomplete required report: {path}")
    return value


def sanitize_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_paths(item) for item in value]
    if isinstance(value, str) and Path(value).is_absolute():
        return Path(value).name
    return value


def artifact_record(path: Path) -> dict[str, Any]:
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def source_snapshot(build_root: Path) -> dict[str, Any]:
    snapshot = read_complete_json(build_root / "source_repository_snapshot.json")
    return {
        str(record["name"]): str(record["commit_sha"])
        for record in snapshot["repositories"]
    }


def validate_release_frame(
    frame: pd.DataFrame,
    target_column: str,
    *,
    private_release: bool,
) -> None:
    required = {
        "lang_code",
        "formosan_sentence",
        target_column,
        "dialect",
        "source",
        "split",
        *PROVENANCE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"Release corpus is missing columns: {missing}")
    if not private_release:
        if not frame["repository"].astype(str).eq("FormosanBank").all():
            raise SystemExit("Public release contains a non-public repository")
        if not frame["source"].astype(str).str.startswith("FormosanBank/").all():
            raise SystemExit("Public release contains a non-public source path")
    evaluation = frame[frame["split"].isin(["test", "validate"])]
    if evaluation["pivot_origin"].astype(str).eq("synthetic").any():
        raise SystemExit("Release contains synthetic evaluation rows")
    if not evaluation["mt_eval_eligible"].fillna(False).astype(bool).all():
        raise SystemExit("Release contains ineligible evaluation rows")
    if not evaluation["row_type"].astype(str).eq("sentence").all():
        raise SystemExit("Release contains non-sentence evaluation rows")


def package_config(
    build_root: Path,
    output_dir: Path,
    config_name: str,
    spec: dict[str, Any],
    *,
    private_release: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_dir = build_root / "formosan_mt_experiments" / "data" / f"splits_{spec['short']}_v1"
    input_path = split_dir / f"big_corpus_{spec['short']}_in_domain_hard.parquet"
    frame = pd.read_parquet(input_path)
    validate_release_frame(
        frame,
        spec["target_column"],
        private_release=private_release,
    )
    ids = pd.Series(range(len(frame)), dtype="int64", name="id")

    release = pd.DataFrame(
        {
            "id": ids,
            "source_lang": frame["lang_code"].astype(str),
            "target_lang": spec["target_code"],
            "source_sentence": frame["formosan_sentence"].astype(str),
            "target_sentence": frame[spec["target_column"]].astype(str),
            "lang_code": frame["lang_code"].astype(str),
            "dialect": frame["dialect"].fillna("unknown").astype(str),
            "source": frame["source"].astype(str),
            "split": frame["split"].astype(str),
        }
    )
    csv_name = f"formosan_{spec['short']}_hf.csv"
    release.to_csv(output_dir / csv_name, index=False, lineterminator="\n")

    provenance = frame[PROVENANCE_COLUMNS].copy()
    provenance.insert(0, "id", ids)
    provenance = provenance.rename(
        columns={"kindOf": "kind_of", "pivot_origin": "translation_origin"}
    )
    provenance_name = f"provenance/formosan_{spec['short']}_rows.parquet"
    provenance.to_parquet(output_dir / provenance_name, index=False, compression="zstd")

    validation = read_complete_json(split_dir / "validation_in_domain_hard.json")
    exposure = read_complete_json(split_dir / "exposure_in_domain_hard.json")
    atomic_write_json(
        output_dir / f"provenance/validation_{spec['short']}.json",
        sanitize_paths(validation),
    )
    atomic_write_json(
        output_dir / f"provenance/exposure_{spec['short']}.json",
        sanitize_paths(exposure),
    )

    split_counts = release["split"].value_counts().to_dict()
    synthetic = frame["pivot_origin"].astype(str).eq("synthetic")
    by_language = release.groupby("lang_code", sort=True).size().to_dict()
    by_language_split = (
        release.groupby(["lang_code", "split"], sort=True)
        .size()
        .unstack(fill_value=0)
        .to_dict("index")
    )
    return frame, {
        "rows": len(frame),
        "splits": {key: int(split_counts.get(key, 0)) for key in ("train", "validate", "test")},
        "synthetic_train": int((synthetic & frame["split"].eq("train")).sum()),
        "by_language": {key: int(value) for key, value in by_language.items()},
        "by_language_and_split": {
            key: {split: int(value) for split, value in values.items()}
            for key, values in by_language_split.items()
        },
        "source_buckets": {
            key: int(value) for key, value in frame["source_bucket"].value_counts().sort_index().items()
        },
        "dialects": {
            key: int(value) for key, value in frame["dialect"].fillna("unknown").value_counts().sort_index().items()
        },
        "translation_origin_and_split": {
            origin: {split: int(value) for split, value in values.items()}
            for origin, values in (
                frame.assign(translation_origin=frame["pivot_origin"].astype(str))
                .groupby(["translation_origin", "split"])
                .size()
                .unstack(fill_value=0)
                .to_dict("index")
                .items()
            )
        },
        "input": artifact_record(input_path),
        "artifacts": {
            csv_name: artifact_record(output_dir / csv_name),
            provenance_name: artifact_record(output_dir / provenance_name),
        },
        "directions": spec["directions"],
    }


def metric_rows(output_dir: Path) -> list[tuple[str, int, float, float, float, float]]:
    labels = {
        "f2en": "Formosan to English",
        "en2f": "English to Formosan",
        "f2zh": "Formosan to Chinese",
        "zh2f": "Chinese to Formosan",
    }
    rows = []
    for short in ("en", "zh"):
        report = json.loads((output_dir / f"provenance/exposure_{short}.json").read_text(encoding="utf-8"))
        for direction, details in report["directions"].items():
            combined = details["combined_evaluation"]
            source = combined["exposure"]["source"]
            rows.append(
                (
                    labels[direction],
                    int(combined["data"]["num_test"]),
                    float(combined["quality"]["tm"]["bleu"]),
                    float(combined["quality"]["tm"]["chrf"]),
                    float(source["mean"]),
                    float(source["at_threshold"]["0.70"]),
                )
            )
    return rows


def build_card(
    metadata: dict[str, Any],
    output_dir: Path,
    *,
    private_release: bool,
) -> str:
    configs = metadata["configs"]
    en = configs["formosan-en"]
    zh = configs["formosan-zh"]
    total = en["rows"] + zh["rows"]
    summary_rows = []
    for name, values in configs.items():
        summary_rows.append(
            f"| `{name}` | {values['rows']:,} | {values['splits']['train']:,} | "
            f"{values['splits']['validate']:,} | {values['splits']['test']:,} | "
            f"{values['synthetic_train']:,} |"
        )
    language_rows = [
        f"| `{code}` | {LANGUAGE_NAMES[code]} | {en['by_language'][code]:,} | {zh['by_language'][code]:,} |"
        for code in LANGUAGE_NAMES
    ]
    exposure_rows = [
        f"| {label} | {count:,} | {bleu:.2f} | {chrf:.2f} | {mean:.3f} | {near:.3%} |"
        for label, count, bleu, chrf, mean, near in metric_rows(output_dir)
    ]
    repo_id = metadata["repo_id"]
    release_label = "Private" if private_release else "Public"
    dataset_name = (
        "FormosanBank Machine Translation (Private)"
        if private_release
        else "FormosanBank Machine Translation"
    )
    access_text = (
        "This is a private internal FormosanBank dataset. Access is limited to "
        "authorized members of the FormosanBank Hugging Face organization."
        if private_release
        else "Public parallel corpora for 15 Indigenous Formosan languages aligned "
        "with English and Mandarin Chinese."
    )
    size_category = "1M<n<10M" if total >= 1_000_000 else "100K<n<1M"
    public_commit = metadata["source_repositories"].get("FormosanBank", "not present")
    source_provenance = (
        f"- FormosanBank commit: `{public_commit}`\n"
        f"- Source repositories: {len(metadata['source_repositories']):,}"
        if private_release
        else f"- Public FormosanBank commit: `{public_commit}`"
    )
    citation_key = f"formosanbank_mt_{'private' if private_release else 'public'}_v3"
    return f"""---
license: other
license_name: formosanbank-terms-ai-use-addendum
license_link: https://ai4commsci.gitbook.io/formosanbank/additional-resources/terms-of-use
pretty_name: {dataset_name}
task_categories:
- translation
language:
{chr(10).join(f'- {code}' for code in [*LANGUAGE_NAMES, 'en', 'zh'])}
size_categories:
- {size_category}
tags:
- noncommercial
- no-commercial-ai
- translation
- machine-translation
- low-resource
- endangered-languages
- formosan-languages
- leakage-controlled
- hard-split
- synthetic-data
library_name: datasets
configs:
- config_name: formosan-en
  data_files:
  - split: train
    path: formosan_en_hf.csv
- config_name: formosan-zh
  data_files:
  - split: train
    path: formosan_zh_hf.csv
---

# {dataset_name}

{access_text} This release uses canonical MT-standardized Formosan text, excludes `Formosan-Taiwan-Bible-Society-Bibles`, and keeps DeepL pivot translations in training only.

Commercial AI use is prohibited without prior written permission. See the [FormosanBank Terms of Use](https://ai4commsci.gitbook.io/formosanbank/additional-resources/terms-of-use).

## Release Summary

| Config | Rows | Train | Validate | Test | Synthetic train |
|---|---:|---:|---:|---:|---:|
{chr(10).join(summary_rows)}
| **Total** | **{total:,}** | **{sum(v['splits']['train'] for v in configs.values()):,}** | **{sum(v['splits']['validate'] for v in configs.values()):,}** | **{sum(v['splits']['test'] for v in configs.values()):,}** | **{sum(v['synthetic_train'] for v in configs.values()):,}** |

Every language is split approximately 85% train, 5% validation, and 10% test. Evaluation contains only eligible human sentence references. Synthetic pivots and short entries are train-only.

## Languages

| Code | Language | English pairs | Chinese pairs |
|---|---|---:|---:|
{chr(10).join(language_rows)}

## Schema

The two main files preserve the established nine-column format: `id`, `source_lang`, `target_lang`, `source_sentence`, `target_sentence`, `lang_code`, `dialect`, `source`, and row-level `split`.

```python
from datasets import DatasetDict, load_dataset

rows = load_dataset("{repo_id}", "formosan-en", split="train")
dataset = DatasetDict({{
    split: rows.filter(lambda row: row["split"] == split)
    for split in ("train", "validate", "test")
}})
```

Use `formosan-zh` for Chinese. Reverse-direction training can swap the sentence columns. Files under `provenance/` map release IDs to source commits and XML records and include independent validation and TAME-MT reports.

## Split Quality

The builder groups exact normalized pairs and punctuation skeletons, blocks one-edit conflicts, and excludes pair exposure at character 3-5 gram Jaccard similarity 0.95 or above. These are row-level hard splits, not document-held-out splits.

| Direction | Eval rows | TM BLEU | TM chrF2 | Mean source exposure | Source >= 0.70 |
|---|---:|---:|---:|---:|---:|
{chr(10).join(exposure_rows)}

TM scores measure nearest-neighbor translation-memory retrieval, not model quality. All four directions have zero exact overlap and zero source, target, or pair exposure at 0.95.

## Provenance

{source_provenance}
- FormosanBank QC commit: `{metadata['qc_revision']}`
- Toolkit commit: `{metadata['toolkit']['commit']}`
- MT standardization: `{metadata['mt_standardization']['id']}`
- Build completed: {metadata['source_build_completed'][:10]}

Artifact hashes and per-language counts are recorded in `provenance/release_metadata.json` and `SHA256SUMS`.

## Limitations

- Sources vary in dialect, genre, translation style, and transcription quality.
- Some human references retain source or linguistic annotations.
- The English training corpus contains substantial synthetic augmentation.
- Similarity controls do not prove semantic or document independence.
- Automatic metrics do not replace evaluation by fluent speakers.

## Citation

```bibtex
@misc{{{citation_key},
  title        = {{FormosanBank Machine Translation {release_label} Corpus}},
  author       = {{FormosanBank contributors}},
  year         = {{2026}},
  howpublished = {{https://huggingface.co/datasets/{repo_id}}}
}}
```

See the [Terms of Use](https://ai4commsci.gitbook.io/formosanbank/additional-resources/terms-of-use) and [AI Use Addendum](https://github.com/FormosanBank/FormosanBank/blob/main/AI-USE-ADDENDUM.md).
"""


def write_checksums(output_dir: Path) -> None:
    paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.name != "SHA256SUMS")
    lines = [f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}" for path in paths]
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_id = args.repo_id or (
        "FormosanBank/formosan-mt-private"
        if args.private
        else "FormosanBank/formosan-mt"
    )
    build_root = args.build_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest = read_complete_json(build_root / "mt_build_manifest.json")
    settings = manifest.get("settings", {})
    expected_public = not args.private
    if settings.get("public") is not expected_public:
        release_type = "private" if args.private else "public"
        raise SystemExit(f"Hugging Face {release_type} release uses the wrong build type")
    if settings.get("exclude_bible") is not True:
        raise SystemExit("Hugging Face releases require a no-Bible build")
    if output_dir in {Path("/"), Path.home(), build_root}:
        raise SystemExit(f"Unsafe output directory: {output_dir}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    (output_dir / "provenance").mkdir(parents=True)

    configs = {}
    for name, spec in CONFIGS.items():
        _, configs[name] = package_config(
            build_root,
            output_dir,
            name,
            spec,
            private_release=args.private,
        )
    metadata = {
        "schema_version": 2,
        "complete": True,
        "created_at": utc_now(),
        "repo_id": repo_id,
        "private": args.private,
        "source_build_completed": manifest["created_at"],
        "source_build_manifest_sha256": sha256_file(build_root / "mt_build_manifest.json"),
        "source_repositories": source_snapshot(build_root),
        "qc_revision": settings["qc_revision"],
        "toolkit": manifest["repository"],
        "pipeline_version": manifest["pipeline_version"],
        "pipeline_config": manifest["pipeline_config"],
        "mt_standardization": manifest["mt_standardization"],
        "configs": configs,
    }
    for short in ("en", "zh"):
        for kind in ("validation", "exposure"):
            name = f"provenance/{kind}_{short}.json"
            metadata.setdefault("artifacts", {})[name] = artifact_record(output_dir / name)
    atomic_write_json(output_dir / "provenance/release_metadata.json", metadata)
    (output_dir / "README.md").write_text(
        build_card(metadata, output_dir, private_release=args.private),
        encoding="utf-8",
    )
    write_checksums(output_dir)

    print(f"Packaged {sum(value['rows'] for value in configs.values()):,} rows in {output_dir}")
    if args.upload:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        info = api.dataset_info(repo_id)
        if bool(info.private) is not args.private:
            expected = "private" if args.private else "public"
            raise SystemExit(f"Refusing upload: {repo_id} is not {expected}")
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=output_dir,
            commit_message=(
                f"Publish {'private' if args.private else 'public'} no-Bible corpus "
                f"from {manifest['repository']['commit'][:12]}"
            ),
        )
        print(commit)


if __name__ == "__main__":
    main()
