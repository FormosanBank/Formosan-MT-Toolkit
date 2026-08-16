#!/usr/bin/env python3
"""Package and optionally publish one four-direction Formosan MT flight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from experiment_config import load_profile, profile_record, sha256_file
from mt_common import CODE_TO_LID, DOMAIN_BUCKETS, FORMOSAN_CODES

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EXPERIMENT_ROOT.parent
MT_STANDARDIZER = PROJECT_ROOT / "scripts/local/mt_standardization.py"
MT_INFERENCE = EXPERIMENT_ROOT / "scripts/formosan_mt_inference.py"
MT_PROFILE = PROJECT_ROOT / "config/mt_standardization.json"


@dataclass(frozen=True)
class Direction:
    code: str
    title: str
    target_lang: str
    source_example: str


DIRECTIONS = {
    "f2en": Direction("f2en", "Formosan to English", "english", "Pa'araw cingra."),
    "en2f": Direction("en2f", "English to Formosan", "english", "He went home."),
    "f2zh": Direction("f2zh", "Formosan to Traditional Chinese", "chinese", "Pa'araw cingra."),
    "zh2f": Direction("zh2f", "Traditional Chinese to Formosan", "chinese", "他回家了。"),
}

NLLB_LIDS = {code: CODE_TO_LID[code] for code in FORMOSAN_CODES}

NLLB_REPOS = {
    "f2en": "nllb200-formosan-en-spm8k",
    "en2f": "nllb200-en-formosan-spm8k",
    "f2zh": "nllb200-formosan-zh-spm8k",
    "zh2f": "nllb200-zh-formosan-spm8k",
}

REQUIRED_FILES = {
    "config.json",
    "experiment_metadata.json",
    "generation_config.json",
    "tokenizer_config.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_slug(recipe_id: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", recipe_id.lower()).strip("_")


def repo_name(direction: str) -> str:
    return NLLB_REPOS[direction]


def validate_checkpoint(path: Path) -> list[Path]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.iterdir() if item.is_file())
    names = {item.name for item in files}
    missing = REQUIRED_FILES - names
    if missing:
        raise RuntimeError(f"Checkpoint {path} is missing: {sorted(missing)}")
    if not any(item.name.endswith(".safetensors") for item in files):
        raise RuntimeError(f"Checkpoint {path} has no safetensors weights")
    if not (
        "sentencepiece.bpe.model" in names
        or "spiece.model" in names
        or "tokenizer.json" in names
    ):
        raise RuntimeError(f"Checkpoint {path} has no tokenizer model")
    return files


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def nllb_usage(spec: Direction) -> str:
    major_lid = "eng_Latn" if spec.target_lang == "english" else "zho_Hant"
    major_tag = "eng" if spec.target_lang == "english" else "zh"
    if spec.code.startswith("f2"):
        source_lid = "NLLB_LIDS[lang_code]"
        target_lid = repr(major_lid)
        prefix = f"<to_{major_tag}> <src_{{lang_code}}>"
    else:
        source_lid = repr(major_lid)
        target_lid = "NLLB_LIDS[lang_code]"
        prefix = f"<to_{{lang_code}}> <src_{major_tag}>"
    normalization_import = (
        "from formosan_mt_inference import normalize_formosan\n"
        if spec.code.startswith("f2")
        else ""
    )
    normalization_line = (
        "    text = normalize_formosan(text, lang_code)\n"
        if spec.code.startswith("f2")
        else ""
    )
    return f"""```python
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
{normalization_import}

model_id = "REPLACE_MODEL_ID"
tokenizer = NllbTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to("cuda" if torch.cuda.is_available() else "cpu")
NLLB_LIDS = {NLLB_LIDS!r}

def translate(text, lang_code, source_bucket="unknown", dialect="default"):
{normalization_line.rstrip()}
    tokenizer.src_lang = {source_lid}
    source_bucket = source_bucket if source_bucket in {DOMAIN_BUCKETS!r} else "unknown"
    domain_tag = f"<dom_{{source_bucket}}>"
    if tokenizer.convert_tokens_to_ids(domain_tag) == tokenizer.unk_token_id:
        domain_tag = "<dom_unknown>"
    dialect_tag = f"<dialect_{{dialect}}>"
    if tokenizer.convert_tokens_to_ids(dialect_tag) == tokenizer.unk_token_id:
        dialect_tag = "<dialect_default>"
    prompt = (
        f"{prefix} {{domain_tag}} {{dialect_tag}} {{text}}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        decoder_start_token_id=tokenizer.eos_token_id,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids({target_lid}),
        max_new_tokens=256,
        num_beams=4,
    )
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

print(translate({spec.source_example!r}, "ami"))
```"""


def corpus_record(manifest: dict, target_lang: str) -> dict:
    corpora = manifest["corpora"]
    if target_lang in corpora:
        return corpora[target_lang]
    suffix = "_en" if target_lang == "english" else "_zh"
    matches = [
        value
        for key, value in corpora.items()
        if key.endswith(suffix)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot identify {target_lang} corpus in publication manifest"
        )
    return matches[0]


def render_card(
    *,
    spec: Direction,
    repo_id: str,
    profile: dict,
    metrics: dict,
    metadata: dict,
    manifest: dict,
    run_stamp: str,
) -> str:
    global_metrics = metrics["global"]
    corpus = corpus_record(manifest, spec.target_lang)
    usage = nllb_usage(spec).replace("REPLACE_MODEL_ID", repo_id)
    by_language = "\n".join(
        f"| `{code}` | {metrics['by_language'][code]['samples']:,} | "
        f"{metrics['by_language'][code]['BLEU']:.2f} | "
        f"{metrics['by_language'][code]['chrF2']:.2f} | "
        f"{metrics['by_language'][code]['TER']:.2f} |"
        for code in FORMOSAN_CODES
        if code in metrics.get("by_language", {})
    )
    validation = metadata.get("validation", {})
    selected = validation.get("generation", {}).get("global", {})
    base = profile["base_model"]["name"]
    base_revision = profile["base_model"]["revision"]
    tokenizer_note = "Formosan-aware 8k SentencePiece extension"
    languages = (*FORMOSAN_CODES, "en" if spec.target_lang == "english" else "zh")
    language_yaml = "\n".join(f"- {code}" for code in languages)
    headline_mode = metrics.get("headline_metadata_mode", "default")
    training = profile["training_defaults"]
    corpus_validation = corpus["validation"]
    confidence = metrics.get("bootstrap_95_ci") or {}
    confidence_metrics = confidence.get("metrics") or {}
    confidence_table = ""
    if confidence_metrics:
        confidence_rows = "\n".join(
            f"| {name} | {values['lower']:.2f} | {values['upper']:.2f} |"
            for name, values in confidence_metrics.items()
        )
        confidence_table = f"""

### Confidence intervals

Stratified bootstrap, {confidence['samples']:,} samples, 95% confidence.

| Metric | Lower | Upper |
|---|---:|---:|
{confidence_rows}
"""
    model_name = repo_id.split("/", 1)[-1]
    return f"""---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: {base}
language:
{language_yaml}
tags:
- translation
- nllb-200
- formosan-languages
- low-resource
metrics:
- bleu
- chrf
- ter
model-index:
- name: {model_name}
  results:
  - task:
      type: translation
      name: Translation
    dataset:
      name: FormosanBank private no-Bible hard test
      type: private-no-bible-hard-test
      split: test
    metrics:
    - type: bleu
      name: sacreBLEU
      value: {global_metrics['BLEU']:.6f}
    - type: chrf
      name: chrF2
      value: {global_metrics['chrF2']:.6f}
    - type: ter
      name: TER
      value: {global_metrics['TER']:.6f}
---

# {model_name}

**Direction:** {spec.title}<br>
**Base model:** [`{base}`](https://huggingface.co/{base})<br>
**Recipe:** `{profile['recipe_id']}`<br>
**Release:** `{run_stamp}`, validation-selected step {metadata['step']:,}

This is a directional model for 15 Formosan languages. It uses the
`private_no_bible` leakage-controlled corpus, {tokenizer_note}, balanced
language/source sampling, direction/language controls, fixed coarse domain
tags, and dialect tags. Domain and dialect metadata use independent training
dropout so `unknown` and `default` are learned inference conditions. The model
weights are public, but the private training corpus is not included.

## Model details

| Item | Value |
|---|---|
| Base revision | `{base_revision}` |
| Training rows | {corpus['splits']['train']:,} |
| Effective batch size | {training['effective_batch_size']:,} |
| Maximum sequence length | {training['max_length']:,} |
| Learning rate | {training['learning_rate']:.2g} |
| Precision | `{training['precision']}` |
| Checkpoint selection | Validation `{training['best_metric']}` |
| Formosan text | `kindOf=standard`, `{profile['mt_standardization']['id']}` |
| Corpus SHA-256 | `{corpus['sha256']}` |
| Training profile SHA-256 | `{metrics['profile']['sha256']}` |

## Usage

{usage}

The control tags are part of the training contract. Use `unknown` and `default`
when source bucket or dialect metadata is unavailable.

## Evaluation

The best checkpoint was selected on validation chrF2. Evaluation contains only
eligible sentence pairs. Human references are preferred within each source;
validated synthetic pivots are used only where human coverage is insufficient.
Lexical entries remain train-only.
The headline result uses `{headline_mode}` metadata controls, so it does not
assume access to test-set domain or dialect labels.

| Split | Rows |
|---|---:|
| Train | {corpus['splits']['train']:,} |
| Test | {corpus['splits']['test']:,} |
| Validate | {corpus['splits']['validate']:,} |

| Scope | BLEU | chrF2 | TER |
|---|---:|---:|---:|
| Hard test | {global_metrics['BLEU']:.2f} | {global_metrics['chrF2']:.2f} | {global_metrics['TER']:.2f} |
| Selection validation | {selected.get('BLEU', float('nan')):.2f} | {selected.get('chrF2', float('nan')):.2f} | {selected.get('TER', float('nan')):.2f} |

Test empty-output rate: {global_metrics.get('empty_output_rate', 0.0):.4%}.
{confidence_table}

| Language | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|
{by_language}

The corpus gate enforces standard-tier Formosan text, 5/10
validation/test proportions from all deduplicated pairs, capacity-aware source
balance, sentence-only evaluation, and
zero exact, skeleton, one-edit, or configured high character n-gram
train/evaluation conflicts. Document overlap is diagnostic. This release
passed all leakage gates: exact
{corpus_validation['exact_overlap']}, skeleton
{corpus_validation['skeleton_overlap']}, one-edit
{corpus_validation['one_edit_conflicts']}, character n-gram
{corpus_validation['character_ngram_conflicts']}. Document overlap:
{corpus_validation['document_overlap']}.

See `eval/metrics.json` for sacreBLEU signatures, per-language, source,
dialect, and length diagnostics. `publication.json` records the corpus,
profile, run, and checkpoint hashes used for this release.

## Intended use

This model supports research, corpus development, and assisted translation for
the 15 included Formosan languages. It is designed for the exact prompt and
generation contract shown above.

## Limitations

Outputs require knowledgeable speaker review. Aggregate metrics hide large
differences among languages and domains. This model is not suitable for
authoritative, medical, legal, or safety-critical translation.
"""


def public_metrics(
    metrics: dict,
    *,
    repo_id: str,
    corpus_name: str,
    target_lang: str,
) -> dict:
    """Remove machine-local paths while preserving evaluation provenance."""
    output = json.loads(json.dumps(metrics))
    output["input"] = f"{corpus_name}:{target_lang}"
    output["model"] = repo_id
    output["tokenizer"] = repo_id
    if isinstance(output.get("profile"), dict):
        output["profile"]["path"] = "training_profile.json"
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-stamp")
    parser.add_argument("--checkpoint", choices=["best", "final"], default="best")
    parser.add_argument("--organization", default="FormosanBank")
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def prepare_direction(
    args: argparse.Namespace,
    spec: Direction,
    *,
    profile: dict,
    manifest: dict,
    run_stamp: str,
) -> dict:
    slug = recipe_slug(profile["recipe_id"])
    source = (
        args.artifact_root
        / f"{slug}_{spec.code}_{run_stamp}"
        / args.checkpoint
    )
    metrics_path = (
        args.metrics_root
        / f"{slug}_{spec.code}_{args.checkpoint}_{run_stamp}"
        / "metrics.json"
    )
    files = validate_checkpoint(source)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(
        (source / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    expected_mt_standard = profile["mt_standardization"]
    if (
        metrics.get("direction") != spec.code
        or metadata.get("direction") != spec.code
        or metrics.get("model_family") != "nllb"
        or metrics.get("mt_standardization") != expected_mt_standard
        or metadata.get("mt_standardization") != expected_mt_standard
    ):
        raise RuntimeError(
            f"Direction/NLLB/MT-standard mismatch for {spec.code}"
        )
    expected_test_rows = int(
        corpus_record(manifest, spec.target_lang)["splits"]["test"]
    )
    if int(metrics.get("samples", -1)) != expected_test_rows:
        raise RuntimeError(
            f"Test row mismatch for {spec.code}: "
            f"{metrics.get('samples')} != {expected_test_rows}"
        )

    name = repo_name(spec.code)
    repo_id = f"{args.organization}/{name}"
    corpus = corpus_record(manifest, spec.target_lang)
    output = args.output_root / name
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for path in files:
        link_or_copy(path, output / path.name)
    for source_path, output_name in (
        (MT_STANDARDIZER, "mt_standardization.py"),
        (MT_INFERENCE, "formosan_mt_inference.py"),
        (MT_PROFILE, "mt_standardization_profile.json"),
        (args.profile, "training_profile.json"),
    ):
        if not source_path.is_file():
            raise RuntimeError(
                f"Missing inference standardization artifact: {source_path}"
            )
        link_or_copy(source_path, output / output_name)
    (output / "eval").mkdir()
    published_metrics = public_metrics(
        metrics,
        repo_id=repo_id,
        corpus_name=manifest["corpus_name"],
        target_lang=spec.target_lang,
    )
    (output / "eval" / "metrics.json").write_text(
        json.dumps(published_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    publication = {
        "schema_version": 1,
        "repo_id": repo_id,
        "direction": spec.code,
        "run_stamp": run_stamp,
        "checkpoint": args.checkpoint,
        "checkpoint_step": metadata["step"],
        "corpus": {
            "name": manifest["corpus_name"],
            "target_lang": spec.target_lang,
            "rows": corpus["rows"],
            "splits": corpus["splits"],
            "sha256": corpus["sha256"],
        },
        "profile_sha256": sha256_file(args.profile),
        "base_model": profile["base_model"],
        "mt_standardization": profile["mt_standardization"],
        "run_contract_sha256": metadata["run_contract_sha256"],
    }
    (output / "publication.json").write_text(
        json.dumps(publication, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        render_card(
            spec=spec,
            repo_id=repo_id,
            profile=profile,
            metrics=metrics,
            metadata=metadata,
            manifest=manifest,
            run_stamp=run_stamp,
        ),
        encoding="utf-8",
    )
    (output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.model filter=lfs diff=lfs merge=lfs -text\n",
        encoding="utf-8",
    )
    return {
        "direction": spec.code,
        "repo_id": repo_id,
        "step": metadata["step"],
        "files": {
            str(path.relative_to(output)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }


def main() -> None:
    args = parse_args()
    profile = load_profile(args.profile)
    if sha256_file(MT_PROFILE) != profile["mt_standardization"]["sha256"]:
        raise RuntimeError(
            "Experiment profile does not pin the repository MT standardization profile"
        )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    run_stamp = args.run_stamp or manifest["run_stamp"]
    if manifest["run_stamp"] != run_stamp:
        raise RuntimeError("Manifest run stamp does not match --run-stamp")
    if manifest.get("recipe_id", profile["recipe_id"]) != profile["recipe_id"]:
        raise RuntimeError("Manifest and profile recipes differ")

    args.output_root.mkdir(parents=True, exist_ok=True)
    release = {
        "schema_version": 3,
        "run_stamp": run_stamp,
        "corpus": manifest["corpus_name"],
        "checkpoint_policy": args.checkpoint,
        "model_family": profile["model_family"],
        "mt_standardization": profile["mt_standardization"],
        "profile": profile_record(args.profile),
        "models": [
            prepare_direction(
                args,
                DIRECTIONS[code],
                profile=profile,
                manifest=manifest,
                run_stamp=run_stamp,
            )
            for code in DIRECTIONS
        ],
    }
    release_path = args.output_root / "release_manifest.json"
    release_path.write_text(
        json.dumps(release, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        for item in release["models"]:
            repo_id = item["repo_id"]
            api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            commit = api.upload_folder(
                repo_id=repo_id,
                folder_path=args.output_root / repo_id.split("/", 1)[1],
                delete_patterns="*",
                commit_message=(
                    f"Publish {profile['recipe_id']} "
                    f"{args.checkpoint} checkpoint ({run_stamp})"
                ),
            )
            item["published_commit"] = commit.oid
            release_path.write_text(
                json.dumps(release, indent=2) + "\n",
                encoding="utf-8",
            )

    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
