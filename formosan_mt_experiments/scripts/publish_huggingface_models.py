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

FORMOSAN_CODES = (
    "ami",
    "bnn",
    "ckv",
    "dru",
    "pwn",
    "pyu",
    "ssf",
    "sxr",
    "szy",
    "tao",
    "tay",
    "trv",
    "tsu",
    "xnb",
    "xsy",
)

NLLB_LIDS = {
    "ami": "ami_Latn",
    "bnn": "bnn_Latn",
    "ckv": "ckv_Latn",
    "dru": "dru_Latn",
    "pwn": "pwn_Latn",
    "pyu": "pyu_Latn",
    "ssf": "ssf_Latn",
    "sxr": "sxr_Latn",
    "szy": "szy_Latn",
    "tao": "tao_Latn",
    "tay": "tay_Latn",
    "trv": "trv_Latn",
    "tsu": "tsu_Latn",
    "xnb": "xnb_Latn",
    "xsy": "xsy_Latn",
}

NLLB_REPOS = {
    "f2en": "nllb200-formosan-en-spm8k",
    "en2f": "nllb200-en-formosan-spm8k",
    "f2zh": "nllb200-formosan-zh-spm8k",
    "zh2f": "nllb200-zh-formosan-spm8k",
}

MADLAD_REPOS = {
    "f2en": "madlad400-3b-formosan-en",
    "en2f": "madlad400-3b-en-formosan",
    "f2zh": "madlad400-3b-formosan-zh",
    "zh2f": "madlad400-3b-zh-formosan",
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


def repo_name(family: str, direction: str) -> str:
    return (
        NLLB_REPOS
        if family == "nllb"
        else MADLAD_REPOS
    )[direction]


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
    prompt = (
        f"{prefix} <dom_{{source_bucket}}> "
        f"<dialect_{{dialect}}> {{text}}"
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


def madlad_usage(spec: Direction) -> str:
    major_selector = "<2en>" if spec.target_lang == "english" else "<2zh_Hant>"
    major_tag = "eng" if spec.target_lang == "english" else "zh"
    if spec.code.startswith("f2"):
        selector = repr(major_selector)
        controls = f"<to_{major_tag}> <src_{{lang_code}}>"
    else:
        selector = 'f"<2{lang_code}>"'
        controls = f"<to_{{lang_code}}> <src_{major_tag}>"
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
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
{normalization_import}

model_id = "REPLACE_MODEL_ID"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
)
model.to("cuda" if torch.cuda.is_available() else "cpu")

def translate(text, lang_code, source_bucket="unknown", dialect="default"):
{normalization_line.rstrip()}
    target = {selector}
    prompt = (
        f"{{target}} {controls} <dom_{{source_bucket}}> "
        f"<dialect_{{dialect}}> {{text}}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
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
    family: str,
    profile: dict,
    metrics: dict,
    metadata: dict,
    manifest: dict,
) -> str:
    global_metrics = metrics["global"]
    corpus = corpus_record(manifest, spec.target_lang)
    usage = (
        nllb_usage(spec)
        if family == "nllb"
        else madlad_usage(spec)
    ).replace("REPLACE_MODEL_ID", repo_id)
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
    family_tag = "nllb-200" if family == "nllb" else "madlad-400"
    tokenizer_note = (
        "Formosan-aware 8k SentencePiece extension"
        if family == "nllb"
        else "native MADLAD 256k SentencePiece plus Formosan target/control tokens"
    )
    return f"""---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: {base}
tags:
- translation
- {family_tag}
- formosan-languages
- low-resource
metrics:
- bleu
- chrf2
- ter
---

# {repo_id.split('/', 1)[-1]}

**Direction:** {spec.title}<br>
**Base model:** [`{base}`](https://huggingface.co/{base})<br>
**Recipe:** `{profile['recipe_id']}`<br>
**Checkpoint:** validation-selected step {metadata['step']:,}

This is a directional model for 15 Formosan languages. It uses the
`private_no_bible` leakage-controlled corpus, {tokenizer_note}, balanced
language/source sampling, and direction/domain/dialect control tags.

## Usage

{usage}

MADLAD selects the target language with the first source token and uses its
configured T5 decoder start token. NLLB selects the target using
`forced_bos_token_id`. Do not interchange these generation contracts.

## Evaluation

The best checkpoint was selected on human validation chrF2. Test references
are human sentence pairs; synthetic pivots and lexical entries are train-only.

| Split | Rows |
|---|---:|
| Train | {corpus['splits']['train']:,} |
| Test | {corpus['splits']['test']:,} |
| Validate | {corpus['splits']['validate']:,} |

| Scope | BLEU | chrF2 | TER |
|---|---:|---:|---:|
| Hard test | {global_metrics['BLEU']:.2f} | {global_metrics['chrF2']:.2f} | {global_metrics['TER']:.2f} |
| Selection validation | {selected.get('BLEU', float('nan')):.2f} | {selected.get('chrF2', float('nan')):.2f} | {selected.get('TER', float('nan')):.2f} |

| Language | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|
{by_language}

The corpus gate enforces standard-tier Formosan text, at least 7.5% test and
2.5% validation per language, human sentence-only evaluation, and zero exact,
skeleton, one-edit, or configured high character n-gram train/evaluation
conflicts. See `eval/metrics.json` for bootstrap confidence intervals and
domain/length diagnostics.

## Limitations

Outputs require knowledgeable speaker review. Aggregate metrics hide large
differences among languages and domains. This model is not suitable for
authoritative, medical, legal, or safety-critical translation.
"""


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
    family = profile["model_family"]
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
        or metrics.get("model_family", family) != family
        or metrics.get("mt_standardization") != expected_mt_standard
        or metadata.get("mt_standardization") != expected_mt_standard
    ):
        raise RuntimeError(
            f"Direction/family/MT-standard mismatch for {spec.code}"
        )
    expected_test_rows = int(
        corpus_record(manifest, spec.target_lang)["splits"]["test"]
    )
    if int(metrics.get("samples", -1)) != expected_test_rows:
        raise RuntimeError(
            f"Test row mismatch for {spec.code}: "
            f"{metrics.get('samples')} != {expected_test_rows}"
        )

    name = repo_name(family, spec.code)
    repo_id = f"{args.organization}/{name}"
    output = args.output_root / name
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True)
    for path in files:
        link_or_copy(path, output / path.name)
    for source_path, output_name in (
        (MT_STANDARDIZER, "mt_standardization.py"),
        (MT_INFERENCE, "formosan_mt_inference.py"),
        (MT_PROFILE, "mt_standardization_profile.json"),
    ):
        if not source_path.is_file():
            raise RuntimeError(
                f"Missing inference standardization artifact: {source_path}"
            )
        link_or_copy(source_path, output / output_name)
    (output / "eval").mkdir()
    shutil.copy2(metrics_path, output / "eval" / "metrics.json")
    (output / "README.md").write_text(
        render_card(
            spec=spec,
            repo_id=repo_id,
            family=family,
            profile=profile,
            metrics=metrics,
            metadata=metadata,
            manifest=manifest,
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
