#!/usr/bin/env python3
"""Build and optionally publish the four production Formosan MT model repos."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = [
    ("Amis", "ami", "ami_Latn"),
    ("Bunun", "bnn", "bnn_Latn"),
    ("Kavalan", "ckv", "ckv_Latn"),
    ("Rukai", "dru", "dru_Latn"),
    ("Paiwan", "pwn", "pwn_Latn"),
    ("Puyuma", "pyu", "pyu_Latn"),
    ("Thao", "ssf", "ssf_Latn"),
    ("Saaroa", "sxr", "sxr_Latn"),
    ("Sakizaya", "szy", "szy_Latn"),
    ("Tao / Yami", "tao", "tao_Latn"),
    ("Atayal", "tay", "tay_Latn"),
    ("Seediq", "trv", "trv_Latn"),
    ("Tsou", "tsu", "tsu_Latn"),
    ("Kanakanavu", "xnb", "xnb_Latn"),
    ("Saisiyat", "xsy", "xsy_Latn"),
]

REQUIRED_MODEL_FILES = {
    "added_tokens.json",
    "config.json",
    "experiment_metadata.json",
    "generation_config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer_config.json",
}


@dataclass(frozen=True)
class Direction:
    code: str
    repo: str
    title: str
    target_family: str
    corpus_key: str
    easy_weight: float
    input_format: str
    example: str
    function_name: str
    source_setup: str
    forced_bos: str
    companion: str


DIRECTIONS = {
    "f2en": Direction(
        code="f2en",
        repo="nllb200-formosan-en-spm8k",
        title="Formosan -> English",
        target_family="English",
        corpus_key="private_no_bible_en",
        easy_weight=0.05,
        input_format="<to_eng> <src_LANG> <dom_BUCKET> <dialect_DIALECT>",
        example="<to_eng> <src_ami> <dom_unknown> <dialect_default> Pa'araw cingra to demak nira.",
        function_name="translate_formosan_to_english",
        source_setup='tokenizer.src_lang = FORMOSAN_TO_LID[lang_code]\n    prompt = f"<to_eng> <src_{lang_code}> <dom_{source_bucket}> <dialect_{dialect}> {text}"',
        forced_bos='tokenizer.convert_tokens_to_ids("eng_Latn")',
        companion="nllb200-en-formosan-spm8k",
    ),
    "en2f": Direction(
        code="en2f",
        repo="nllb200-en-formosan-spm8k",
        title="English -> Formosan",
        target_family="Formosan",
        corpus_key="private_no_bible_en",
        easy_weight=0.15,
        input_format="<to_LANG> <src_eng> <dom_BUCKET> <dialect_DIALECT>",
        example="<to_ami> <src_eng> <dom_unknown> <dialect_default> He went home.",
        function_name="translate_english_to_formosan",
        source_setup='tokenizer.src_lang = "eng_Latn"\n    prompt = f"<to_{lang_code}> <src_eng> <dom_{source_bucket}> <dialect_{dialect}> {text}"',
        forced_bos="tokenizer.convert_tokens_to_ids(FORMOSAN_TO_LID[lang_code])",
        companion="nllb200-formosan-en-spm8k",
    ),
    "f2zh": Direction(
        code="f2zh",
        repo="nllb200-formosan-zh-spm8k",
        title="Formosan -> Traditional Chinese",
        target_family="Traditional Chinese",
        corpus_key="private_no_bible_zh",
        easy_weight=0.05,
        input_format="<to_zh> <src_LANG> <dom_BUCKET> <dialect_DIALECT>",
        example="<to_zh> <src_ami> <dom_unknown> <dialect_default> Pa'araw cingra to demak nira.",
        function_name="translate_formosan_to_chinese",
        source_setup='tokenizer.src_lang = FORMOSAN_TO_LID[lang_code]\n    prompt = f"<to_zh> <src_{lang_code}> <dom_{source_bucket}> <dialect_{dialect}> {text}"',
        forced_bos='tokenizer.convert_tokens_to_ids("zho_Hant")',
        companion="nllb200-zh-formosan-spm8k",
    ),
    "zh2f": Direction(
        code="zh2f",
        repo="nllb200-zh-formosan-spm8k",
        title="Traditional Chinese -> Formosan",
        target_family="Formosan",
        corpus_key="private_no_bible_zh",
        easy_weight=0.15,
        input_format="<to_LANG> <src_zh> <dom_BUCKET> <dialect_DIALECT>",
        example="<to_ami> <src_zh> <dom_unknown> <dialect_default> 他回家了。",
        function_name="translate_chinese_to_formosan",
        source_setup='tokenizer.src_lang = "zho_Hant"\n    prompt = f"<to_{lang_code}> <src_zh> <dom_{source_bucket}> <dialect_{dialect}> {text}"',
        forced_bos="tokenizer.convert_tokens_to_ids(FORMOSAN_TO_LID[lang_code])",
        companion="nllb200-formosan-zh-spm8k",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-stamp", default="20260712-232900")
    parser.add_argument("--organization", default="FormosanBank")
    parser.add_argument("--publish", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def usage_block(spec: Direction) -> str:
    source_example = "Pa'araw cingra to demak nira." if spec.code.startswith("f2") else (
        "He went home." if spec.code == "en2f" else "他回家了。"
    )
    return f'''```python
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

model_id = "FormosanBank/{spec.repo}"
tokenizer = NllbTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to("cuda" if torch.cuda.is_available() else "cpu")

FORMOSAN_TO_LID = {{
    "ami": "ami_Latn", "bnn": "bnn_Latn", "ckv": "ckv_Latn", "dru": "dru_Latn",
    "pwn": "pwn_Latn", "pyu": "pyu_Latn", "ssf": "ssf_Latn", "sxr": "sxr_Latn",
    "szy": "szy_Latn", "tao": "tao_Latn", "tay": "tay_Latn", "trv": "trv_Latn",
    "tsu": "tsu_Latn", "xnb": "xnb_Latn", "xsy": "xsy_Latn",
}}

def {spec.function_name}(text: str, lang_code: str, source_bucket: str = "unknown", dialect: str = "default") -> str:
    {spec.source_setup}
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(model.device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id={spec.forced_bos},
        decoder_start_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print({spec.function_name}({source_example!r}, "ami"))
```'''


def render_card(spec: Direction, metrics: dict, metadata: dict, manifest: dict) -> str:
    corpus = manifest["corpora"][spec.corpus_key]
    training = manifest["training"]
    global_metrics = metrics["global"]
    major_hf_code, major_name, major_lid = (
        ("eng", "English", "eng_Latn") if spec.corpus_key.endswith("_en") else ("zh", "Traditional Chinese", "zho_Hant")
    )
    language_tags = "\n".join(f"- {code}" for _, code, _ in LANGUAGES)
    language_table = "\n".join(
        [f"| {major_name} | `{major_lid}` |", *(f"| {name} | `{lid}` |" for name, _, lid in LANGUAGES)]
    )
    result_table = "\n".join(
        f"| {name} | `{lid}` | {metrics['by_language'][code]['samples']:,} | "
        f"{metrics['by_language'][code]['BLEU']:.2f} | {metrics['by_language'][code]['chrF2']:.2f} | "
        f"{metrics['by_language'][code]['TER']:.2f} |"
        for name, code, lid in LANGUAGES
    )
    validation = metadata["validation"]
    validation_global = validation["generation"]["global"]
    synthetic_test = 6923 if spec.corpus_key.endswith("_en") else 0
    human_test = corpus["splits"]["test"] - synthetic_test
    synthetic_note = (
        f"The English hard test contains {human_test:,} original-reference rows and {synthetic_test:,} "
        "DeepL-pivoted reference rows. Synthetic rows were admitted only when the available human sentence groups "
        "could not satisfy a language's minimum evaluation floor; metrics should not be described as human-only."
        if synthetic_test
        else "All hard-test references in this Chinese corpus are original rather than pivot-generated."
    )
    dataset_name = f"FormosanBank {spec.target_family} private no-Bible hard split"
    return f'''---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
language:
- {major_hf_code}
{language_tags}
tags:
- translation
- nllb
- nllb-200
- low-resource
- endangered-languages
- formosan-languages
- sentencepiece
- private-no-bible
metrics:
- bleu
- chrf2
- ter
model-index:
- name: {spec.repo}
  results:
  - task:
      name: Machine Translation
      type: translation
    dataset:
      name: {dataset_name}
      type: custom
    metrics:
    - name: BLEU
      type: bleu
      value: {global_metrics['BLEU']:.4f}
      args:
        direction: {spec.code}
        samples: {global_metrics['samples']}
        tokenize: {metrics['bleu_tokenize']}
    - name: chrF2
      type: chrf2
      value: {global_metrics['chrF2']:.4f}
      args:
        direction: {spec.code}
        samples: {global_metrics['samples']}
    - name: TER
      type: ter
      value: {global_metrics['TER']:.4f}
      args:
        direction: {spec.code}
        samples: {global_metrics['samples']}
---

# {spec.repo}

**Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)  
**Direction:** **{spec.title}**  
**Companion model:** [`FormosanBank/{spec.companion}`](https://huggingface.co/FormosanBank/{spec.companion})  
**Release:** private no-Bible SPM8k flight `{manifest['run_stamp']}`, validation-selected step `{metadata['step']:,}`

This directional checkpoint replaces the earlier release with the strongest `private_no_bible` model from the
fully rebuilt FormosanBank MT pipeline. It uses an 8,192-piece Formosan-aware SentencePiece extension and explicit
direction, source-language, source-domain, and dialect control tags.

## Supported Languages

| Language | NLLB code |
|---|---|
{language_table}

## Input Format

Prefix every source with:

`{spec.input_format}`

Example:

`{spec.example}`

Use `<dom_unknown>` and `<dialect_default>` when metadata is unavailable.

## Usage

Use the slow `NllbTokenizer` (`use_fast=False` with `AutoTokenizer`). These checkpoints were trained with
`transformers==4.56.1`; fast-tokenizer added-token IDs can differ from the slow tokenizer IDs used in training.
NLLB generation must start with the tokenizer EOS ID and force the target-language BOS ID.

{usage_block(spec)}

## Checkpoint Selection

The published checkpoint was selected **only on validation chrF2**, not on the hard test set.

| Selection step | Validation samples | Validation loss | Perplexity | BLEU | chrF2 | TER |
|---:|---:|---:|---:|---:|---:|---:|
| {metadata['step']:,} | {validation['generation']['samples']:,} | {validation['mean_token_loss']:.4f} | {validation['ppl']:.2f} | {validation_global['BLEU']:.2f} | {validation_global['chrF2']:.2f} | {validation_global['TER']:.2f} |

Validation generation sampled 128 rows per Formosan language every 10,000 updates. The full hard test was evaluated
only after selection.

## Training Setup

| Setting | Value |
|---|---|
| Corpus | `private_no_bible` ({spec.target_family}) |
| Base model | `facebook/nllb-200-distilled-600M` |
| Maximum updates | {training['steps']:,} |
| Published best step | {metadata['step']:,} |
| Microbatch / accumulation | {training['batch_size']} / {training['grad_accum_steps']} |
| Effective batch | {training['batch_size'] * training['grad_accum_steps']} |
| Maximum length | {training['max_length']} |
| Learning rate | `{training['learning_rate']}` |
| Precision | `{training['precision']}` |
| Easy-source weight | {spec.easy_weight} |
| Language sampling alpha | 0.5 |
| Metadata control tags | enabled and validated as single tokenizer IDs |

## Corpus and Split Integrity

| Total | Train | Test | Validate | Minimum per-language test | Minimum per-language validate |
|---:|---:|---:|---:|---:|---:|
| {corpus['rows']:,} | {corpus['splits']['train']:,} | {corpus['splits']['test']:,} | {corpus['splits']['validate']:,} | 7.5% | 2.5% |

The exact `Formosan-Taiwan-Bible-Society-Bibles` repository is excluded. Lexical entries are train-only. Independent
validation found zero normalized source, target, or pair overlap; zero punctuation/spacing skeleton overlap; and zero
one-edit source or target conflicts across train and evaluation. Connected similarity groups are assigned as units so
variants cannot be split independently merely because they are not exact duplicates.

{synthetic_note}

## Hard-Test Results

SacreBLEU was computed with `{metrics['bleu_tokenize']}` tokenization; chrF uses beta 2; TER is lower-is-better.

| Direction | Samples | BLEU | chrF2 | TER | Exact match | Empty output |
|---|---:|---:|---:|---:|---:|---:|
| {spec.title} | {global_metrics['samples']:,} | {global_metrics['BLEU']:.2f} | {global_metrics['chrF2']:.2f} | {global_metrics['TER']:.2f} | {100 * global_metrics['exact_match_rate']:.2f}% | {100 * global_metrics['empty_output_rate']:.2f}% |

### Per-Language Results

| Language | Code | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|---:|
{result_table}

Full source-bucket and length-bin breakdowns are in [`eval/metrics.json`](eval/metrics.json).

## Intended Use

- Research, teaching, and prototyping for Formosan-language machine translation.
- Draft translation assistance where knowledgeable speakers can review the output.
- Comparative low-resource MT evaluation on the documented leakage-controlled split.

## Limitations

- Output may be incorrect, ungrammatical, incomplete, or culturally inappropriate.
- Formosan generation is draft-only and requires speaker review.
- Aggregate scores across 15 languages conceal substantial per-language variation.
- This model is unsuitable for legal, medical, safety-critical, or authoritative community-facing use without expert review.
- Hard-split scores are not directly comparable with earlier evaluations that allowed stronger train-test similarity.

## License

Released under `cc-by-nc-4.0`. Underlying corpus sources may impose additional restrictions. Confirm the rights needed
for your use case.

## Citation

```bibtex
@misc{{formosanbank_{spec.repo.replace('-', '_')}_2026,
  title  = {{{spec.repo}: Directional NLLB-200 MT on the FormosanBank private no-Bible corpus}},
  author = {{FormosanBank contributors}},
  year   = {{2026}},
  url    = {{https://huggingface.co/FormosanBank/{spec.repo}}}
}}
```
'''


def prepare_direction(args: argparse.Namespace, spec: Direction, manifest: dict) -> dict:
    source = args.artifact_root / f"v1_spm8192_{spec.code}_{args.run_stamp}" / "best"
    metrics_path = args.metrics_root / f"{spec.code}.metrics.json"
    if not source.is_dir():
        raise FileNotFoundError(source)
    present = {path.name for path in source.iterdir() if path.is_file()}
    if present != REQUIRED_MODEL_FILES:
        raise RuntimeError(f"Unexpected files for {spec.code}: missing={REQUIRED_MODEL_FILES - present}, extra={present - REQUIRED_MODEL_FILES}")
    metrics = json.loads(metrics_path.read_text())
    metadata = json.loads((source / "experiment_metadata.json").read_text())
    if metrics["direction"] != spec.code or metadata["direction"] != spec.code:
        raise RuntimeError(f"Direction mismatch for {spec.code}")
    if metrics["samples"] != manifest["corpora"][spec.corpus_key]["splits"]["test"]:
        raise RuntimeError(f"Test row mismatch for {spec.code}")

    output = args.output_root / spec.repo
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    for name in sorted(REQUIRED_MODEL_FILES):
        link_or_copy(source / name, output / name)
    (output / "eval").mkdir()
    shutil.copy2(metrics_path, output / "eval" / "metrics.json")
    (output / "README.md").write_text(render_card(spec, metrics, metadata, manifest))
    (output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
        "*.model filter=lfs diff=lfs merge=lfs -text\n"
    )
    return {
        "direction": spec.code,
        "repo_id": f"{args.organization}/{spec.repo}",
        "step": metadata["step"],
        "files": {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(output.rglob("*")) if path.is_file()},
    }


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest["run_stamp"] != args.run_stamp:
        raise RuntimeError("Manifest run stamp does not match --run-stamp")
    args.output_root.mkdir(parents=True, exist_ok=True)
    release = {
        "schema_version": 1,
        "run_stamp": args.run_stamp,
        "corpus": "private_no_bible",
        "checkpoint_policy": "best validation chrF2",
        "models": [prepare_direction(args, DIRECTIONS[code], manifest) for code in DIRECTIONS],
    }
    release_path = args.output_root / "release_manifest.json"
    release_path.write_text(json.dumps(release, indent=2) + "\n")

    if args.publish:
        from huggingface_hub import HfApi

        api = HfApi()
        for item in release["models"]:
            repo_id = item["repo_id"]
            repo_dir = args.output_root / repo_id.split("/", 1)[1]
            info = api.model_info(repo_id)
            commit = api.upload_folder(
                repo_id=repo_id,
                folder_path=repo_dir,
                delete_patterns="*",
                parent_commit=info.sha,
                commit_message=f"Replace with private no-Bible SPM8k best checkpoint ({args.run_stamp})",
            )
            item["published_commit"] = commit.oid
            release_path.write_text(json.dumps(release, indent=2) + "\n")

    print(json.dumps(release, indent=2))


if __name__ == "__main__":
    main()
