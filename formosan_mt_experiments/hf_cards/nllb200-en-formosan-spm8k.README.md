---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
language:
- eng
- ami
- bnn
- ckv
- dru
- pwn
- pyu
- ssf
- sxr
- szy
- tao
- tay
- trv
- tsu
- xnb
- xsy
tags:
- translation
- nllb
- nllb-200
- low-resource
- endangered-languages
- formosan-languages
- seq2seq
- encoder-decoder
- sentencepiece
metrics:
- bleu
- chrf2
- ter
model-index:
- name: nllb200-en-formosan-spm8k
  results:
  - task:
      name: Machine Translation
      type: translation
    dataset:
      name: FormosanBank English Parallel Corpus, leakage-controlled in-domain hard split
      type: custom
    metrics:
    - name: BLEU
      type: bleu
      value: 5.77
      args:
        direction: eng_Latn-to-formosan
        samples: 36559
    - name: chrF2
      type: chrf2
      value: 30.24
      args:
        direction: eng_Latn-to-formosan
        samples: 36559
    - name: TER
      type: ter
      value: 88.72
      args:
        direction: eng_Latn-to-formosan
        samples: 36559
---

# nllb200-en-formosan-spm8k

**Repo:** `FormosanBank/nllb200-en-formosan-spm8k`  
**Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)  
**Direction:** **English -> Formosan**  
**Companion reverse-direction model:** [`FormosanBank/nllb200-formosan-en-spm8k`](https://huggingface.co/FormosanBank/nllb200-formosan-en-spm8k)

This is a directional NLLB-200 distilled 600M checkpoint for FormosanBank machine translation. It uses an 8k SentencePiece vocabulary extension plus FormosanBank metadata/control tags. Use the companion model for the reverse direction.

## Supported Languages

| Language | NLLB code |
|---|---|
| English | `eng_Latn` |
| Amis | `ami_Latn` |
| Bunun | `bnn_Latn` |
| Kavalan | `ckv_Latn` |
| Rukai | `dru_Latn` |
| Paiwan | `pwn_Latn` |
| Puyuma | `pyu_Latn` |
| Thao | `ssf_Latn` |
| Saaroa | `sxr_Latn` |
| Sakizaya | `szy_Latn` |
| Tao / Yami | `tao_Latn` |
| Atayal | `tay_Latn` |
| Seediq | `trv_Latn` |
| Tsou | `tsu_Latn` |
| Kanakanavu | `xnb_Latn` |
| Saisiyat | `xsy_Latn` |

## Input Format

This model was trained and evaluated with metadata control tags. Prefix the English source text with:

`<to_LANG> <src_eng> <dom_BUCKET> <dialect_DIALECT>`

Example with unknown metadata:

`<to_ami> <src_eng> <dom_unknown> <dialect_default> He went home.`

If source bucket or dialect is unknown, use `<dom_unknown>` and `<dialect_default>`. If you know the training source bucket or dialect, using the matching tag may improve quality.

## Usage

Tested with `transformers` 4.56.x. Use the slow `NllbTokenizer` or `AutoTokenizer.from_pretrained(model_id, use_fast=False)`. In `transformers` 4.56.x, fast-tokenizer added-token IDs can differ from the slow tokenizer IDs used during training.

For NLLB generation, keep `decoder_start_token_id=tokenizer.eos_token_id` and set `forced_bos_token_id` to the target language ID.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

model_id = "FormosanBank/nllb200-en-formosan-spm8k"
tokenizer = NllbTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to("cuda" if torch.cuda.is_available() else "cpu")

FORMOSAN_TO_LID = {
    "ami": "ami_Latn", "bnn": "bnn_Latn", "ckv": "ckv_Latn", "dru": "dru_Latn",
    "pwn": "pwn_Latn", "pyu": "pyu_Latn", "ssf": "ssf_Latn", "sxr": "sxr_Latn",
    "szy": "szy_Latn", "tao": "tao_Latn", "tay": "tay_Latn", "trv": "trv_Latn",
    "tsu": "tsu_Latn", "xnb": "xnb_Latn", "xsy": "xsy_Latn",
}

def translate_english_to_formosan(text: str, lang_code: str, source_bucket: str = "unknown", dialect: str = "default") -> str:
    tokenizer.src_lang = "eng_Latn"
    prompt = f"<to_{lang_code}> <src_eng> <dom_{source_bucket}> <dialect_{dialect}> {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(model.device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(FORMOSAN_TO_LID[lang_code]),
        decoder_start_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(translate_english_to_formosan("He went home.", "ami"))
```

## Training Setup

| Setting | Value |
|---|---|
| Corpus | FormosanBank English Parallel Corpus, leakage-controlled in-domain hard split |
| Direction | `en2f` |
| Base model | `facebook/nllb-200-distilled-600M` |
| Tokenizer | 8k Formosan SentencePiece extension |
| Steps | 300,000 |
| Batch size | 16 |
| Gradient accumulation | 4 |
| Effective batch size | 64 |
| Max sequence length | 384 |
| Learning rate | 2e-05 |
| Warmup steps | 4,000 |
| Precision | `bf16` |
| Label smoothing | 0.1 |
| Easy-source weight | 0.15 |
| Language sampling alpha | 0.5 |
| Metadata tags | enabled and validated as single tokenizer IDs |

This repo publishes the final 300k-step checkpoint because it scored highest on the held-out hard test set among the evaluated final/best checkpoints.

## Evaluation

Evaluation used the held-out `in_domain_hard` test split with no normalized source, target, or pair overlap against train. These scores are intentionally lower than leaky or near-duplicate splits and are intended as a harder MT benchmark.

### Global Metrics

| Direction | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|
| English -> Formosan | 36,559 | 5.77 | 30.24 | 88.72 |

### Per-Language Metrics

| Language | Code | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|---:|
| Amis | `ami_Latn` | 4,660 | 3.79 | 24.78 | 86.88 |
| Bunun | `bnn_Latn` | 3,195 | 3.92 | 29.63 | 90.85 |
| Kavalan | `ckv_Latn` | 1,622 | 10.24 | 36.52 | 80.51 |
| Rukai | `dru_Latn` | 3,876 | 2.98 | 26.89 | 98.55 |
| Paiwan | `pwn_Latn` | 3,124 | 4.82 | 30.85 | 88.71 |
| Puyuma | `pyu_Latn` | 2,112 | 12.34 | 36.58 | 82.49 |
| Thao | `ssf_Latn` | 1,183 | 9.55 | 37.93 | 80.26 |
| Saaroa | `sxr_Latn` | 1,139 | 2.25 | 35.91 | 92.09 |
| Sakizaya | `szy_Latn` | 1,523 | 7.80 | 32.13 | 82.79 |
| Tao / Yami | `tao_Latn` | 1,450 | 4.74 | 28.14 | 89.76 |
| Atayal | `tay_Latn` | 4,185 | 2.79 | 24.15 | 94.04 |
| Seediq | `trv_Latn` | 4,455 | 5.73 | 29.14 | 86.05 |
| Tsou | `tsu_Latn` | 1,209 | 4.52 | 30.16 | 90.13 |
| Kanakanavu | `xnb_Latn` | 1,506 | 7.56 | 38.49 | 92.23 |
| Saisiyat | `xsy_Latn` | 1,320 | 13.31 | 41.36 | 83.65 |

Full source-bucket and length-bin breakdowns are available in [`eval/metrics.json`](eval/metrics.json).

## Intended Use

- Research, teaching, and prototyping for Formosan-language MT.
- Draft translation assistance where review by knowledgeable speakers is available.
- Comparative evaluation of low-resource MT methods on leakage-controlled FormosanBank splits.

## Limitations

- Outputs can be incorrect, ungrammatical, incomplete, or culturally inappropriate.
- Generation into Formosan languages is especially difficult and should be treated as draft-only.
- This model is not suitable for legal, medical, safety-critical, or authoritative community-facing use without expert review.
- Evaluation uses a hard split; BLEU should not be compared directly to older leaky or near-duplicate split results.

## License

Released under `cc-by-nc-4.0`. Some underlying corpus sources may carry additional restrictions. Use this model only for non-commercial research and educational purposes unless you have confirmed broader rights for your use case.

## Citation

```bibtex
@misc{formosanbank_nllb200_en_formosan_spm8k,
  title  = {nllb200-en-formosan-spm8k: Directional NLLB-200 MT for the FormosanBank English Parallel Corpus, leakage-controlled in-domain hard split},
  author = {FormosanBank contributors},
  year   = {2026},
  url    = {https://huggingface.co/FormosanBank/nllb200-en-formosan-spm8k}
}
```
