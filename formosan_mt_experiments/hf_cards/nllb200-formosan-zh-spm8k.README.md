---
license: cc-by-nc-4.0
library_name: transformers
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
language:
- zh
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
- traditional-chinese
metrics:
- bleu
- chrf2
- ter
model-index:
- name: nllb200-formosan-zh-spm8k
  results:
  - task:
      name: Machine Translation
      type: translation
    dataset:
      name: FormosanBank Chinese Parallel Corpus, leakage-controlled in-domain hard split
      type: custom
    metrics:
    - name: BLEU
      type: bleu
      value: 9.79
      args:
        direction: formosan-to-zho_Hant
        samples: 37435
        tokenize: zh
    - name: chrF2
      type: chrf2
      value: 11.77
      args:
        direction: formosan-to-zho_Hant
        samples: 37435
    - name: TER
      type: ter
      value: 109.29
      args:
        direction: formosan-to-zho_Hant
        samples: 37435
---

# nllb200-formosan-zh-spm8k

**Repo:** `FormosanBank/nllb200-formosan-zh-spm8k`  
**Base model:** [`facebook/nllb-200-distilled-600M`](https://huggingface.co/facebook/nllb-200-distilled-600M)  
**Direction:** **Formosan -> Traditional Chinese**  
**Companion reverse-direction model:** [`FormosanBank/nllb200-zh-formosan-spm8k`](https://huggingface.co/FormosanBank/nllb200-zh-formosan-spm8k)

This is a directional NLLB-200 distilled 600M checkpoint for FormosanBank machine translation. It uses an 8k SentencePiece vocabulary extension plus FormosanBank metadata/control tags. Use the companion model for the reverse direction.

## Supported Languages

| Language | NLLB code |
|---|---|
| Traditional Chinese | `zho_Hant` |
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

This model was trained and evaluated with metadata control tags. Prefix the source text in one of the supported Formosan languages with:

`<to_zh> <src_LANG> <dom_BUCKET> <dialect_DIALECT>`

Example with unknown metadata:

`<to_zh> <src_ami> <dom_unknown> <dialect_default> Pa'araw cingra to demak nira.`

If source bucket or dialect is unknown, use `<dom_unknown>` and `<dialect_default>`. If you know the training source bucket or dialect, using the matching tag may improve quality.

## Usage

Tested with `transformers` 4.56.x. Use the slow `NllbTokenizer` or `AutoTokenizer.from_pretrained(model_id, use_fast=False)`. In `transformers` 4.56.x, fast-tokenizer added-token IDs can differ from the slow tokenizer IDs used during training.

For NLLB generation, keep `decoder_start_token_id=tokenizer.eos_token_id` and set `forced_bos_token_id` to the target language ID.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer

model_id = "FormosanBank/nllb200-formosan-zh-spm8k"
tokenizer = NllbTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
model.to("cuda" if torch.cuda.is_available() else "cpu")

FORMOSAN_TO_LID = {
    "ami": "ami_Latn", "bnn": "bnn_Latn", "ckv": "ckv_Latn", "dru": "dru_Latn",
    "pwn": "pwn_Latn", "pyu": "pyu_Latn", "ssf": "ssf_Latn", "sxr": "sxr_Latn",
    "szy": "szy_Latn", "tao": "tao_Latn", "tay": "tay_Latn", "trv": "trv_Latn",
    "tsu": "tsu_Latn", "xnb": "xnb_Latn", "xsy": "xsy_Latn",
}

def translate_formosan_to_chinese(text: str, lang_code: str, source_bucket: str = "unknown", dialect: str = "default") -> str:
    tokenizer.src_lang = FORMOSAN_TO_LID[lang_code]
    prompt = f"<to_zh> <src_{lang_code}> <dom_{source_bucket}> <dialect_{dialect}> {text}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=384).to(model.device)
    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids("zho_Hant"),
        decoder_start_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.15,
        length_penalty=1.0,
        early_stopping=True,
    )
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

print(translate_formosan_to_chinese("Pa'araw cingra to demak nira.", "ami"))
```

## Training Setup

| Setting | Value |
|---|---|
| Corpus | FormosanBank Chinese Parallel Corpus, leakage-controlled in-domain hard split |
| Direction | `f2zh` |
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
| Easy-source weight | 0.05 |
| Language sampling alpha | 0.5 |
| Metadata tags | enabled and validated as single tokenizer IDs |

This repo publishes the final 300k-step checkpoint because it scored highest on the held-out hard test set among the evaluated final/best checkpoints.

## Evaluation

Evaluation used the held-out `in_domain_hard` test split with no normalized source, target, or pair overlap against train. These scores are intentionally lower than leaky or near-duplicate splits and are intended as a harder MT benchmark.

### Global Metrics

| Direction | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|
| Formosan -> Traditional Chinese | 37,435 | 9.79 | 11.77 | 109.29 |

### Per-Language Metrics

| Language | Code | Samples | BLEU | chrF2 | TER |
|---|---:|---:|---:|---:|---:|
| Amis | `ami_Latn` | 4,866 | 9.68 | 11.26 | 107.56 |
| Bunun | `bnn_Latn` | 3,223 | 9.44 | 10.52 | 105.04 |
| Kavalan | `ckv_Latn` | 1,832 | 14.27 | 15.11 | 108.67 |
| Rukai | `dru_Latn` | 3,846 | 5.57 | 8.57 | 108.67 |
| Paiwan | `pwn_Latn` | 3,221 | 7.86 | 9.95 | 106.33 |
| Puyuma | `pyu_Latn` | 2,225 | 15.77 | 16.93 | 124.55 |
| Thao | `ssf_Latn` | 1,180 | 14.73 | 16.36 | 106.69 |
| Saaroa | `sxr_Latn` | 1,106 | 9.17 | 12.79 | 108.07 |
| Sakizaya | `szy_Latn` | 1,601 | 11.42 | 14.83 | 104.58 |
| Tao / Yami | `tao_Latn` | 1,455 | 6.93 | 10.38 | 110.85 |
| Atayal | `tay_Latn` | 4,293 | 6.83 | 9.07 | 108.89 |
| Seediq | `trv_Latn` | 4,573 | 10.09 | 11.96 | 109.35 |
| Tsou | `tsu_Latn` | 1,250 | 8.56 | 11.42 | 126.84 |
| Kanakanavu | `xnb_Latn` | 1,552 | 12.96 | 15.19 | 107.91 |
| Saisiyat | `xsy_Latn` | 1,212 | 14.45 | 16.32 | 106.80 |

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
@misc{formosanbank_nllb200_formosan_zh_spm8k,
  title  = {nllb200-formosan-zh-spm8k: Directional NLLB-200 MT for the FormosanBank Chinese Parallel Corpus, leakage-controlled in-domain hard split},
  author = {FormosanBank contributors},
  year   = {2026},
  url    = {https://huggingface.co/FormosanBank/nllb200-formosan-zh-spm8k}
}
```
