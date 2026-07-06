# Fine-tuning NLLB-200 for a New Low-Resource Language in 2026

A practical successor to David Dale's NLLB tutorial, updated for current `transformers` and written as a public Colab workflow.

**Author:** Hunter Scheppat, Boston College  
**Project:** AI4CommSci Lab / FormosanBank  
**Associated notebook:** `NLLB_200_MT.ipynb`

NLLB-200 is still one of the most useful open baselines for low-resource machine translation. The model already knows 200 language varieties, but many languages are missing, underrepresented, or poorly tokenized. This post shows how to adapt NLLB-200 to a new language pair with modern Hugging Face tooling.

The worked example uses Atayal from the public FormosanBank dataset, but the workflow is general. If you have a parallel corpus for another language, you can use the same steps by changing the dataset columns and language codes.

## What this tutorial covers

This tutorial shows how to:

- Load a public Hugging Face dataset or your own CSV.
- Convert the data into a simple parallel-text schema.
- Build or validate train, validation, and test splits.
- Check for exact train/eval leakage.
- Extend NLLB's SentencePiece tokenizer with corpus-specific pieces.
- Add a new NLLB-style language code such as `tay_Latn`.
- Add optional control tags for direction, source language, domain, and dialect.
- Fine-tune `facebook/nllb-200-distilled-600M` directionally.
- Evaluate with BLEU and chrF2.
- Generate correctly with current `transformers`.
- Save the model and tokenizer for reuse or publication.

The main change from older tutorials is that we train one direction at a time. For a two-way translator, train two checkpoints:

- source language -> target language
- target language -> source language

Directional training is easier to debug, easier to evaluate, and avoids several common language-ID mistakes.

## 1. Environment

Use a GPU runtime. A Colab T4 can run smoke tests and short fine-tuning runs. A full model will need more time.

```python
!nvidia-smi -L || echo "No GPU found. In Colab, go to Runtime -> Change runtime type -> GPU."
!pip install -q "transformers>=4.56,<4.57" sentencepiece sacremoses sacrebleu datasets pandas numpy tqdm protobuf
```

Imports:

```python
import os
import random
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torch.nn.functional as F
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
from tqdm.auto import trange
from transformers import (
    Adafactor,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    NllbTokenizer,
    get_constant_schedule_with_warmup,
)

try:
    from torch.amp import GradScaler, autocast

    def make_scaler(enabled: bool):
        return GradScaler("cuda", enabled=enabled)

    def amp_context(enabled: bool, dtype):
        return autocast("cuda", dtype=dtype, enabled=enabled)
except Exception:
    from torch.cuda.amp import GradScaler, autocast

    def make_scaler(enabled: bool):
        return GradScaler(enabled=enabled)

    def amp_context(enabled: bool, dtype):
        return autocast(dtype=dtype, enabled=enabled)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Transformers:", __import__("transformers").__version__)
print("Torch:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
```

## 2. Data contract

The training code expects a simple table:

```text
source_sentence,target_sentence,source_lang,target_lang,split,source,dialect
```

Required columns:

- `source_sentence`: input text.
- `target_sentence`: reference translation.
- `source_lang`: short source language code, such as `tay`.
- `target_lang`: short target language code, such as `en`.
- `split`: `train`, `validate`, or `test`.

Optional but useful columns:

- `source`: corpus, document, book, website, or data source label.
- `dialect`: dialect, variety, orthography, or region label.

For your own corpus, a CSV like this is enough:

```python
df = pd.read_csv("my_parallel_corpus.csv")
df = df.rename(columns={
    "my_source_text_column": "source_sentence",
    "my_target_text_column": "target_sentence",
})
```

For the worked example, load the public FormosanBank dataset:

```python
HF_DATASET = "FormosanBank/formosan-mt"
HF_CONFIG = "formosan-en"

raw = load_dataset(HF_DATASET, HF_CONFIG, split="train")
df = raw.to_pandas()

df.head()
```

Some public datasets expose `source_sentence` and `target_sentence` directly. Others have language-specific columns. This helper normalizes either format:

```python
def normalize_parallel_schema(df: pd.DataFrame, target_lang: str = "en") -> pd.DataFrame:
    cols = set(df.columns)
    if {"source_lang", "target_lang", "source_sentence", "target_sentence"}.issubset(cols):
        out = df.copy()
        out["source_lang"] = out["source_lang"].astype(str).str.lower()
        out["target_lang"] = out["target_lang"].astype(str).str.lower()
        for col, default in [("source", ""), ("dialect", "default"), ("split", "train")]:
            if col not in out.columns:
                out[col] = default
        return out

    target_col = "english_sentence" if target_lang == "en" else "chinese_sentence"
    if {"lang_code", "formosan_sentence", target_col}.issubset(cols):
        out = df[df[target_col].notna()].copy()
        out = out.rename(columns={
            "formosan_sentence": "source_sentence",
            target_col: "target_sentence",
        })
        out["source_lang"] = out["lang_code"].astype(str).str.lower()
        out["target_lang"] = target_lang
        for col, default in [("source", ""), ("dialect", "default"), ("split", "train")]:
            if col not in out.columns:
                out[col] = default
        return out

    raise ValueError(f"Unsupported schema: {sorted(cols)}")
```

Filter one language pair:

```python
SOURCE_LANG = "tay"
TARGET_LANG = "en"
SOURCE_LID = "tay_Latn"
TARGET_LID = "eng_Latn"
DIRECTION = "src2tgt"

df = normalize_parallel_schema(df, target_lang=TARGET_LANG)
pair_df = df[
    (df["source_lang"].astype(str).str.lower() == SOURCE_LANG)
    & (df["target_lang"].astype(str).str.lower() == TARGET_LANG)
].copy()

pair_df["split"] = pair_df["split"].astype(str).str.lower()
pair_df["source"] = pair_df["source"].fillna("").astype(str)
pair_df["dialect"] = pair_df["dialect"].fillna("default").astype(str)

train_df = pair_df[pair_df["split"] == "train"].reset_index(drop=True)
val_df = pair_df[pair_df["split"].isin(["validate", "valid", "val"])].reset_index(drop=True)
test_df = pair_df[pair_df["split"] == "test"].reset_index(drop=True)

len(train_df), len(val_df), len(test_df)
```

## 3. Splits and leakage

For low-resource corpora, random row splits are often too easy. Parallel corpora may contain repeated examples, dictionary entries, lesson templates, or near duplicates from the same source document.

At minimum, check that train and eval do not share exact normalized source strings, target strings, or source-target pairs:

```python
SRC_COL = "source_sentence"
TGT_COL = "target_sentence"

def normalize_text(x) -> str:
    text = unicodedata.normalize("NFKC", "" if pd.isna(x) else str(x))
    text = text.casefold()
    return " ".join(text.split())

def leakage_report(train, *eval_sets):
    eval_df = pd.concat(eval_sets, ignore_index=True)
    train_src = set(train[SRC_COL].map(normalize_text))
    eval_src = set(eval_df[SRC_COL].map(normalize_text))
    train_tgt = set(train[TGT_COL].map(normalize_text))
    eval_tgt = set(eval_df[TGT_COL].map(normalize_text))
    train_pair = set(train[SRC_COL].map(normalize_text) + "\u241f" + train[TGT_COL].map(normalize_text))
    eval_pair = set(eval_df[SRC_COL].map(normalize_text) + "\u241f" + eval_df[TGT_COL].map(normalize_text))
    return {
        "source_overlap": len(train_src & eval_src),
        "target_overlap": len(train_tgt & eval_tgt),
        "pair_overlap": len(train_pair & eval_pair),
    }

leakage_report(train_df, val_df, test_df)
```

For your own corpus, prefer holding out by document, source file, speaker, or collection when that metadata exists. Exact overlap should be zero before you trust the evaluation.

## 4. Language IDs

NLLB uses language tokens such as:

```text
eng_Latn
zho_Hant
```

For a new language, choose a code that follows the same shape:

```text
abc_Latn
abc_Cyrl
abc_Arab
```

The code does not need to be an official NLLB language if you add it to the tokenizer as a special token. It must be a single known token after setup.

```python
def ensure_token(tokenizer, token: str) -> int:
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid == tokenizer.unk_token_id or tokenizer.convert_ids_to_tokens(tid) != token:
        raise ValueError(f"{token} is not a single known tokenizer token")
    return int(tid)
```

Do not rely on old `lang_code_to_id` internals. Use `convert_tokens_to_ids`.

## 5. Optional control tags

Control tags make the training task explicit. They are especially useful for multilingual or multi-domain corpora.

Example source prefix:

```text
<to_eng> <src_tay> <dom_unknown> <dialect_default>
```

The tags mean:

- `<to_eng>`: target side is English.
- `<src_tay>`: source language is Atayal.
- `<dom_unknown>`: broad source/domain label is unknown.
- `<dialect_default>`: dialect label is unknown or default.

For your own corpus, you can simplify this to only direction and source-language tags. Keep tags short and add them as special tokens.

```python
def safe_tag_value(value: object, default: str = "default", max_len: int = 48) -> str:
    import re
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = default
    return text[:max_len].strip("_") or default

def source_bucket(source: object) -> str:
    s = "" if pd.isna(source) else str(source)
    s_lower = s.lower()
    if "dictionary" in s_lower or "dict" in s_lower:
        return "dictionary"
    if "picture_book" in s_lower:
        return "picture_book"
    if "picture_story" in s_lower:
        return "picture_story"
    if "reading" in s_lower or "writing" in s_lower:
        return "reading_writing"
    if "culture" in s_lower:
        return "culture"
    return "unknown"

def build_prefix(row, direction: str) -> str:
    src_short = SOURCE_LANG if direction == "src2tgt" else TARGET_LANG
    tgt_short = TARGET_LANG if direction == "src2tgt" else SOURCE_LANG
    domain_tag = f"<dom_{safe_tag_value(source_bucket(row.get('source', '')))}>"
    dialect_tag = f"<dialect_{safe_tag_value(row.get('dialect', 'default'))}>"
    return f"<to_{tgt_short}> <src_{src_short}> {domain_tag} {dialect_tag}"
```

## 6. Train a SentencePiece extension

NLLB already has a large multilingual SentencePiece model. For an underrepresented language, the base tokenizer may split words into too many tiny pieces. The fix is not to throw away NLLB's tokenizer. Instead, train a small SentencePiece model on your corpus and append missing normal pieces to NLLB's existing model.

```python
BASE_MODEL_NAME = "facebook/nllb-200-distilled-600M"
SPM_VOCAB = 8000
MIN_CHAR_FREQ = 3

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return unicodedata.normalize("NFKC", s)

def build_spm_corpus(texts: List[str], path: str) -> Counter:
    with open(path, "w", encoding="utf-8") as f:
        for t in texts:
            t = clean_text(t)
            if t.strip():
                f.write(t + "\n")
    char_counts = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            for ch in line.rstrip("\n"):
                if ch != " ":
                    char_counts[ch] += 1
    return char_counts

def merge_spm_models(nllb_tokenizer, merged_spm_out: str, corpus_txt_path: str, spm_vocab: int, required_chars: str = "") -> int:
    spm_prefix = merged_spm_out.replace(".model", "") + "_tmp"
    spm.SentencePieceTrainer.train(
        input=corpus_txt_path,
        model_prefix=spm_prefix,
        vocab_size=int(spm_vocab),
        character_coverage=1.0,
        num_threads=max(1, os.cpu_count() or 1),
        add_dummy_prefix=False,
        max_sentencepiece_length=128,
        pad_id=0,
        eos_id=1,
        unk_id=2,
        bos_id=-1,
        required_chars=required_chars,
        train_extremely_large_corpus=False,
    )

    sp_trained = spm.SentencePieceProcessor(model_file=f"{spm_prefix}.model")
    added_spm = sp_pb2_model.ModelProto()
    added_spm.ParseFromString(sp_trained.serialized_model_proto())

    base_spm = sp_pb2_model.ModelProto()
    base_spm.ParseFromString(nllb_tokenizer.sp_model.serialized_model_proto())

    existing = {p.piece for p in base_spm.pieces}
    prev_min_score = base_spm.pieces[-1].score
    added = 0
    for p in added_spm.pieces:
        if getattr(p, "type", 1) != 1:
            continue
        if p.piece not in existing:
            new_p = base_spm.pieces.add()
            new_p.piece = p.piece
            new_p.score = p.score + prev_min_score
            added += 1

    with open(merged_spm_out, "wb") as f:
        f.write(base_spm.SerializeToString())

    for ext in (".model", ".vocab"):
        try:
            os.remove(f"{spm_prefix}{ext}")
        except OSError:
            pass

    return added
```

## 7. Add tokens and resize the model

Use the slow tokenizer for this setup. Added-token behavior can differ between fast and slow tokenizers, and training should use the same tokenizer that you publish.

```python
def add_special_tokens_preserve_mask(tokenizer, new_tokens: List[str]):
    mask = tokenizer.mask_token
    existing = list(tokenizer.additional_special_tokens or [])
    merged = []
    for tok in existing + list(new_tokens):
        if tok and tok != mask and tok not in merged:
            merged.append(tok)
    if mask:
        merged.append(mask)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": merged},
        replace_additional_special_tokens=True,
    )

def reload_tokenizer_with_spm_and_tokens(base_tokenizer, new_spm_path: str, special_tokens: List[str]) -> NllbTokenizer:
    tmp = tempfile.mkdtemp(prefix="nllb_tok_")
    base_tokenizer.save_pretrained(tmp)
    shutil.copy(new_spm_path, os.path.join(tmp, "sentencepiece.bpe.model"))
    tok = NllbTokenizer.from_pretrained(tmp, use_fast=False)
    add_special_tokens_preserve_mask(tok, special_tokens)
    shutil.rmtree(tmp, ignore_errors=True)
    return tok

def warm_start_new_rows(model, tokenizer_old, tokenizer_new) -> int:
    old_vocab = tokenizer_old.get_vocab()
    new_vocab = tokenizer_new.get_vocab()
    new_tokens = sorted(set(new_vocab) - set(old_vocab))
    if not new_tokens:
        return 0

    emb = model.get_input_embeddings().weight.data
    unk_old = tokenizer_old.unk_token_id
    for tok in new_tokens:
        new_id = tokenizer_new.convert_tokens_to_ids(tok)
        old_ids = tokenizer_old(tok, add_special_tokens=False).input_ids
        if not old_ids:
            old_ids = [unk_old]
        emb[new_id] = emb[old_ids].mean(0)
    return len(new_tokens)

def seed_new_language_code(model, tokenizer, new_lid: str, seed_lid: str) -> bool:
    emb = model.get_input_embeddings().weight.data
    src_id = tokenizer.convert_tokens_to_ids(seed_lid)
    tgt_id = tokenizer.convert_tokens_to_ids(new_lid)
    if src_id == tokenizer.unk_token_id or tgt_id == tokenizer.unk_token_id:
        return False
    emb[tgt_id] = emb[src_id].clone()
    return True
```

Apply the tokenizer update:

```python
tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_NAME)

train_texts = train_df[SRC_COL].astype(str).tolist() + train_df[TGT_COL].astype(str).tolist()
corpus_path = "spm_training_corpus.txt"
char_counts = build_spm_corpus(train_texts, corpus_path)
required_chars = "".join(ch for ch, n in char_counts.items() if n >= MIN_CHAR_FREQ)

merged_spm_path = "merged_nllb_spm.model"
added_pieces = merge_spm_models(
    tokenizer_base,
    merged_spm_path,
    corpus_path,
    spm_vocab=SPM_VOCAB,
    required_chars=required_chars,
)

special_tokens = [SOURCE_LID, TARGET_LID]
for frame in [train_df, val_df, test_df]:
    for _, row in frame.iterrows():
        special_tokens.extend(build_prefix(row, "src2tgt").split())
        special_tokens.extend(build_prefix(row, "tgt2src").split())
special_tokens = sorted(set(special_tokens))

tokenizer = reload_tokenizer_with_spm_and_tokens(tokenizer_base, merged_spm_path, special_tokens)
model.resize_token_embeddings(len(tokenizer))
warmed = warm_start_new_rows(model, tokenizer_base, tokenizer)
seeded = seed_new_language_code(model, tokenizer, SOURCE_LID, seed_lid=TARGET_LID)

model.config.decoder_start_token_id = tokenizer.eos_token_id
if getattr(model, "generation_config", None) is not None:
    model.generation_config.decoder_start_token_id = tokenizer.eos_token_id

model.to(device)

print({"added_spm_pieces": added_pieces, "warm_started_rows": warmed, "seeded_language_code": seeded})
```

Validate tokens:

```python
for tok in [SOURCE_LID, TARGET_LID, f"<src_{SOURCE_LANG}>", f"<to_{TARGET_LANG}>"]:
    print(tok, ensure_token(tokenizer, tok))
```

## 8. Prepare directional rows

The same data can train either direction. This tutorial defaults to source -> target.

```python
def make_directional_frame(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    out = df.copy()
    prefixes = out.apply(lambda row: build_prefix(row, direction), axis=1)
    if direction == "src2tgt":
        out["src_text"] = prefixes + " " + out[SRC_COL].astype(str)
        out["tgt_text"] = out[TGT_COL].astype(str)
        out["src_lid"] = SOURCE_LID
        out["tgt_lid"] = TARGET_LID
    elif direction == "tgt2src":
        out["src_text"] = prefixes + " " + out[TGT_COL].astype(str)
        out["tgt_text"] = out[SRC_COL].astype(str)
        out["src_lid"] = TARGET_LID
        out["tgt_lid"] = SOURCE_LID
    else:
        raise ValueError(direction)
    return out

train_dir = make_directional_frame(train_df, DIRECTION)
val_dir = make_directional_frame(val_df, DIRECTION)
test_dir = make_directional_frame(test_df, DIRECTION)
```

## 9. Train

Current `transformers` can create shifted decoder inputs from labels. During training, set `tokenizer.src_lang`, set `tokenizer.tgt_lang`, tokenize with `text_target`, and pass `labels`.

```python
def encode_batch(tokenizer, src_texts, tgt_texts, src_lid, tgt_lid, max_length, device):
    tokenizer.src_lang = src_lid
    enc = tokenizer(
        list(src_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )

    tokenizer.tgt_lang = tgt_lid
    labels = tokenizer(
        text_target=list(tgt_texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_token_type_ids=False,
    )["input_ids"]

    labels[labels == tokenizer.pad_token_id] = -100
    return {k: v.to(device) for k, v in enc.items()}, labels.to(device)
```

Run a short training loop:

```python
STEPS = 500
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 2
MAX_LENGTH = 192
LEARNING_RATE = 2e-5
WARMUP_STEPS = 50
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0
PRECISION = "fp16" if device.type == "cuda" else "fp32"

optimizer = Adafactor(
    model.parameters(),
    lr=LEARNING_RATE,
    scale_parameter=False,
    relative_step=False,
    warmup_init=False,
    weight_decay=1e-3,
)
scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)

use_amp = PRECISION in {"fp16", "bf16"} and device.type == "cuda"
amp_dtype = torch.float16 if PRECISION == "fp16" else torch.bfloat16
scaler = make_scaler(PRECISION == "fp16" and device.type == "cuda")

src_lid = train_dir["src_lid"].iloc[0]
tgt_lid = train_dir["tgt_lid"].iloc[0]
running = []
model.train()

pbar = trange(1, STEPS + 1, dynamic_ncols=True)
for step in pbar:
    optimizer.zero_grad(set_to_none=True)
    update_loss = 0.0

    for _ in range(GRAD_ACCUM_STEPS):
        batch = train_dir.sample(n=BATCH_SIZE, replace=True)
        enc, labels = encode_batch(
            tokenizer,
            batch["src_text"].tolist(),
            batch["tgt_text"].tolist(),
            src_lid,
            tgt_lid,
            MAX_LENGTH,
            device,
        )

        with amp_context(use_amp, amp_dtype):
            outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], labels=labels)
            loss = F.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                label_smoothing=LABEL_SMOOTHING,
            )
            loss = loss / GRAD_ACCUM_STEPS

        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        update_loss += float(loss.detach().item())

    if scaler.is_enabled():
        scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

    if scaler.is_enabled():
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    scheduler.step()

    running.append(update_loss)
    pbar.set_postfix(loss=f"{np.mean(running[-100:]):.4f}")
```

For a real model, increase `STEPS`, use a larger effective batch size if memory allows, and evaluate periodically.

## 10. Evaluate

For this NLLB setup, generation should use EOS as the decoder start token and the target language as `forced_bos_token_id`.

```python
@torch.no_grad()
def generate_batch(texts, src_lid, tgt_lid, batch_size=16, max_length=256, max_new_tokens=128, num_beams=4):
    forced_id = ensure_token(tokenizer, tgt_lid)
    outputs = []
    model.eval()

    for start in range(0, len(texts), batch_size):
        batch = [str(x) for x in texts[start:start + batch_size]]
        tokenizer.src_lang = src_lid
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_token_type_ids=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        gen = model.generate(
            **enc,
            forced_bos_token_id=forced_id,
            decoder_start_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
            length_penalty=1.0,
        )
        outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outputs

if len(test_dir):
    eval_src_lid = test_dir["src_lid"].iloc[0]
    eval_tgt_lid = test_dir["tgt_lid"].iloc[0]
    hyp = generate_batch(test_dir["src_text"].tolist(), eval_src_lid, eval_tgt_lid)
    ref = test_dir["tgt_text"].astype(str).tolist()
    bleu = BLEU(tokenize="13a", effective_order=True).corpus_score(hyp, [ref]).score
    chrf = CHRF().corpus_score(hyp, [ref]).score
    print({"BLEU": bleu, "chrF2": chrf, "samples": len(ref)})
```

Inspect examples:

```python
for i in random.sample(range(len(test_dir)), min(5, len(test_dir))):
    print("SRC:", test_dir.iloc[i]["src_text"])
    print("REF:", ref[i])
    print("HYP:", hyp[i])
    print("---")
```

## 11. Inference helper

```python
def translate(text: str, direction: str = DIRECTION, source_label: str = "unknown", dialect: str = "default") -> str:
    row = {"source": source_label, "dialect": dialect}
    prefix = build_prefix(row, direction)

    if direction == "src2tgt":
        src_lid, tgt_lid = SOURCE_LID, TARGET_LID
    elif direction == "tgt2src":
        src_lid, tgt_lid = TARGET_LID, SOURCE_LID
    else:
        raise ValueError(direction)

    prompt = f"{prefix} {text}"
    tokenizer.src_lang = src_lid
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
    out = model.generate(
        **enc,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lid),
        decoder_start_token_id=tokenizer.eos_token_id,
        max_new_tokens=128,
        num_beams=4,
    )
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]
```

## 12. Save and publish

Save the model and tokenizer together:

```python
OUT_DIR = Path(f"nllb_{SOURCE_LANG}_{TARGET_LANG}_{DIRECTION}_finetuned")
OUT_DIR.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(OUT_DIR)
model.save_pretrained(OUT_DIR)

print("Saved to:", OUT_DIR.resolve())
```

Reload:

```python
tok = NllbTokenizer.from_pretrained(OUT_DIR)
mdl = AutoModelForSeq2SeqLM.from_pretrained(OUT_DIR).to(device)
```

If you publish to the Hugging Face Hub, include a model card with:

- Base model.
- Dataset and split policy.
- Direction.
- Language codes.
- Control-tag format.
- `transformers` version.
- Generation settings.
- BLEU and chrF2.
- Known limitations.

## 13. What we learned from the Formosan work

The FormosanBank experiments used this same general pattern across multiple Formosan languages:

- Better data mattered more than any single modeling trick.
- Directional models were easier to control than one mixed bidirectional checkpoint.
- A SentencePiece extension reduced tokenizer fragmentation.
- Metadata tags helped keep multilingual and multi-domain data explicit.
- Leakage-controlled splits made scores lower but more honest.
- English or Chinese -> Formosan remained harder than Formosan -> English or Chinese.

Those lessons apply beyond Formosan languages. For any low-resource MT project, first improve the corpus and evaluation split, then adapt the tokenizer, then tune the model.

## 14. Checklist for your own language

Before training:

- Your CSV has `source_sentence`, `target_sentence`, `source_lang`, `target_lang`, and `split`.
- Train/eval exact overlap is zero for source, target, and pairs.
- You picked a new NLLB-style language code.
- All language codes and control tags are single tokenizer IDs.
- You trained the SentencePiece extension on train text only.
- You are training one direction at a time.

During generation:

- Set `tokenizer.src_lang` to the source language ID.
- Set `forced_bos_token_id` to the target language ID.
- Set `decoder_start_token_id` to `tokenizer.eos_token_id`.
- Use the same tokenizer that was saved with the model.

## Conclusion

NLLB-200 can be adapted to languages it did not originally support, but the details matter. The reliable recipe is:

1. Clean and split the data carefully.
2. Add a tokenizer extension instead of only adding a language code.
3. Add the new language ID and any control tags as special tokens.
4. Train directional checkpoints.
5. Evaluate on a leakage-controlled test set.
6. Publish the model and tokenizer together.

This gives a reproducible baseline that other researchers can inspect, rerun, and adapt to their own languages.
