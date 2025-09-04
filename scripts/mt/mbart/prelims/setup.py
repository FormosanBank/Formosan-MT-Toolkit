#!/usr/bin/env python3
"""
MBART PRELIMS SETUP 

Prepare an mBart tokenizer for Machine Translation with custom language support.

This script:
1. Loads the pre-trained mBart-50 model and tokenizer
2. Adds custom language tokens for your source language
3. Identifies unknown tokens in your parallel corpus
4. Patches the tokenizer with the unknown tokens
5. Saves the customized tokenizer and model

Examples
--------
python setup.py --src-lang amis --tgt-lang chinese --input corpus_ready.csv
python setup.py --src-lang paiwan --tgt-lang english --input pwn_en_ready.csv --output-prefix pwn_en
python setup.py --src-lang tsou --tgt-lang chinese --input tsou_zh.csv --device cpu
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path
from typing import Dict, Tuple, Set

import pandas as pd
import torch
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from tqdm.auto import tqdm


# ─────────────────────────────  language maps  ───────────────────────────────
# Map language names to mBart-50 language codes
MBART_LANGUAGE_MAP: Dict[str, str] = {
    
    # Formosan languages (custom tokens) NEED TO BE UPDATED
    "amis": "ami_XX",
    "paiwan": "pwn_XX", 
    "tsou": "tsu_XX",
    "bunun": "bnn_XX",
    "rukai": "dru_XX",
    "puyuma": "pyu_XX",
    "atayal": "tay_XX",
    "seediq": "trv_XX",
    
    # Existing mBart-50 languages
    "chinese": "zh_CN",
    "english": "en_XX",
    "japanese": "ja_XX",
    "korean": "ko_KR",
    "thai": "th_TH",
    "vietnamese": "vi_VN",
    "indonesian": "id_ID",
    "malay": "ms_MY",
    "tagalog": "tl_XX",
}

# Languages that need custom tokens (not in pre-trained mBart-50)
CUSTOM_LANGUAGES = {
    "amis", "paiwan", "tsou", "bunun", "rukai", "puyuma", "atayal", "seediq"
}
# ─────────────────────────────────────────────────────────────────────────────


def get_mbart_code(lang_name: str) -> str:
    """Get the mBart language code for a given language name."""
    lang_lower = lang_name.lower()
    if lang_lower not in MBART_LANGUAGE_MAP:
        sys.exit(f"❌  Unsupported language: {lang_name}. "
                f"Supported: {', '.join(MBART_LANGUAGE_MAP.keys())}")
    return MBART_LANGUAGE_MAP[lang_lower]


def load_corpus(input_path: Path) -> pd.DataFrame:
    """Load the parallel corpus CSV file."""
    try:
        df = pd.read_csv(input_path)
        print(f"✅  Loaded {len(df):,} rows from {input_path}")
        return df
    except Exception as e:
        sys.exit(f"❌  Error loading corpus: {e}")


def setup_tokenizer_and_model(src_lang_code: str, device: str) -> Tuple[MBart50Tokenizer, MBartForConditionalGeneration]:
    """Load and configure mBart tokenizer and model."""
    print("🔧  Loading mBart-50 tokenizer and model...")
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    
    # Add custom language token if needed
    if src_lang_code.endswith("_XX") and src_lang_code not in tokenizer.lang_code_to_id:
        print(f"➕  Adding custom language token: {src_lang_code}")
        if src_lang_code not in tokenizer.additional_special_tokens:
            tokenizer.add_special_tokens({'additional_special_tokens': [src_lang_code]})
            model.resize_token_embeddings(tokenizer.vocab_size + len(tokenizer.added_tokens_encoder))
            lang_token_id = tokenizer.convert_tokens_to_ids(src_lang_code)
            tokenizer.lang_code_to_id[src_lang_code] = lang_token_id
            print(f"   ✅  Added {src_lang_code} with ID {lang_token_id}")
    
    # Move to device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    print(f"📱  Model loaded on: {device}")
    
    return tokenizer, model


def test_generation(tokenizer: MBart50Tokenizer, 
                   model: MBartForConditionalGeneration,
                   df: pd.DataFrame,
                   src_lang_code: str,
                   tgt_lang_code: str,
                   device: str) -> None:
    """Test the model with sample generation."""
    print("\n🧪  Testing generation with sample sentences...")
    
    # Get sample sentences from the corpus
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split']]
    
    if len(lang_cols) < 2:
        print("⚠️  Warning: Not enough language columns for generation testing")
        return
    
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    
    # Get a few sample sentences
    samples = df.dropna(subset=[src_col, tgt_col]).head(2)
    
    for idx, row in samples.iterrows():
        src_text = str(row[src_col]).strip()
        tgt_text = str(row[tgt_col]).strip()
        
        if not src_text or not tgt_text:
            continue
            
        print(f"\n📝  Test {idx + 1}:")
        print(f"   Source ({src_col}): {src_text}")
        print(f"   Expected ({tgt_col}): {tgt_text}")
        
        try:
            # Generate translation
            tokenizer.src_lang = src_lang_code
            encoded = tokenizer(src_text, return_tensors="pt").to(device)
            generated = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang_code],
                max_length=50,
                num_beams=5
            )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            print(f"   Generated: {generated_text}")
        except Exception as e:
            print(f"   ❌  Generation failed: {e}")


def find_unknown_tokens(texts: list, tokenizer: MBart50Tokenizer, lang_code: str) -> Tuple[int, Dict[str, int]]:
    """Find unknown tokens in the given texts."""
    total_unk_count = 0
    all_unknown_tokens = {}
    
    print(f"🔍  Analyzing unknown tokens for {lang_code}...")
    
    for text in tqdm(texts, desc=f"Processing {lang_code}"):
        if pd.isna(text):
            continue
            
        text = str(text)
        tokenizer.src_lang = lang_code
        tokenized = tokenizer(text, return_tensors="pt")
        input_ids = tokenized.input_ids[0]
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        
        for token, id_ in zip(tokens, input_ids):
            if id_ == tokenizer.unk_token_id:
                # Identify single-character unknown tokens
                for char in text:
                    char_ids = tokenizer(char, return_tensors="pt").input_ids[0]
                    if tokenizer.unk_token_id in char_ids:
                        total_unk_count += 1
                        all_unknown_tokens[char] = all_unknown_tokens.get(char, 0) + 1
    
    return total_unk_count, all_unknown_tokens


def analyze_unknown_tokens(df: pd.DataFrame, 
                         tokenizer: MBart50Tokenizer,
                         src_lang_code: str,
                         tgt_lang_code: str,
                         output_prefix: str) -> str:
    """Analyze and save unknown tokens from both languages."""
    print("\n📊  Analyzing unknown tokens in corpus...")
    
    # Get language columns
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split']]
    
    if len(lang_cols) < 2:
        sys.exit("❌  Error: Need at least 2 language columns in CSV")
    
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    
    # Analyze unknown tokens for both languages
    src_unk_count, src_unknown_tokens = find_unknown_tokens(
        df[src_col].dropna().tolist(), tokenizer, src_lang_code
    )
    tgt_unk_count, tgt_unknown_tokens = find_unknown_tokens(
        df[tgt_col].dropna().tolist(), tokenizer, tgt_lang_code
    )
    
    # Save results
    unknown_tokens_df = pd.DataFrame({
        "language": [src_col, tgt_col],
        "language_code": [src_lang_code, tgt_lang_code],
        "total_unk_count": [src_unk_count, tgt_unk_count],
        "unknown_tokens": [src_unknown_tokens, tgt_unknown_tokens],
    })
    
    unknown_tokens_path = f"{output_prefix}_unknown_tokens.csv"
    unknown_tokens_df.to_csv(unknown_tokens_path, index=False)
    
    print(f"📈  Unknown token analysis:")
    print(f"   {src_col} ({src_lang_code}): {src_unk_count:,} unknown tokens")
    print(f"   {tgt_col} ({tgt_lang_code}): {tgt_unk_count:,} unknown tokens")
    print(f"   💾  Saved analysis to: {unknown_tokens_path}")
    
    return unknown_tokens_path


def patch_tokenizer_with_unknown_tokens(tokenizer: MBart50Tokenizer,
                                      model: MBartForConditionalGeneration,
                                      unknown_tokens_path: str) -> None:
    """Add unknown tokens to the tokenizer vocabulary."""
    print("\n🔧  Patching tokenizer with unknown tokens...")
    
    # Load unknown tokens analysis
    unk_df = pd.read_csv(unknown_tokens_path)
    unk_df["parsed_unknown_tokens"] = unk_df["unknown_tokens"].apply(
        lambda x: ast.literal_eval(x) if pd.notna(x) else {}
    )
    
    # Collect all unique unknown tokens
    all_unknown_tokens = set()
    for tokens_dict in unk_df["parsed_unknown_tokens"]:
        all_unknown_tokens.update(tokens_dict.keys())
    
    # Filter out tokens that already exist
    existing_vocab = set(tokenizer.fairseq_tokens_to_ids.keys()) | set(tokenizer.added_tokens_encoder.keys())
    tokens_to_add = [token for token in all_unknown_tokens if token not in existing_vocab]
    
    if not tokens_to_add:
        print("   ✅  No new tokens to add - all characters already in vocabulary")
        return
    
    print(f"   ➕  Adding {len(tokens_to_add)} new tokens to vocabulary")
    added_count = tokenizer.add_tokens(tokens_to_add)
    print(f"   ✅  Successfully added {added_count} new tokens")
    
    # Resize model embeddings
    model.resize_token_embeddings(tokenizer.vocab_size + len(tokenizer.added_tokens_encoder))
    print(f"   📏  Resized model embeddings to {tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)}")


def save_customized_model(tokenizer: MBart50Tokenizer,
                         model: MBartForConditionalGeneration,
                         output_prefix: str) -> None:
    """Save the customized tokenizer and model."""
    print("\n💾  Saving customized tokenizer and model...")
    
    tokenizer_path = f"{output_prefix}_custom_tokenizer"
    model_path = f"{output_prefix}_custom_model"
    
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)
    
    print(f"   ✅  Tokenizer saved to: {tokenizer_path}")
    print(f"   ✅  Model saved to: {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup mBart tokenizer for custom language MT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--src-lang", required=True,
        help="Source language name (e.g., 'amis', 'paiwan')"
    )
    parser.add_argument(
        "--tgt-lang", required=True,
        help="Target language name (e.g., 'chinese', 'english')"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to parallel corpus CSV file (from filter_split_corpus.py)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-prefix", default=None,
        help="Prefix for output files (default: src_tgt format)"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to use for model [default: %(default)s]"
    )
    parser.add_argument(
        "--skip-generation-test", action="store_true",
        help="Skip generation testing with sample sentences"
    )
    parser.add_argument(
        "--skip-unknown-analysis", action="store_true",
        help="Skip unknown token analysis and patching"
    )
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Set output prefix
    if args.output_prefix is None:
        args.output_prefix = f"{args.src_lang}_{args.tgt_lang}_mbart"
    
    # Get mBart language codes
    src_lang_code = get_mbart_code(args.src_lang)
    tgt_lang_code = get_mbart_code(args.tgt_lang)
    
    print(f"🌐  Language mapping:")
    print(f"   Source: {args.src_lang} → {src_lang_code}")
    print(f"   Target: {args.tgt_lang} → {tgt_lang_code}")
    
    # Load corpus
    df = load_corpus(args.input)
    
    # Setup tokenizer and model
    tokenizer, model = setup_tokenizer_and_model(src_lang_code, device)
    
    # Test generation
    if not args.skip_generation_test:
        test_generation(tokenizer, model, df, src_lang_code, tgt_lang_code, device)
    
    # Analyze and patch unknown tokens
    if not args.skip_unknown_analysis:
        unknown_tokens_path = analyze_unknown_tokens(
            df, tokenizer, src_lang_code, tgt_lang_code, args.output_prefix
        )
        patch_tokenizer_with_unknown_tokens(tokenizer, model, unknown_tokens_path)
    
    # Save customized model
    save_customized_model(tokenizer, model, args.output_prefix)
    
    print("\n🎉  Setup complete! Your customized mBart model is ready for training.")


if __name__ == "__main__":
    main()