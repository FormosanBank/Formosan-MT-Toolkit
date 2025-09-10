# MULTILINGUAL MBART TOKENIZER SETUP FOR ALL FORMOSAN LANGUAGES 
#!/usr/bin/env python3
"""
MBART MULTILINGUAL TOKENIZER SETUP

Prepare an mBart tokenizer for multilingual Machine Translation with ALL Formosan languages.

This script:
1. Loads the pre-trained mBart-50 model and tokenizer
2. Adds ALL Formosan language tokens at once
3. Identifies unknown tokens across the entire multilingual corpus
4. Patches the tokenizer with all unknown tokens in one batch
5. Saves the unified multilingual tokenizer and model

Examples
--------
python tokenizer_langs.py --input ../../../raw_corpora/big_corpus_combined.csv

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
# Map language codes to mBart language tokens
FORMOSAN_LANGUAGE_MAP: Dict[str, str] = {
    # All Formosan languages (custom tokens to be added)
    "ami": "ami_XX",  # Amis
    "bnn": "bnn_XX",  # Bunun
    "ckv": "ckv_XX",  # Kavalan
    "dru": "dru_XX",  # Rukai
    "pwn": "pwn_XX",  # Paiwan
    "pyu": "pyu_XX",  # Puyuma
    "ssf": "ssf_XX",  # Thao
    "sxr": "sxr_XX",  # Saaroa
    "szy": "szy_XX",  # Sakizaya
    "tao": "tao_XX",  # Yami/Tao
    "tay": "tay_XX",  # Atayal
    "trv": "trv_XX",  # Seediq (Note: also used for Truku)
    "tsu": "tsu_XX",  # Tsou
    "xnb": "xnb_XX",  # Kanakanavu
    "xsy": "xsy_XX",  # Saisiyat
}

# Target languages already in mBart-50
TARGET_LANGUAGE_MAP: Dict[str, str] = {
    "english": "en_XX",
    "chinese": "zh_CN",
}

# All Formosan languages need custom tokens (not in pre-trained mBart-50)
FORMOSAN_LANGUAGES = set(FORMOSAN_LANGUAGE_MAP.keys())
# ─────────────────────────────────────────────────────────────────────────────


# Old function removed - no longer needed for multilingual approach


def load_multilingual_corpus(input_path: Path) -> Tuple[pd.DataFrame, Set[str]]:
    """Load the big multilingual corpus and extract unique language codes."""
    try:
        df = pd.read_csv(input_path)
        print(f"✅  Loaded {len(df):,} rows from {input_path}")
        
        # Extract unique language codes
        unique_langs = set(df['lang_code'].unique())
        print(f"🌍  Found {len(unique_langs)} languages: {', '.join(sorted(unique_langs))}")
        
        # Validate all languages are supported
        unsupported = unique_langs - FORMOSAN_LANGUAGES
        if unsupported:
            print(f"⚠️   Warning: Unsupported language codes found: {unsupported}")
            print("    These will be skipped during processing")
            
        supported_langs = unique_langs & FORMOSAN_LANGUAGES
        print(f"✅  Processing {len(supported_langs)} supported Formosan languages")
        
        return df, supported_langs
    except Exception as e:
        sys.exit(f"❌  Error loading corpus: {e}")


def setup_multilingual_tokenizer_and_model(supported_languages: Set[str], device: str) -> Tuple[MBart50Tokenizer, MBartForConditionalGeneration]:
    """Load mBart tokenizer and add ALL Formosan language tokens at once."""
    print("🔧  Loading mBart-50 tokenizer and model...")
    tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")
    
    # Add all Formosan language tokens at once
    print(f"🌍  Adding ALL Formosan language tokens...")
    new_language_tokens = []
    
    for lang_code in sorted(supported_languages):
        mbart_token = FORMOSAN_LANGUAGE_MAP[lang_code]
        if mbart_token not in tokenizer.lang_code_to_id:
            new_language_tokens.append(mbart_token)
    
    if new_language_tokens:
        print(f"➕  Adding {len(new_language_tokens)} new language tokens: {', '.join(new_language_tokens)}")
        
        # Add all new language tokens as special tokens
        tokenizer.add_special_tokens({'additional_special_tokens': new_language_tokens})
        
        # Resize model embeddings once for all new tokens
        new_vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
        model.resize_token_embeddings(new_vocab_size)
        print(f"📏  Resized model embeddings to {new_vocab_size}")
        
        # Update language code mappings
        for token in new_language_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token)
            tokenizer.lang_code_to_id[token] = token_id
            print(f"   ✅  Added {token} with ID {token_id}")
    else:
        print("✅  All language tokens already present in tokenizer")
    
    # Move to device
    device_obj = torch.device(device)
    model = model.to(device_obj)
    print(f"📱  Model loaded on: {device}")
    
    return tokenizer, model


# Old test_generation function removed - not needed for setup phase


def analyze_multilingual_unknown_tokens(df: pd.DataFrame, 
                                       tokenizer: MBart50Tokenizer,
                                       supported_languages: Set[str]) -> Dict[str, int]:
    """Find ALL unknown tokens across all Formosan sentences and Chinese sentences."""
    print("🔍  Analyzing unknown tokens across all languages and text types...")
    
    all_unknown_tokens = {}
    
    # Process Formosan sentences for all languages
    print("📝  Processing Formosan sentences...")
    formosan_texts = df['formosan_sentence'].dropna().tolist()
    
    # Set a default language for tokenization (we'll use the first supported language)
    default_lang = sorted(supported_languages)[0] if supported_languages else 'ami'
    default_lang_token = FORMOSAN_LANGUAGE_MAP.get(default_lang, 'ami_XX')
    
    # For Formosan text, we need to use one of the Formosan language tokens
    for text in tqdm(formosan_texts, desc="Formosan sentences"):
        if pd.isna(text) or not str(text).strip():
            continue
            
        text = str(text)
        try:
            tokenizer.src_lang = default_lang_token
            tokenized = tokenizer(text, return_tensors="pt")
            input_ids = tokenized.input_ids[0]
            
            # Find characters that get tokenized as UNK
            for char in text:
                char_ids = tokenizer(char, return_tensors="pt").input_ids[0]
                if tokenizer.unk_token_id in char_ids:
                    all_unknown_tokens[char] = all_unknown_tokens.get(char, 0) + 1
        except Exception as e:
            continue  # Skip problematic texts
    
    # Process Chinese sentences 
    print("🇨🇳  Processing Chinese sentences...")
    chinese_texts = df['chinese_sentence'].dropna().tolist()
    
    for text in tqdm(chinese_texts, desc="Chinese sentences"):
        if pd.isna(text) or not str(text).strip():
            continue
            
        text = str(text)
        try:
            tokenizer.src_lang = "zh_CN"  # Use Chinese language token
            tokenized = tokenizer(text, return_tensors="pt")
            
            # Find characters that get tokenized as UNK
            for char in text:
                char_ids = tokenizer(char, return_tensors="pt").input_ids[0]
                if tokenizer.unk_token_id in char_ids:
                    all_unknown_tokens[char] = all_unknown_tokens.get(char, 0) + 1
        except Exception as e:
            continue  # Skip problematic texts
    
    print(f"📊  Found {len(all_unknown_tokens)} unique unknown characters/tokens")
    print(f"🔢  Total unknown token instances: {sum(all_unknown_tokens.values()):,}")
    
    return all_unknown_tokens


# Old function removed - replaced with analyze_multilingual_unknown_tokens


def add_unknown_tokens_to_tokenizer(tokenizer: MBart50Tokenizer,
                                   model: MBartForConditionalGeneration,
                                   unknown_tokens: Dict[str, int]) -> None:
    """Add all unknown tokens to the tokenizer vocabulary in one batch."""
    print("\n🔧  Adding unknown tokens to tokenizer vocabulary...")
    
    if not unknown_tokens:
        print("   ✅  No unknown tokens to add")
        return
    
    # Filter out tokens that already exist in the tokenizer
    existing_vocab = set(tokenizer.fairseq_tokens_to_ids.keys()) | set(tokenizer.added_tokens_encoder.keys())
    tokens_to_add = [token for token in unknown_tokens.keys() if token not in existing_vocab]
    
    if not tokens_to_add:
        print("   ✅  All unknown characters already exist in vocabulary")
        return
    
    # Sort by frequency (most common first) for better training
    tokens_to_add.sort(key=lambda x: unknown_tokens[x], reverse=True)
    
    print(f"   ➕  Adding {len(tokens_to_add)} new character tokens to vocabulary")
    print(f"   📊  Most common unknown characters: {', '.join(tokens_to_add[:10])}")
    
    # Add all tokens at once
    added_count = tokenizer.add_tokens(tokens_to_add)
    print(f"   ✅  Successfully added {added_count} new tokens")
    
    # Resize model embeddings once for all new tokens
    final_vocab_size = tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
    model.resize_token_embeddings(final_vocab_size)
    print(f"   📏  Final vocabulary size: {final_vocab_size}")


def save_multilingual_model(tokenizer: MBart50Tokenizer,
                          model: MBartForConditionalGeneration,
                          supported_languages: Set[str],
                          output_prefix: str) -> None:
    """Save the multilingual customized tokenizer and model."""
    print("\n💾  Saving multilingual tokenizer and model...")
    
    tokenizer_path = f"{output_prefix}_tokenizer"
    model_path = f"{output_prefix}_model"
    
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(model_path)
    
    # Save language mapping info
    lang_info = {
        "supported_languages": sorted(list(supported_languages)),
        "language_tokens": {lang: FORMOSAN_LANGUAGE_MAP[lang] for lang in supported_languages},
        "vocab_size": tokenizer.vocab_size + len(tokenizer.added_tokens_encoder)
    }
    
    import json
    with open(f"{output_prefix}_language_info.json", 'w') as f:
        json.dump(lang_info, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅  Tokenizer saved to: {tokenizer_path}")
    print(f"   ✅  Model saved to: {model_path}")
    print(f"   📝  Language info saved to: {output_prefix}_language_info.json")
    print(f"   🌍  Supports {len(supported_languages)} languages: {', '.join(sorted(supported_languages))}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Setup multilingual mBart tokenizer for ALL Formosan languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to big_corpus_combined.csv file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output-prefix", default="formosan_multilingual_mbart",
        help="Prefix for output files [default: %(default)s]"
    )
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to use for model [default: %(default)s]"
    )
    parser.add_argument(
        "--skip-unknown-analysis", action="store_true",
        help="Skip unknown token analysis and patching"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.input.exists():
        sys.exit(f"❌  Input file not found: {args.input}")
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("🚀  Setting up multilingual mBart tokenizer for ALL Formosan languages...")
    print(f"📁  Input corpus: {args.input}")
    print(f"📱  Device: {device}")
    print(f"📂  Output prefix: {args.output_prefix}")
    
    # Load multilingual corpus and extract language codes
    df, supported_languages = load_multilingual_corpus(args.input)
    
    # Setup tokenizer with ALL Formosan language tokens
    tokenizer, model = setup_multilingual_tokenizer_and_model(supported_languages, device)
    
    # Analyze and add unknown tokens across all languages
    if not args.skip_unknown_analysis:
        unknown_tokens = analyze_multilingual_unknown_tokens(df, tokenizer, supported_languages)
        add_unknown_tokens_to_tokenizer(tokenizer, model, unknown_tokens)
    
    # Save the multilingual model
    save_multilingual_model(tokenizer, model, supported_languages, args.output_prefix)
    
    print("\n🎉  Multilingual setup complete!")
    print(f"🌍  Your tokenizer now supports ALL {len(supported_languages)} Formosan languages")
    print(f"🚀  Ready for multilingual machine translation training!")


if __name__ == "__main__":
    main()
