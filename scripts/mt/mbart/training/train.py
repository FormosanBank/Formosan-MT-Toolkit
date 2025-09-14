#!/usr/bin/env python3
"""
MBART TRAINING SCRIPT

Train an mBart model for Machine Translation using a custom tokenizer and parallel corpus.

This script:
1. Loads a custom mBart tokenizer and model (from custom_tokenizer_langs.py output)
2. Loads a parallel corpus CSV file with train/validation/test splits
3. Trains the model with bidirectional translation sampling
4. Saves checkpoints at regular intervals
5. Handles OOM errors gracefully

Examples
--------
# Basic training with Paiwan-Chinese
python train.py --src-lang paiwan --tgt-lang chinese \\
                --tokenizer pwn_zh_mbart_custom_tokenizer \\
                --model pwn_zh_mbart_custom_model \\
                --input pwn_zh_ready.csv

# Custom training parameters
python train.py --src-lang amis --tgt-lang english \\
                --tokenizer ami_en_custom_tokenizer \\
                --model ami_en_custom_model \\
                --input ami_en_ready.csv \\
                --batch-size 16 --learning-rate 3e-5 --steps 100000 \\
                --output-dir ami_en_checkpoints

# Training on CPU (for testing)
python train.py --src-lang tsou --tgt-lang chinese \\
                --tokenizer tsu_zh_tokenizer --model tsu_zh_model \\
                --input tsu_zh_ready.csv --device cpu --steps 1000
"""
from __future__ import annotations

import argparse
import os
import random
import gc
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import trange
from transformers import (
    MBartForConditionalGeneration, 
    MBart50Tokenizer, 
    Adafactor, 
    get_constant_schedule_with_warmup
)


# ─────────────────────────────  language maps  ───────────────────────────────
# Map language names to mBart-50 language codes (same as setup.py)
MBART_LANGUAGE_MAP: Dict[str, str] = {
    
    # Formosan languages (custom tokens)
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


def load_corpus(input_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and split the parallel corpus CSV file."""
    try:
        df = pd.read_csv(input_path)
        print(f"✅  Loaded {len(df):,} rows from {input_path}")
        
        # Split into train/validation/test
        df_train = df[df['split'] == 'train'].copy()
        df_val = df[df['split'] == 'validate'].copy()
        df_test = df[df['split'] == 'test'].copy()
        
        print(f"📊  Data splits: {len(df_train):,} train, {len(df_val):,} validation, {len(df_test):,} test")
        
        if len(df_train) == 0:
            sys.exit("❌  No training data found. Make sure CSV has 'split' column with 'train' entries.")
        
        return df_train, df_val, df_test
    except Exception as e:
        sys.exit(f"❌  Error loading corpus: {e}")


def load_tokenizer_and_model(tokenizer_path: str, model_path: str, device: str) -> Tuple[MBart50Tokenizer, MBartForConditionalGeneration]:
    """Load the custom tokenizer and model."""
    try:
        print(f"🔧  Loading tokenizer from: {tokenizer_path}")
        tokenizer = MBart50Tokenizer.from_pretrained(tokenizer_path)
        
        print(f"🔧  Loading model from: {model_path}")
        model = MBartForConditionalGeneration.from_pretrained(model_path)
        
        # Ensure model embeddings match tokenizer
        model.resize_token_embeddings(len(tokenizer))
        
        # Move to device
        device_obj = torch.device(device)
        model = model.to(device_obj)
        
        print(f"📱  Model loaded on: {device}")
        print(f"📏  Tokenizer vocab size: {len(tokenizer):,}")
        
        return tokenizer, model
    except Exception as e:
        sys.exit(f"❌  Error loading tokenizer/model: {e}")


def restore_custom_language_codes(tokenizer: MBart50Tokenizer, src_lang_code: str, tgt_lang_code: str) -> None:
    """Restore custom language codes to tokenizer mappings (needed after reload)."""
    custom_codes = [code for code in [src_lang_code, tgt_lang_code] if code.endswith('_XX')]
    
    for lang_code in custom_codes:
        if lang_code not in tokenizer.lang_code_to_id:
            print(f"🔧  Restoring custom language code: {lang_code}")
            lang_token_id = tokenizer.convert_tokens_to_ids(lang_code)
            tokenizer.lang_code_to_id[lang_code] = lang_token_id
            tokenizer.id_to_lang_code[lang_token_id] = lang_code


def cleanup() -> None:
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_batch_pairs(
    batch_size: int, 
    data: pd.DataFrame, 
    src_col: str, 
    tgt_col: str, 
    src_lang_code: str, 
    tgt_lang_code: str
) -> Tuple[List[str], List[str], str, str]:
    """Generate a batch of translation pairs with random direction."""
    
    # Randomly choose translation direction
    lang_pairs = [(src_col, src_lang_code, tgt_col, tgt_lang_code),
                  (tgt_col, tgt_lang_code, src_col, src_lang_code)]
    source_col, source_lang, target_col, target_lang = random.choice(lang_pairs)
    
    # Sample batch
    batch_indices = np.random.choice(len(data), batch_size, replace=True)
    source_texts = []
    target_texts = []
    
    for idx in batch_indices:
        row = data.iloc[idx]
        source_text = str(row[source_col]).strip()
        target_text = str(row[target_col]).strip()
        
        if source_text and target_text:
            source_texts.append(source_text)
            target_texts.append(target_text)
    
    return source_texts, target_texts, source_lang, target_lang


def train_model(
    tokenizer: MBart50Tokenizer,
    model: MBartForConditionalGeneration,
    df_train: pd.DataFrame,
    src_col: str,
    tgt_col: str,
    src_lang_code: str,
    tgt_lang_code: str,
    args: argparse.Namespace
) -> None:
    """Train the mBart model."""
    
    # Setup optimizer and scheduler
    optimizer = Adafactor(
        model.parameters(),
        scale_parameter=False,
        relative_step=False,
        lr=args.learning_rate,
        clip_threshold=args.clip_threshold,
        weight_decay=args.weight_decay,
    )
    
    scheduler = get_constant_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_steps
    )
    
    # Training setup
    device = next(model.parameters()).device
    losses = []
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀  Starting training...")
    print(f"   Device: {device}")
    print(f"   Training steps: {args.steps:,}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Max length: {args.max_length}")
    
    model.train()
    
    # Training loop
    pbar = trange(args.steps, desc="Training")
    
    for step in pbar:
        try:
            # Get batch
            source_texts, target_texts, source_lang, target_lang = get_batch_pairs(
                args.batch_size, df_train, src_col, tgt_col, src_lang_code, tgt_lang_code
            )
            
            if not source_texts or not target_texts:
                continue
            
            # Tokenize source
            tokenizer.src_lang = source_lang
            source_encoded = tokenizer(
                source_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=args.max_length
            ).to(device)
            
            # Tokenize target
            tokenizer.src_lang = target_lang
            target_encoded = tokenizer(
                target_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=args.max_length
            ).to(device)
            
            # Set padding tokens to -100 (ignore in loss)
            target_encoded.input_ids[target_encoded.input_ids == tokenizer.pad_token_id] = -100
            
            # Forward pass
            loss = model(**source_encoded, labels=target_encoded.input_ids).loss
            
            # Backward pass
            loss.backward()
            losses.append(loss.item())
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            
            # Progress reporting
            if step % args.log_interval == 0 and len(losses) > 0:
                recent_losses = losses[-args.log_interval:] if len(losses) >= args.log_interval else losses
                avg_loss = np.mean(recent_losses)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "direction": f"{source_lang}→{target_lang}"})
            
            # Save checkpoint
            if step % args.save_interval == 0 and step > 0:
                save_dir = os.path.join(args.output_dir, f"{args.model_name}-step-{step}")
                print(f"\n💾  Saving checkpoint at step {step}...")
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
                print(f"   ✅  Checkpoint saved to: {save_dir}")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n⚠️  OOM at step {step}: {e}")
                print("   Cleaning up and continuing...")
                optimizer.zero_grad(set_to_none=True)
                cleanup()
                continue
            else:
                raise e
    
    # Save final model
    final_save_dir = os.path.join(args.output_dir, f"{args.model_name}-final")
    print(f"\n💾  Saving final model...")
    os.makedirs(final_save_dir, exist_ok=True)
    model.save_pretrained(final_save_dir)
    tokenizer.save_pretrained(final_save_dir)
    print(f"   ✅  Final model saved to: {final_save_dir}")
    
    print(f"\n🎉  Training completed! Final average loss: {np.mean(losses[-100:]):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train mBart model for custom language MT",
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
        "--tokenizer", required=True,
        help="Path to custom tokenizer directory (from setup.py)"
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to custom model directory (from setup.py)"
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to parallel corpus CSV file with train/val/test splits"
    )
    
    # Training parameters
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Training batch size [default: %(default)s]"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=5e-5,
        help="Learning rate [default: %(default)s]"
    )
    parser.add_argument(
        "--steps", type=int, default=60000,
        help="Number of training steps [default: %(default)s]"
    )
    parser.add_argument(
        "--max-length", type=int, default=128,
        help="Maximum sequence length [default: %(default)s]"
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000,
        help="Warmup steps for scheduler [default: %(default)s]"
    )
    parser.add_argument(
        "--clip-threshold", type=float, default=1.0,
        help="Gradient clipping threshold [default: %(default)s]"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01,
        help="Weight decay [default: %(default)s]"
    )
    
    # Output and logging
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for checkpoints (default: src_tgt_checkpoints)"
    )
    parser.add_argument(
        "--model-name", default=None,
        help="Base name for saved models (default: mbart-src-tgt)"
    )
    parser.add_argument(
        "--save-interval", type=int, default=5000,
        help="Save checkpoint every N steps [default: %(default)s]"
    )
    parser.add_argument(
        "--log-interval", type=int, default=1000,
        help="Log progress every N steps [default: %(default)s]"
    )
    
    # Device
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to use for training [default: %(default)s]"
    )
    
    args = parser.parse_args()
    
    # Set defaults based on language pair
    if args.output_dir is None:
        args.output_dir = f"{args.src_lang}_{args.tgt_lang}_checkpoints"
    
    if args.model_name is None:
        args.model_name = f"mbart-{args.src_lang}-{args.tgt_lang}"
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Get mBart language codes
    src_lang_code = get_mbart_code(args.src_lang)
    tgt_lang_code = get_mbart_code(args.tgt_lang)
    
    print(f"🌐  Language mapping:")
    print(f"   Source: {args.src_lang} → {src_lang_code}")
    print(f"   Target: {args.tgt_lang} → {tgt_lang_code}")
    
    # Load corpus
    df_train, df_val, df_test = load_corpus(args.input)
    
    # Get language column names from corpus
    text_cols = df_train.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split']]
    
    if len(lang_cols) < 2:
        sys.exit("❌  Error: Need at least 2 language columns in CSV")
    
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    print(f"📝  Using columns: '{src_col}' → '{tgt_col}'")
    
    # Load tokenizer and model
    tokenizer, model = load_tokenizer_and_model(args.tokenizer, args.model, device)
    
    # Restore custom language codes
    restore_custom_language_codes(tokenizer, src_lang_code, tgt_lang_code)
    
    # Start training
    train_model(
        tokenizer, model, df_train, src_col, tgt_col, 
        src_lang_code, tgt_lang_code, args
    )


if __name__ == "__main__":
    main()