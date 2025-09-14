#!/usr/bin/env python3
"""
STEP 4: 

Filter and split parallel corpus for Machine Translation training.

This script performs several cleaning operations on a parallel corpus CSV:
1. Applies Moses punctuation normalization (following NLLB approach)
2. Removes non-printable characters and normalizes text
3. Removes exact duplicate sentence pairs
4. Applies fertility heuristic based on length ratios
5. Splits the corpus into train/validation/test sets

Examples
--------
python filter_split_corpus.py --input corpus.csv --output corpus_ready.csv
python filter_split_corpus.py --input amis_zh.csv --output amis_zh_ready.csv --min-ratio 0.3 --max-ratio 5.0
python filter_split_corpus.py --input corpus.csv --train-ratio 0.9 --val-ratio 0.05 --test-ratio 0.05
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import re
import sys
import unicodedata
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm
from sacremoses import MosesPunctNormalizer


def get_non_printing_char_replacer(replace_by: str = " "):
    """Create a function to replace non-printable characters (NLLB approach)."""
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line: str, verbose: bool = False) -> str:
        replaced = []
        for char in line:
            # Skip caret (^) and apostrophe (') as per NLLB approach
            if char in "^'":
                continue
            if ord(char) in non_printable_map:
                replaced.append((char, replace_by))
        
        if verbose:
            for old, new in replaced:
                print(f"Non-printable: Replacing '{old}' with '{new}'")
        
        return line.translate(non_printable_map)

    return replace_non_printing_char


def preprocess_text(text: str, mpn: MosesPunctNormalizer, verbose: bool = False) -> str:
    """
    Preprocess text following NLLB approach:
    1. Moses punctuation normalization
    2. Non-printable character removal
    3. Unicode NFKC normalization
    
    Parameters
    ----------
    text : str
        Input text to clean
    mpn : MosesPunctNormalizer
        Moses punctuation normalizer instance
    verbose : bool
        Whether to print cleaning operations
        
    Returns
    -------
    str
        Cleaned text
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    original = text
    
    # Step 1: Moses punctuation normalization
    clean = text
    if verbose:
        for pattern, sub in mpn.substitutions:
            matches = pattern.findall(clean)
            for match in matches:
                print(f"MosesPunctNormalizer: Replacing '{match}' with '{sub}'")
    
    # Apply all Moses substitutions
    for pattern, sub in mpn.substitutions:
        clean = pattern.sub(sub, clean)
    
    # Step 2: Remove non-printable characters
    replace_nonprint = get_non_printing_char_replacer(" ")
    clean = replace_nonprint(clean, verbose)
    
    # Step 3: Unicode normalization (NFKC)
    if verbose:
        normalized_chars = []
        for char in clean:
            normalized_char = unicodedata.normalize("NFKC", char)
            if char != normalized_char:
                normalized_chars.append((char, normalized_char))
        for old, new in normalized_chars:
            print(f"Unicode NFKC: Replacing '{old}' with '{new}'")
    
    clean = unicodedata.normalize("NFKC", clean)
    
    # Step 4: Clean up extra whitespace
    clean = re.sub(r'\s+', ' ', clean).strip()
    
    if verbose and original != clean:
        print(f"Final result: '{original}' → '{clean}'")
    
    return clean


def process_chunk(args):
    """Process a chunk of text data with Moses normalization."""
    chunk_data, column_name, verbose = args
    
    # Initialize Moses normalizer for this worker
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [
        (re.compile(r), sub) for r, sub in mpn.substitutions
    ]
    
    # Process each text in the chunk
    processed = []
    for text in chunk_data:
        processed.append(preprocess_text(text, mpn, verbose))
    
    return processed


def clean_corpus_text_parallel(df: pd.DataFrame, verbose: bool = False, n_workers: int = None) -> pd.DataFrame:
    """Clean text in all string columns using NLLB approach with multiprocessing."""
    df = df.copy()
    string_columns = df.select_dtypes(include=['object']).columns
    
    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    # For small datasets, parallel processing has too much overhead - use single thread
    if len(df) < 1000:
        print(f"📝  Dataset too small ({len(df)} rows) - using single-threaded processing for better performance")
        return clean_corpus_text(df, verbose)
    
    print(f"🚀  Using {n_workers} CPU cores for parallel text cleaning")
    
    for col in string_columns:
        if col not in ['source', 'kindOf']:  # Don't clean metadata columns
            print(f"🧼  Cleaning text in column '{col}' (Moses + NLLB approach)...")
            
            # Split data into chunks for parallel processing
            data = df[col].tolist()
            # Use more reasonable chunking: minimum 100 rows per chunk, maximum n_workers * 2 chunks
            min_chunk_size = 100
            max_chunks = n_workers * 2
            chunk_size = max(min_chunk_size, len(data) // max_chunks)
            chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            
            print(f"📦  Processing {len(data):,} rows in {len(chunks)} chunks...")
            
            # Prepare arguments for each chunk
            chunk_args = [(chunk, col, verbose) for chunk in chunks]
            
            # Process chunks in parallel
            with mp.Pool(n_workers) as pool:
                print("⏳  Processing chunks...")
                chunk_results = list(tqdm(
                    pool.imap(process_chunk, chunk_args),
                    total=len(chunks),
                    desc=f"Chunks",
                    unit="chunk"
                ))
            
            # Flatten results back into a single list
            results = []
            for chunk_result in chunk_results:
                results.extend(chunk_result)
            
            print(f"✅  Processed {len(results):,} rows for column '{col}'")
            
            # Update the dataframe with cleaned text
            df[col] = results
    
    return df


def clean_corpus_text(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Clean text in all string columns using NLLB approach (single-threaded fallback)."""
    df = df.copy()
    string_columns = df.select_dtypes(include=['object']).columns
    
    # Initialize Moses punctuation normalizer (using English rules as default)
    print("🔧  Initializing Moses punctuation normalizer...")
    mpn = MosesPunctNormalizer(lang="en")
    # Compile substitutions for efficiency
    mpn.substitutions = [
        (re.compile(r), sub) for r, sub in mpn.substitutions
    ]
    
    # Enable pandas progress bars
    tqdm.pandas()
    
    for col in string_columns:
        if col not in ['source', 'kindOf']:  # Don't clean metadata columns
            print(f"🧼  Cleaning text in column '{col}' (Moses + NLLB approach)...")
            df[col] = df[col].progress_apply(lambda x: preprocess_text(x, mpn, verbose))
    
    return df


def load_data(filepath: Path) -> pd.DataFrame:
    """Load the parallel corpus CSV file."""
    try:
        df = pd.read_csv(filepath)
        print(f"✅  Loaded {len(df):,} rows from {filepath}")
        return df
    except Exception as e:
        sys.exit(f"❌  Error loading data: {e}")


def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate sentence pairs, retaining one instance."""
    initial_len = len(df)
    
    # Get the source and target columns (assume first two are the language pairs)
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split']]
    
    if len(lang_cols) >= 2:
        df_cleaned = df.drop_duplicates(subset=lang_cols[:2], keep='first')
    else:
        df_cleaned = df.drop_duplicates(keep='first')
    
    duplicates_removed = initial_len - len(df_cleaned)
    print(f"🗑️   Removed {duplicates_removed:,} exact duplicate sentence pairs")
    return df_cleaned


def apply_fertility_heuristic(
    df: pd.DataFrame, 
    min_ratio: float = 0.2, 
    max_ratio: float = 7.0
) -> pd.DataFrame:
    """
    Remove sentence pairs where the length ratio is outside specified bounds.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with language columns
    min_ratio : float
        Minimum acceptable length ratio (target/source)
    max_ratio : float  
        Maximum acceptable length ratio (target/source)
        
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame
    """
    df_work = df.copy()
    
    # Get the source and target columns (assume first two are the language pairs)
    text_cols = df.select_dtypes(include=['object']).columns
    lang_cols = [col for col in text_cols if col not in ['source', 'kindOf', 'split']]
    
    if len(lang_cols) < 2:
        print("⚠️  Warning: Cannot apply fertility heuristic - need at least 2 language columns")
        return df
        
    src_col, tgt_col = lang_cols[0], lang_cols[1]
    
    # Calculate lengths
    df_work['src_length'] = df_work[src_col].astype(str).apply(len)
    df_work['tgt_length'] = df_work[tgt_col].astype(str).apply(len)
    
    # Remove empty sentences
    df_work = df_work[(df_work['src_length'] > 0) & (df_work['tgt_length'] > 0)]
    
    # Calculate ratio
    df_work['length_ratio'] = df_work['tgt_length'] / df_work['src_length']
    
    # Apply heuristic
    condition = (df_work['length_ratio'] >= min_ratio) & (df_work['length_ratio'] <= max_ratio)
    filtered_df = df_work[condition]
    
    removed = len(df_work) - len(filtered_df)
    print(f"📏  Removed {removed:,} sentence pairs with length ratio outside [{min_ratio:.2f}, {max_ratio:.2f}]")
    
    # Drop auxiliary columns
    filtered_df = filtered_df.drop(columns=['src_length', 'tgt_length', 'length_ratio'])
    
    return filtered_df


def split_corpus(
    df: pd.DataFrame,
    train_ratio: float = 0.96,
    val_ratio: float = 0.02,
    test_ratio: float = 0.02,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Split corpus into train/validation/test sets.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    train_ratio : float
        Proportion for training set
    val_ratio : float
        Proportion for validation set  
    test_ratio : float
        Proportion for test set
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    pd.DataFrame
        DataFrame with 'split' column added
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 0.001:
        sys.exit(f"❌  Split ratios must sum to 1.0, got {total_ratio:.3f}")
    
    # Shuffle the data
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Calculate split points
    n_total = len(df_shuffled)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Split the data
    train = df_shuffled[:n_train].copy()
    val = df_shuffled[n_train:n_train + n_val].copy()
    test = df_shuffled[n_train + n_val:].copy()
    
    # Add split labels
    train['split'] = 'train'
    val['split'] = 'validate'
    test['split'] = 'test'
    
    # Combine back together
    result = pd.concat([train, val, test], ignore_index=True)
    
    print(f"📊  Split corpus: {len(train):,} train, {len(val):,} validation, {len(test):,} test")
    
    return result


def save_data(df: pd.DataFrame, output_path: Path) -> None:
    """Save the processed DataFrame to CSV."""
    try:
        df.to_csv(output_path, index=False)
        print(f"✅  Saved processed corpus to {output_path} ({len(df):,} rows)")
    except Exception as e:
        sys.exit(f"❌  Error saving data: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and split parallel corpus for MT training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument(
        "--input", type=Path, required=True,
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output", type=Path, required=True, 
        help="Path to save processed CSV file"
    )
    
    # Filtering options
    parser.add_argument(
        "--min-ratio", type=float, default=0.2,
        help="Minimum acceptable length ratio (target/source) [default: %(default)s]"
    )
    parser.add_argument(
        "--max-ratio", type=float, default=8.0,
        help="Maximum acceptable length ratio (target/source) [default: %(default)s]"
    )
    parser.add_argument(
        "--no-clean-text", action="store_true",
        help="Skip text cleaning (Moses normalization, non-printable chars)"
    )
    parser.add_argument(
        "--no-dedup", action="store_true", 
        help="Skip duplicate removal"
    )
    parser.add_argument(
        "--no-fertility", action="store_true",
        help="Skip fertility heuristic filtering"
    )
    
    # Splitting options  
    parser.add_argument(
        "--train-ratio", type=float, default=0.96,
        help="Training set ratio [default: %(default)s]"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.02, 
        help="Validation set ratio [default: %(default)s]"
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.02,
        help="Test set ratio [default: %(default)s]"
    )
    parser.add_argument(
        "--no-split", action="store_true",
        help="Skip train/val/test splitting"
    )
    parser.add_argument(
        "--random-state", type=int, default=42,
        help="Random seed for splitting [default: %(default)s]"
    )
    
    # Performance options
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of CPU cores to use (default: all available)"
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Disable parallel processing (use single thread)"
    )
    
    # Other options
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed cleaning operations"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input)
    
    # FOR TESTING: ONLY RUN THE FIRST 100 ROWS
    # COMMENT THIS OUT FOR FULL RUN
    df = df.sample(100)
    
    # Clean text (always includes Moses normalization unless explicitly disabled)
    if not args.no_clean_text:
        if args.no_parallel:
            df = clean_corpus_text(df, args.verbose)
        else:
            df = clean_corpus_text_parallel(df, args.verbose, args.workers)
    
    # Remove duplicates  
    if not args.no_dedup:
        df = remove_exact_duplicates(df)
    
    # Apply fertility heuristic
    if not args.no_fertility:
        df = apply_fertility_heuristic(df, args.min_ratio, args.max_ratio)
    
    # Split corpus
    if not args.no_split:
        df = split_corpus(df, args.train_ratio, args.val_ratio, args.test_ratio, args.random_state)
    
    # Save result
    save_data(df, args.output)


if __name__ == "__main__":
    main()