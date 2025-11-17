# -*- coding: utf-8 -*-
"""
add_embeddings.py
Add sentence and word embeddings to video analysis CSV

This script:
1. Reads CSV with 'description' and 'keywords' columns
2. Generates description embeddings using sBERT (all-mpnet-base-v2) → 768 dimensions
3. Generates keywords embeddings using word2vec → 300 dimensions (averaged)
4. Outputs new CSV with all embedding columns added

Usage:
    python add_embeddings.py --input video_analysis.csv --output video_with_embeddings.csv
"""

import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import sys
import os

# Sentence Transformers for sBERT
from sentence_transformers import SentenceTransformer

# Gensim for word2vec
import gensim.downloader as api

warnings.filterwarnings('ignore')


def load_sbert_model(device='cuda'):
    """
    Load sBERT model: all-mpnet-base-v2
    This is the same model family used in the OSF paper
    Output: 768 dimensions
    """
    print("=" * 80)
    print("Loading sBERT model (all-mpnet-base-v2)...")
    print("=" * 80)
    
    try:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        print(f"✓ sBERT model loaded successfully on {device}")
        print(f"  Model: all-mpnet-base-v2")
        print(f"  Output dimensions: 768")
        return model
    except Exception as e:
        print(f"✗ Error loading sBERT model: {e}")
        print("\nTrying to download model...")
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device=device)
        print(f"✓ sBERT model downloaded and loaded on {device}")
        return model


def load_word2vec_model():
    """
    Load word2vec model: word2vec-google-news-300
    This is the standard word2vec model used in the OSF paper
    Output: 300 dimensions per word
    """
    print("\n" + "=" * 80)
    print("Loading word2vec model (word2vec-google-news-300)...")
    print("=" * 80)
    print("⚠️  First time: This will download ~1.6GB model (may take a few minutes)")
    print("    Subsequent runs will use cached model")
    
    try:
        model = api.load('word2vec-google-news-300')
        print(f"✓ word2vec model loaded successfully")
        print(f"  Model: word2vec-google-news-300")
        print(f"  Output dimensions: 300 per word")
        print(f"  Vocabulary size: {len(model)} words")
        return model
    except Exception as e:
        print(f"✗ Error loading word2vec model: {e}")
        sys.exit(1)


def generate_description_embeddings(descriptions, model, batch_size=32):
    """
    Generate sBERT embeddings for descriptions
    
    Args:
        descriptions: List of description strings
        model: SentenceTransformer model
        batch_size: Batch size for encoding
    
    Returns:
        numpy array of shape (n_descriptions, 768)
    """
    print("\n" + "=" * 80)
    print("Generating description embeddings with sBERT...")
    print("=" * 80)
    print(f"Total descriptions: {len(descriptions)}")
    print(f"Batch size: {batch_size}")
    
    # Encode all descriptions
    embeddings = model.encode(
        descriptions, 
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"  Expected: ({len(descriptions)}, 768)")
    
    return embeddings


def generate_keywords_embeddings(keywords_list, model):
    """
    Generate word2vec embeddings for keywords with average pooling
    
    Args:
        keywords_list: List of keyword strings (e.g., "boy, couch, box")
        model: Gensim word2vec model
    
    Returns:
        numpy array of shape (n_frames, 300)
    """
    print("\n" + "=" * 80)
    print("Generating keywords embeddings with word2vec...")
    print("=" * 80)
    print(f"Total keyword sets: {len(keywords_list)}")
    print("Strategy: Average pooling over available word vectors")
    
    embeddings = []
    missing_words_count = 0
    total_words_count = 0
    
    for keywords_str in tqdm(keywords_list, desc="Processing keywords"):
        # Split keywords by comma
        keywords = [kw.strip().lower() for kw in keywords_str.split(',')]
        
        # Collect vectors for words that exist in vocabulary
        vectors = []
        for word in keywords:
            total_words_count += 1
            try:
                # Try to get vector for the word
                vector = model[word]
                vectors.append(vector)
            except KeyError:
                # Word not in vocabulary
                missing_words_count += 1
                # Try common variations
                variations = [
                    word.replace(' ', ''),  # Remove spaces
                    word.replace('-', ''),  # Remove hyphens
                    word.replace('_', ''),  # Remove underscores
                ]
                for var in variations:
                    try:
                        vector = model[var]
                        vectors.append(vector)
                        missing_words_count -= 1  # Found alternative
                        break
                    except KeyError:
                        continue
        
        # Average pooling
        if len(vectors) > 0:
            avg_vector = np.mean(vectors, axis=0)
        else:
            # If no words found, use zero vector
            avg_vector = np.zeros(300)
        
        embeddings.append(avg_vector)
    
    embeddings = np.array(embeddings)
    
    print(f"✓ Generated embeddings shape: {embeddings.shape}")
    print(f"  Expected: ({len(keywords_list)}, 300)")
    print(f"  Words found in vocabulary: {total_words_count - missing_words_count}/{total_words_count} ({100*(total_words_count-missing_words_count)/total_words_count:.1f}%)")
    print(f"  Words NOT found: {missing_words_count}")
    
    return embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Add sBERT and word2vec embeddings to video analysis CSV"
    )
    parser.add_argument(
        '--input', 
        required=True, 
        help='Input CSV file with description and keywords columns'
    )
    parser.add_argument(
        '--output', 
        required=True, 
        help='Output CSV file with added embedding columns'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to run sBERT model (default: cuda)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for sBERT encoding (default: 32)'
    )
    
    args = parser.parse_args()
    
    # ============================================================================
    # Step 1: Check if input file exists
    # ============================================================================
    if not os.path.exists(args.input):
        print(f"✗ Error: Input file not found: {args.input}")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("VIDEO EMBEDDING GENERATOR")
    print("=" * 80)
    print(f"Input file:  {args.input}")
    print(f"Output file: {args.output}")
    print(f"Device:      {args.device}")
    print("=" * 80)
    
    # ============================================================================
    # Step 2: Load CSV
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 1/5: Loading CSV file...")
    print("=" * 80)
    
    try:
        df = pd.read_csv(args.input)
        print(f"✓ CSV loaded successfully")
        print(f"  Total rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error loading CSV: {e}")
        sys.exit(1)
    
    # Check required columns
    required_cols = ['frame_index', 'timestamp_sec', 'description', 'keywords']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"✗ Error: Missing required columns: {missing_cols}")
        sys.exit(1)
    
    print(f"✓ Required columns present: {required_cols}")
    
    # ============================================================================
    # Step 3: Load models
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 2/5: Loading embedding models...")
    print("=" * 80)
    
    # Load sBERT
    sbert_model = load_sbert_model(device=args.device)
    
    # Load word2vec
    w2v_model = load_word2vec_model()
    
    # ============================================================================
    # Step 4: Generate description embeddings
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 3/5: Generating description embeddings...")
    print("=" * 80)
    
    descriptions = df['description'].tolist()
    desc_embeddings = generate_description_embeddings(
        descriptions, 
        sbert_model, 
        batch_size=args.batch_size
    )
    
    # ============================================================================
    # Step 5: Generate keywords embeddings
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 4/5: Generating keywords embeddings...")
    print("=" * 80)
    
    keywords = df['keywords'].tolist()
    kw_embeddings = generate_keywords_embeddings(keywords, w2v_model)
    
    # ============================================================================
    # Step 6: Add embeddings to dataframe and save
    # ============================================================================
    print("\n" + "=" * 80)
    print("Step 5/5: Creating output CSV...")
    print("=" * 80)
    
    # Create column names for description embeddings
    desc_cols = [f'desc_emb_{i}' for i in range(768)]
    desc_df = pd.DataFrame(desc_embeddings, columns=desc_cols)
    
    # Create column names for keywords embeddings
    kw_cols = [f'kw_emb_{i}' for i in range(300)]
    kw_df = pd.DataFrame(kw_embeddings, columns=kw_cols)
    
    # Combine with original dataframe
    # Order: frame_index, timestamp_sec, description, keywords, desc_emb_*, kw_emb_*
    output_df = pd.concat([
        df[['frame_index', 'timestamp_sec', 'description', 'keywords']],
        desc_df,
        kw_df
    ], axis=1)
    
    # Add other original columns if they exist
    other_cols = [col for col in df.columns if col not in ['frame_index', 'timestamp_sec', 'description', 'keywords']]
    if other_cols:
        output_df = pd.concat([output_df, df[other_cols]], axis=1)
    
    print(f"Output dataframe shape: {output_df.shape}")
    print(f"  Rows: {len(output_df)}")
    print(f"  Total columns: {len(output_df.columns)}")
    print(f"    - Original columns: {len(df.columns)}")
    print(f"    - Description embedding columns: 768")
    print(f"    - Keywords embedding columns: 300")
    
    # Save to CSV
    try:
        output_df.to_csv(args.output, index=False)
        print(f"\n✓ Output saved to: {args.output}")
    except Exception as e:
        print(f"\n✗ Error saving output: {e}")
        sys.exit(1)
    
    # ============================================================================
    # Summary
    # ============================================================================
    print("\n" + "=" * 80)
    print("✓✓✓ EMBEDDING GENERATION COMPLETE! ✓✓✓")
    print("=" * 80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nEmbedding Summary:")
    print(f"  • Description embeddings: 768 dimensions (sBERT all-mpnet-base-v2)")
    print(f"  • Keywords embeddings:    300 dimensions (word2vec-google-news-300)")
    print(f"  • Total frames processed: {len(output_df)}")
    print(f"  • Total embedding dims:   1068 (768 + 300)")
    print("\nYou can now use this CSV for fMRI encoding analysis!")
    print("=" * 80)


if __name__ == "__main__":
    main()
