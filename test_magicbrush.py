#!/usr/bin/env python3
"""
Direct download from MagicBrush dataset - trying multiple methods.
"""

import os
import sys

print("Method 1: Using datasets library with cache...")
try:
    from datasets import load_dataset

    # Set cache directory
    cache_dir = "./hf_cache"
    os.makedirs(cache_dir, exist_ok=True)

    print("Loading MagicBrush dataset (this may take a while)...")
    dataset = load_dataset('osunlp/MagicBrush', split='train', cache_dir=cache_dir)

    print(f"✓ Successfully loaded! Dataset has {len(dataset)} samples")
    print(f"First sample keys: {list(dataset[0].keys())}")
    print(f"\nFirst sample preview:")
    for key, value in dataset[0].items():
        if isinstance(value, str):
            print(f"  {key}: {value[:100] if len(value) > 100 else value}")
        else:
            print(f"  {key}: {type(value)}")

    sys.exit(0)

except Exception as e:
    print(f"✗ Method 1 failed: {e}\n")

print("Method 2: Downloading parquet files directly...")
try:
    import requests
    import pandas as pd
    from io import BytesIO

    # Try to get parquet file URL from HuggingFace
    parquet_url = "https://huggingface.co/datasets/osunlp/MagicBrush/resolve/main/data/train-00000-of-00001.parquet"

    print(f"Downloading from: {parquet_url}")
    response = requests.get(parquet_url, timeout=60)
    response.raise_for_status()

    print("Parsing parquet file...")
    df = pd.read_parquet(BytesIO(response.content))

    print(f"✓ Successfully loaded! DataFrame has {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst row preview:")
    print(df.iloc[0])

    sys.exit(0)

except Exception as e:
    print(f"✗ Method 2 failed: {e}\n")

print("Method 3: Using streaming...")
try:
    from datasets import load_dataset

    print("Loading MagicBrush with streaming...")
    dataset = load_dataset('osunlp/MagicBrush', split='train', streaming=True)

    print("Getting first sample...")
    first_sample = next(iter(dataset))

    print(f"✓ Streaming works!")
    print(f"First sample keys: {list(first_sample.keys())}")

    sys.exit(0)

except Exception as e:
    print(f"✗ Method 3 failed: {e}\n")

print("\n❌ All methods failed. Please check network connectivity.")
sys.exit(1)
