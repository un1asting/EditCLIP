#!/usr/bin/env python3
"""
Load samples from a locally downloaded MagicBrush or InstructPix2Pix dataset.

Usage:
    python3 load_from_local_dataset.py --dataset_path /path/to/dataset
    python3 load_from_local_dataset.py --parquet_file /path/to/data.parquet
"""

import argparse
import json
import random
from pathlib import Path
from PIL import Image
import sys

random.seed(42)

def classify_edit_type(instruction):
    """Classify edit type based on instruction text."""
    instruction_lower = instruction.lower()

    add_keywords = ['add', 'insert', 'place', 'put', 'draw', 'include']
    remove_keywords = ['remove', 'delete', 'erase', 'take away', 'eliminate', 'get rid']
    color_keywords = ['color', 'colour', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                      'black', 'white', 'pink', 'brown', 'gray', 'grey']
    style_keywords = ['style', 'painting', 'watercolor', 'sketch', 'cartoon', 'artistic',
                      'oil painting', 'pencil', 'drawing']

    if any(keyword in instruction_lower for keyword in add_keywords):
        return 'object_add'
    elif any(keyword in instruction_lower for keyword in remove_keywords):
        return 'object_remove'
    elif any(keyword in instruction_lower for keyword in style_keywords):
        return 'style_transfer'
    elif any(keyword in instruction_lower for keyword in color_keywords):
        return 'color_change'
    elif len(instruction.split()) <= 5:
        return 'small_edit'
    else:
        return 'large_edit'

def load_from_datasets_library(dataset_path):
    """Load using HuggingFace datasets library."""
    try:
        from datasets import load_from_disk, load_dataset

        # Try loading from disk first
        try:
            dataset = load_from_disk(dataset_path)
            print(f"✓ Loaded dataset from disk: {len(dataset)} samples")
            return dataset
        except:
            pass

        # Try loading as HF dataset
        dataset = load_dataset(dataset_path, split='train')
        print(f"✓ Loaded HuggingFace dataset: {len(dataset)} samples")
        return dataset

    except Exception as e:
        print(f"✗ Error loading with datasets library: {e}")
        return None

def load_from_parquet(parquet_file):
    """Load from parquet file."""
    try:
        import pandas as pd
        df = pd.read_parquet(parquet_file)
        print(f"✓ Loaded parquet file: {len(df)} rows")
        return df.to_dict('records')
    except Exception as e:
        print(f"✗ Error loading parquet: {e}")
        return None

def select_diverse_samples(dataset, num_per_type=5):
    """Select diverse samples stratified by edit type."""
    samples_by_type = {
        'object_add': [],
        'object_remove': [],
        'color_change': [],
        'style_transfer': [],
        'small_edit': [],
        'large_edit': []
    }

    print("Classifying samples...")
    for idx, item in enumerate(dataset):
        if idx >= 1000:  # Limit scanning
            break

        # Get instruction
        instruction = item.get('instruction') or item.get('edit_prompt') or item.get('edit', '')
        if not instruction:
            continue

        edit_type = classify_edit_type(instruction)

        # Get images
        source_img = item.get('source_img') or item.get('input_image') or item.get('original_image')
        target_img = item.get('target_img') or item.get('edited_image') or item.get('output_image')

        if source_img is not None and target_img is not None:
            samples_by_type[edit_type].append({
                'idx': idx,
                'instruction': instruction,
                'source_img': source_img,
                'target_img': target_img
            })

    print("\nDistribution:")
    for edit_type, items in samples_by_type.items():
        print(f"  {edit_type}: {len(items)} samples")

    # Select samples
    selected = []
    for edit_type in samples_by_type:
        available = samples_by_type[edit_type]
        count = min(num_per_type, len(available))
        if count > 0:
            sampled = random.sample(available, count)
            selected.extend([(edit_type, s) for s in sampled])

    return selected

def save_samples(selected, output_dir):
    """Save selected samples to disk."""
    output_dir = Path(output_dir)
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    output_data = []
    success_count = 0

    print("\nProcessing samples...")
    for i, (edit_type, sample) in enumerate(selected):
        sample_id = f"{i+1:03d}"

        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        edited_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        try:
            # Save source image
            source_img = sample['source_img']
            if isinstance(source_img, Image.Image):
                source_img.save(source_path)
            elif hasattr(source_img, 'save'):
                source_img.save(source_path)
            else:
                print(f"  ✗ [{sample_id}] Unsupported source image type: {type(source_img)}")
                continue

            # Save target image
            target_img = sample['target_img']
            if isinstance(target_img, Image.Image):
                target_img.save(edited_path)
            elif hasattr(target_img, 'save'):
                target_img.save(edited_path)
            else:
                print(f"  ✗ [{sample_id}] Unsupported target image type: {type(target_img)}")
                continue

            entry = {
                "id": sample_id,
                "source_img": source_path,
                "edited_img": edited_path,
                "instruction": sample['instruction'],
                "edit_type": edit_type,
                "dataset_source": "local",
                "dataset_idx": sample['idx']
            }

            output_data.append(entry)
            success_count += 1
            print(f"  ✓ [{sample_id}] {edit_type}: {sample['instruction'][:50]}...")

        except Exception as e:
            print(f"  ✗ [{sample_id}] Error: {e}")
            continue

    # Save JSON
    if success_count > 0:
        output_file = output_dir / 'samples.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Successfully saved {success_count} samples to {output_file}")

        # Print summary
        type_counts = {}
        for entry in output_data:
            edit_type = entry['edit_type']
            type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

        print("\nFinal distribution:")
        for edit_type, count in sorted(type_counts.items()):
            print(f"  {edit_type}: {count} samples")

        return True
    else:
        print("\n❌ No samples were successfully saved")
        return False

def main():
    parser = argparse.ArgumentParser(description='Load samples from local dataset')
    parser.add_argument('--dataset_path', type=str, help='Path to local dataset directory')
    parser.add_argument('--parquet_file', type=str, help='Path to parquet file')
    parser.add_argument('--output_dir', type=str, default='evaluation_samples',
                       help='Output directory (default: evaluation_samples)')
    parser.add_argument('--num_per_type', type=int, default=5,
                       help='Number of samples per edit type (default: 5)')

    args = parser.parse_args()

    if not args.dataset_path and not args.parquet_file:
        print("Error: Please provide either --dataset_path or --parquet_file")
        parser.print_help()
        sys.exit(1)

    print("=" * 60)
    print("Loading Samples from Local Dataset")
    print("=" * 60)

    # Load dataset
    dataset = None
    if args.parquet_file:
        dataset = load_from_parquet(args.parquet_file)
    elif args.dataset_path:
        dataset = load_from_datasets_library(args.dataset_path)

    if dataset is None:
        print("\n❌ Failed to load dataset")
        sys.exit(1)

    # Select diverse samples
    selected = select_diverse_samples(dataset, args.num_per_type)

    if len(selected) < 10:
        print("\n❌ Not enough diverse samples found")
        sys.exit(1)

    # Save samples
    success = save_samples(selected, args.output_dir)

    if success:
        print("\n✅ Done! You can now use the evaluation scripts.")
    else:
        sys.exit(1)

if __name__ == '__main__':
    main()
