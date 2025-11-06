#!/usr/bin/env python3
"""
Download 50 diverse samples from MagicBrush dataset.

Run this on your LOCAL MACHINE or a server with HuggingFace access:
    pip install datasets pillow
    python3 download_magicbrush_50.py
"""

import json
import random
from pathlib import Path
from PIL import Image
from datasets import load_dataset

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

def select_diverse_samples(dataset, total_samples=50):
    """Select diverse samples from dataset."""
    samples_by_type = {
        'object_add': [],
        'object_remove': [],
        'color_change': [],
        'style_transfer': [],
        'small_edit': [],
        'large_edit': []
    }

    print(f"Scanning dataset (total: {len(dataset)} samples)...")

    # Scan through dataset
    for idx, item in enumerate(dataset):
        if idx >= 2000:  # Scan first 2000 for efficiency
            break

        # Get instruction
        instruction = item.get('instruction', '')
        if not instruction:
            continue

        # Get images
        source_img = item.get('source_img')
        target_img = item.get('target_img')

        if source_img is None or target_img is None:
            continue

        # Classify
        edit_type = classify_edit_type(instruction)

        samples_by_type[edit_type].append({
            'idx': idx,
            'instruction': instruction,
            'source_img': source_img,
            'target_img': target_img,
            'turn_index': item.get('turn_index', 0),
            'img_id': item.get('img_id', ''),
        })

        if idx % 100 == 0:
            print(f"  Processed {idx} samples...")

    # Print distribution
    print("\nDataset distribution:")
    for edit_type, items in samples_by_type.items():
        print(f"  {edit_type}: {len(items)} samples")

    # Calculate how many to sample from each type
    num_per_type = total_samples // 6  # 6 types
    remainder = total_samples % 6

    selected = []
    for i, edit_type in enumerate(samples_by_type.keys()):
        available = samples_by_type[edit_type]

        # Add one more to first few types to reach total
        count = num_per_type + (1 if i < remainder else 0)
        count = min(count, len(available))

        if count > 0:
            sampled = random.sample(available, count)
            selected.extend(sampled)

    return selected

def main():
    print("=" * 60)
    print("Downloading 50 Samples from MagicBrush Dataset")
    print("=" * 60)

    # Create output directory
    output_dir = Path('evaluation_samples')
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Load MagicBrush dataset
    print("\nLoading MagicBrush dataset...")
    print("(This may take a few minutes on first run)")

    try:
        dataset = load_dataset('osunlp/MagicBrush', split='train')
        print(f"✓ Loaded {len(dataset)} samples from MagicBrush")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nPlease ensure you have:")
        print("  1. Internet connection")
        print("  2. Installed: pip install datasets")
        print("  3. Enough disk space for dataset cache")
        return

    # Select diverse samples
    print("\nSelecting 50 diverse samples...")
    selected = select_diverse_samples(dataset, total_samples=50)

    print(f"\nSelected {len(selected)} samples")

    # Save samples
    output_data = []
    success_count = 0

    print("\nProcessing and saving images...")
    for i, sample in enumerate(selected):
        sample_id = f"{i+1:03d}"

        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        edited_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        try:
            # Get images
            source_img = sample['source_img']
            target_img = sample['target_img']

            # Convert to RGB if needed
            if isinstance(source_img, Image.Image):
                if source_img.mode != 'RGB':
                    source_img = source_img.convert('RGB')
                source_img.save(source_path, 'JPEG')
            else:
                raise ValueError(f"Unexpected image type: {type(source_img)}")

            if isinstance(target_img, Image.Image):
                if target_img.mode != 'RGB':
                    target_img = target_img.convert('RGB')
                target_img.save(edited_path, 'JPEG')
            else:
                raise ValueError(f"Unexpected image type: {type(target_img)}")

            # Classify edit type
            edit_type = classify_edit_type(sample['instruction'])

            # Create entry
            entry = {
                "id": sample_id,
                "source_img": source_path,
                "edited_img": edited_path,
                "instruction": sample['instruction'],
                "edit_type": edit_type,
                "dataset_source": "MagicBrush",
                "dataset_idx": sample['idx'],
                "turn_index": sample['turn_index'],
                "img_id": sample['img_id']
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
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Successfully saved {success_count} samples to {output_file}")

        # Print summary
        type_counts = {}
        for entry in output_data:
            edit_type = entry['edit_type']
            type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

        print("\nFinal distribution by edit type:")
        for edit_type, count in sorted(type_counts.items()):
            print(f"  {edit_type}: {count} samples")

        print("\n📦 To use these samples:")
        print("  1. Zip the evaluation_samples/ folder")
        print("  2. Upload to your server")
        print("  3. Run: python3 verify_samples.py")

    else:
        print("\n❌ No samples were successfully saved")

if __name__ == '__main__':
    main()
