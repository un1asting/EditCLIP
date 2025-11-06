#!/usr/bin/env python3
"""
Fetch real images from MagicBrush/InstructPix2Pix by directly accessing
HuggingFace dataset viewer API or parquet files.
"""

import os
import json
import random
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import time

random.seed(42)

def fetch_from_hf_dataset_viewer(dataset_name, split='train', limit=100):
    """
    Fetch data from HuggingFace dataset viewer API.
    """
    base_url = f"https://datasets-server.huggingface.co/rows"
    params = {
        'dataset': dataset_name,
        'config': 'default',
        'split': split,
        'offset': 0,
        'length': limit
    }

    try:
        print(f"Fetching from HuggingFace API: {dataset_name}...")
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if 'rows' in data:
            print(f"✓ Found {len(data['rows'])} rows")
            return data['rows'], dataset_name.split('/')[-1]
        else:
            print(f"✗ No rows found in response")
            return None, None

    except Exception as e:
        print(f"✗ Error fetching {dataset_name}: {e}")
        return None, None

def download_image_from_url(url, save_path):
    """Download and save image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))

        # Convert to RGB if necessary
        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path, 'JPEG')
        return True
    except Exception as e:
        print(f"  ✗ Error downloading image: {e}")
        return False

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

def process_magicbrush_row(row):
    """Process a MagicBrush dataset row."""
    try:
        row_data = row.get('row', row)

        instruction = row_data.get('instruction', '')
        source_img_url = row_data.get('source_img', {}).get('src') if isinstance(row_data.get('source_img'), dict) else None
        target_img_url = row_data.get('target_img', {}).get('src') if isinstance(row_data.get('target_img'), dict) else None

        if source_img_url and target_img_url and instruction:
            return {
                'instruction': instruction,
                'source_url': source_img_url,
                'target_url': target_img_url,
                'edit_type': classify_edit_type(instruction)
            }
    except Exception as e:
        pass
    return None

def process_instructpix2pix_row(row):
    """Process an InstructPix2Pix dataset row."""
    try:
        row_data = row.get('row', row)

        instruction = row_data.get('edit_prompt', row_data.get('instruction', ''))

        # Try different possible field names
        source_img = row_data.get('input_image', row_data.get('original_image'))
        target_img = row_data.get('edited_image', row_data.get('output_image'))

        source_url = source_img.get('src') if isinstance(source_img, dict) else None
        target_url = target_img.get('src') if isinstance(target_img, dict) else None

        if source_url and target_url and instruction:
            return {
                'instruction': instruction,
                'source_url': source_url,
                'target_url': target_url,
                'edit_type': classify_edit_type(instruction)
            }
    except Exception as e:
        pass
    return None

def select_diverse_samples(processed_rows, num_per_type=5):
    """Select diverse samples stratified by edit type."""
    samples_by_type = {
        'object_add': [],
        'object_remove': [],
        'color_change': [],
        'style_transfer': [],
        'small_edit': [],
        'large_edit': []
    }

    for item in processed_rows:
        if item:
            edit_type = item['edit_type']
            samples_by_type[edit_type].append(item)

    print("\nDistribution:")
    for edit_type, items in samples_by_type.items():
        print(f"  {edit_type}: {len(items)} samples")

    selected = []
    for edit_type in samples_by_type:
        available = samples_by_type[edit_type]
        count = min(num_per_type, len(available))
        if count > 0:
            sampled = random.sample(available, count)
            selected.extend(sampled)

    return selected

def main():
    output_dir = Path('evaluation_samples')
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading Real Images from HuggingFace Datasets")
    print("=" * 60)

    # Try MagicBrush first
    rows, dataset_name = fetch_from_hf_dataset_viewer('osunlp/MagicBrush', limit=200)

    if rows is None:
        # Try InstructPix2Pix
        print("\nTrying InstructPix2Pix dataset...")
        rows, dataset_name = fetch_from_hf_dataset_viewer('timbrooks/instructpix2pix-clip-filtered', limit=200)

    if rows is None:
        # Try smaller dataset
        print("\nTrying smaller InstructPix2Pix dataset...")
        rows, dataset_name = fetch_from_hf_dataset_viewer('fusing/instructpix2pix-1000-samples', limit=200)

    if rows is None:
        print("\n❌ Could not fetch data from any dataset.")
        print("Please check internet connection or try manual download.")
        return

    # Process rows
    print(f"\nProcessing {len(rows)} rows from {dataset_name}...")
    processed = []

    for row in rows:
        if 'magicbrush' in dataset_name.lower():
            item = process_magicbrush_row(row)
        else:
            item = process_instructpix2pix_row(row)

        if item:
            processed.append(item)

    print(f"Successfully processed {len(processed)} samples")

    if len(processed) < 10:
        print("❌ Not enough samples to create diverse dataset")
        return

    # Select diverse samples
    selected = select_diverse_samples(processed, num_per_type=5)
    print(f"\nSelected {len(selected)} diverse samples")

    # Download images
    output_data = []
    success_count = 0

    print("\nDownloading images...")
    for i, sample in enumerate(selected):
        sample_id = f"{i+1:03d}"

        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        edited_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        print(f"\n[{sample_id}] {sample['edit_type']}: {sample['instruction'][:60]}...")

        # Download source image
        if download_image_from_url(sample['source_url'], source_path):
            print(f"  ✓ Downloaded source image")
        else:
            print(f"  ✗ Failed to download source image")
            continue

        # Download edited image
        if download_image_from_url(sample['target_url'], edited_path):
            print(f"  ✓ Downloaded edited image")
        else:
            print(f"  ✗ Failed to download edited image")
            continue

        entry = {
            "id": sample_id,
            "source_img": source_path,
            "edited_img": edited_path,
            "instruction": sample['instruction'],
            "edit_type": sample['edit_type'],
            "dataset_source": dataset_name
        }

        output_data.append(entry)
        success_count += 1

        time.sleep(0.2)  # Rate limiting

    # Save to JSON
    if success_count > 0:
        output_file = output_dir / 'samples.json'
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Successfully downloaded {success_count} samples to {output_file}")

        # Print summary
        type_counts = {}
        for entry in output_data:
            edit_type = entry['edit_type']
            type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

        print("\nFinal distribution by edit type:")
        for edit_type, count in sorted(type_counts.items()):
            print(f"  {edit_type}: {count} samples")
    else:
        print("\n❌ No samples were successfully downloaded")

if __name__ == '__main__':
    main()
