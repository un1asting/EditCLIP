#!/usr/bin/env python3
"""
Download real samples from MagicBrush or InstructPix2Pix dataset using alternative methods.
"""

import os
import json
from pathlib import Path
from PIL import Image
import requests
from io import BytesIO
import time

def download_image_from_url(url, save_path):
    """Download image from URL."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def try_datasets_api():
    """Try using the HuggingFace datasets library with different configurations."""
    try:
        from datasets import load_dataset
        import os

        # Set environment variables for better connectivity
        os.environ['HF_DATASETS_OFFLINE'] = '0'

        print("Attempting to load MagicBrush dataset...")
        dataset = load_dataset('osunlp/MagicBrush', split='train[:50]')
        print(f"✓ Successfully loaded {len(dataset)} samples from MagicBrush")
        return dataset, 'magicbrush'
    except Exception as e:
        print(f"✗ MagicBrush failed: {e}\n")

    try:
        print("Attempting to load InstructPix2Pix dataset...")
        dataset = load_dataset('timbrooks/instructpix2pix-clip-filtered', split='train[:50]')
        print(f"✓ Successfully loaded {len(dataset)} samples from InstructPix2Pix")
        return dataset, 'instructpix2pix'
    except Exception as e:
        print(f"✗ InstructPix2Pix failed: {e}\n")

    try:
        print("Attempting to load smaller InstructPix2Pix dataset...")
        dataset = load_dataset('fusing/instructpix2pix-1000-samples', split='train[:50]')
        print(f"✓ Successfully loaded {len(dataset)} samples from InstructPix2Pix-1000")
        return dataset, 'instructpix2pix-small'
    except Exception as e:
        print(f"✗ InstructPix2Pix-1000 failed: {e}\n")

    return None, None

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

def select_diverse_samples(dataset, dataset_name, num_samples=30):
    """Select diverse samples from dataset."""
    samples_by_type = {
        'object_add': [],
        'object_remove': [],
        'color_change': [],
        'style_transfer': [],
        'small_edit': [],
        'large_edit': []
    }

    print(f"\nClassifying {len(dataset)} samples...")

    for idx, item in enumerate(dataset):
        # Get instruction field
        instruction = None
        if 'instruction' in item:
            instruction = item['instruction']
        elif 'edit_prompt' in item:
            instruction = item['edit_prompt']
        elif 'edit' in item:
            instruction = item['edit']

        if not instruction:
            continue

        edit_type = classify_edit_type(instruction)
        samples_by_type[edit_type].append((idx, item, instruction))

    print("\nDistribution:")
    for edit_type, items in samples_by_type.items():
        print(f"  {edit_type}: {len(items)} samples")

    # Select 5 from each category
    import random
    random.seed(42)

    selected = []
    for edit_type in samples_by_type:
        available = samples_by_type[edit_type]
        count = min(5, len(available))
        if count > 0:
            sampled = random.sample(available, count)
            selected.extend([(edit_type, idx, item, inst) for idx, item, inst in sampled])

    return selected

def main():
    # Create output directory
    output_dir = Path('evaluation_samples')
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    # Try to load dataset
    dataset, dataset_name = try_datasets_api()

    if dataset is None:
        print("\n❌ Could not load any dataset. Using demo samples instead.")
        print("   Please check your internet connection or try running:")
        print("   python3 create_demo_samples.py")
        return

    # Select diverse samples
    selected_samples = select_diverse_samples(dataset, dataset_name)

    print(f"\n📥 Processing {len(selected_samples)} samples...")

    output_data = []
    success_count = 0

    for i, (edit_type, idx, item, instruction) in enumerate(selected_samples):
        sample_id = f"{i+1:03d}"

        # Determine field names
        if dataset_name == 'magicbrush':
            source_field = 'source_img'
            target_field = 'target_img'
        else:
            source_field = 'input_image' if 'input_image' in item else 'original_image'
            target_field = 'edited_image' if 'edited_image' in item else 'output_image'

        source_img = item.get(source_field)
        target_img = item.get(target_field)

        if source_img is None or target_img is None:
            print(f"⚠ Skipping sample {sample_id}: missing images")
            continue

        # Save images
        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        target_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        try:
            # Save source image
            if isinstance(source_img, Image.Image):
                source_img.save(source_path)
            elif isinstance(source_img, str) and source_img.startswith('http'):
                download_image_from_url(source_img, source_path)
            else:
                source_img.save(source_path)

            # Save target image
            if isinstance(target_img, Image.Image):
                target_img.save(target_path)
            elif isinstance(target_img, str) and target_img.startswith('http'):
                download_image_from_url(target_img, target_path)
            else:
                target_img.save(target_path)

            entry = {
                "id": sample_id,
                "source_img": source_path,
                "edited_img": target_path,
                "instruction": instruction,
                "edit_type": edit_type,
                "dataset_source": dataset_name,
                "dataset_idx": idx
            }

            output_data.append(entry)
            success_count += 1
            print(f"✓ {sample_id}: {edit_type} - {instruction[:60]}...")

        except Exception as e:
            print(f"✗ Error processing sample {sample_id}: {e}")
            continue

        # Small delay to avoid rate limiting
        time.sleep(0.1)

    # Save to JSON
    output_file = output_dir / 'samples.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Successfully saved {success_count} samples to {output_file}")

    # Print summary
    type_counts = {}
    for entry in output_data:
        edit_type = entry['edit_type']
        type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

    print("\nFinal distribution by edit type:")
    for edit_type, count in sorted(type_counts.items()):
        print(f"  {edit_type}: {count} samples")

if __name__ == '__main__':
    main()
