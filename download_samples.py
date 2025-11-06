#!/usr/bin/env python3
"""
Download 30 diverse samples from MagicBrush and InstructPix2Pix datasets,
stratified by edit type for testing LLM and EditCLIP correlation.
"""

import os
import json
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO

# Set random seed for reproducibility
random.seed(42)

def download_and_save_image(image, save_path):
    """Download and save an image."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if isinstance(image, Image.Image):
        image.save(save_path)
    elif isinstance(image, str):
        # If it's a URL
        response = requests.get(image)
        img = Image.open(BytesIO(response.content))
        img.save(save_path)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    return save_path

def classify_edit_type(instruction):
    """
    Classify edit type based on instruction text.
    Categories: object_add, object_remove, color_change, style_transfer, small_edit, large_edit
    """
    instruction_lower = instruction.lower()

    # Object additions
    add_keywords = ['add', 'insert', 'place', 'put', 'draw', 'include']
    # Object removals
    remove_keywords = ['remove', 'delete', 'erase', 'take away', 'eliminate', 'get rid']
    # Color changes
    color_keywords = ['color', 'colour', 'red', 'blue', 'green', 'yellow', 'orange', 'purple',
                      'black', 'white', 'pink', 'brown', 'gray', 'grey']
    # Style transfers
    style_keywords = ['style', 'painting', 'watercolor', 'sketch', 'cartoon', 'artistic',
                      'oil painting', 'pencil', 'drawing']

    # Check for specific patterns
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

def select_stratified_samples(dataset, target_counts):
    """
    Select samples stratified by edit type.
    target_counts: dict mapping edit_type to number of samples needed
    """
    samples_by_type = {edit_type: [] for edit_type in target_counts.keys()}

    # Classify all samples
    print("Classifying dataset samples...")
    for idx, item in enumerate(dataset):
        if idx >= 5000:  # Limit scanning to first 5000 for efficiency
            break

        # Handle different dataset structures
        if 'instruction' in item:
            instruction = item['instruction']
        elif 'edit_prompt' in item:
            instruction = item['edit_prompt']
        else:
            continue

        edit_type = classify_edit_type(instruction)

        # Store the item with its index
        item_data = {
            'idx': idx,
            'instruction': instruction,
            'item': item
        }
        samples_by_type[edit_type].append(item_data)

    # Print distribution
    print("\nEdit type distribution:")
    for edit_type, items in samples_by_type.items():
        print(f"  {edit_type}: {len(items)} samples")

    # Select samples for each type
    selected_samples = []
    for edit_type, count in target_counts.items():
        available = samples_by_type[edit_type]
        if len(available) < count:
            print(f"Warning: Only {len(available)} samples available for {edit_type}, requested {count}")
            selected = available
        else:
            selected = random.sample(available, count)

        selected_samples.extend(selected)

    return selected_samples

def main():
    # Define target counts per edit type
    target_counts = {
        'object_add': 5,
        'object_remove': 5,
        'color_change': 5,
        'style_transfer': 5,
        'small_edit': 5,
        'large_edit': 5
    }

    # Create output directory
    output_dir = Path('evaluation_samples')
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MagicBrush dataset...")
    try:
        # Try MagicBrush first with streaming
        dataset = load_dataset('osunlp/MagicBrush', split='train', streaming=False, trust_remote_code=True)
        dataset_name = 'magicbrush'
        print(f"Loaded MagicBrush with {len(dataset)} samples")
    except Exception as e:
        print(f"Failed to load MagicBrush: {e}")
        print("\nTrying InstructPix2Pix dataset...")
        try:
            dataset = load_dataset('timbrooks/instructpix2pix-clip-filtered', split='train', streaming=False, trust_remote_code=True)
            dataset_name = 'instructpix2pix'
            print(f"Loaded InstructPix2Pix with {len(dataset)} samples")
        except Exception as e:
            print(f"Failed to load InstructPix2Pix: {e}")
            print("\nTrying smaller test dataset...")
            try:
                # Try a smaller, more accessible dataset
                dataset = load_dataset('fusing/instructpix2pix-1000-samples', split='train', trust_remote_code=True)
                dataset_name = 'instructpix2pix-small'
                print(f"Loaded InstructPix2Pix-1000 with {len(dataset)} samples")
            except Exception as e:
                print(f"Failed to load InstructPix2Pix-1000: {e}")
                return

    # Select stratified samples
    print("\nSelecting stratified samples...")
    selected_samples = select_stratified_samples(dataset, target_counts)

    print(f"\nSelected {len(selected_samples)} samples")

    # Process and save samples
    output_data = []

    for i, sample_data in enumerate(selected_samples):
        item = sample_data['item']
        instruction = sample_data['instruction']

        sample_id = f"{i+1:03d}"

        # Determine image column names based on dataset
        if dataset_name == 'magicbrush':
            source_img_key = 'source_img'
            target_img_key = 'target_img'
        else:  # instructpix2pix or instructpix2pix-small
            # Try different possible column names
            if 'input_image' in item:
                source_img_key = 'input_image'
            elif 'original_image' in item:
                source_img_key = 'original_image'
            else:
                source_img_key = 'image'

            if 'edited_image' in item:
                target_img_key = 'edited_image'
            elif 'output_image' in item:
                target_img_key = 'output_image'
            else:
                target_img_key = 'edited_image'

        # Get images
        source_img = item.get(source_img_key)
        edited_img = item.get(target_img_key)

        if source_img is None or edited_img is None:
            print(f"Warning: Missing images for sample {sample_id}, skipping")
            continue

        # Save images
        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        edited_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        try:
            download_and_save_image(source_img, source_path)
            download_and_save_image(edited_img, edited_path)

            # Classify edit type
            edit_type = classify_edit_type(instruction)

            # Create entry
            entry = {
                "id": sample_id,
                "source_img": source_path,
                "edited_img": edited_path,
                "instruction": instruction,
                "edit_type": edit_type,
                "dataset_source": dataset_name
            }

            output_data.append(entry)
            print(f"Processed sample {sample_id}: {edit_type}")

        except Exception as e:
            print(f"Error processing sample {sample_id}: {e}")
            continue

    # Save to JSON
    output_file = output_dir / 'samples.json'
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Successfully saved {len(output_data)} samples to {output_file}")

    # Print summary
    print("\nSummary by edit type:")
    type_counts = {}
    for entry in output_data:
        edit_type = entry['edit_type']
        type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

    for edit_type, count in sorted(type_counts.items()):
        print(f"  {edit_type}: {count} samples")

if __name__ == '__main__':
    main()
