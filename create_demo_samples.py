#!/usr/bin/env python3
"""
Create a demo dataset with 30 diverse samples for testing LLM and EditCLIP correlation.
Since we cannot access HuggingFace datasets, we'll create placeholder entries with realistic
instructions that can be filled in with actual images later.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# Set random seed for reproducibility
random.seed(42)

def create_placeholder_image(width, height, text, color='gray'):
    """Create a placeholder image with text."""
    img = Image.new('RGB', (width, height), color=color)
    draw = ImageDraw.Draw(img)

    # Calculate text position (center)
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2

    # Draw text
    draw.text((x, y), text, fill='white')

    return img

# Curated realistic editing instructions stratified by type
SAMPLE_DATA = [
    # Object additions (5 samples)
    {
        "instruction": "add a red apple on the table",
        "edit_type": "object_add",
        "description": "Add a red apple to an empty table"
    },
    {
        "instruction": "place a white cat next to the sofa",
        "edit_type": "object_add",
        "description": "Add a cat in a living room scene"
    },
    {
        "instruction": "insert a blue umbrella in the person's hand",
        "edit_type": "object_add",
        "description": "Add an umbrella to a person portrait"
    },
    {
        "instruction": "put a vase with flowers on the windowsill",
        "edit_type": "object_add",
        "description": "Add flowers to interior scene"
    },
    {
        "instruction": "add birds flying in the sky",
        "edit_type": "object_add",
        "description": "Add birds to outdoor landscape"
    },

    # Object removals (5 samples)
    {
        "instruction": "remove the car from the street",
        "edit_type": "object_remove",
        "description": "Remove car from street scene"
    },
    {
        "instruction": "delete the person from the photo",
        "edit_type": "object_remove",
        "description": "Remove person from image"
    },
    {
        "instruction": "erase the text from the sign",
        "edit_type": "object_remove",
        "description": "Clean text from signage"
    },
    {
        "instruction": "take away the lamp from the desk",
        "edit_type": "object_remove",
        "description": "Remove object from furniture"
    },
    {
        "instruction": "remove the clouds from the sky",
        "edit_type": "object_remove",
        "description": "Clear sky in landscape"
    },

    # Color changes (5 samples)
    {
        "instruction": "change the car to red color",
        "edit_type": "color_change",
        "description": "Change vehicle color"
    },
    {
        "instruction": "make the walls blue",
        "edit_type": "color_change",
        "description": "Recolor interior walls"
    },
    {
        "instruction": "turn the dress from white to black",
        "edit_type": "color_change",
        "description": "Change clothing color"
    },
    {
        "instruction": "make the grass more green",
        "edit_type": "color_change",
        "description": "Enhance grass color"
    },
    {
        "instruction": "change the sky from blue to orange sunset",
        "edit_type": "color_change",
        "description": "Change sky color/lighting"
    },

    # Style transfers (5 samples)
    {
        "instruction": "make it look like a watercolor painting",
        "edit_type": "style_transfer",
        "description": "Apply watercolor style"
    },
    {
        "instruction": "convert to pencil sketch style",
        "edit_type": "style_transfer",
        "description": "Apply sketch style"
    },
    {
        "instruction": "make it look like a van gogh painting",
        "edit_type": "style_transfer",
        "description": "Apply Van Gogh artistic style"
    },
    {
        "instruction": "turn into cartoon style",
        "edit_type": "style_transfer",
        "description": "Apply cartoon/animation style"
    },
    {
        "instruction": "make it look like an oil painting",
        "edit_type": "style_transfer",
        "description": "Apply oil painting style"
    },

    # Small edits (5 samples)
    {
        "instruction": "brighten image",
        "edit_type": "small_edit",
        "description": "Increase brightness"
    },
    {
        "instruction": "add smile",
        "edit_type": "small_edit",
        "description": "Make person smile"
    },
    {
        "instruction": "blur background",
        "edit_type": "small_edit",
        "description": "Apply background blur"
    },
    {
        "instruction": "open eyes",
        "edit_type": "small_edit",
        "description": "Open closed eyes"
    },
    {
        "instruction": "rotate left",
        "edit_type": "small_edit",
        "description": "Rotate image orientation"
    },

    # Large/complex edits (5 samples)
    {
        "instruction": "change the daytime scene to nighttime with street lights on",
        "edit_type": "large_edit",
        "description": "Transform day to night with lighting changes"
    },
    {
        "instruction": "replace the summer trees with autumn colored leaves and add fallen leaves on ground",
        "edit_type": "large_edit",
        "description": "Seasonal transformation with multiple elements"
    },
    {
        "instruction": "transform the empty room into a cozy living space with furniture and decorations",
        "edit_type": "large_edit",
        "description": "Complete room transformation"
    },
    {
        "instruction": "change the modern building facade to a classical architectural style",
        "edit_type": "large_edit",
        "description": "Architectural style transformation"
    },
    {
        "instruction": "convert the indoor scene to an outdoor garden setting with natural lighting",
        "edit_type": "large_edit",
        "description": "Complete scene/environment transformation"
    }
]

def main():
    # Create output directory
    output_dir = Path('evaluation_samples')
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)

    print("Creating demo dataset with 30 diverse samples...")

    output_data = []

    for i, sample in enumerate(SAMPLE_DATA):
        sample_id = f"{i+1:03d}"

        # Create placeholder images
        source_path = f"evaluation_samples/images/{sample_id}_source.jpg"
        edited_path = f"evaluation_samples/images/{sample_id}_edited.jpg"

        # Create simple placeholder images
        source_img = create_placeholder_image(512, 512, f"Source #{sample_id}", 'lightgray')
        edited_img = create_placeholder_image(512, 512, f"Edited #{sample_id}", 'lightblue')

        source_img.save(source_path)
        edited_img.save(edited_path)

        # Create entry
        entry = {
            "id": sample_id,
            "source_img": source_path,
            "edited_img": edited_path,
            "instruction": sample["instruction"],
            "edit_type": sample["edit_type"],
            "description": sample["description"],
            "dataset_source": "demo"
        }

        output_data.append(entry)
        print(f"✓ Created sample {sample_id}: {sample['edit_type']} - {sample['instruction']}")

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

    print("\n📝 Note: Placeholder images have been created.")
    print("   To use real images, replace the placeholder images in evaluation_samples/images/")
    print("   Or provide actual dataset images and re-run with dataset access.")

if __name__ == '__main__':
    main()
