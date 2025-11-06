#!/usr/bin/env python3
"""
Verify that all samples in evaluation_samples are correctly loaded.
"""

import json
from pathlib import Path
from PIL import Image

def verify_samples():
    """Verify all samples can be loaded."""
    samples_file = Path('evaluation_samples/samples.json')

    if not samples_file.exists():
        print("❌ samples.json not found")
        return False

    with open(samples_file, 'r') as f:
        samples = json.load(f)

    print(f"Found {len(samples)} samples in samples.json")
    print("\nVerifying images...")

    errors = []
    warnings = []

    for sample in samples:
        sample_id = sample['id']
        source_path = sample['source_img']
        edited_path = sample['edited_img']

        # Check source image
        try:
            source_img = Image.open(source_path)
            w, h = source_img.size

            # Check if it's a placeholder (small file size or single color)
            if w == 512 and h == 512:
                # Check if it's mostly one color (placeholder)
                colors = source_img.getcolors(maxcolors=10)
                if colors and len(colors) <= 5:
                    warnings.append(f"[{sample_id}] Source image might be a placeholder (limited colors)")

            print(f"✓ [{sample_id}] Source: {w}x{h} - {source_img.mode}")

        except FileNotFoundError:
            errors.append(f"[{sample_id}] Source image not found: {source_path}")
            print(f"✗ [{sample_id}] Source image not found")
        except Exception as e:
            errors.append(f"[{sample_id}] Error loading source: {e}")
            print(f"✗ [{sample_id}] Error loading source: {e}")

        # Check edited image
        try:
            edited_img = Image.open(edited_path)
            w, h = edited_img.size

            # Check if it's a placeholder
            if w == 512 and h == 512:
                colors = edited_img.getcolors(maxcolors=10)
                if colors and len(colors) <= 5:
                    warnings.append(f"[{sample_id}] Edited image might be a placeholder (limited colors)")

            print(f"✓ [{sample_id}] Edited: {w}x{h} - {edited_img.mode}")

        except FileNotFoundError:
            errors.append(f"[{sample_id}] Edited image not found: {edited_path}")
            print(f"✗ [{sample_id}] Edited image not found")
        except Exception as e:
            errors.append(f"[{sample_id}] Error loading edited: {e}")
            print(f"✗ [{sample_id}] Error loading edited: {e}")

    # Print summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    if errors:
        print(f"\n❌ Found {len(errors)} errors:")
        for error in errors:
            print(f"  {error}")

    if warnings:
        print(f"\n⚠️  Found {len(warnings)} warnings:")
        for warning in warnings:
            print(f"  {warning}")

    if not errors and not warnings:
        print("\n✅ All samples verified successfully!")
        return True
    elif not errors:
        print("\n⚠️  Verification completed with warnings (likely placeholder images)")
        return True
    else:
        print("\n❌ Verification failed")
        return False

    # Check distribution
    print("\n" + "=" * 60)
    print("Sample Distribution")
    print("=" * 60)

    type_counts = {}
    for sample in samples:
        edit_type = sample['edit_type']
        type_counts[edit_type] = type_counts.get(edit_type, 0) + 1

    for edit_type, count in sorted(type_counts.items()):
        print(f"  {edit_type}: {count} samples")

if __name__ == '__main__':
    verify_samples()
