#!/usr/bin/env python3
"""
Quick analysis of MagicBrush data - no heavy dependencies required.
Shows data structure and prepares for correlation testing.
"""

import json
from pathlib import Path

def analyze_magicbrush_data():
    """Analyze the MagicBrush dataset structure."""

    print("="*60)
    print("MagicBrush Data Analysis")
    print("="*60)

    # Load data
    with open('magicbrush_data/data.json', 'r') as f:
        data = json.load(f)

    print(f"\n✓ Total samples: {len(data)}")

    # Analyze instructions
    instructions = [s['instruction'] for s in data]
    avg_length = sum(len(inst.split()) for inst in instructions) / len(instructions)

    print(f"\n📝 Instruction Statistics:")
    print(f"  Average length: {avg_length:.1f} words")
    print(f"  Shortest: {min(len(inst.split()) for inst in instructions)} words")
    print(f"  Longest: {max(len(inst.split()) for inst in instructions)} words")

    # Check images
    print(f"\n🖼️  Image Files:")
    image_dir = Path('magicbrush_data/images')
    source_images = list(image_dir.glob('*_source.jpg'))
    target_images = list(image_dir.glob('*_target.jpg'))

    print(f"  Source images: {len(source_images)}")
    print(f"  Target images: {len(target_images)}")
    print(f"  All pairs present: {'✓' if len(source_images) == len(target_images) == len(data) else '✗'}")

    # Show some examples
    print(f"\n📋 Sample Instructions:")
    for i, sample in enumerate(data[:5]):
        print(f"  {i+1}. [{sample['id']:02d}] {sample['instruction']}")

    print(f"\n💡 Ready for correlation testing!")
    print(f"   Use: python3 test_correlation_demo.py")
    print(f"   (Requires: torch, transformers, scipy)")

if __name__ == '__main__':
    analyze_magicbrush_data()
