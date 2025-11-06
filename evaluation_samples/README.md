# Evaluation Samples for LLM and EditCLIP Correlation Testing

This directory contains 30 diverse image editing samples stratified by edit type for testing the correlation between LLM and EditCLIP evaluations.

## Dataset Structure

- **samples.json**: Metadata for all 30 samples
- **images/**: Directory containing source and edited images (60 total files)

## Sample Distribution

The dataset is stratified across 6 edit types with 5 samples each:

| Edit Type | Count | Description |
|-----------|-------|-------------|
| object_add | 5 | Adding objects to scenes (e.g., "add a red apple on the table") |
| object_remove | 5 | Removing objects from scenes (e.g., "remove the car from the street") |
| color_change | 5 | Changing colors of objects (e.g., "change the car to red color") |
| style_transfer | 5 | Artistic style transformations (e.g., "make it look like a watercolor painting") |
| small_edit | 5 | Minor modifications (e.g., "brighten image", "blur background") |
| large_edit | 5 | Complex multi-element edits (e.g., "change daytime scene to nighttime") |

## JSON Format

Each sample in `samples.json` has the following structure:

```json
{
  "id": "001",
  "source_img": "evaluation_samples/images/001_source.jpg",
  "edited_img": "evaluation_samples/images/001_edited.jpg",
  "instruction": "add a red apple on the table",
  "edit_type": "object_add",
  "description": "Add a red apple to an empty table",
  "dataset_source": "demo"
}
```

## Usage

### Loading the Dataset

```python
import json
from PIL import Image

# Load metadata
with open('evaluation_samples/samples.json', 'r') as f:
    samples = json.load(f)

# Access a sample
sample = samples[0]
source_img = Image.open(sample['source_img'])
edited_img = Image.open(sample['edited_img'])
instruction = sample['instruction']
edit_type = sample['edit_type']
```

### For Correlation Testing

Use this dataset to:
1. Evaluate editing quality with EditCLIP
2. Evaluate editing quality with LLM (e.g., GPT-4V, Claude)
3. Compare the two evaluation scores
4. Calculate correlation metrics (Pearson, Spearman, etc.)

## Notes

- **Current Status**: Placeholder images have been created
- **Next Steps**: Replace placeholder images with real dataset images from MagicBrush or InstructPix2Pix
- **Alternative**: Use the provided scripts to download real images when dataset access is available

## Scripts

- `create_demo_samples.py`: Creates this demo dataset with placeholder images
- `download_real_samples.py`: Downloads real images from HuggingFace datasets (requires internet access)
- `download_samples.py`: Alternative dataset download script

## Dataset Sources

The instructions are curated to be representative of:
- **MagicBrush**: https://huggingface.co/datasets/osunlp/MagicBrush
- **InstructPix2Pix**: https://huggingface.co/datasets/timbrooks/instructpix2pix-clip-filtered
