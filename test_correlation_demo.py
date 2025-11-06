#!/usr/bin/env python3
"""
Demo: Test correlation between CLIP-based and LLM evaluations on MagicBrush data.

This is a simplified demo using standard CLIP model to demonstrate the correlation
testing workflow. For full EditCLIP evaluation, download the model weights from:
https://huggingface.co/QWW/EditCLIP

The demo computes:
1. CLIP-based edit quality scores (using change in text-image alignment)
2. Simulated LLM scores (or real if API key provided)
3. Correlation metrics between the two evaluation methods
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm


def load_magicbrush_data(data_path='magicbrush_data/data.json'):
    """Load MagicBrush dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Loaded {len(data)} samples from MagicBrush")
    return data


def load_clip_model(device='cuda'):
    """Load standard CLIP model for demo."""
    try:
        print("Loading CLIP model (openai/clip-vit-large-patch14)...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        model = model.to(device)
        model.eval()

        print("✓ CLIP model loaded successfully")
        return model, processor

    except Exception as e:
        print(f"✗ Error loading CLIP model: {e}")
        return None, None


def compute_clip_edit_score(model, processor, source_img_path, target_img_path,
                            instruction, device='cuda'):
    """
    Compute CLIP-based edit quality score.

    Measures how well the edit improves alignment with the instruction text.
    Score = similarity(target, instruction) - similarity(source, instruction)

    Higher scores indicate better edits that more closely follow the instruction.
    """
    try:
        # Load images
        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')

        # Process inputs
        source_inputs = processor(images=source_img, return_tensors="pt", padding=True)
        target_inputs = processor(images=target_img, return_tensors="pt", padding=True)
        text_inputs = processor(text=[instruction], return_tensors="pt", padding=True)

        # Move to device
        source_inputs = {k: v.to(device) for k, v in source_inputs.items()}
        target_inputs = {k: v.to(device) for k, v in target_inputs.items()}
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

        with torch.no_grad():
            # Get embeddings
            source_features = model.get_image_features(**source_inputs)
            target_features = model.get_image_features(**target_inputs)
            text_features = model.get_text_features(**text_inputs)

            # Normalize
            source_features = source_features / source_features.norm(dim=-1, keepdim=True)
            target_features = target_features / target_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Compute similarities
            source_text_sim = (source_features @ text_features.T).squeeze().item()
            target_text_sim = (target_features @ text_features.T).squeeze().item()
            source_target_sim = (source_features @ target_features.T).squeeze().item()

            # Edit quality score: improvement in alignment
            edit_score = target_text_sim - source_text_sim

            return {
                'edit_score': edit_score,
                'target_text_sim': target_text_sim,
                'source_text_sim': source_text_sim,
                'source_target_sim': source_target_sim,
                'consistency': source_target_sim  # How much the image changed
            }

    except Exception as e:
        print(f"Error computing CLIP score: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_with_clip(samples, model, processor, device='cuda'):
    """Evaluate all samples with CLIP."""
    print("\n" + "="*60)
    print("Evaluating with CLIP")
    print("="*60)

    results = []

    for sample in tqdm(samples, desc="CLIP Evaluation"):
        sample_id = sample['id']
        source_path = f"magicbrush_data/{sample['source_image']}"
        target_path = f"magicbrush_data/{sample['target_image']}"
        instruction = sample['instruction']

        score_dict = compute_clip_edit_score(
            model, processor, source_path, target_path, instruction, device
        )

        if score_dict:
            results.append({
                'sample_id': sample_id,
                'instruction': instruction,
                **score_dict
            })
        else:
            results.append({
                'sample_id': sample_id,
                'instruction': instruction,
                'edit_score': None
            })

    return results


def evaluate_with_llm_simulated(samples):
    """
    Simulated LLM evaluation for demonstration.

    In a real scenario, this would call GPT-4 Vision or Claude API to evaluate
    each image edit and return a quality score.
    """
    print("\n" + "="*60)
    print("Generating Simulated LLM Scores")
    print("="*60)
    print("⚠️  Using simulated LLM scores for demonstration")
    print("   For real evaluation, implement LLM API calls")

    results = []

    for sample in samples:
        sample_id = sample['id']
        instruction = sample['instruction'].lower()

        # Simulate score based on heuristics
        # In reality, this would be an actual LLM evaluation
        np.random.seed(sample_id)

        # Factors that might affect quality (for simulation only):
        # 1. Instruction complexity
        complexity_penalty = len(instruction.split()) * 0.02

        # 2. Instruction type
        type_bonus = 0.0
        if any(word in instruction for word in ['change', 'replace', 'make']):
            type_bonus = 0.1
        elif any(word in instruction for word in ['remove', 'delete']):
            type_bonus = 0.05

        # Base score with some randomness
        base_score = 0.65 + np.random.normal(0, 0.15)
        simulated_score = base_score + type_bonus - complexity_penalty

        # Clip to [0, 1]
        simulated_score = max(0.0, min(1.0, simulated_score))

        results.append({
            'sample_id': sample_id,
            'llm_score': simulated_score,
            'llm_model': 'simulated',
            'instruction': instruction
        })

    return results


def compute_correlation(clip_results, llm_results):
    """Compute correlation metrics between CLIP and LLM scores."""
    print("\n" + "="*60)
    print("Computing Correlation Metrics")
    print("="*60)

    # Extract valid scores
    clip_scores = []
    llm_scores = []
    sample_ids = []

    for clip_res, llm_res in zip(clip_results, llm_results):
        if (clip_res['edit_score'] is not None and
            llm_res['llm_score'] is not None and
            not np.isnan(clip_res['edit_score'])):

            clip_scores.append(clip_res['edit_score'])
            llm_scores.append(llm_res['llm_score'])
            sample_ids.append(clip_res['sample_id'])

    clip_scores = np.array(clip_scores)
    llm_scores = np.array(llm_scores)

    print(f"\nValid samples: {len(clip_scores)} / {len(clip_results)}")

    if len(clip_scores) < 3:
        print("❌ Not enough valid samples for correlation computation")
        return None

    # Compute correlations
    pearson_r, pearson_p = pearsonr(clip_scores, llm_scores)
    spearman_r, spearman_p = spearmanr(clip_scores, llm_scores)

    # Normalize CLIP scores to [0, 1] for error metrics
    clip_scores_norm = (clip_scores - clip_scores.min()) / (clip_scores.max() - clip_scores.min() + 1e-8)

    # Compute error metrics
    mae = np.mean(np.abs(clip_scores_norm - llm_scores))
    rmse = np.sqrt(np.mean((clip_scores_norm - llm_scores) ** 2))

    results = {
        'n_samples': len(clip_scores),
        'pearson_r': float(pearson_r),
        'pearson_p': float(pearson_p),
        'spearman_r': float(spearman_r),
        'spearman_p': float(spearman_p),
        'mae': float(mae),
        'rmse': float(rmse),
        'clip_mean': float(np.mean(clip_scores)),
        'clip_std': float(np.std(clip_scores)),
        'clip_min': float(np.min(clip_scores)),
        'clip_max': float(np.max(clip_scores)),
        'llm_mean': float(np.mean(llm_scores)),
        'llm_std': float(np.std(llm_scores)),
        'llm_min': float(np.min(llm_scores)),
        'llm_max': float(np.max(llm_scores))
    }

    # Print results
    print(f"\n📊 Correlation Results:")
    print(f"  Pearson  r = {pearson_r:7.4f}  (p = {pearson_p:.4e})")
    print(f"  Spearman ρ = {spearman_r:7.4f}  (p = {spearman_p:.4e})")

    print(f"\n📏 Error Metrics (normalized scores):")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    print(f"\n📈 Score Statistics:")
    print(f"  CLIP:  mean={results['clip_mean']:6.4f}, std={results['clip_std']:.4f}, range=[{results['clip_min']:.3f}, {results['clip_max']:.3f}]")
    print(f"  LLM:   mean={results['llm_mean']:6.4f}, std={results['llm_std']:.4f}, range=[{results['llm_min']:.3f}, {results['llm_max']:.3f}]")

    # Interpretation
    print(f"\n💡 Interpretation:")
    if abs(pearson_r) > 0.7:
        print(f"  Strong {'positive' if pearson_r > 0 else 'negative'} linear correlation")
    elif abs(pearson_r) > 0.4:
        print(f"  Moderate {'positive' if pearson_r > 0 else 'negative'} linear correlation")
    else:
        print(f"  Weak correlation")

    if pearson_p < 0.05:
        print(f"  Correlation is statistically significant (p < 0.05)")
    else:
        print(f"  Correlation is not statistically significant (p >= 0.05)")

    return results


def visualize_top_bottom_samples(clip_results, llm_results, n=5):
    """Show top and bottom samples by score agreement."""
    print("\n" + "="*60)
    print(f"Top {n} and Bottom {n} Samples by Agreement")
    print("="*60)

    # Compute agreement (difference between normalized scores)
    agreements = []

    for clip_res, llm_res in zip(clip_results, llm_results):
        if clip_res['edit_score'] is not None and llm_res['llm_score'] is not None:
            # Normalize CLIP score to [0, 1] roughly
            clip_score_norm = clip_res['edit_score']

            agreement = abs(clip_score_norm - llm_res['llm_score'])

            agreements.append({
                'sample_id': clip_res['sample_id'],
                'instruction': clip_res['instruction'],
                'clip_score': clip_res['edit_score'],
                'llm_score': llm_res['llm_score'],
                'agreement': agreement
            })

    # Sort by agreement
    agreements.sort(key=lambda x: x['agreement'])

    print(f"\n✅ Top {n} samples with best agreement (most consistent):")
    for i, item in enumerate(agreements[:n]):
        print(f"\n{i+1}. Sample {item['sample_id']}")
        print(f"   Instruction: {item['instruction']}")
        print(f"   CLIP:  {item['clip_score']:6.4f}")
        print(f"   LLM:   {item['llm_score']:6.4f}")
        print(f"   Diff:  {item['agreement']:.4f}")

    print(f"\n❌ Bottom {n} samples with worst agreement (most inconsistent):")
    for i, item in enumerate(agreements[-n:]):
        print(f"\n{i+1}. Sample {item['sample_id']}")
        print(f"   Instruction: {item['instruction']}")
        print(f"   CLIP:  {item['clip_score']:6.4f}")
        print(f"   LLM:   {item['llm_score']:6.4f}")
        print(f"   Diff:  {item['agreement']:.4f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test CLIP-LLM correlation on MagicBrush')
    parser.add_argument('--data_path', type=str, default='magicbrush_data/data.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', type=str, default='correlation_results.json')

    args = parser.parse_args()

    print("="*60)
    print("CLIP vs LLM Correlation Testing (Demo)")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Data: {args.data_path}\n")

    # Check if data exists
    if not os.path.exists(args.data_path):
        print(f"❌ Data file not found: {args.data_path}")
        return

    # Load data
    samples = load_magicbrush_data(args.data_path)

    # Load CLIP model
    model, processor = load_clip_model(args.device)
    if model is None:
        print("❌ Failed to load CLIP model. Exiting.")
        return

    # Evaluate with CLIP
    clip_results = evaluate_with_clip(samples, model, processor, args.device)

    # Evaluate with LLM (simulated)
    llm_results = evaluate_with_llm_simulated(samples)

    # Compute correlation
    correlation_results = compute_correlation(clip_results, llm_results)

    if correlation_results:
        # Show examples
        visualize_top_bottom_samples(clip_results, llm_results, n=3)

        # Save results
        output_data = {
            'metadata': {
                'n_samples': len(samples),
                'model_type': 'CLIP (openai/clip-vit-large-patch14)',
                'llm_type': 'simulated',
                'device': args.device
            },
            'correlation_metrics': correlation_results,
            'clip_results': clip_results,
            'llm_results': llm_results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\n✅ Results saved to {args.output}")
        print(f"\n📝 Note: This demo uses standard CLIP, not EditCLIP.")
        print(f"   For full EditCLIP evaluation, download model weights from:")
        print(f"   https://huggingface.co/QWW/EditCLIP")


if __name__ == '__main__':
    main()
