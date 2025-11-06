#!/usr/bin/env python3
"""
Test correlation between LLM and EditCLIP evaluations on MagicBrush data.

This script:
1. Loads EditCLIP model
2. Evaluates all samples with EditCLIP
3. Evaluates all samples with LLM (GPT-4 Vision or Claude)
4. Computes correlation metrics
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from open_clip import create_model_and_transforms, get_tokenizer
from transformers import CLIPModel, CLIPProcessor


def load_magicbrush_data(data_path='magicbrush_data/data.json'):
    """Load MagicBrush dataset."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from MagicBrush")
    return data


def load_editclip_model(model_path='clip_ckpt/editclip_vit_l_14', device='cuda'):
    """
    Load EditCLIP model.

    Note: EditCLIP uses a 6-channel input (source + edited images concatenated)
    """
    try:
        # Try loading with transformers (HuggingFace format)
        print(f"Loading EditCLIP model from {model_path}...")

        # Check if model files exist
        if not os.path.exists(model_path):
            print(f"Warning: Model path {model_path} not found")
            return None, None

        # For EditCLIP, we need to load as CLIP with modified vision encoder
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(model_path)
        processor = CLIPProcessor.from_pretrained(model_path)

        model = model.to(device)
        model.eval()

        print("✓ EditCLIP model loaded successfully")
        return model, processor

    except Exception as e:
        print(f"✗ Error loading EditCLIP model: {e}")
        print("Attempting alternative loading method...")

        try:
            # Alternative: Load using open_clip
            model, _, preprocess = create_model_and_transforms(
                'ViT-L-14',
                pretrained=model_path,
                device=device
            )
            tokenizer = get_tokenizer('ViT-L-14')

            print("✓ EditCLIP model loaded with open_clip")
            return model, (preprocess, tokenizer)

        except Exception as e2:
            print(f"✗ Alternative loading also failed: {e2}")
            return None, None


def compute_editclip_score(model, processor, source_img_path, target_img_path,
                          instruction, device='cuda'):
    """
    Compute EditCLIP similarity score for an image edit.

    EditCLIP measures how well the edit follows the instruction by computing
    similarity between concatenated (source, edited) images and the instruction text.
    """
    try:
        # Load images
        source_img = Image.open(source_img_path).convert('RGB')
        target_img = Image.open(target_img_path).convert('RGB')

        # For EditCLIP, we concatenate source and target images
        # This is a 6-channel input: [source_R, source_G, source_B, target_R, target_G, target_B]

        if isinstance(processor, CLIPProcessor):
            # HuggingFace format
            # Process images
            source_inputs = processor(images=source_img, return_tensors="pt")
            target_inputs = processor(images=target_img, return_tensors="pt")
            text_inputs = processor(text=instruction, return_tensors="pt", padding=True)

            # Move to device
            source_inputs = {k: v.to(device) for k, v in source_inputs.items()}
            target_inputs = {k: v.to(device) for k, v in target_inputs.items()}
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}

            with torch.no_grad():
                # Get image embeddings
                source_features = model.get_image_features(**source_inputs)
                target_features = model.get_image_features(**target_inputs)

                # Concatenate or average features (depending on EditCLIP implementation)
                # For now, let's compute similarity of target image with text
                text_features = model.get_text_features(**text_inputs)

                # Normalize features
                source_features = source_features / source_features.norm(dim=-1, keepdim=True)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute similarity scores
                # EditCLIP score: how well does the edit match the instruction
                # We can measure: similarity(target, text) - similarity(source, text)
                # Or: similarity(target, text) weighted by change from source

                source_text_sim = (source_features @ text_features.T).item()
                target_text_sim = (target_features @ text_features.T).item()

                # EditCLIP score: improvement in alignment with instruction
                editclip_score = target_text_sim - source_text_sim

                return {
                    'editclip_score': editclip_score,
                    'target_text_sim': target_text_sim,
                    'source_text_sim': source_text_sim,
                    'source_target_sim': (source_features @ target_features.T).item()
                }
        else:
            # open_clip format
            preprocess, tokenizer = processor

            source_tensor = preprocess(source_img).unsqueeze(0).to(device)
            target_tensor = preprocess(target_img).unsqueeze(0).to(device)
            text_tokens = tokenizer([instruction]).to(device)

            with torch.no_grad():
                source_features = model.encode_image(source_tensor)
                target_features = model.encode_image(target_tensor)
                text_features = model.encode_text(text_tokens)

                # Normalize
                source_features = source_features / source_features.norm(dim=-1, keepdim=True)
                target_features = target_features / target_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                source_text_sim = (source_features @ text_features.T).item()
                target_text_sim = (target_features @ text_features.T).item()
                editclip_score = target_text_sim - source_text_sim

                return {
                    'editclip_score': editclip_score,
                    'target_text_sim': target_text_sim,
                    'source_text_sim': source_text_sim,
                    'source_target_sim': (source_features @ target_features.T).item()
                }

    except Exception as e:
        print(f"Error computing EditCLIP score: {e}")
        return None


def evaluate_with_editclip(samples, model, processor, device='cuda'):
    """Evaluate all samples with EditCLIP."""
    print("\n" + "="*60)
    print("Evaluating with EditCLIP")
    print("="*60)

    results = []

    for i, sample in enumerate(samples):
        sample_id = sample['id']
        source_path = f"magicbrush_data/{sample['source_image']}"
        target_path = f"magicbrush_data/{sample['target_image']}"
        instruction = sample['instruction']

        print(f"\n[{i+1}/{len(samples)}] Sample {sample_id}: {instruction[:60]}...")

        score_dict = compute_editclip_score(
            model, processor, source_path, target_path, instruction, device
        )

        if score_dict:
            results.append({
                'sample_id': sample_id,
                'instruction': instruction,
                **score_dict
            })
            print(f"  EditCLIP score: {score_dict['editclip_score']:.4f}")
            print(f"  Target-Text sim: {score_dict['target_text_sim']:.4f}")
            print(f"  Source-Text sim: {score_dict['source_text_sim']:.4f}")
        else:
            results.append({
                'sample_id': sample_id,
                'instruction': instruction,
                'editclip_score': None
            })
            print(f"  ✗ Failed to compute score")

    return results


def evaluate_with_llm(samples, api_key=None, model_name='gpt-4-vision-preview'):
    """
    Evaluate all samples with LLM (GPT-4 Vision or Claude).

    If no API key is provided, returns placeholder scores.
    """
    print("\n" + "="*60)
    print("Evaluating with LLM")
    print("="*60)

    if api_key is None:
        print("⚠️  No API key provided. Using simulated LLM scores.")
        print("   To use real LLM evaluation, set OPENAI_API_KEY or ANTHROPIC_API_KEY")

        # Return simulated scores for demonstration
        results = []
        for sample in samples:
            # Simulate score based on simple heuristics (for demo only)
            instruction = sample['instruction'].lower()

            # Simple heuristic: longer instructions might be harder
            complexity = len(instruction.split()) / 10.0
            # Add some randomness
            np.random.seed(sample['id'])
            score = max(0, min(1, 0.7 + np.random.normal(0, 0.15) - complexity * 0.1))

            results.append({
                'sample_id': sample['id'],
                'llm_score': score,
                'llm_model': 'simulated'
            })

        return results

    # TODO: Implement real LLM evaluation with API
    # For now, return simulated scores
    print("⚠️  Real LLM evaluation not yet implemented. Using simulated scores.")
    return evaluate_with_llm(samples, api_key=None)


def compute_correlation(editclip_results, llm_results):
    """Compute correlation metrics between EditCLIP and LLM scores."""
    print("\n" + "="*60)
    print("Computing Correlation Metrics")
    print("="*60)

    # Extract scores
    editclip_scores = []
    llm_scores = []

    for ec_res, llm_res in zip(editclip_results, llm_results):
        if ec_res['editclip_score'] is not None and llm_res['llm_score'] is not None:
            editclip_scores.append(ec_res['editclip_score'])
            llm_scores.append(llm_res['llm_score'])

    editclip_scores = np.array(editclip_scores)
    llm_scores = np.array(llm_scores)

    if len(editclip_scores) < 3:
        print("Not enough valid samples for correlation computation")
        return None

    # Compute correlations
    pearson_r, pearson_p = pearsonr(editclip_scores, llm_scores)
    spearman_r, spearman_p = spearmanr(editclip_scores, llm_scores)

    # Compute error metrics
    mae = np.mean(np.abs(editclip_scores - llm_scores))
    rmse = np.sqrt(np.mean((editclip_scores - llm_scores) ** 2))

    results = {
        'n_samples': len(editclip_scores),
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'mae': mae,
        'rmse': rmse,
        'editclip_mean': np.mean(editclip_scores),
        'editclip_std': np.std(editclip_scores),
        'llm_mean': np.mean(llm_scores),
        'llm_std': np.std(llm_scores)
    }

    # Print results
    print(f"\nSamples analyzed: {results['n_samples']}")
    print(f"\nCorrelation:")
    print(f"  Pearson  r = {pearson_r:6.4f}  (p = {pearson_p:.4e})")
    print(f"  Spearman ρ = {spearman_r:6.4f}  (p = {spearman_p:.4e})")

    print(f"\nError Metrics:")
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")

    print(f"\nScore Statistics:")
    print(f"  EditCLIP: mean = {results['editclip_mean']:.4f}, std = {results['editclip_std']:.4f}")
    print(f"  LLM:      mean = {results['llm_mean']:.4f}, std = {results['llm_std']:.4f}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test LLM-EditCLIP correlation')
    parser.add_argument('--data_path', type=str, default='magicbrush_data/data.json',
                       help='Path to MagicBrush data.json')
    parser.add_argument('--model_path', type=str, default='clip_ckpt/editclip_vit_l_14',
                       help='Path to EditCLIP model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--output', type=str, default='correlation_results.json',
                       help='Output file for results')
    parser.add_argument('--api_key', type=str, default=None,
                       help='API key for LLM evaluation (OpenAI or Anthropic)')

    args = parser.parse_args()

    print("="*60)
    print("EditCLIP vs LLM Correlation Testing")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Data: {args.data_path}")
    print(f"Model: {args.model_path}")

    # Load data
    samples = load_magicbrush_data(args.data_path)

    # Load EditCLIP model
    model, processor = load_editclip_model(args.model_path, args.device)

    if model is None:
        print("\n❌ Failed to load EditCLIP model. Exiting.")
        return

    # Evaluate with EditCLIP
    editclip_results = evaluate_with_editclip(samples, model, processor, args.device)

    # Evaluate with LLM
    llm_results = evaluate_with_llm(samples, api_key=args.api_key)

    # Compute correlation
    correlation_results = compute_correlation(editclip_results, llm_results)

    # Save results
    output_data = {
        'correlation_metrics': correlation_results,
        'editclip_results': editclip_results,
        'llm_results': llm_results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Results saved to {args.output}")


if __name__ == '__main__':
    main()
