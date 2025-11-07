"""
Compare EditCLIP and LLM evaluation results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import os

def load_results():
    """Load both evaluation results"""
    with open('editclip_results.json', 'r', encoding='utf-8') as f:
        editclip_data = json.load(f)

    with open('llm_results.json', 'r', encoding='utf-8') as f:
        llm_data = json.load(f)

    return editclip_data, llm_data

def normalize_scores(editclip_scores, llm_scores):
    """
    Normalize scores to 0-1 range for fair comparison
    EditCLIP scores are already in 0-1 range
    LLM scores are in 0-10 range
    """
    editclip_normalized = np.array(editclip_scores)
    llm_normalized = np.array(llm_scores) / 10.0
    return editclip_normalized, llm_normalized

def analyze_correlation(editclip_data, llm_data):
    """Analyze correlation between two evaluation methods"""

    # Match samples by id
    editclip_dict = {item['id']: item for item in editclip_data['results']}
    llm_dict = {item['id']: item for item in llm_data['results']}

    # Get common samples
    common_ids = sorted(set(editclip_dict.keys()) & set(llm_dict.keys()))

    editclip_scores = [editclip_dict[id]['editclip_score'] for id in common_ids]
    llm_scores = [llm_dict[id]['llm_score'] for id in common_ids]

    # Normalize scores
    editclip_norm, llm_norm = normalize_scores(editclip_scores, llm_scores)

    # Calculate correlations
    pearson_corr, pearson_p = pearsonr(editclip_norm, llm_norm)
    spearman_corr, spearman_p = spearmanr(editclip_norm, llm_norm)

    return {
        'common_ids': common_ids,
        'editclip_scores': editclip_scores,
        'llm_scores': llm_scores,
        'editclip_normalized': editclip_norm,
        'llm_normalized': llm_norm,
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p
    }

def find_disagreements(editclip_data, llm_data, threshold=0.3):
    """Find cases where two methods disagree significantly"""

    editclip_dict = {item['id']: item for item in editclip_data['results']}
    llm_dict = {item['id']: item for item in llm_data['results']}

    common_ids = sorted(set(editclip_dict.keys()) & set(llm_dict.keys()))

    disagreements = []

    for id in common_ids:
        editclip_score = editclip_dict[id]['editclip_score']
        llm_score = llm_dict[id]['llm_score'] / 10.0  # Normalize to 0-1

        diff = abs(editclip_score - llm_score)

        if diff > threshold:
            disagreements.append({
                'id': id,
                'instruction': editclip_dict[id]['instruction'],
                'editclip_score': editclip_score,
                'llm_score': llm_score,
                'difference': diff,
                'llm_reasoning': llm_dict[id].get('reasoning', 'N/A')
            })

    # Sort by difference
    disagreements.sort(key=lambda x: x['difference'], reverse=True)

    return disagreements

def generate_report(editclip_data, llm_data):
    """Generate comprehensive comparison report"""

    print("="*80)
    print("EditCLIP vs LLM Evaluation Comparison Report")
    print("="*80)

    # Basic statistics
    print("\n1. Basic Statistics")
    print("-" * 80)
    print(f"{'Metric':<30} {'EditCLIP':<25} {'LLM':<25}")
    print("-" * 80)
    print(f"{'Model':<30} {editclip_data['statistics']['model']:<25} {llm_data['statistics']['model']:<25}")
    print(f"{'Number of Samples':<30} {editclip_data['statistics']['num_samples']:<25} {llm_data['statistics']['num_samples']:<25}")
    print(f"{'Mean Score':<30} {editclip_data['statistics']['mean_score']:.4f}{'':<20} {llm_data['statistics']['mean_score']:.4f}{'':<20}")
    print(f"{'Std Score':<30} {editclip_data['statistics']['std_score']:.4f}{'':<20} {llm_data['statistics']['std_score']:.4f}{'':<20}")
    print(f"{'Min Score':<30} {editclip_data['statistics']['min_score']:.4f}{'':<20} {llm_data['statistics']['min_score']:.4f}{'':<20}")
    print(f"{'Max Score':<30} {editclip_data['statistics']['max_score']:.4f}{'':<20} {llm_data['statistics']['max_score']:.4f}{'':<20}")
    print(f"{'Score Range':<30} {'0-1':<25} {llm_data['statistics']['score_range']:<25}")

    # Correlation analysis
    print("\n2. Correlation Analysis")
    print("-" * 80)
    corr_results = analyze_correlation(editclip_data, llm_data)
    print(f"Pearson Correlation:  {corr_results['pearson_correlation']:.4f} (p={corr_results['pearson_p_value']:.4e})")
    print(f"Spearman Correlation: {corr_results['spearman_correlation']:.4f} (p={corr_results['spearman_p_value']:.4e})")

    if corr_results['pearson_correlation'] > 0.7:
        print("\n✓ Strong positive correlation - Both methods agree well")
    elif corr_results['pearson_correlation'] > 0.4:
        print("\n⚠ Moderate positive correlation - Some agreement but notable differences")
    else:
        print("\n✗ Weak correlation - Methods measure different aspects")

    # Disagreement analysis
    print("\n3. Top Disagreements (Normalized score difference > 0.3)")
    print("-" * 80)
    disagreements = find_disagreements(editclip_data, llm_data, threshold=0.3)

    if disagreements:
        print(f"Found {len(disagreements)} significant disagreements:\n")
        for i, item in enumerate(disagreements[:10], 1):  # Show top 10
            print(f"{i}. Sample ID: {item['id']}")
            print(f"   Instruction: {item['instruction']}")
            print(f"   EditCLIP Score: {item['editclip_score']:.4f}")
            print(f"   LLM Score (normalized): {item['llm_score']:.4f}")
            print(f"   Difference: {item['difference']:.4f}")
            print(f"   LLM Reasoning: {item['llm_reasoning'][:150]}...")
            print()
    else:
        print("No significant disagreements found.")

    # Key differences
    print("\n4. Key Differences Between Methods")
    print("-" * 80)
    print("""
EditCLIP Evaluation:
- Metric: CLIP embedding similarity between (source_image, target_image) pair and instruction
- Strengths:
  * Fast and scalable
  * Objective and reproducible
  * No API costs
  * Based on learned visual-text representations
- Limitations:
  * May not capture fine-grained semantic details
  * Limited understanding of complex instructions
  * Cannot reason about context

LLM (Claude) Evaluation:
- Metric: Multi-faceted scoring (instruction compliance, edit quality, preservation)
- Strengths:
  * Deep semantic understanding
  * Can reason about complex edits
  * Provides detailed explanations
  * Considers multiple evaluation aspects
- Limitations:
  * Slower and more expensive
  * May have subjective variations
  * Requires API access
  * Potential hallucination in visual understanding
""")

    print("\n5. Recommendations")
    print("-" * 80)
    print("""
- Use EditCLIP for: Large-scale evaluations, real-time feedback, consistent baselines
- Use LLM for: Detailed analysis, understanding failure cases, human-aligned evaluation
- Best practice: Combine both methods for comprehensive evaluation
  * EditCLIP for quantitative metrics
  * LLM for qualitative insights and edge case analysis
""")

    # Save detailed comparison
    comparison_data = {
        'correlation_analysis': {
            'pearson_correlation': float(corr_results['pearson_correlation']),
            'pearson_p_value': float(corr_results['pearson_p_value']),
            'spearman_correlation': float(corr_results['spearman_correlation']),
            'spearman_p_value': float(corr_results['spearman_p_value'])
        },
        'disagreements': disagreements,
        'sample_comparisons': []
    }

    # Add sample-by-sample comparison
    for id in corr_results['common_ids']:
        editclip_dict = {item['id']: item for item in editclip_data['results']}
        llm_dict = {item['id']: item for item in llm_data['results']}

        comparison_data['sample_comparisons'].append({
            'id': id,
            'instruction': editclip_dict[id]['instruction'],
            'editclip_score': editclip_dict[id]['editclip_score'],
            'llm_score': llm_dict[id]['llm_score'],
            'llm_score_normalized': llm_dict[id]['llm_score'] / 10.0,
            'difference': abs(editclip_dict[id]['editclip_score'] - llm_dict[id]['llm_score'] / 10.0)
        })

    with open('comparison_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("Detailed comparison saved to: comparison_analysis.json")
    print("="*80)

    return corr_results, disagreements

def plot_comparison(editclip_data, llm_data):
    """Create visualization comparing both methods"""
    try:
        corr_results = analyze_correlation(editclip_data, llm_data)

        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(corr_results['editclip_normalized'], corr_results['llm_normalized'], alpha=0.6)
        plt.plot([0, 1], [0, 1], 'r--', label='Perfect Agreement')
        plt.xlabel('EditCLIP Score (normalized)')
        plt.ylabel('LLM Score (normalized)')
        plt.title(f"EditCLIP vs LLM Evaluation\nPearson r={corr_results['pearson_correlation']:.3f}")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Distribution comparison
        plt.subplot(1, 2, 2)
        plt.hist(corr_results['editclip_normalized'], alpha=0.5, bins=15, label='EditCLIP', density=True)
        plt.hist(corr_results['llm_normalized'], alpha=0.5, bins=15, label='LLM', density=True)
        plt.xlabel('Score (normalized)')
        plt.ylabel('Density')
        plt.title('Score Distribution Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('evaluation_comparison.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved to: evaluation_comparison.png")

    except Exception as e:
        print(f"\nNote: Could not generate plot (matplotlib not available or error): {e}")

def main():
    # Load data
    editclip_data, llm_data = load_results()

    # Generate report
    corr_results, disagreements = generate_report(editclip_data, llm_data)

    # Create visualization
    plot_comparison(editclip_data, llm_data)

if __name__ == "__main__":
    main()
