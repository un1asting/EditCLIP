"""
Visualize EditCLIP evaluation results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

def load_editclip_results():
    """Load EditCLIP results"""
    with open('editclip_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_results(data):
    """Analyze EditCLIP results"""
    results = data['results']
    scores = [r['editclip_score'] for r in results]

    # Sort by score
    results_sorted = sorted(results, key=lambda x: x['editclip_score'], reverse=True)

    print("="*80)
    print("EditCLIP Evaluation Results Analysis")
    print("="*80)

    # Statistics
    print("\n1. Overall Statistics")
    print("-"*80)
    print(f"Number of samples: {data['statistics']['num_samples']}")
    print(f"Mean score: {data['statistics']['mean_score']:.4f}")
    print(f"Std score: {data['statistics']['std_score']:.4f}")
    print(f"Min score: {data['statistics']['min_score']:.4f}")
    print(f"Max score: {data['statistics']['max_score']:.4f}")
    print(f"Median score: {np.median(scores):.4f}")

    # Top samples
    print("\n2. Top 5 Samples (Highest EditCLIP Scores)")
    print("-"*80)
    for i, item in enumerate(results_sorted[:5], 1):
        print(f"{i}. ID {item['id']}: {item['editclip_score']:.4f}")
        print(f"   Instruction: {item['instruction']}")
        print()

    # Bottom samples
    print("3. Bottom 5 Samples (Lowest EditCLIP Scores)")
    print("-"*80)
    for i, item in enumerate(results_sorted[-5:], 1):
        print(f"{i}. ID {item['id']}: {item['editclip_score']:.4f}")
        print(f"   Instruction: {item['instruction']}")
        print()

    # Score distribution
    print("4. Score Distribution")
    print("-"*80)
    bins = [(0.1, 0.15), (0.15, 0.2), (0.2, 0.25), (0.25, 0.3), (0.3, 0.35)]
    for low, high in bins:
        count = sum(1 for s in scores if low <= s < high)
        percentage = count / len(scores) * 100
        bar = '█' * int(percentage / 2)
        print(f"[{low:.2f}, {high:.2f}): {count:2d} samples ({percentage:5.1f}%) {bar}")

    return results_sorted, scores

def create_visualizations(data, results_sorted, scores):
    """Create visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Score distribution histogram
    ax = axes[0, 0]
    ax.hist(scores, bins=15, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.3f}')
    ax.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(scores):.3f}')
    ax.set_xlabel('EditCLIP Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Score by sample ID
    ax = axes[0, 1]
    ids = [r['id'] for r in data['results']]
    sample_scores = [r['editclip_score'] for r in data['results']]
    colors = ['green' if s > 0.25 else 'orange' if s > 0.15 else 'red' for s in sample_scores]
    ax.scatter(ids, sample_scores, c=colors, alpha=0.6, s=50)
    ax.axhline(np.mean(scores), color='blue', linestyle='--', linewidth=1.5, label='Mean', alpha=0.7)
    ax.set_xlabel('Sample ID', fontsize=11)
    ax.set_ylabel('EditCLIP Score', fontsize=11)
    ax.set_title('Scores by Sample ID', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Top and bottom samples
    ax = axes[1, 0]
    top_5 = results_sorted[:5]
    bottom_5 = results_sorted[-5:]
    all_samples = top_5 + bottom_5

    y_pos = np.arange(len(all_samples))
    scores_plot = [s['editclip_score'] for s in all_samples]
    colors_plot = ['green']*5 + ['red']*5

    ax.barh(y_pos, scores_plot, color=colors_plot, alpha=0.7)
    ax.set_yticks(y_pos)
    labels = [f"ID {s['id']}: {s['instruction'][:30]}..." for s in all_samples]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel('EditCLIP Score', fontsize=11)
    ax.set_title('Top 5 & Bottom 5 Samples', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    # 4. Box plot and statistics
    ax = axes[1, 1]
    bp = ax.boxplot([scores], vert=True, patch_artist=True, labels=['EditCLIP'])
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)

    # Add statistics text
    stats_text = f"""
    Statistics:
    ────────────
    Mean:   {np.mean(scores):.4f}
    Median: {np.median(scores):.4f}
    Std:    {np.std(scores):.4f}
    Min:    {np.min(scores):.4f}
    Max:    {np.max(scores):.4f}
    Q1:     {np.percentile(scores, 25):.4f}
    Q3:     {np.percentile(scores, 75):.4f}
    """
    ax.text(1.3, np.mean(scores), stats_text, fontsize=10,
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax.set_ylabel('EditCLIP Score', fontsize=11)
    ax.set_title('Score Statistics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('editclip_analysis.png', dpi=300, bbox_inches='tight')
    print("\n" + "="*80)
    print("Visualization saved to: editclip_analysis.png")
    print("="*80)

def main():
    # Load data
    data = load_editclip_results()

    # Analyze results
    results_sorted, scores = analyze_results(data)

    # Create visualizations
    create_visualizations(data, results_sorted, scores)

if __name__ == "__main__":
    main()
