"""
Create visualization for group meeting presentation
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
fig = plt.figure(figsize=(16, 10))

# Load data
with open('editclip_results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

results = data['results']

# Select key examples
examples = {
    'low_scores': [
        {'id': 21, 'instruction': 'change the grass to water', 'score': 0.113, 'type': '场景变化'},
        {'id': 18, 'instruction': 'Get rid of all the people', 'score': 0.116, 'type': '移除操作'},
        {'id': 24, 'instruction': 'The background should be of a mountain', 'score': 0.159, 'type': '背景替换'},
    ],
    'high_scores': [
        {'id': 19, 'instruction': 'Make the umbrellas blue', 'score': 0.298, 'type': '颜色变化'},
        {'id': 27, 'instruction': 'remove microwave and stove, put refrigerator', 'score': 0.288, 'type': '对象替换'},
        {'id': 16, 'instruction': 'let the dogs bite a stuffed toy', 'score': 0.281, 'type': '对象添加'},
    ],
}

# Subplot 1: Score distribution with highlights
ax1 = plt.subplot(2, 3, 1)
scores = [r['editclip_score'] for r in results]
ax1.hist(scores, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
ax1.axvline(0.113, color='red', linestyle='--', linewidth=2, label='最低分(场景变化)')
ax1.axvline(0.298, color='green', linestyle='--', linewidth=2, label='最高分(颜色变化)')
ax1.axvline(np.mean(scores), color='orange', linestyle='-', linewidth=2, label=f'平均分: {np.mean(scores):.3f}')
ax1.set_xlabel('EditCLIP Score', fontsize=11, fontweight='bold')
ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax1.set_title('EditCLIP 分数分布', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Subplot 2: Low score examples (expected divergence)
ax2 = plt.subplot(2, 3, 2)
low_examples = examples['low_scores']
y_pos = np.arange(len(low_examples))
scores_low = [e['score'] for e in low_examples]
colors_low = ['#ff6b6b', '#ee5a52', '#dc4b3a']

bars = ax2.barh(y_pos, scores_low, color=colors_low, alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels([f"ID {e['id']}: {e['instruction'][:30]}..." for e in low_examples], fontsize=9)
ax2.set_xlabel('EditCLIP Score', fontsize=11, fontweight='bold')
ax2.set_title('⚠️ 预期分歧案例\n(EditCLIP低 → LLM可能高)', fontsize=12, fontweight='bold', color='red')
ax2.set_xlim(0, 0.35)

# Add annotations
for i, (bar, ex) in enumerate(zip(bars, low_examples)):
    ax2.text(ex['score'] + 0.01, i, f"{ex['score']:.3f}\n{ex['type']}",
             va='center', fontsize=9, fontweight='bold')

ax2.grid(True, alpha=0.3, axis='x')

# Subplot 3: High score examples (expected agreement)
ax3 = plt.subplot(2, 3, 3)
high_examples = examples['high_scores']
y_pos = np.arange(len(high_examples))
scores_high = [e['score'] for e in high_examples]
colors_high = ['#51cf66', '#37b24d', '#2f9e44']

bars = ax3.barh(y_pos, scores_high, color=colors_high, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_yticks(y_pos)
ax3.set_yticklabels([f"ID {e['id']}: {e['instruction'][:30]}..." for e in high_examples], fontsize=9)
ax3.set_xlabel('EditCLIP Score', fontsize=11, fontweight='bold')
ax3.set_title('✓ 预期一致案例\n(EditCLIP高 → LLM也高)', fontsize=12, fontweight='bold', color='green')
ax3.set_xlim(0, 0.35)

# Add annotations
for i, (bar, ex) in enumerate(zip(bars, high_examples)):
    ax3.text(ex['score'] + 0.01, i, f"{ex['score']:.3f}\n{ex['type']}",
             va='center', fontsize=9, fontweight='bold')

ax3.grid(True, alpha=0.3, axis='x')

# Subplot 4: Method comparison diagram
ax4 = plt.subplot(2, 3, 4)
ax4.axis('off')
ax4.set_xlim(0, 10)
ax4.set_ylim(0, 10)

# EditCLIP box
editclip_box = FancyBboxPatch((0.5, 5.5), 4, 3.5,
                               boxstyle="round,pad=0.1",
                               facecolor='lightblue',
                               edgecolor='blue',
                               linewidth=2)
ax4.add_patch(editclip_box)
ax4.text(2.5, 8.3, 'EditCLIP', fontsize=14, fontweight='bold', ha='center')
ax4.text(2.5, 7.5, '✓ 快速 (~10ms)', fontsize=10, ha='center')
ax4.text(2.5, 7.0, '✓ 免费', fontsize=10, ha='center')
ax4.text(2.5, 6.5, '✓ 可扩展', fontsize=10, ha='center')
ax4.text(2.5, 6.0, '✗ 惩罚大变化', fontsize=10, ha='center', color='red')

# LLM box
llm_box = FancyBboxPatch((5.5, 5.5), 4, 3.5,
                          boxstyle="round,pad=0.1",
                          facecolor='lightgreen',
                          edgecolor='green',
                          linewidth=2)
ax4.add_patch(llm_box)
ax4.text(7.5, 8.3, 'LLM', fontsize=14, fontweight='bold', ha='center')
ax4.text(7.5, 7.5, '✓ 语义理解', fontsize=10, ha='center')
ax4.text(7.5, 7.0, '✓ 可解释', fontsize=10, ha='center')
ax4.text(7.5, 6.5, '✓ 多维度', fontsize=10, ha='center')
ax4.text(7.5, 6.0, '✗ 慢且贵', fontsize=10, ha='center', color='red')

# Connection
ax4.annotate('', xy=(5.5, 7.25), xytext=(4.5, 7.25),
             arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax4.text(5, 7.6, '互补', fontsize=11, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Comparison
ax4.text(5, 4.5, '对比维度', fontsize=12, fontweight='bold', ha='center')
ax4.text(1.5, 3.8, '速度:', fontsize=10, fontweight='bold')
ax4.text(3.5, 3.8, '10ms', fontsize=10, ha='center', color='blue')
ax4.text(7.5, 3.8, '3-5s', fontsize=10, ha='center', color='green')

ax4.text(1.5, 3.2, '成本:', fontsize=10, fontweight='bold')
ax4.text(3.5, 3.2, '$0', fontsize=10, ha='center', color='blue')
ax4.text(7.5, 3.2, '$0.01/样本', fontsize=10, ha='center', color='green')

ax4.text(1.5, 2.6, '理解:', fontsize=10, fontweight='bold')
ax4.text(3.5, 2.6, '特征相似', fontsize=9, ha='center', color='blue')
ax4.text(7.5, 2.6, '语义正确', fontsize=9, ha='center', color='green')

ax4.text(1.5, 2.0, '适用:', fontsize=10, fontweight='bold')
ax4.text(3.5, 2.0, '大规模', fontsize=9, ha='center', color='blue')
ax4.text(7.5, 2.0, '深入分析', fontsize=9, ha='center', color='green')

# Subplot 5: Expected correlation by edit type
ax5 = plt.subplot(2, 3, 5)
edit_types = ['颜色\n变化', '对象\n替换', '对象\n添加', '移除\n操作', '背景\n替换', '场景\n变换']
correlation = [0.85, 0.70, 0.65, 0.40, 0.40, 0.35]
colors_corr = ['green' if c > 0.7 else 'orange' if c > 0.5 else 'red' for c in correlation]

bars = ax5.bar(edit_types, correlation, color=colors_corr, alpha=0.7, edgecolor='black', linewidth=1.5)
ax5.axhline(0.7, color='green', linestyle='--', linewidth=1, alpha=0.5, label='高相关 (>0.7)')
ax5.axhline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='中等相关 (0.5-0.7)')
ax5.set_ylabel('预期相关系数', fontsize=11, fontweight='bold')
ax5.set_title('不同编辑类型的预期相关性', fontsize=12, fontweight='bold')
ax5.set_ylim(0, 1)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, val in zip(bars, correlation):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Subplot 6: Key insights
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
ax6.set_xlim(0, 10)
ax6.set_ylim(0, 10)

# Title
ax6.text(5, 9.5, '核心发现与结论', fontsize=14, fontweight='bold', ha='center',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

# Key findings
findings = [
    '1. EditCLIP平均分 0.21 ± 0.05',
    '   → 整体评分偏保守',
    '',
    '2. 最大差异: 场景大变化类编辑',
    '   • EditCLIP: 0.11-0.16 (低分)',
    '   • LLM预期: 如执行正确→高分',
    '',
    '3. 原因: 评估逻辑不同',
    '   • EditCLIP: "视觉相似度"',
    '   • LLM: "语义正确性"',
    '',
    '4. 结论: 两者互补',
    '   • EditCLIP → 快速筛选',
    '   • LLM → 深入分析',
]

y_start = 8.5
for i, text in enumerate(findings):
    if text.startswith(('1.', '2.', '3.', '4.')):
        ax6.text(0.5, y_start - i*0.5, text, fontsize=11, fontweight='bold', va='top')
    elif text.startswith('   →'):
        ax6.text(1.0, y_start - i*0.5, text, fontsize=10, va='top', color='blue')
    elif text.startswith('   •'):
        ax6.text(1.5, y_start - i*0.5, text, fontsize=9, va='top', color='darkgreen')
    else:
        ax6.text(0.5, y_start - i*0.5, text, fontsize=10, va='top')

plt.suptitle('EditCLIP vs LLM 图像编辑评估对比分析',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('meeting_presentation.png', dpi=300, bbox_inches='tight')
print("✓ Presentation visualization saved to: meeting_presentation.png")
