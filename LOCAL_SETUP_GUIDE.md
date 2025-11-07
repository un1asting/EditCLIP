# 本地运行LLM-EditCLIP Correlation测试指南

## 快速开始（10分钟）

### 步骤1: 克隆仓库

```bash
# 克隆您的EditCLIP仓库
git clone https://github.com/un1asting/EditCLIP.git
cd EditCLIP

# 切换到correlation测试分支
git checkout claude/test-llm-editclip-correlation-011CUrUh32wYnBtinfWY3zPk
```

### 步骤2: 安装依赖

```bash
# 创建虚拟环境（推荐）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖包
pip install torch torchvision transformers scipy tqdm pillow

# 或者使用国内镜像加速
pip install torch torchvision transformers scipy tqdm pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 步骤3: 运行测试

```bash
# 方案A: 使用标准CLIP测试（推荐开始）
python3 test_correlation_demo.py

# 方案B: 先快速查看数据（无需等待模型下载）
python3 analyze_data.py
```

就这么简单！🎉

---

## 详细说明

### ✅ 方案A: 使用标准CLIP（无需EditCLIP权重）

**优点**:
- ✅ 无需下载额外模型权重
- ✅ 立即可以运行
- ✅ 完整展示correlation测试流程

**运行**:
```bash
python3 test_correlation_demo.py
```

**预期输出**:
```
============================================================
CLIP vs LLM Correlation Testing (Demo)
============================================================
Device: cuda (或 cpu)
Data: magicbrush_data/data.json

Loading CLIP model (openai/clip-vit-large-patch14)...
✓ CLIP model loaded successfully
✓ Loaded 30 samples from MagicBrush

Evaluating with CLIP
============================================================
CLIP Evaluation: 100%|██████████| 30/30 [00:15<00:00]

Generating Simulated LLM Scores
============================================================
⚠️  Using simulated LLM scores for demonstration

Computing Correlation Metrics
============================================================

Valid samples: 30 / 30

📊 Correlation Results:
  Pearson  r =  0.xxxx  (p = x.xxxe-xx)
  Spearman ρ =  0.xxxx  (p = x.xxxe-xx)

📏 Error Metrics:
  MAE  = 0.xxxx
  RMSE = 0.xxxx

✅ Results saved to correlation_results.json
```

**查看结果**:
```bash
# 查看correlation指标
cat correlation_results.json | jq '.correlation_metrics'

# 查看完整结果
cat correlation_results.json | jq '.'
```

### 🔧 方案B: 使用真实EditCLIP（需要下载权重）

如果您想使用真实的EditCLIP模型：

**步骤1: 下载EditCLIP权重**

访问: https://huggingface.co/QWW/EditCLIP

下载以下文件到 `clip_ckpt/editclip_vit_l_14/`:
- `model.safetensors`
- `config.json`
- 其他配置文件（如果有）

**步骤2: 运行EditCLIP版本**
```bash
python3 test_correlation.py --model_path clip_ckpt/editclip_vit_l_14
```

---

## 📊 理解输出结果

### correlation_results.json 结构

```json
{
  "metadata": {
    "n_samples": 30,
    "model_type": "CLIP (openai/clip-vit-large-patch14)",
    "llm_type": "simulated"
  },
  "correlation_metrics": {
    "n_samples": 30,
    "pearson_r": 0.xxxx,        // Pearson相关系数
    "pearson_p": 0.xxxx,        // 统计显著性
    "spearman_r": 0.xxxx,       // Spearman相关系数
    "spearman_p": 0.xxxx,
    "mae": 0.xxxx,              // 平均绝对误差
    "rmse": 0.xxxx              // 均方根误差
  },
  "clip_results": [
    {
      "sample_id": 0,
      "instruction": "change the table for a dog",
      "edit_score": 0.xxxx,           // CLIP编辑质量分数
      "target_text_sim": 0.xxxx,      // 编辑后图片与指令的相似度
      "source_text_sim": 0.xxxx,      // 原图与指令的相似度
      "source_target_sim": 0.xxxx     // 原图与编辑后的相似度
    },
    ...
  ],
  "llm_results": [
    {
      "sample_id": 0,
      "llm_score": 0.xxxx,            // LLM评估分数
      "llm_model": "simulated"
    },
    ...
  ]
}
```

### 如何解读相关系数

| Pearson r | 含义 | 说明 |
|-----------|------|------|
| 0.7 ~ 1.0 | 强正相关 | CLIP和LLM评估高度一致 |
| 0.4 ~ 0.7 | 中等相关 | 有一定一致性，但存在差异 |
| 0.0 ~ 0.4 | 弱相关 | 评估差异较大 |
| p < 0.05 | 统计显著 | 相关性不是偶然产生的 |

---

## 🎨 可视化结果（可选）

创建散点图查看相关性：

```bash
# 创建可视化脚本
cat > visualize_results.py << 'EOF'
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# 读取结果
with open('correlation_results.json') as f:
    data = json.load(f)

clip_scores = [r['edit_score'] for r in data['clip_results'] if r['edit_score'] is not None]
llm_scores = [r['llm_score'] for r in data['llm_results'] if r['llm_score'] is not None]

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(clip_scores, llm_scores, alpha=0.6, s=100)

# 添加趋势线
z = np.polyfit(clip_scores, llm_scores, 1)
p = np.poly1d(z)
plt.plot(sorted(clip_scores), p(sorted(clip_scores)), "r--", alpha=0.8, label='Trend line')

# 计算相关系数
r, p_val = pearsonr(clip_scores, llm_scores)

plt.xlabel('CLIP Edit Score', fontsize=12)
plt.ylabel('LLM Score', fontsize=12)
plt.title(f'CLIP vs LLM Evaluation Correlation\nPearson r = {r:.3f}, p = {p_val:.4f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300)
print("✅ Plot saved to correlation_plot.png")
plt.show()
EOF

# 安装matplotlib（如果还没有）
pip install matplotlib

# 运行可视化
python3 visualize_results.py
```

---

## 🔄 下一步：集成真实LLM评估

当前使用模拟LLM分数。要使用真实LLM：

### 方案1: 使用OpenAI GPT-4V

```python
# 修改 test_correlation_demo.py 中的 evaluate_with_llm 函数

import openai
import base64

def evaluate_with_llm_gpt4v(samples, api_key):
    """使用GPT-4V评估"""
    openai.api_key = api_key
    results = []

    for sample in samples:
        # 读取图片
        source_path = f"magicbrush_data/{sample['source_image']}"
        target_path = f"magicbrush_data/{sample['target_image']}"

        # Base64编码
        with open(source_path, 'rb') as f:
            source_b64 = base64.b64encode(f.read()).decode()
        with open(target_path, 'rb') as f:
            target_b64 = base64.b64encode(f.read()).decode()

        # 调用GPT-4V
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Rate how well this edit follows the instruction: '{sample['instruction']}'. Source image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{source_b64}"}},
                    {"type": "text", "text": "Edited image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{target_b64}"}},
                    {"type": "text", "text": "Rate from 0 to 1 (0=poor, 1=perfect). Reply with only a number."}
                ]
            }],
            max_tokens=10
        )

        score = float(response.choices[0].message.content.strip())
        results.append({
            'sample_id': sample['id'],
            'llm_score': score,
            'llm_model': 'gpt-4-vision-preview'
        })

    return results
```

### 方案2: 使用Anthropic Claude

```python
import anthropic
import base64

def evaluate_with_llm_claude(samples, api_key):
    """使用Claude评估"""
    client = anthropic.Anthropic(api_key=api_key)
    results = []

    for sample in samples:
        # 读取并编码图片
        source_path = f"magicbrush_data/{sample['source_image']}"
        target_path = f"magicbrush_data/{sample['target_image']}"

        with open(source_path, 'rb') as f:
            source_b64 = base64.b64encode(f.read()).decode()
        with open(target_path, 'rb') as f:
            target_b64 = base64.b64encode(f.read()).decode()

        # 调用Claude
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Rate how well this edit follows the instruction: '{sample['instruction']}'.\n\nSource image:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": source_b64}},
                    {"type": "text", "text": "Edited image:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": target_b64}},
                    {"type": "text", "text": "Rate from 0 to 1 (0=poor edit, 1=perfect edit). Reply with only a number."}
                ]
            }]
        )

        score = float(message.content[0].text.strip())
        results.append({
            'sample_id': sample['id'],
            'llm_score': score,
            'llm_model': 'claude-3-5-sonnet'
        })

    return results
```

使用真实LLM：
```bash
# 设置API key
export OPENAI_API_KEY="your-key-here"
# 或
export ANTHROPIC_API_KEY="your-key-here"

# 运行（需要修改脚本使用上述函数）
python3 test_correlation_demo.py --api_key $OPENAI_API_KEY
```

---

## 💡 常见问题

### Q: 没有GPU怎么办？
A: 脚本会自动检测并使用CPU。速度会慢一些但完全可用。

### Q: 安装很慢？
A: 使用国内镜像：
```bash
pip install torch torchvision -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### Q: 想要更多样本？
A: 运行 `download_magicbrush_50.py` 下载50个样本：
```bash
python3 download_magicbrush_50.py
```

### Q: 如何只测试特定样本？
A: 编辑 `magicbrush_data/data.json`，只保留您想测试的样本。

---

## ✅ 检查清单

开始前确认：
- [ ] Git已安装
- [ ] Python 3.8+ 已安装
- [ ] 已克隆EditCLIP仓库
- [ ] 已切换到正确分支
- [ ] 已安装依赖包
- [ ] magicbrush_data/ 文件夹存在

运行测试：
- [ ] 运行 `python3 test_correlation_demo.py`
- [ ] 查看 correlation_results.json
- [ ] 理解相关性结果

可选进阶：
- [ ] 可视化结果
- [ ] 集成真实LLM
- [ ] 使用EditCLIP权重

---

祝测试顺利！如果遇到问题，可以查看 `README_CORRELATION_TEST.md` 获取更多帮助。🎉
