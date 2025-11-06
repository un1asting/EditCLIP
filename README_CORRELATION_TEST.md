# LLM-EditCLIP Correlation Testing

## 📊 数据准备完成

✅ **MagicBrush数据**: 30个真实样本
- 数据文件: `magicbrush_data/data.json`
- 图片: `magicbrush_data/images/` (60张图片，30对source/target)
- 平均指令长度: 5.9词

## 🔧 可用的测试脚本

### 1. **test_correlation_demo.py** (推荐)
完整的correlation测试脚本，使用标准CLIP模型

**功能**:
- 加载CLIP模型评估所有样本
- 生成LLM评估分数（可模拟或真实API）
- 计算Pearson和Spearman相关系数
- 生成详细分析报告

**运行**:
```bash
# 需要先安装依赖
pip install torch torchvision transformers scipy tqdm

# 运行测试
python3 test_correlation_demo.py

# 指定输出文件
python3 test_correlation_demo.py --output my_results.json
```

**输出**: `correlation_results.json` 包含:
- Pearson相关系数
- Spearman相关系数
- MAE, RMSE等误差指标
- 每个样本的详细分数
- Top/Bottom样本分析

### 2. **test_correlation.py**
使用EditCLIP模型的完整版本（需要下载模型权重）

**使用前需要**:
```bash
# 下载EditCLIP模型权重
# 从 https://huggingface.co/QWW/EditCLIP
# 解压到 clip_ckpt/editclip_vit_l_14/
```

### 3. **analyze_data.py**
快速数据分析脚本（无需PyTorch）

```bash
python3 analyze_data.py
```

## 📈 评估方法

### CLIP/EditCLIP评估
计算编辑质量分数：
```
edit_score = similarity(target_image, instruction) - similarity(source_image, instruction)
```

- 正分数：编辑改善了与指令的对齐
- 负分数：编辑使图像偏离了指令
- 分数越高，编辑质量越好

### LLM评估
可以使用GPT-4 Vision或Claude评估编辑质量：
1. 展示source和target图片
2. 提供editing指令
3. 让LLM打分(0-1或1-10)

**当前状态**: 脚本包含模拟LLM评估。要使用真实API，需要：
- 设置API key
- 实现API调用逻辑（已有模板）

## 📊 相关性指标

脚本会计算：

| 指标 | 说明 | 理想值 |
|-----|------|--------|
| **Pearson r** | 线性相关 | 接近±1 |
| **Spearman ρ** | 单调相关 | 接近±1 |
| **p-value** | 显著性 | <0.05 |
| **MAE** | 平均绝对误差 | 越小越好 |
| **RMSE** | 均方根误差 | 越小越好 |

### 相关系数解释
- **|r| > 0.7**: 强相关
- **0.4 < |r| < 0.7**: 中等相关
- **|r| < 0.4**: 弱相关
- **p < 0.05**: 统计显著

## 🚀 快速开始

```bash
# 1. 安装依赖（如果还没安装）
pip install torch torchvision transformers scipy tqdm

# 2. 运行correlation测试
python3 test_correlation_demo.py

# 3. 查看结果
cat correlation_results.json | jq '.correlation_metrics'
```

## 📝 预期输出示例

```
============================================================
CLIP vs LLM Correlation Testing (Demo)
============================================================
Device: cuda
Data: magicbrush_data/data.json

✓ Loaded 30 samples from MagicBrush
✓ CLIP model loaded successfully

Evaluating with CLIP
============================================================
[进度条显示处理每个样本...]

Generating Simulated LLM Scores
============================================================
⚠️  Using simulated LLM scores for demonstration

Computing Correlation Metrics
============================================================

Valid samples: 30 / 30

📊 Correlation Results:
  Pearson  r =  0.xxxx  (p = x.xxxe-xx)
  Spearman ρ =  0.xxxx  (p = x.xxxe-xx)

📏 Error Metrics (normalized scores):
  MAE  = 0.xxxx
  RMSE = 0.xxxx

📈 Score Statistics:
  CLIP:  mean=0.xxxx, std=0.xxxx, range=[xxx, xxx]
  LLM:   mean=0.xxxx, std=0.xxxx, range=[xxx, xxx]

💡 Interpretation:
  [自动解释相关性强度和显著性]

✅ Top 3 samples with best agreement
  [显示最一致的样本]

❌ Bottom 3 samples with worst agreement
  [显示最不一致的样本]

✅ Results saved to correlation_results.json
```

## 📦 输出文件结构

`correlation_results.json`:
```json
{
  "metadata": {
    "n_samples": 30,
    "model_type": "CLIP (openai/clip-vit-large-patch14)",
    "llm_type": "simulated",
    "device": "cuda"
  },
  "correlation_metrics": {
    "n_samples": 30,
    "pearson_r": 0.xxxx,
    "pearson_p": 0.xxxx,
    "spearman_r": 0.xxxx,
    "spearman_p": 0.xxxx,
    "mae": 0.xxxx,
    "rmse": 0.xxxx
  },
  "clip_results": [...],
  "llm_results": [...]
}
```

## 🔄 下一步

1. **当前**: 使用CLIP + 模拟LLM测试流程
2. **改进1**: 使用EditCLIP模型（下载权重）
3. **改进2**: 集成真实LLM API（GPT-4V/Claude）
4. **分析**: 按编辑类型分析相关性差异

## ⚠️ 注意事项

- CLIP模型首次加载会下载约1.7GB
- GPU推荐但非必需（CPU也可运行）
- 模拟LLM分数仅用于演示流程
- 真实评估需要LLM API access

## 📚 相关资源

- **EditCLIP论文**: https://arxiv.org/abs/2503.20318
- **EditCLIP权重**: https://huggingface.co/QWW/EditCLIP
- **MagicBrush数据集**: https://huggingface.co/datasets/osunlp/MagicBrush
