# Image Editing Evaluation: EditCLIP vs LLM

本项目提供了两种图像编辑质量评估方法的对比分析。

## 评估方法

### 1. EditCLIP 评估
- **原理**: 使用 CLIP 模型计算编辑前后图像对与文本指令的匹配度
- **输出**: 0-1 范围的相似度分数
- **优势**: 快速、可扩展、客观
- **文件**: `editclip_results.json`

### 2. LLM (Claude) 评估
- **原理**: 使用视觉语言模型直接分析图像和指令
- **输出**: 0-10 分的综合评分 + 详细推理
- **评估维度**:
  - 指令遵循度 (Instruction Compliance)
  - 编辑质量 (Edit Quality)
  - 内容保留度 (Preservation)
- **优势**: 语义理解深入、可解释性强
- **文件**: `llm_results.json` (需要运行评估脚本生成)

## 使用方法

### 步骤 1: 安装依赖

```bash
pip install anthropic pillow numpy scipy matplotlib tqdm
```

### 步骤 2: 运行 LLM 评估

```bash
# 设置 API Key
export ANTHROPIC_API_KEY='your-api-key-here'

# 运行评估
cd magicbrush_data
python llm_evaluation.py
```

这将生成 `llm_results.json` 文件。

### 步骤 3: 对比分析

```bash
python compare_evaluations.py
```

这将生成：
- 终端输出的详细对比报告
- `comparison_analysis.json`: 详细的对比数据
- `evaluation_comparison.png`: 可视化对比图

## 预期结果

### EditCLIP 结果 (已有)
- 30 个样本
- 平均分: 0.2145 ± 0.0496
- 分数范围: 0.113 - 0.298

### LLM 评估结果 (待生成)
- 提供 0-10 分的综合评分
- 包含每个样本的详细推理
- 分析指令遵循、编辑质量、内容保留三个维度

## 分析内容

对比分析脚本将提供：

1. **基础统计**: 两种方法的均值、标准差、范围等
2. **相关性分析**: Pearson 和 Spearman 相关系数
3. **差异案例**: 两种方法评分差异最大的样本
4. **方法对比**: 各自的优势和局限性
5. **使用建议**: 不同场景下的方法选择

## 注意事项

- LLM 评估需要 Anthropic API key (费用约 $0.01-0.02/样本)
- 评估 30 个样本大约需要 1-2 分钟
- 建议先在少量样本上测试
